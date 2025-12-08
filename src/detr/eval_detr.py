"""
Evaluation script for DETR model with DDP support.
Computes mAP (COCO metrics) and classification metrics on the validation set.
Uses batched inference distributed across multiple GPUs for efficiency.

Supports multi-node evaluation via torchrun or srun.
"""

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
import json
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformers import AutoModelForObjectDetection, AutoImageProcessor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataset_detr import SparkDetectionDatasetDETR, collate_fn_detr

import warnings
warnings.filterwarnings("ignore")


def setup_distributed():
    """Initialize distributed environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        # On this cluster, each task sees exactly one GPU via CUDA_VISIBLE_DEVICES
        # so the only valid device index is 0 in each process.
        local_rank = 0
    else:
        return 0, 1, 0, False

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank, True



def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def gather_objects(local_data, world_size):
    """Gather Python objects from all ranks to rank 0."""
    if world_size == 1:
        return local_data
    
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_data)
    
    # Flatten the list of lists
    result = []
    for data in gathered:
        result.extend(data)
    return result


def main():
    # =========================================================================
    # DISTRIBUTED SETUP
    # =========================================================================
    rank, world_size, local_rank, is_distributed = setup_distributed()
    is_main = (rank == 0)
    
    if is_main:
        print("=" * 70)
        print("DETR DISTRIBUTED EVALUATION")
        print("=" * 70)
        print(f"World size: {world_size} GPUs")
        print(f"Distributed: {is_distributed}")
        print("=" * 70 + "\n")
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    MODEL_PATH = "model_weights/detr_best"
    DATA_ROOT = "/project/scratch/p200981/spark2024"
    VAL_CSV = f"{DATA_ROOT}/val.csv"
    IMAGE_ROOT = f"{DATA_ROOT}/images"
    OUTPUT_DIR = "evaluation_results"
    CLASSIFICATION_THRESHOLD = 0.3
    BATCH_SIZE = 64  # Per GPU
    NUM_WORKERS = 4
    
    if is_main:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main:
        print(f"Using device: {device}")
    
    # =========================================================================
    # LOAD MODEL
    # =========================================================================
    if is_main:
        print(f"\nLoading model from {MODEL_PATH}...")
    
    model = AutoModelForObjectDetection.from_pretrained(MODEL_PATH)
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    processor.do_resize = True
    processor.size = {
        "shortest_edge": 800,
        "longest_edge": 1333,
    }
    model.to(device)
    model.eval()
    
    if is_main:
        print("✓ Model loaded and ready\n")
    
    # Get label mappings from model config
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {v: int(k) for k, v in model.config.id2label.items()}
    num_classes = len(id2label)
    
    if is_main:
        print(f"Number of classes: {num_classes}")
        print(f"id2label: {id2label}")
    
    # =========================================================================
    # LOAD VALIDATION DATASET
    # =========================================================================
    if is_main:
        print(f"\nLoading validation dataset...")
    
    val_dataset = SparkDetectionDatasetDETR(
        csv_path=VAL_CSV,
        image_root=IMAGE_ROOT,
        split="val",
        image_processor=processor
    )
    
    # Use DistributedSampler to split data across GPUs
    if is_distributed:
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False  # Important: keep order for correct index mapping
        )
    else:
        val_sampler = None
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn_detr,
        sampler=val_sampler
    )
    
    # Load CSV for ground truth info (all ranks need this for index mapping)
    val_df = pd.read_csv(VAL_CSV)
    
    if is_main:
        print(f"Total validation samples: {len(val_dataset)}")
        print(f"Samples per GPU: ~{len(val_dataset) // world_size}")
        print(f"Batches per GPU: {len(val_loader)}")
    
    # =========================================================================
    # BATCHED INFERENCE (each GPU processes its subset)
    # =========================================================================
    if is_main:
        print("\nRunning distributed inference on validation set...")
    
    # Each rank collects its local results
    local_results = []  # List of dicts with global_idx, gt_class, pred_class, coco_image, coco_ann, coco_dets
    
    # Get the indices this rank will process
    if is_distributed:
        # DistributedSampler provides indices
        sampler_indices = list(val_sampler)
    else:
        sampler_indices = list(range(len(val_dataset)))
    
    local_idx = 0  # Track position in sampler_indices
    
    with torch.no_grad():
        iterator = tqdm(val_loader, desc=f"[Rank {rank}] Inference", disable=not is_main)
        for batch_idx, batch in enumerate(iterator):
            pixel_values = batch["pixel_values"].to(device)
            labels_batch = batch["labels"]
            batch_size_actual = pixel_values.shape[0]
            
            # Forward pass
            outputs = model(pixel_values=pixel_values)
            
            # Process each sample in batch
            for i in range(batch_size_actual):
                # Get global index from sampler
                global_idx = sampler_indices[local_idx]
                local_idx += 1
                
                sample_labels = labels_batch[i]
                
                # Get ground truth info from CSV using global index
                row = val_df.iloc[global_idx]
                gt_class_name = row["Class"]
                gt_class_id = label2id.get(gt_class_name, -1)
                
                bbox = ast.literal_eval(row["Bounding box"])
                xmin, ymin, xmax, ymax = bbox
                box_w = xmax - xmin
                box_h = ymax - ymin
                
                # Get original image size from labels (orig_size is [H, W])
                orig_size = sample_labels.get("orig_size", None)
                if orig_size is not None:
                    img_h, img_w = orig_size.tolist()
                else:
                    img_h, img_w = pixel_values.shape[2], pixel_values.shape[3]
                
                # COCO ground truth
                coco_image = {
                    "id": global_idx,
                    "file_name": row["Image name"],
                    "width": int(img_w),
                    "height": int(img_h),
                }
                
                coco_ann = {
                    "id": global_idx,  # Will be reassigned later
                    "image_id": global_idx,
                    "category_id": gt_class_id,
                    "bbox": [float(xmin), float(ymin), float(box_w), float(box_h)],
                    "area": float(box_w * box_h),
                    "iscrowd": 0
                }
                
                # Post-process predictions for this sample
                single_outputs = type(outputs)(
                    logits=outputs.logits[i:i+1],
                    pred_boxes=outputs.pred_boxes[i:i+1]
                )
                
                target_sizes = torch.tensor([[img_h, img_w]]).to(device)
                results = processor.post_process_object_detection(
                    single_outputs,
                    target_sizes=target_sizes,
                    threshold=0.0
                )[0]
                
                # Store all detections for COCO mAP
                coco_dets = []
                for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
                    x1, y1, x2, y2 = box.tolist()
                    coco_dets.append({
                        "image_id": global_idx,
                        "category_id": int(label.item()),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score.item()),
                    })
                
                # For classification: get best prediction above threshold
                mask = results["scores"] >= CLASSIFICATION_THRESHOLD
                if mask.any():
                    filtered_scores = results["scores"][mask]
                    filtered_labels = results["labels"][mask]
                    best_idx = filtered_scores.argmax()
                    pred_class = int(filtered_labels[best_idx].item())
                else:
                    pred_class = -1
                
                local_results.append({
                    "global_idx": global_idx,
                    "gt_class": gt_class_id,
                    "pred_class": pred_class,
                    "coco_image": coco_image,
                    "coco_ann": coco_ann,
                    "coco_dets": coco_dets,
                })
    
    if is_main:
        print(f"\n[Rank 0] Processed {len(local_results)} samples locally")
    
    # =========================================================================
    # GATHER RESULTS FROM ALL RANKS
    # =========================================================================
    if is_distributed:
        if is_main:
            print("\nGathering results from all GPUs...")
        dist.barrier()
        all_results = gather_objects(local_results, world_size)
        if is_main:
            print(f"✓ Gathered {len(all_results)} total results")
    else:
        all_results = local_results
    
    # =========================================================================
    # ONLY RANK 0 COMPUTES FINAL METRICS
    # =========================================================================
    if is_main:
        # Sort by global_idx to ensure consistent ordering
        all_results.sort(key=lambda x: x["global_idx"])
        
        # Build final structures
        coco_images = []
        coco_annotations = []
        coco_detections = []
        gt_classes = []
        pred_classes = []
        
        ann_id = 1
        for r in all_results:
            gt_classes.append(r["gt_class"])
            pred_classes.append(r["pred_class"])
            coco_images.append(r["coco_image"])
            
            ann = r["coco_ann"]
            ann["id"] = ann_id
            coco_annotations.append(ann)
            ann_id += 1
            
            coco_detections.extend(r["coco_dets"])
        
        # =====================================================================
        # SAVE COCO FILES
        # =====================================================================
        gt_file = os.path.join(OUTPUT_DIR, "coco_gt.json")
        dt_file = os.path.join(OUTPUT_DIR, "coco_dt.json")
        
        categories = [{"id": i, "name": id2label[i]} for i in sorted(id2label.keys())]
        
        coco_dict = {
            "info": {"description": "SparkDetectionDataset Validation"},
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": categories
        }
        
        with open(gt_file, "w") as f:
            json.dump(coco_dict, f)
        
        with open(dt_file, "w") as f:
            json.dump(coco_detections, f)
        
        print(f"\n  GT: {len(coco_images)} images, {len(coco_annotations)} annotations")
        print(f"  Predictions: {len(coco_detections)} detections")
        print("✓ COCO files created\n")
        
        # =====================================================================
        # COCO mAP EVALUATION
        # =====================================================================
        print("=" * 70)
        print("COCO mAP EVALUATION")
        print("=" * 70)
        
        coco_gt = COCO(gt_file)
        coco_dt = coco_gt.loadRes(dt_file)
        
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        print("\n================== DETECTION RESULTS ==================")
        print(f"mAP@50-95  (AP)      : {coco_eval.stats[0]:.4f}")
        print(f"mAP@50     (AP50)    : {coco_eval.stats[1]:.4f}")
        print(f"mAP@75     (AP75)    : {coco_eval.stats[2]:.4f}")
        print(f"mAR@1      (AR1)     : {coco_eval.stats[6]:.4f}")
        print(f"mAR@10     (AR10)    : {coco_eval.stats[7]:.4f}")
        print(f"mAR@100    (AR100)   : {coco_eval.stats[8]:.4f}")
        print("========================================================\n")
        
        # =====================================================================
        # CLASSIFICATION METRICS
        # =====================================================================
        print("=" * 70)
        print("CLASSIFICATION METRICS")
        print(f"(Each image -> 1 predicted class, threshold={CLASSIFICATION_THRESHOLD})")
        print("=" * 70)
        
        gt_classes = np.array(gt_classes)
        pred_classes = np.array(pred_classes)
        
        all_class_ids = sorted(id2label.keys())
        class_names = [id2label[i] for i in all_class_ids]
        
        # Per-Class Metrics
        print(f"\n{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 62)
        
        class_precisions = []
        class_recalls = []
        class_f1s = []
        class_supports = []
        
        for cls_id in all_class_ids:
            class_name = id2label[cls_id]
            
            tp = np.sum((gt_classes == cls_id) & (pred_classes == cls_id))
            fp = np.sum((gt_classes != cls_id) & (pred_classes == cls_id))
            fn = np.sum((gt_classes == cls_id) & (pred_classes != cls_id))
            
            support = np.sum(gt_classes == cls_id)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_precisions.append(precision)
            class_recalls.append(recall)
            class_f1s.append(f1)
            class_supports.append(support)
            
            print(f"{class_name:<20} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10}")
        
        print("-" * 62)
        
        # Global Metrics
        global_accuracy = np.sum(gt_classes == pred_classes) / len(gt_classes)
        
        macro_precision = np.mean(class_precisions)
        macro_recall = np.mean(class_recalls)
        macro_f1 = np.mean(class_f1s)
        
        total_support = np.sum(class_supports)
        weighted_precision = np.sum(np.array(class_precisions) * np.array(class_supports)) / total_support
        weighted_recall = np.sum(np.array(class_recalls) * np.array(class_supports)) / total_support
        weighted_f1 = np.sum(np.array(class_f1s) * np.array(class_supports)) / total_support
        
        print(f"\n{'GLOBAL METRICS':^62}")
        print("=" * 62)
        print(f"{'Global Accuracy:':<30} {global_accuracy:.4f}")
        print(f"{'Macro Precision:':<30} {macro_precision:.4f}")
        print(f"{'Macro Recall:':<30} {macro_recall:.4f}")
        print(f"{'Macro F1:':<30} {macro_f1:.4f}")
        print("-" * 62)
        print(f"{'Weighted Precision:':<30} {weighted_precision:.4f}")
        print(f"{'Weighted Recall:':<30} {weighted_recall:.4f}")
        print(f"{'Weighted F1:':<30} {weighted_f1:.4f}")
        print("=" * 62)
        
        no_detection_count = np.sum(pred_classes == -1)
        print(f"\nTotal images: {len(gt_classes)}")
        print(f"Correct: {np.sum(gt_classes == pred_classes)} | "
              f"Incorrect: {np.sum((gt_classes != pred_classes) & (pred_classes != -1))} | "
              f"No detection: {no_detection_count}")
        print(f"Detection rate: {(len(gt_classes) - no_detection_count) / len(gt_classes):.4f}")
        
        # =====================================================================
        # CONFUSION MATRIX
        # =====================================================================
        print("\n" + "=" * 70)
        print("CONFUSION MATRIX")
        print("=" * 70)
        
        display_labels = class_names + ["NoDetection"]
        all_class_ids_extended = all_class_ids + [-1]
        
        cm = confusion_matrix(gt_classes, pred_classes, labels=all_class_ids_extended)
        
        fig, ax = plt.subplots(figsize=(14, 12))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(ax=ax, cmap='Blues', values_format='d', xticks_rotation=45)
        ax.set_title(f"Confusion Matrix (threshold={CLASSIFICATION_THRESHOLD})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Confusion matrix saved to {cm_path}")
        
        # =====================================================================
        # SAVE METRICS TO FILE
        # =====================================================================
        metrics_file = os.path.join(OUTPUT_DIR, "metrics.txt")
        with open(metrics_file, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("DETR EVALUATION RESULTS\n")
            f.write(f"Distributed evaluation with {world_size} GPUs\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("DETECTION METRICS (COCO)\n")
            f.write("-" * 70 + "\n")
            f.write(f"mAP@50-95  (AP)      : {coco_eval.stats[0]:.4f}\n")
            f.write(f"mAP@50     (AP50)    : {coco_eval.stats[1]:.4f}\n")
            f.write(f"mAP@75     (AP75)    : {coco_eval.stats[2]:.4f}\n")
            f.write(f"mAR@1      (AR1)     : {coco_eval.stats[6]:.4f}\n")
            f.write(f"mAR@10     (AR10)    : {coco_eval.stats[7]:.4f}\n")
            f.write(f"mAR@100    (AR100)   : {coco_eval.stats[8]:.4f}\n\n")
            
            f.write("CLASSIFICATION METRICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Classification Threshold: {CLASSIFICATION_THRESHOLD}\n\n")
            
            f.write(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}\n")
            f.write("-" * 62 + "\n")
            for i, cls_id in enumerate(all_class_ids):
                f.write(f"{id2label[cls_id]:<20} {class_precisions[i]:>10.4f} {class_recalls[i]:>10.4f} "
                        f"{class_f1s[i]:>10.4f} {class_supports[i]:>10}\n")
            f.write("-" * 62 + "\n\n")
            
            f.write("GLOBAL METRICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Global Accuracy:     {global_accuracy:.4f}\n")
            f.write(f"Macro Precision:     {macro_precision:.4f}\n")
            f.write(f"Macro Recall:        {macro_recall:.4f}\n")
            f.write(f"Macro F1:            {macro_f1:.4f}\n")
            f.write(f"Weighted Precision:  {weighted_precision:.4f}\n")
            f.write(f"Weighted Recall:     {weighted_recall:.4f}\n")
            f.write(f"Weighted F1:         {weighted_f1:.4f}\n\n")
            
            f.write(f"Total images: {len(gt_classes)}\n")
            f.write(f"Correct: {np.sum(gt_classes == pred_classes)}\n")
            f.write(f"Incorrect: {np.sum((gt_classes != pred_classes) & (pred_classes != -1))}\n")
            f.write(f"No detection: {no_detection_count}\n")
            f.write(f"Detection rate: {(len(gt_classes) - no_detection_count) / len(gt_classes):.4f}\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"✓ Metrics saved to {metrics_file}")
        print("\n✅ Distributed evaluation complete!")
    
    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
