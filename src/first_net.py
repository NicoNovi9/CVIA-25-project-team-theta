import os
import ast
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class SparkDetectionDataset(Dataset):
    def __init__(self, csv_path, image_root, split, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.split = split   # "train" oppure "val"
        self.transform = transform

        # mappa classe -> indice intero
        class_names = sorted(self.df["Class"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # path corretto: images/Class/train/image_xxx_img.jpg
        img_path = os.path.join(
            self.image_root,
            row["Class"],
            self.split,
            row["Image name"]
        )

        img = Image.open(img_path).convert("RGB")

        bbox = torch.tensor(ast.literal_eval(row["Bounding box"]), dtype=torch.float32)
        label = torch.tensor(self.class_to_idx[row["Class"]], dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, bbox, label


if __name__ == "__main__":
    DATA_ROOT = "/project/scratch/p200981/spark2024"

    dataset = SparkDetectionDataset(
        csv_path=f"{DATA_ROOT}/train.csv",
        image_root=f"{DATA_ROOT}/images",
        split="train"
    )

    img, bbox, label = dataset[0]
    print("Image:", img)
    print("BBox:", bbox)
    print("Label:", label)

