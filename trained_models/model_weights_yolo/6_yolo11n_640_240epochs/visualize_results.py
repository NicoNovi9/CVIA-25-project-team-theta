import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("results.csv")

# Ensure required columns exist
required_columns = [
    "epoch",
    "train/box_loss",
    "val/box_loss",
    "train/cls_loss",
    "val/cls_loss",
]

missing = [c for c in required_columns if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# -------- Plot box loss --------
plt.figure()
plt.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss")
plt.plot(df["epoch"], df["val/box_loss"], label="Val Box Loss")
plt.xlabel("Epoch")
plt.ylabel("Box Loss")
plt.title("Train vs Validation Box Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("box_loss_plot.png")

# -------- Plot classification loss --------
plt.figure()
plt.plot(df["epoch"], df["train/cls_loss"], label="Train Cls Loss")
plt.plot(df["epoch"], df["val/cls_loss"], label="Val Cls Loss")
plt.xlabel("Epoch")
plt.ylabel("Classification Loss")
plt.title("Train vs Validation Classification Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cls_loss_plot.png")