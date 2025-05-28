import torch
from pathlib import Path
import yaml
import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def get_device():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    return device

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"log_{now}.log")
    logger = logging.getLogger(f"mae_logger_{now}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode="a")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def save_images(predictions, gts, save_dir="output", idx="idx"):
    n = len(predictions)

    pred_np_list = []
    gt_np_list = []

    for pred, gt in zip(predictions, gts):
        pred_np = pred.detach().cpu()
        gt_np = gt.detach().cpu()

        if pred_np.dim() == 4:
            pred_np = pred_np[0]
        if gt_np.dim() == 4:
            gt_np = gt_np[0]

        # Convert from (C, H, W) to (H, W, C)
        pred_np = np.transpose(pred_np.numpy(), (1, 2, 0))
        gt_np = np.transpose(gt_np.numpy(), (1, 2, 0))

        # Normalize if needed
        pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-5)
        gt_np = (gt_np - gt_np.min()) / (gt_np.max() - gt_np.min() + 1e-5)

        pred_np_list.append(pred_np)
        gt_np_list.append(gt_np)

    fig, axs = plt.subplots(n, 2, figsize=(8, 4 * n))

    for i in range(n):
        axs[i, 0].imshow(pred_np_list[i])
        axs[i, 0].set_title(f"Prediction")
        axs[i, 0].axis("off")
        axs[i, 1].imshow(gt_np_list[i])
        axs[i, 1].set_title(f"Ground Truth")
        axs[i, 1].axis("off")
    fig.suptitle(str(idx), fontsize=16)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{idx}.png"))
    plt.close(fig)
