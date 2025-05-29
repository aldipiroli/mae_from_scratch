import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def get_device():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
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


def plot_single_image(tensor, filename):
    import numpy as np
    from PIL import Image

    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)

    img = tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = Image.fromarray((img * 255).astype(np.uint8))
    img.save(filename)


def save_images(predictions, gts, save_dir="output", idx="idx"):
    n = len(predictions)
    if n == 1:
        predictions = [predictions, predictions]
        gts = [gts, gts]
        n = 2

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

    fig, axs = plt.subplots(2, n, figsize=(4 * n, 8))

    for i in range(n):
        axs[0, i].imshow(pred_np_list[i])
        axs[0, i].set_title(f"Prediction {i+1}")
        axs[0, i].axis("off")
        axs[1, i].imshow(gt_np_list[i])
        axs[1, i].set_title(f"Ground Truth {i+1}")
        axs[1, i].axis("off")
    fig.suptitle(str(idx), fontsize=16)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{idx}.png"))
    plt.close(fig)
