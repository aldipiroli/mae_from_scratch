import torch
from pathlib import Path


def get_device():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    return device


def save_model(model, model_dir, patch_size):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(model_dir) / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved Model in: {model_path}")