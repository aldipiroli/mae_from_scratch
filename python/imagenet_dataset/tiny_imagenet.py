import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from imagenet_dataset.download_tiny_imagenet import download_tiny_imagenet
from pathlib import Path


class TinyImageNet(Dataset):
    def __init__(self, root, split="train"):
        self.root = Path(root)
        self.split = split
        self.data = []
        self.labels = []
        self.class_to_idx = {}
        self.transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ]
        )
        if not os.path.isdir(self.root / "tiny-imagenet-200"):
            download_tiny_imagenet(self.root)
        self.root = self.root / "tiny-imagenet-200"

        wnids_path = os.path.join(self.root, "wnids.txt")
        with open(wnids_path, "r") as f:
            wnids = [line.strip() for line in f.readlines()]
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}

        if split == "train":
            for wnid in wnids:
                img_dir = os.path.join(self.root, "train", wnid, "images")
                for img_name in os.listdir(img_dir):
                    self.data.append(os.path.join(img_dir, img_name))
                    self.labels.append(self.class_to_idx[wnid])
        elif split == "val":
            val_img_dir = os.path.join(self.root, "val", "images")
            val_annotations = os.path.join(self.root, "val", "val_annotations.txt")
            img_to_wnid = {}
            with open(val_annotations, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    img_to_wnid[parts[0]] = parts[1]
            for img_name in os.listdir(val_img_dir):
                self.data.append(os.path.join(val_img_dir, img_name))
                self.labels.append(self.class_to_idx[img_to_wnid[img_name]])
        else:
            raise ValueError("split must be 'train' or 'val'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
