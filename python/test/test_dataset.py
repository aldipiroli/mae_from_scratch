import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.tiny_imagenet import TinyImageNet
from torch.utils.data import DataLoader

def test_tiny_imagenet_dataset():
    for curr_split in ["train", "val"]:
        dataset = TinyImageNet(root="../data", split=curr_split)
        batch_size = 8
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        it = iter(loader)
        images, labels = next(it)
        assert images.shape == (batch_size, 3, 64, 64)
        assert labels.shape == (batch_size,)
