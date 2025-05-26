import sys
import os
import torch
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.tiny_imagenet import TinyImageNet
from torch.utils.data import DataLoader

def test_tiny_imagenet_dataset():
    dataset = TinyImageNet(root='../data/tiny-imagenet-200', split='train')
    batch_size = 8
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    it = iter(loader)
    images, labels = next(it)
    # assert images.shape == torch.tensor([batch_size, 3, 64, 64])
    # assert labels.shape == torch.tensor([batch_size])
