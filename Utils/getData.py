import os

import cv2 as cv
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

EXTS = (".png", ".jpg", ".jpeg", ".bmp")


class getImageLabel(Dataset):
    """Membaca dataset berformat: folder/<nama_kelas>/<gambar>."""

    def __init__(self, folder="./Dataset", size=100):
        self.classes = sorted(
            d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))
        )
        self.images, self.labels = [], []

        for i, name in enumerate(self.classes):
            path = os.path.join(folder, name)
            for f in sorted(os.listdir(path)):
                if not f.lower().endswith(EXTS):
                    continue
                img = cv.imread(os.path.join(path, f))
                if img is None:
                    continue
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                # resize + normalisasi ke rentang 0-1
                img = cv.resize(img, (size, size)).astype(np.float32) / 255.0
                self.images.append(img)
                self.labels.append(i)

    def __getitem__(self, item):
        # (H, W, C) -> (C, H, W) agar langsung bisa masuk Conv2d
        x = torch.from_numpy(self.images[item]).permute(2, 0, 1)
        y = torch.tensor(self.labels[item], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.images)


def get_loaders(folder="./Dataset", batch_size=32, size=100, test_size=0.2, seed=42):
    """Split train/test terpisah (stratified) dengan seed tetap,
    sehingga train.py dan test.py memakai pembagian data yang sama."""
    data = getImageLabel(folder, size)
    idx_train, idx_test = train_test_split(
        np.arange(len(data)),
        test_size=test_size,
        stratify=data.labels,
        random_state=seed,
    )
    train_loader = DataLoader(Subset(data, idx_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(Subset(data, idx_test), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, data.classes
