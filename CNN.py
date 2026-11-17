import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Simple CNN dari slide: 2 blok Conv-ReLU-MaxPool + 1 FC layer.
    Output berupa logits (CrossEntropyLoss sudah memuat log-softmax)."""

    def __init__(self, input_c=3, num_classes=5, img_size=100, dropout=0.5):
        super().__init__()
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(input_c, 20, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
        )
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(20, 50, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
        )
        # hitung ukuran flatten otomatis (img 100 -> 50*9*9)
        with torch.no_grad():
            n = self.convBlock2(self.convBlock1(torch.zeros(1, input_c, img_size, img_size))).numel()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n, num_classes)

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.dropout(torch.flatten(x, 1))
        return self.fc(x)
