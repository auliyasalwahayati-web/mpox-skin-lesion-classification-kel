import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam

from Models.CNN import SimpleCNN
from Models.ResNet import ResNet18Classifier
from Utils.getData import get_loaders


def build_model(name, num_classes):
    if name == "cnn":
        return SimpleCNN(input_c=3, num_classes=num_classes, img_size=100, dropout=0.5)
    return ResNet18Classifier(num_classes=num_classes, pretrained=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn", "resnet"], default="cnn")
    p.add_argument("--data", default="./Dataset")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=None)
    a = p.parse_args()

    lr = a.lr or (1e-3 if a.model == "cnn" else 1e-4)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader, classes = get_loaders(a.data, a.batch)
    model = build_model(a.model, len(classes)).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()

    os.makedirs("Checkpoints", exist_ok=True)
    loss_train_all, loss_test_all = [], []

    for epoch in range(a.epochs):
        model.train()
        train_loss = 0.0
        for src, trg in train_loader:
            src, trg = src.to(device), trg.to(device)
            pred = model(src)
            loss = loss_function(pred, trg)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for src, trg in test_loader:
                src, trg = src.to(device), trg.to(device)
                test_loss += loss_function(model(src), trg).item()

        loss_train_all.append(train_loss / len(train_loader))
        loss_test_all.append(test_loss / len(test_loader))
        print(f"[{a.model}] epoch {epoch + 1}/{a.epochs} "
              f"train loss = {loss_train_all[-1]:.4f}, test loss = {loss_test_all[-1]:.4f}")

    torch.save(
        {
            "epoch": a.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "classes": classes,
        },
        f"./Checkpoints/{a.model}.pt",
    )

    plt.plot(range(a.epochs), loss_train_all, color="#931a00", label="Training")
    plt.plot(range(a.epochs), loss_test_all, color="#3399e6", label="Testing")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"./training_{a.model}.png")


if __name__ == "__main__":
    main()
