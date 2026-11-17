import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from train import build_model
from Utils.getData import get_loaders


def evaluate(name, test_loader, classes, device):
    ckpt = torch.load(f"./Checkpoints/{name}.pt", map_location=device)
    model = build_model(name, len(classes)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    prediction, ground_truth = [], []
    with torch.no_grad():
        for src, trg in test_loader:
            pred = model(src.to(device))
            prediction.extend(torch.argmax(pred, dim=1).cpu().numpy())
            ground_truth.extend(trg.numpy())

    cf = confusion_matrix(ground_truth, prediction)
    print(f"\n== {name} ==\n{cf}")

    df_cm = pd.DataFrame(cf / np.sum(cf, axis=1)[:, None], index=classes, columns=classes)
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(f"confusion_matrix_{name}.png")
    plt.close()

    acc = accuracy_score(ground_truth, prediction)
    prec, rec, f1, _ = precision_recall_fscore_support(
        ground_truth, prediction, average="weighted", zero_division=0
    )
    return {"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="./Dataset")
    p.add_argument("--models", nargs="+", default=["cnn", "resnet"])
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_loader, classes = get_loaders(a.data, 32)

    results = pd.DataFrame([evaluate(m, test_loader, classes, device) for m in a.models])
    print("\n", results.round(4).to_string(index=False))
    results.to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()
