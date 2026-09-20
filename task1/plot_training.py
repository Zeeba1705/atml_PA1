import argparse
import json
import os
import matplotlib.pyplot as plt


def main(results_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for name in ["resnet", "vit", "clip"]:
        path = os.path.join(results_dir, f"{name}_history.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            h = json.load(f)
        epochs = range(1, len(h["train_acc"]) + 1)
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, h["train_acc"], label="Train")
        plt.plot(epochs, h["val_acc"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{name.upper()} linear head")
        plt.legend()
        plt.tight_layout()
        out = os.path.join(output_dir, f"{name}_head_training.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved:", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--output_dir", required=True)
    a = p.parse_args()
    main(a.results_dir, a.output_dir)
