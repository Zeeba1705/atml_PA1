import argparse
import os
import torch
from task1.models.linear_heads import train_linear_head, save_head_and_history, MODEL_DIMS


def main(feature_dir, checkpoint_dir, results_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_labels = torch.load(os.path.join(feature_dir, "train_labels.pt"), map_location="cpu")
    val_labels = torch.load(os.path.join(feature_dir, "val_labels.pt"), map_location="cpu")

    for name in ["resnet", "vit", "clip"]:
        train_features = torch.load(os.path.join(feature_dir, f"{name}_train.pt"), map_location="cpu")
        val_features = torch.load(os.path.join(feature_dir, f"{name}_val.pt"), map_location="cpu")
        head, history = train_linear_head(
            train_features, train_labels, val_features, val_labels,
            input_dim=MODEL_DIMS[name], device=device,
        )
        save_head_and_history(head, history, name, checkpoint_dir, results_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--feature_dir", required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--results_dir", required=True)
    a = p.parse_args()
    main(a.feature_dir, a.checkpoint_dir, a.results_dir)
