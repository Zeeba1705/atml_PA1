import argparse
import os
import torch
import matplotlib.pyplot as plt
from task1.analysis.representation import joint_tsne, joint_umap


def plot_projection(clean_xy, transformed_xy, labels, title, output_path):
    labels = labels.numpy()
    plt.figure(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    for class_id in sorted(set(labels.tolist())):
        mask = labels == class_id
        color = cmap(class_id % 10)
        plt.scatter(clean_xy[mask, 0], clean_xy[mask, 1], s=12, marker="o", alpha=0.55, color=color)
        plt.scatter(transformed_xy[mask, 0], transformed_xy[mask, 1], s=14, marker="x", alpha=0.75, color=color)
    plt.scatter([], [], marker="o", label="Clean")
    plt.scatter([], [], marker="x", label="Transformed")
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main(cache_dir, output_dir, methods, interventions):
    os.makedirs(output_dir, exist_ok=True)
    for model in ["resnet", "vit", "clip"]:
        for intervention in interventions:
            path = os.path.join(cache_dir, f"{model}_{intervention}.pt")
            if not os.path.exists(path):
                print("Missing:", path)
                continue
            obj = torch.load(path, map_location="cpu")
            clean = obj["clean"].float().numpy()
            changed = obj["transformed"].float().numpy()
            labels = obj["labels"]

            if "tsne" in methods:
                cxy, txy = joint_tsne(clean, changed)
                out = os.path.join(output_dir, f"{model}_{intervention}_tsne.png")
                plot_projection(cxy, txy, labels, f"{model.upper()} — {intervention} — t-SNE", out)
                print("Saved:", out)

            if "umap" in methods:
                cxy, txy = joint_umap(clean, changed)
                out = os.path.join(output_dir, f"{model}_{intervention}_umap.png")
                plot_projection(cxy, txy, labels, f"{model.upper()} — {intervention} — UMAP", out)
                print("Saved:", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--methods", nargs="+", default=["tsne", "umap"], choices=["tsne", "umap"])
    p.add_argument("--interventions", nargs="+", default=["grayscale", "cue_conflict", "translation_32", "patch_shuffle"])
    a = p.parse_args()
    main(a.cache_dir, a.output_dir, a.methods, a.interventions)
