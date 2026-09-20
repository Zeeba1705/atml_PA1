import argparse
import json
import os
import matplotlib.pyplot as plt


def main(clean_path, color_path, patch_path, output_dir):
    with open(clean_path) as f: clean = json.load(f)
    with open(color_path) as f: color = json.load(f)
    with open(patch_path) as f: patch = json.load(f)
    os.makedirs(output_dir, exist_ok=True)

    models = list(clean.keys())
    conditions = ["clean", "grayscale", "palette", "patch_shuffle"]
    values = {
        "clean": [clean[m]["accuracy"] for m in models],
        "grayscale": [color["grayscale"][m]["accuracy"] for m in models],
        "palette": [color["palette"][m]["accuracy"] for m in models],
        "patch_shuffle": [patch[m]["patch_accuracy"] for m in models],
    }

    x = list(range(len(models)))
    width = 0.18
    plt.figure(figsize=(9, 4.8))
    for j, condition in enumerate(conditions):
        offset = (j - 1.5) * width
        plt.bar([v + offset for v in x], values[condition], width=width, label=condition.replace("_", " "))
    plt.xticks(x, models)
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "intervention_accuracy.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clean_path", required=True)
    p.add_argument("--color_path", required=True)
    p.add_argument("--patch_path", required=True)
    p.add_argument("--output_dir", required=True)
    a = p.parse_args()
    main(a.clean_path, a.color_path, a.patch_path, a.output_dir)
