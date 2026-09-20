import argparse
import json
import os
import matplotlib.pyplot as plt


def main(results_path, output_dir):
    with open(results_path, "r") as f:
        results = json.load(f)
    os.makedirs(output_dir, exist_ok=True)
    shifts = [0, 8, 16, 32]

    for metric, ylabel, filename in [
        ("accuracy", "Accuracy", "translation_accuracy.png"),
        ("consistency", "Prediction consistency", "translation_consistency.png"),
    ]:
        plt.figure(figsize=(7, 4.5))
        for model, values in results.items():
            ys = [values[str(x)][metric] for x in shifts]
            plt.plot(shifts, ys, marker="o", label=model)
        plt.xlabel("Translation (pixels)")
        plt.ylabel(ylabel)
        plt.xticks(shifts)
        plt.legend()
        plt.tight_layout()
        path = os.path.join(output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved:", path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_path", required=True)
    p.add_argument("--output_dir", required=True)
    a = p.parse_args()
    main(a.results_path, a.output_dir)
