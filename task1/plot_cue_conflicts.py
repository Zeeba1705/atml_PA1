import argparse
import json
import os
from PIL import Image
import matplotlib.pyplot as plt


def main(results_path, metadata_path, image_dir, output_dir, model="ResNet", n=8):
    with open(results_path) as f:
        results = json.load(f)
    with open(metadata_path) as f:
        metadata = {x["candidate_id"]: x for x in json.load(f)}

    rows = [x for x in results["examples"] if x["model"] == model]
    
    shape = [x for x in rows if x["prediction"] == x["shape"]]
    texture = [x for x in rows if x["prediction"] == x["texture"]]
    other = [x for x in rows if x["prediction"] not in {x["shape"], x["texture"]}]
    chosen = (shape[:max(1, n // 3)] + texture[:max(1, n // 3)] + other[:max(1, n // 3)])
    chosen += [x for x in rows if x not in chosen][:max(0, n - len(chosen))]
    chosen = chosen[:n]

    cols = 4
    rows_n = (len(chosen) + cols - 1) // cols
    plt.figure(figsize=(3.2 * cols, 3.2 * rows_n))
    for i, item in enumerate(chosen):
        meta = metadata[item["candidate_id"]]
        img = Image.open(os.path.join(image_dir, meta["filename"])).convert("RGB")
        ax = plt.subplot(rows_n, cols, i + 1)
        ax.imshow(img)
        ax.set_title(f"shape={item['shape']}\ntexture={item['texture']}\npred={item['prediction']}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"cue_conflict_examples_{model.lower().replace(' ', '_')}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_path", required=True)
    p.add_argument("--metadata_path", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model", default="ResNet")
    p.add_argument("--n", type=int, default=8)
    a = p.parse_args()
    main(a.results_path, a.metadata_path, a.image_dir, a.output_dir, a.model, a.n)
