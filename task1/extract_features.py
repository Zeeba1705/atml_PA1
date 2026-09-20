import argparse
import os
import torch
from torch.utils.data import DataLoader
from task1.data.stl10 import load_stl10, load_split, IndexedTransformDataset
from task1.models.backbones import load_backbones
from task1.transforms import COMMON_TRANSFORM, IMAGENET_NORMALIZE, CLIP_NORMALIZE
from task1.analysis.feature_similarity import extract_features


def main(data_root, split_path, output_dir, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, _ = load_stl10(data_root, download=False)
    split = load_split(split_path)
    resnet, vit, clip_model = load_backbones(device)

    os.makedirs(output_dir, exist_ok=True)
    specs = {
        "resnet": (resnet, IMAGENET_NORMALIZE, False),
        "vit": (vit, IMAGENET_NORMALIZE, False),
        "clip": (clip_model, CLIP_NORMALIZE, True),
    }

    labels_saved = False
    for name, (model, norm, is_clip) in specs.items():
        for split_name, idx_key in [("train", "train_idx"), ("val", "val_idx")]:
            ds = IndexedTransformDataset(train, split[idx_key], COMMON_TRANSFORM, norm)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
            feats, labels = extract_features(model, loader, device, is_clip=is_clip)
            torch.save(feats, os.path.join(output_dir, f"{name}_{split_name}.pt"))
            if not labels_saved or split_name == "val":
                torch.save(labels, os.path.join(output_dir, f"{split_name}_labels.pt"))
            print(name, split_name, tuple(feats.shape))
        labels_saved = True


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--split_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=64)
    a = p.parse_args()
    main(a.data_root, a.split_path, a.output_dir, a.batch_size)
