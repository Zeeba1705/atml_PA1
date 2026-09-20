import argparse
import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from task1.data.stl10 import load_stl10, load_split, IndexedTransformDataset
from task1.models.backbones import load_backbones
from task1.transforms import COMMON_TRANSFORM, GRAYSCALE_TRANSFORM, IMAGENET_NORMALIZE, CLIP_NORMALIZE, make_patch_orders, patch_shuffle
from task1.run_translation import TranslationDataset
from task1.run_patch_shuffle import PatchDataset
from task1.analysis.feature_similarity import extract_features, cosine_stability


class CueStyledDataset(Dataset):
    def __init__(self, metadata, image_dir, normalize, class_to_idx):
        self.metadata = metadata
        self.image_dir = image_dir
        self.normalize = normalize
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, i):
        item = self.metadata[i]
        img = Image.open(os.path.join(self.image_dir, item["filename"])).convert("RGB")
        img = self.normalize(TF.to_tensor(TF.resize(img, [224, 224])))
        return img, self.class_to_idx[item["shape_class"]]


class CueCleanDataset(Dataset):
    def __init__(self, base_dataset, metadata, normalize):
        self.base_dataset = base_dataset
        self.metadata = metadata
        self.normalize = normalize

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, i):
        item = self.metadata[i]
        img, label = self.base_dataset[int(item["shape_idx"])]
        img = self.normalize(TF.to_tensor(TF.resize(img, [224, 224])))
        return img, label


def _save_pair(cache_dir, model_name, intervention, clean, transformed, labels):
    os.makedirs(cache_dir, exist_ok=True)
    torch.save({
        "clean": clean,
        "transformed": transformed,
        "labels": labels,
    }, os.path.join(cache_dir, f"{model_name}_{intervention}.pt"))


def main(data_root, split_path, cue_metadata_path, cue_image_dir, output_path, cache_dir, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test = load_stl10(data_root, download=False)
    indices = load_split(split_path)["test_subset_idx"]
    with open(cue_metadata_path, "r") as f:
        cue_metadata = json.load(f)

    resnet, vit, clip_model = load_backbones(device)
    class_to_idx = {name: i for i, name in enumerate(test.classes)}

    specs = {
        "resnet": (resnet, IMAGENET_NORMALIZE, False),
        "vit": (vit, IMAGENET_NORMALIZE, False),
        "clip": (clip_model, CLIP_NORMALIZE, True),
    }
    results = {name: {} for name in specs}

    for name, (model, norm, is_clip) in specs.items():
        clean_ds = IndexedTransformDataset(test, indices, COMMON_TRANSFORM, norm)
        clean_loader = DataLoader(clean_ds, batch_size=batch_size, shuffle=False, num_workers=2)
        clean_feats, labels = extract_features(model, clean_loader, device, is_clip=is_clip)

        gray_ds = IndexedTransformDataset(test, indices, GRAYSCALE_TRANSFORM, norm)
        gray_loader = DataLoader(gray_ds, batch_size=batch_size, shuffle=False, num_workers=2)
        gray_feats, _ = extract_features(model, gray_loader, device, is_clip=is_clip)
        results[name]["grayscale"] = cosine_stability(clean_feats, gray_feats)
        _save_pair(cache_dir, name, "grayscale", clean_feats, gray_feats, labels)

        trans_scores = {}
        for shift in [8, 16, 32]:
            direction_scores = []
            direction_feats = []
            for dx, dy in [(shift, 0), (-shift, 0), (0, shift), (0, -shift)]:
                ds = TranslationDataset(test, indices, dx, dy, norm)
                loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
                feats, _ = extract_features(model, loader, device, is_clip=is_clip)
                direction_scores.append(cosine_stability(clean_feats, feats))
                direction_feats.append(feats)
            trans_scores[str(shift)] = float(sum(direction_scores) / 4)
            # For visualization, cache the mean feature across the four directions at this displacement.
            mean_shifted = torch.stack(direction_feats, dim=0).mean(dim=0)
            _save_pair(cache_dir, name, f"translation_{shift}", clean_feats, mean_shifted, labels)
        results[name]["translation"] = trans_scores

        orders = make_patch_orders(indices)
        patch_ds = PatchDataset(test, indices, norm, orders)
        patch_loader = DataLoader(patch_ds, batch_size=batch_size, shuffle=False, num_workers=2)
        patch_feats, _ = extract_features(model, patch_loader, device, is_clip=is_clip)
        results[name]["patch_shuffle"] = cosine_stability(clean_feats, patch_feats)
        _save_pair(cache_dir, name, "patch_shuffle", clean_feats, patch_feats, labels)

        cue_clean_ds = CueCleanDataset(test, cue_metadata, norm)
        cue_style_ds = CueStyledDataset(cue_metadata, cue_image_dir, norm, class_to_idx)
        cue_clean_loader = DataLoader(cue_clean_ds, batch_size=batch_size, shuffle=False, num_workers=2)
        cue_style_loader = DataLoader(cue_style_ds, batch_size=batch_size, shuffle=False, num_workers=2)
        cue_clean, cue_labels = extract_features(model, cue_clean_loader, device, is_clip=is_clip)
        cue_styled, _ = extract_features(model, cue_style_loader, device, is_clip=is_clip)
        results[name]["cue_conflict"] = cosine_stability(cue_clean, cue_styled)
        _save_pair(cache_dir, name, "cue_conflict", cue_clean, cue_styled, cue_labels)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--split_path", required=True)
    p.add_argument("--cue_metadata_path", required=True)
    p.add_argument("--cue_image_dir", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    a = p.parse_args()
    main(a.data_root, a.split_path, a.cue_metadata_path, a.cue_image_dir, a.output_path, a.cache_dir, a.batch_size)
