import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from task1.data.stl10 import load_stl10, load_split
from task1.models.backbones import load_backbones, make_clip_text_features
from task1.models.linear_heads import load_heads
from task1.transforms import IMAGENET_NORMALIZE, CLIP_NORMALIZE, translate_reflect


class TranslationDataset(Dataset):
    def __init__(self, base_dataset, indices, dx, dy, normalize):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.dx = dx
        self.dy = dy
        self.normalize = normalize

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img, label = self.base_dataset[idx]
        import torchvision.transforms.functional as TF
        img = translate_reflect(img, self.dx, self.dy)
        img = self.normalize(TF.to_tensor(img))
        return img, label


def predict_linear(backbone, head, loader, device, is_clip=False):
    preds, labels_all = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            if is_clip:
                features = backbone.encode_image(images)
                features = features / features.norm(dim=-1, keepdim=True)
            else:
                features = backbone(images)
            preds.append(head(features.float()).argmax(1).cpu())
            labels_all.append(labels)
    return torch.cat(preds), torch.cat(labels_all)


def predict_zeroshot(clip_model, text_features, loader, device):
    preds, labels_all = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = clip_model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            logits = clip_model.logit_scale.exp() * features @ text_features.T
            preds.append(logits.argmax(1).cpu())
            labels_all.append(labels)
    return torch.cat(preds), torch.cat(labels_all)


def main(data_root, split_path, checkpoint_dir, output_path, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test = load_stl10(data_root, download=False)
    split = load_split(split_path)
    test_indices = split["test_subset_idx"]
    resnet, vit, clip_model = load_backbones(device)
    heads = load_heads(checkpoint_dir, device)
    text_features = make_clip_text_features(clip_model, test.classes, device)

    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            summary = json.load(f)
    else:
        summary = {name: {} for name in ["ResNet", "ViT", "CLIP Linear", "CLIP Zero-shot"]}

    directions = {
        "right": lambda d: (d, 0),
        "left": lambda d: (-d, 0),
        "down": lambda d: (0, d),
        "up": lambda d: (0, -d),
    }

    clean_predictions = {}
    specs = {
        "ResNet": (resnet, heads["resnet"], IMAGENET_NORMALIZE, False, False),
        "ViT": (vit, heads["vit"], IMAGENET_NORMALIZE, False, False),
        "CLIP Linear": (clip_model, heads["clip"], CLIP_NORMALIZE, True, False),
        "CLIP Zero-shot": (clip_model, None, CLIP_NORMALIZE, True, True),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for model_name, (backbone, head, norm, is_clip, zero) in specs.items():
        clean_ds = TranslationDataset(test, test_indices, 0, 0, norm)
        clean_loader = DataLoader(clean_ds, batch_size=batch_size, shuffle=False, num_workers=2)
        if zero:
            clean_preds, clean_labels = predict_zeroshot(clip_model, text_features, clean_loader, device)
        else:
            clean_preds, clean_labels = predict_linear(backbone, head, clean_loader, device, is_clip=is_clip)
        clean_predictions[model_name] = clean_preds
        summary[model_name]["0"] = {
            "accuracy": float((clean_preds == clean_labels).float().mean().item()),
            "consistency": 1.0,
        }

        for shift in [8, 16, 32]:
            if str(shift) in summary[model_name]:
                print(model_name, shift, "already done")
                continue
            values = []
            for direction in ["right", "left", "down", "up"]:
                dx, dy = directions[direction](shift)
                ds = TranslationDataset(test, test_indices, dx, dy, norm)
                loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
                if zero:
                    preds, labels = predict_zeroshot(clip_model, text_features, loader, device)
                else:
                    preds, labels = predict_linear(backbone, head, loader, device, is_clip=is_clip)
                acc = (preds == labels).float().mean().item()
                cons = (preds == clean_predictions[model_name]).float().mean().item()
                values.append((acc, cons))
                print(model_name, shift, direction, round(acc, 4), round(cons, 4))

            summary[model_name][str(shift)] = {
                "accuracy": float(np.mean([x[0] for x in values])),
                "consistency": float(np.mean([x[1] for x in values])),
            }
            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2)

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--split_path", required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    a = p.parse_args()
    main(a.data_root, a.split_path, a.checkpoint_dir, a.output_path, a.batch_size)
