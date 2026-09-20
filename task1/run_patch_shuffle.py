import argparse
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from task1.data.stl10 import load_stl10, load_split
from task1.models.backbones import load_backbones, make_clip_text_features
from task1.models.linear_heads import load_heads
from task1.transforms import IMAGENET_NORMALIZE, CLIP_NORMALIZE, make_patch_orders, patch_shuffle
from task1.run_translation import predict_linear, predict_zeroshot, TranslationDataset


class PatchDataset(Dataset):
    def __init__(self, dataset, indices, normalize, patch_orders):
        self.dataset = dataset
        self.indices = list(indices)
        self.normalize = normalize
        self.patch_orders = patch_orders

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img, label = self.dataset[idx]
        img = patch_shuffle(img, self.patch_orders[int(idx)])
        return self.normalize(img), label


def main(data_root, split_path, checkpoint_dir, output_path, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test = load_stl10(data_root, download=False)
    indices = load_split(split_path)["test_subset_idx"]
    orders = make_patch_orders(indices)
    resnet, vit, clip_model = load_backbones(device)
    heads = load_heads(checkpoint_dir, device)
    text_features = make_clip_text_features(clip_model, test.classes, device)

    im_patch = DataLoader(PatchDataset(test, indices, IMAGENET_NORMALIZE, orders), batch_size=batch_size, shuffle=False, num_workers=2)
    clip_patch = DataLoader(PatchDataset(test, indices, CLIP_NORMALIZE, orders), batch_size=batch_size, shuffle=False, num_workers=2)
    clean_im = DataLoader(TranslationDataset(test, indices, 0, 0, IMAGENET_NORMALIZE), batch_size=batch_size, shuffle=False, num_workers=2)
    clean_clip = DataLoader(TranslationDataset(test, indices, 0, 0, CLIP_NORMALIZE), batch_size=batch_size, shuffle=False, num_workers=2)

    clean = {}
    clean["ResNet"], labels = predict_linear(resnet, heads["resnet"], clean_im, device)
    clean["ViT"], _ = predict_linear(vit, heads["vit"], clean_im, device)
    clean["CLIP Linear"], _ = predict_linear(clip_model, heads["clip"], clean_clip, device, is_clip=True)
    clean["CLIP Zero-shot"], _ = predict_zeroshot(clip_model, text_features, clean_clip, device)

    patch = {}
    patch["ResNet"], labels = predict_linear(resnet, heads["resnet"], im_patch, device)
    patch["ViT"], _ = predict_linear(vit, heads["vit"], im_patch, device)
    patch["CLIP Linear"], _ = predict_linear(clip_model, heads["clip"], clip_patch, device, is_clip=True)
    patch["CLIP Zero-shot"], _ = predict_zeroshot(clip_model, text_features, clip_patch, device)

    results = {}
    for name in patch:
        clean_acc = (clean[name] == labels).float().mean().item()
        patch_acc = (patch[name] == labels).float().mean().item()
        results[name] = {
            "patch_accuracy": float(patch_acc),
            "accuracy_drop": float(clean_acc - patch_acc),
            "consistency": float((patch[name] == clean[name]).float().mean().item()),
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--split_path", required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    a = p.parse_args()
    main(a.data_root, a.split_path, a.checkpoint_dir, a.output_path, a.batch_size)
