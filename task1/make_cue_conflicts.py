import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from task1.data.stl10 import load_stl10
from task1.configs.cue_conflicts import CUE_PAIRS, KEPT_IDS

SEED = 6304


def build_candidates(test_dataset, per_direction=25):
    rng = np.random.default_rng(SEED)
    class_to_idx = {name: i for i, name in enumerate(test_dataset.classes)}
    targets = np.array(test_dataset.labels)
    by_class = {name: np.where(targets == idx)[0] for name, idx in class_to_idx.items()}
    directed = []
    for a, b in CUE_PAIRS:
        directed.extend([(a, b), (b, a)])

    candidates = []
    for shape_class, texture_class in directed:
        shape_idx = rng.choice(by_class[shape_class], size=per_direction, replace=False)
        texture_idx = rng.choice(by_class[texture_class], size=per_direction, replace=False)
        for s, t in zip(shape_idx, texture_idx):
            candidates.append({
                "shape_class": shape_class,
                "texture_class": texture_class,
                "shape_idx": int(s),
                "texture_idx": int(t),
            })
    return candidates


def load_adain(adain_repo, device):
    import sys
    sys.path.insert(0, adain_repo)
    import net
    from function import adaptive_instance_normalization

    decoder = net.decoder
    vgg = net.vgg
    decoder.load_state_dict(torch.load(os.path.join(adain_repo, "models/decoder.pth"), map_location=device))
    vgg.load_state_dict(torch.load(os.path.join(adain_repo, "models/vgg_normalised.pth"), map_location=device))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    return vgg.to(device).eval(), decoder.to(device).eval(), adaptive_instance_normalization


def transfer(content_img, style_img, vgg, decoder, adain_fn, device, alpha=1.0):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    content = transform(content_img).unsqueeze(0).to(device)
    style = transform(style_img).unsqueeze(0).to(device)
    with torch.no_grad():
        content_f = vgg(content)
        style_f = vgg(style)
        feat = adain_fn(content_f, style_f)
        feat = alpha * feat + (1 - alpha) * content_f
        out = decoder(feat)
    return out.squeeze(0).cpu().clamp(0, 1)


def main(data_root, adain_repo, output_dir, alpha=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test = load_stl10(data_root, download=False)
    vgg, decoder, adain_fn = load_adain(adain_repo, device)
    candidates = build_candidates(test)

    candidate_dir = os.path.join(output_dir, "candidate_images")
    os.makedirs(candidate_dir, exist_ok=True)
    metadata = []

    for i, item in enumerate(tqdm(candidates)):
        content, _ = test[item["shape_idx"]]
        style, _ = test[item["texture_idx"]]
        image = transfer(content, style, vgg, decoder, adain_fn, device, alpha=alpha)
        filename = f"candidate_{i:03d}.png"
        path = os.path.join(candidate_dir, filename)
        save_image(image, path)
        metadata.append({"candidate_id": i, "filename": filename, **item})

    with open(os.path.join(output_dir, "candidate_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    valid = [x for x in metadata if x["candidate_id"] in set(KEPT_IDS)]
    with open(os.path.join(output_dir, "valid_conflicts.json"), "w") as f:
        json.dump(valid, f, indent=2)

    print("Candidates:", len(metadata), "Accepted:", len(valid))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--adain_repo", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--alpha", type=float, default=1.0)
    a = p.parse_args()
    main(a.data_root, a.adain_repo, a.output_dir, a.alpha)
