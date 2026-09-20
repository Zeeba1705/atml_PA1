import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from task1.data.stl10 import load_stl10, load_split, IndexedTransformDataset
from task1.models.backbones import load_backbones, make_clip_text_features
from task1.models.linear_heads import load_heads
from task1.transforms import GRAYSCALE_TRANSFORM, IMAGENET_NORMALIZE, CLIP_NORMALIZE, make_lab_palette_transfer
from task1.analysis.feature_similarity import extract_features
from task1.evaluation.metrics import evaluate_linear_head, classification_metrics, pred_consistency, json_ready


def _evaluate_condition(name, image_transform, test, indices, models, heads, text_features, device, batch_size):
    resnet, vit, clip_model = models
    im_ds = IndexedTransformDataset(test, indices, image_transform, IMAGENET_NORMALIZE)
    clip_ds = IndexedTransformDataset(test, indices, image_transform, CLIP_NORMALIZE)
    im_loader = DataLoader(im_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    clip_loader = DataLoader(clip_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    r_feats, labels = extract_features(resnet, im_loader, device)
    v_feats, _ = extract_features(vit, im_loader, device)
    c_feats, _ = extract_features(clip_model, clip_loader, device, is_clip=True)

    r = evaluate_linear_head(heads["resnet"], r_feats, labels, device)
    v = evaluate_linear_head(heads["vit"], v_feats, labels, device)
    c = evaluate_linear_head(heads["clip"], c_feats, labels, device)

    with torch.no_grad():
        logits = clip_model.logit_scale.exp() * c_feats.float().to(device) @ text_features.T
        probs = torch.softmax(logits, dim=1)
        conf, preds = probs.max(1)
    z = classification_metrics(preds.cpu(), labels, conf.cpu())
    z["predictions"] = preds.cpu()

    return {"ResNet": r, "ViT": v, "CLIP Linear": c, "CLIP Zero-shot": z}


def main(data_root, split_path, checkpoint_dir, clean_path, output_path, reference_index=10, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test = load_stl10(data_root, download=False)
    split = load_split(split_path)
    indices = split["test_subset_idx"]
    models = load_backbones(device)
    heads = load_heads(checkpoint_dir, device)
    text_features = make_clip_text_features(models[2], test.classes, device)

    with open(clean_path, "r") as f:
        clean_metrics = json.load(f)

    # Recompute clean predictions because JSON intentionally does not store full prediction arrays.
    from task1.transforms import COMMON_TRANSFORM
    clean = _evaluate_condition("clean", COMMON_TRANSFORM, test, indices, models, heads, text_features, device, batch_size)

    grayscale = _evaluate_condition("grayscale", GRAYSCALE_TRANSFORM, test, indices, models, heads, text_features, device, batch_size)

    reference_img, _ = test[reference_index]
    palette_transform = make_lab_palette_transfer(reference_img)
    palette = _evaluate_condition("palette", palette_transform, test, indices, models, heads, text_features, device, batch_size)

    final = {"grayscale": {}, "palette": {}, "palette_reference_index": int(reference_index)}
    for condition_name, condition in [("grayscale", grayscale), ("palette", palette)]:
        for model_name in clean:
            final[condition_name][model_name] = {
                **json_ready(condition[model_name]),
                "accuracy_change": float(condition[model_name]["accuracy"] - clean_metrics[model_name]["accuracy"]),
                "consistency": pred_consistency(clean[model_name]["predictions"], condition[model_name]["predictions"]),
            }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final, f, indent=2)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--split_path", required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--clean_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--reference_index", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    a = p.parse_args()
    main(a.data_root, a.split_path, a.checkpoint_dir, a.clean_path, a.output_path, a.reference_index, a.batch_size)
