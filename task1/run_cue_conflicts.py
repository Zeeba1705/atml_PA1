import argparse
import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from task1.data.stl10 import load_stl10
from task1.models.backbones import load_backbones, make_clip_text_features
from task1.models.linear_heads import load_heads
from task1.transforms import IMAGENET_NORMALIZE, CLIP_NORMALIZE
from task1.evaluation.metrics import cue_metrics


class CueConflictDataset(Dataset):
    def __init__(self, metadata, image_dir, class_to_idx, normalize):
        self.metadata = metadata
        self.image_dir = image_dir
        self.class_to_idx = class_to_idx
        self.normalize = normalize
        self.base = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, i):
        item = self.metadata[i]
        image = Image.open(os.path.join(self.image_dir, item["filename"])).convert("RGB")
        image = self.normalize(self.base(image))
        return image, self.class_to_idx[item["shape_class"]], self.class_to_idx[item["texture_class"]], item["candidate_id"]


def predict_linear(backbone, head, loader, device, is_clip=False):
    preds, shapes, textures, ids = [], [], [], []
    with torch.no_grad():
        for images, shape, texture, candidate_ids in loader:
            images = images.to(device)
            if is_clip:
                feats = backbone.encode_image(images)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            else:
                feats = backbone(images)
            preds.append(head(feats.float()).argmax(1).cpu())
            shapes.append(shape); textures.append(texture); ids.append(candidate_ids)
    return torch.cat(preds), torch.cat(shapes), torch.cat(textures), torch.cat(ids)


def main(data_root, checkpoint_dir, metadata_path, image_dir, output_path, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test = load_stl10(data_root, download=False)
    class_to_idx = {name: i for i, name in enumerate(test.classes)}
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    resnet, vit, clip_model = load_backbones(device)
    heads = load_heads(checkpoint_dir, device)
    text_features = make_clip_text_features(clip_model, test.classes, device)

    im_ds = CueConflictDataset(metadata, image_dir, class_to_idx, IMAGENET_NORMALIZE)
    clip_ds = CueConflictDataset(metadata, image_dir, class_to_idx, CLIP_NORMALIZE)
    im_loader = DataLoader(im_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    clip_loader = DataLoader(clip_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    outputs = {}
    for name, backbone, head, loader, is_clip in [
        ("ResNet", resnet, heads["resnet"], im_loader, False),
        ("ViT", vit, heads["vit"], im_loader, False),
        ("CLIP Linear", clip_model, heads["clip"], clip_loader, True),
    ]:
        preds, shape, texture, ids = predict_linear(backbone, head, loader, device, is_clip)
        outputs[name] = (preds, shape, texture, ids)

    zp, zs, zt, zi = [], [], [], []
    with torch.no_grad():
        for images, shape, texture, ids in clip_loader:
            images = images.to(device)
            feats = clip_model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            logits = clip_model.logit_scale.exp() * feats @ text_features.T
            zp.append(logits.argmax(1).cpu()); zs.append(shape); zt.append(texture); zi.append(ids)
    outputs["CLIP Zero-shot"] = (torch.cat(zp), torch.cat(zs), torch.cat(zt), torch.cat(zi))

    results = {}
    per_example = []
    for name, (preds, shape, texture, ids) in outputs.items():
        results[name] = cue_metrics(preds, shape, texture)
        for p, s, t, cid in zip(preds.tolist(), shape.tolist(), texture.tolist(), ids.tolist()):
            per_example.append({"model": name, "candidate_id": cid, "prediction": test.classes[p], "shape": test.classes[s], "texture": test.classes[t]})

    payload = {"summary": results, "examples": per_example}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--metadata_path", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--batch_size", type=int, default=64)
    a = p.parse_args()
    main(a.data_root, a.checkpoint_dir, a.metadata_path, a.image_dir, a.output_path, a.batch_size)
