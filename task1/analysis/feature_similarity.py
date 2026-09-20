import torch
import torch.nn.functional as F


def extract_features(model, loader, device, is_clip=False):
    feats, labels = [], []
    model.eval()
    with torch.no_grad():
        for images, y in loader:
            images = images.to(device)
            if is_clip:
                x = model.encode_image(images)
                x = x / x.norm(dim=-1, keepdim=True)
            else:
                x = model(images)
            feats.append(x.cpu())
            labels.append(y.cpu())
    return torch.cat(feats), torch.cat(labels)


def cosine_stability(clean_feats, changed_feats):
    return float(F.cosine_similarity(clean_feats, changed_feats, dim=1).mean().item())
