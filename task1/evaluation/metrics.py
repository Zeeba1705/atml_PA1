import torch
from sklearn.metrics import f1_score


def classification_metrics(predictions, labels, confidence=None):
    accuracy = (predictions == labels).float().mean().item()
    macro_f1 = f1_score(labels.numpy(), predictions.numpy(), average="macro")
    result = {"accuracy": float(accuracy), "macro_f1": float(macro_f1)}
    if confidence is not None:
        result["mean_confidence"] = float(confidence.mean().item())
    return result


def evaluate_linear_head(head, features, labels, device):
    head.eval()
    with torch.no_grad():
        logits = head(features.float().to(device))
        probs = torch.softmax(logits, dim=1)
        confidence, predictions = probs.max(dim=1)
    predictions = predictions.cpu()
    confidence = confidence.cpu()
    out = classification_metrics(predictions, labels, confidence)
    out["predictions"] = predictions
    out["confidence"] = confidence
    return out


def pred_consistency(clean, transformed):
    return float((clean == transformed).float().mean().item())


def cue_metrics(predictions, shape_labels, texture_labels):
    shape_mask = predictions == shape_labels
    texture_mask = predictions == texture_labels
    n_shape = int(shape_mask.sum().item())
    n_texture = int(texture_mask.sum().item())
    n_total = len(predictions)
    n_other = n_total - n_shape - n_texture
    denom = n_shape + n_texture
    return {
        "shape": n_shape,
        "texture": n_texture,
        "other": n_other,
        "shape_bias": 100.0 * n_shape / denom if denom else 0.0,
        "coverage": 100.0 * denom / n_total,
    }


def json_ready(result):
    return {k: v for k, v in result.items() if k not in {"predictions", "confidence"}}
