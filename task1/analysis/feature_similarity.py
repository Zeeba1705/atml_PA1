import torch
import torch.nn.functional as F


def cosine_stability(clean_features,transformed_features):
    if clean_features.shape != transformed_features.shape:
        raise ValueError("Clean and transformed features must have the same shape.")

    similarities= F.cosine_similarity(
        clean_features.float(),
        transformed_features.float(),
        dim=1
    )

    return {
        "mean":similarities.mean().item(),
        "std":similarities.std(unbiased=False).item(),
        "per_example":similarities
    }
