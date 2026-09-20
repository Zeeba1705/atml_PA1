import numpy as np
from sklearn.manifold import TSNE


def joint_tsne(clean_features, transformed_features, seed=6304,
               perplexity=30, learning_rate="auto", init="pca"):
    combined = np.concatenate([clean_features, transformed_features], axis=0)
    projector = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=perplexity,
        learning_rate=learning_rate,
        init=init,
    )
    coords = projector.fit_transform(combined)
    n = len(clean_features)
    return coords[:n], coords[n:]


def joint_umap(clean_features, transformed_features, seed=6304,
               n_neighbors=15, min_dist=0.1, metric="cosine"):
    try:
        import umap
    except ImportError as e:
        raise ImportError("Install UMAP with: pip install umap-learn") from e

    combined = np.concatenate([clean_features, transformed_features], axis=0)
    projector = umap.UMAP(
        n_components=2,
        random_state=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
    )
    coords = projector.fit_transform(combined)
    n = len(clean_features)
    return coords[:n], coords[n:]
