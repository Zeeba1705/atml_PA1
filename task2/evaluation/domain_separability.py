import torch
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def collect_features(backbone, loader, device):
    backbone.eval()

    features = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            batch_features = backbone(images)

            features.append(
                batch_features.cpu()
            )

    return torch.cat(features, dim=0).numpy()


def domain_separability(backbone,source_val,target_loader,device,seed=6304):
    source_features = []
    for domain in source_val:
        feats= collect_features(
            backbone,
            source_val[domain],
            device
        )

        source_features.append(feats)

    source_features = np.concatenate(source_features,axis=0)

    target_features = collect_features( backbone,target_loader,device)

    sample_size = min(
    source_features.shape[0],
    target_features.shape[0]
)

    rng = np.random.default_rng(seed)

    source_order = rng.permutation(
        source_features.shape[0]
    )

    target_order = rng.permutation(
        target_features.shape[0]
    )

    source_features = source_features[
        source_order[:sample_size]
    ]

    target_features = target_features[
        target_order[:sample_size]
    ]

    X = np.concatenate(
        [source_features, target_features],
        axis=0
    )

    y = np.concatenate(
        [
            np.zeros(sample_size),
            np.ones(sample_size)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=seed,
        stratify=y
    )

    probe = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=seed
    )

    probe.fit(
        X_train,
        y_train
    )

    predictions = probe.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return accuracy