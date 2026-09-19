"""
Quick diagnostic: does raw (unnormalized) backbone feature magnitude
differ systematically by domain? No training involved — just loads an
existing checkpoint and reports mean L2 norm of raw features per domain.

Usage:
    python -m task2.check_feature_norms --checkpoint_path <path_to_dann_best.pt> --pacs_root <path>
"""
import argparse
import numpy as np
import torch

from shared.pacs_protocol import make_pacs_loaders, SOURCE_DOMAINS
from task2.models.backbone import ResNetBackbone
from task2.evaluation.domain_separability import collect_features


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, source_val, _, target_eval = make_pacs_loaders(pacs_root=args.pacs_root)

    backbone = ResNetBackbone().to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    backbone.load_state_dict(checkpoint["backbone"])
    backbone.eval()

    print(f"Checkpoint: {args.checkpoint_path} (epoch {checkpoint.get('epoch', '?')})\n")

    norms_by_domain = {}
    for domain in SOURCE_DOMAINS:
        feats = collect_features(backbone, source_val[domain], device)
        norms = np.linalg.norm(feats, axis=1)
        norms_by_domain[domain] = norms
        print(f"{domain:15s} mean norm={norms.mean():.4f}  std={norms.std():.4f}  n={len(norms)}")

    target_feats = collect_features(backbone, target_eval, device)
    target_norms = np.linalg.norm(target_feats, axis=1)
    norms_by_domain["target"] = target_norms
    print(f"{'target':15s} mean norm={target_norms.mean():.4f}  std={target_norms.std():.4f}  n={len(target_norms)}")

    source_all = np.concatenate([norms_by_domain[d] for d in SOURCE_DOMAINS])
    print("\n--- summary ---")
    print(f"mean source norm: {source_all.mean():.4f}")
    print(f"mean target norm: {target_norms.mean():.4f}")
    print(f"ratio (target/source): {target_norms.mean() / source_all.mean():.4f}")

    # quick 1D separability check: how well would a naive norm threshold separate them?
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    n = min(len(source_all), len(target_norms))
    rng = np.random.default_rng(6304)
    X = np.concatenate([
        rng.choice(source_all, n, replace=False),
        rng.choice(target_norms, n, replace=False),
    ]).reshape(-1, 1)
    y = np.concatenate([np.zeros(n), np.ones(n)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=6304, stratify=y
    )
    probe = LogisticRegression(class_weight="balanced", random_state=6304)
    probe.fit(X_train, y_train)
    acc = accuracy_score(y_test, probe.predict(X_test))
    print(f"\ndomain separability using ONLY feature norm (1D): {acc:.4f}")
    print("(compare this to the full-feature domain_separability score you already have)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--pacs_root", type=str, required=True)
    args = parser.parse_args()
    main(args)