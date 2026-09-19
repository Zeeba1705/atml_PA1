import os
import json
import random
import argparse
import numpy as np
import torch
from shared.pacs_protocol import make_pacs_loaders, SOURCE_DOMAINS
from task2.models.backbone import ResNetBackbone
from task2.models.classifier_head import ClassifierHead
from task2.methods.source_only import src_train
from task2.evaluation.metrics import evaluate_model

SEED=6304

def main(args):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    os.makedirs(args.output_dir, exist_ok=True)

    source_train, source_val, target_train, target_eval = make_pacs_loaders(
        pacs_root=args.pacs_root
    )

    backbone= ResNetBackbone().to(device)
    classifier= ClassifierHead().to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer= torch.optim.AdamW(list(backbone.parameters())+list(classifier.parameters()), lr=1e-4, weight_decay=1e-4)

    max_epoc=30
    patience=5
    history=[]

    best_f1=-1
    epochs_noimprov = 0

    for epoch in range(max_epoc):

        print(f"\n--- Epoch {epoch + 1} ---")

        source_iters = {}
        for domain in SOURCE_DOMAINS:
            source_iters[domain] = iter(source_train[domain])

        num_steps = max(
            len(source_train[d])
            for d in SOURCE_DOMAINS
        )

        losses = []

        for _ in range(num_steps):

            batch_loss = src_train(
                backbone=backbone,
                classifier=classifier,
                src_iter=source_iters,
                src_loader=source_train,
                optimizer=optimizer,
                criterion=criterion,
                device=device
            )

            losses.append(batch_loss)

        train_loss = sum(losses) / len(losses)

        source_scores = {}

        for domain in SOURCE_DOMAINS:

            scores = evaluate_model(
                backbone,
                classifier,
                source_val[domain],
                device
            )

            source_scores[domain] = scores

        f1_scores = [
            source_scores[d]["macro_f1"]
            for d in SOURCE_DOMAINS
        ]

        mean_f1 = sum(f1_scores) / len(f1_scores)

        print("average train loss:", round(train_loss, 4))

        for domain in SOURCE_DOMAINS:
            acc = source_scores[domain]["accuracy"]
            f1 = source_scores[domain]["macro_f1"]

            print(
                f"{domain}: "
                f"accuracy={acc:.4f}, "
                f"macro_f1={f1:.4f}"
            )

        print("mean source macro F1:", round(mean_f1, 4))

        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "mean_source_f1": mean_f1,
            "source_validation": source_scores
        }

        history.append(epoch_info)

        improved = mean_f1 > best_f1

        if improved:

            best_f1 = mean_f1
            epochs_noimprov= 0

            save_path = os.path.join(
                args.output_dir,
                "source_only_best.pt"
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "backbone": backbone.state_dict(),
                    "classifier": classifier.state_dict(),
                    "best_f1": best_f1
                },
                save_path
            )

            print("new best checkpoint saved")

        else:

            epochs_noimprov += 1

            print(
                f"no improvement "
                f"({epochs_noimprov}/{patience})"
            )

        history_path = os.path.join(
            args.output_dir,
            "source_only_history.json"
        )

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        if epochs_noimprov >= patience:
            print("stopping early")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pacs_root",
        type=str,
        default="datasets/PACS/kfold"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="task2/results"
    )

    args = parser.parse_args()

    main(args)


