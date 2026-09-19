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
from task2.methods.dan import dan_train
import math

from task2.models.domain_discriminator import DomainDiscriminator
from task2.methods.dann import dann_train
from task2.methods.cdan import cdan_train

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

    print("Method:", args.method)

    if args.method == "dan":
        print("lambda_mmd:", args.lambda_mmd)

    if args.method == "dan":
        run_name = f"dan_lambda_{args.lambda_mmd}"
    else:
        run_name = args.method

    source_train, source_val, target_train, target_eval = make_pacs_loaders(
        pacs_root=args.pacs_root
    )

    backbone= ResNetBackbone().to(device)
    classifier= ClassifierHead().to(device)

    discriminator= None

    if args.method == "dann":
        discriminator = DomainDiscriminator(input_dim=512).to(device)

    elif args.method == "cdan":
        discriminator = DomainDiscriminator(input_dim=512 * 7).to(device)

    criterion= torch.nn.CrossEntropyLoss()

    params=(list(backbone.parameters())+ list(classifier.parameters()))

    if args.method in ["dann", "cdan"]:
        params+=list(discriminator.parameters())

    optimizer= torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)

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

        if args.method in ["dan", "dann", "cdan"]:
            target_iter=iter(target_train)

        num_steps= max(
            len(source_train[d])
            for d in SOURCE_DOMAINS
        )

        losses= []
        cls_losses=[]
        alignment_losses=[]
        domain_losses=[]
        domain_accs=[]
        feature_vars = []

        for step in range(num_steps):
            if args.method=="source_only":

                batch_loss= src_train(
                    backbone=backbone,
                    classifier=classifier,
                    src_iter=source_iters,
                    src_loader=source_train,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device
                )

                losses.append(batch_loss)

            elif args.method=="dan":
                try:
                    target_imgs, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_train)
                    target_imgs, _ = next(target_iter)

                total_loss, cls_loss, alignment_loss = dan_train(
                    backbone=backbone,
                    classifier=classifier,
                    src_iter=source_iters,
                    src_loader=source_train,
                    target_imgs=target_imgs,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    lambda_mmd=args.lambda_mmd
                )

                losses.append(total_loss)
                cls_losses.append(cls_loss)
                alignment_losses.append(alignment_loss)

            elif args.method == "dann":
                try:
                    target_imgs, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_train)
                    target_imgs, _ = next(target_iter)

                total_steps = max_epoc * num_steps

                current_step = epoch * num_steps+ step

                p = current_step / (total_steps-1)

                alpha = (2.0 / (1.0 + math.exp(-10 * p))- 1.0)

                total_loss, cls_loss, domain_loss, domain_acc, feature_var= dann_train(
                    backbone=backbone,
                    classifier=classifier,
                    discriminator=discriminator,
                    src_iter=source_iters,
                    src_loader=source_train,
                    target_imgs=target_imgs,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    alpha=alpha
                )

                losses.append(total_loss)
                cls_losses.append(cls_loss)
                domain_losses.append(domain_loss)
                domain_accs.append(domain_acc)
                feature_vars.append(feature_var)
            
            elif args.method == "cdan":
                try:
                    target_imgs, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_train)
                    target_imgs, _ = next(target_iter)

                total_steps = max_epoc * num_steps
                current_step = epoch * num_steps + step

                p = current_step / (total_steps - 1)

                alpha = (
                    2.0 / (1.0 + math.exp(-10 * p))
                    - 1.0
                )

                total_loss, cls_loss, domain_loss, domain_acc, feature_var = cdan_train(
                    backbone=backbone,
                    classifier=classifier,
                    discriminator=discriminator,
                    src_iter=source_iters,
                    src_loader=source_train,
                    target_imgs=target_imgs,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    alpha=alpha
                )

                losses.append(total_loss)
                cls_losses.append(cls_loss)
                domain_losses.append(domain_loss)
                domain_accs.append(domain_acc)
                feature_vars.append(feature_var)


        train_loss = sum(losses) / len(losses)

        if args.method == "dan":
            avg_cls_loss = sum(cls_losses) / len(cls_losses)
            avg_alignment_loss = sum(alignment_losses) / len(alignment_losses)
        
        if args.method in ["dann", "cdan"]:
            avg_cls_loss = sum(cls_losses) / len(cls_losses)
            avg_domain_loss = sum(domain_losses) / len(domain_losses)
            avg_domain_acc = sum(domain_accs) / len(domain_accs)
            avg_feature_var= sum(feature_vars)/ len(feature_vars)

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
        if args.method == "dan":
            print("classification loss:", round(avg_cls_loss, 4))
            print("MMD loss:", round(avg_alignment_loss, 4))
        
        if args.method in ["dann", "cdan"]:
            print("classification loss:", round(avg_cls_loss, 4))
            print("domain loss:", round(avg_domain_loss, 4))
            print("domain accuracy:", round(avg_domain_acc, 4))
            print("feature variance:", round(avg_feature_var, 4))

            print("alpha:", round(alpha, 4))

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
        if args.method == "dan":
            epoch_info["classification_loss"] = avg_cls_loss
            epoch_info["mmd_loss"] = avg_alignment_loss

        if args.method in ["dann", "cdan"]:
            epoch_info["classification_loss"] = avg_cls_loss
            epoch_info["domain_loss"] = avg_domain_loss
            epoch_info["domain_accuracy"] = avg_domain_acc
            epoch_info["feature_variance"] = avg_feature_var

            epoch_info["alpha"] = alpha

        history.append(epoch_info)

        improved = mean_f1 > best_f1

        if improved:

            best_f1 = mean_f1
            epochs_noimprov= 0

            save_path = os.path.join(
                args.output_dir,
                f"{run_name}_best.pt"
            )

            checkpoint={
                    "epoch": epoch + 1,
                    "backbone": backbone.state_dict(),
                    "classifier": classifier.state_dict(),
                    "best_f1": best_f1
                }
            
            if args.method in ["dann", "cdan"]:
                checkpoint["discriminator"]= discriminator.state_dict()
            
            torch.save(checkpoint, save_path)



        else:

            epochs_noimprov += 1

            print(
                f"no improvement "
                f"({epochs_noimprov}/{patience})"
            )

        history_path = os.path.join(
            args.output_dir,
            f"{run_name}_history.json"
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

    parser.add_argument(
    "--method",
    type=str,
    choices=["source_only", "dan", "dann", "cdan"],
    default="source_only"
    )

    parser.add_argument(
        "--lambda_mmd",
        type=float,
        default=1.0
    )

    args = parser.parse_args()

    main(args)


