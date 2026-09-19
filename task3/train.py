import os
import json
import random
import argparse
import numpy as np
import torch

from shared.pacs_protocol import make_source_loaders, SOURCE_DOMAINS
from task2.models.backbone import ResNetBackbone, freeze_batchnorm_stats
from task2.models.classifier_head import ClassifierHead
from task2.evaluation.metrics import evaluate_model
from task3.methods.dan_dg import dan_dg_loss


SEED= 6304

def next_load(loader, iterator):
    try:
        batch= next(iterator)

    except StopIteration:
        iterator= iter(loader)
        batch= next(iterator)

    return batch, iterator


def train_dan_dg(args,source_train,source_val,device):
    backbone= ResNetBackbone().to(device)
    classifier= ClassifierHead().to(device)

    optimizer= torch.optim.AdamW(
        list(backbone.parameters()) + list(classifier.parameters()),
        lr=1e-4,
        weight_decay=1e-4
    )

    best_f1= -1
    patience_count= 0
    history= []

    num_steps= max(len(source_train[d]) for d in SOURCE_DOMAINS)

    run_name= f"dan_dg_lambda_{args.lambda_dg}"

    checkpoint_path= os.path.join(args.output_dir,f"{run_name}_best.pt")
    history_path= os.path.join(args.output_dir,f"{run_name}_history.json")

    for epoch in range(30):
        print(f"\n--- Epoch {epoch + 1} ---")

        backbone.train()
        classifier.train()
        freeze_batchnorm_stats(backbone)

        iterators= {domain: iter(source_train[domain]) for domain in SOURCE_DOMAINS}

        total_loss_sum= 0
        classification_loss_sum= 0
        mmd_loss_sum= 0

        for step in range(num_steps):
            batches= {}

            for domain in SOURCE_DOMAINS:
                batch, iterators[domain]= next_load(source_train[domain],iterators[domain])
                batches[domain]= batch

            optimizer.zero_grad()

            total_loss, classification_loss, alignment_loss= dan_dg_loss(
                backbone,
                classifier,
                batches,
                device,
                lambda_dg=args.lambda_dg
            )

            total_loss.backward()
            optimizer.step()

            total_loss_sum+= total_loss.item()
            classification_loss_sum+= classification_loss.item()
            mmd_loss_sum+= alignment_loss.item()

        average_loss= total_loss_sum / num_steps
        average_classification_loss= classification_loss_sum / num_steps
        average_mmd_loss= mmd_loss_sum / num_steps

        print("average train loss:", round(average_loss,4))
        print("classification loss:", round(average_classification_loss,4))
        print("MMD loss:", round(average_mmd_loss,4))

        source_results= {}

        for domain in SOURCE_DOMAINS:
            scores= evaluate_model(
                backbone,
                classifier,
                source_val[domain],
                device
            )

            source_results[domain]= scores

        mean_f1= sum(source_results[d]["macro_f1"] for d in SOURCE_DOMAINS) / len(SOURCE_DOMAINS)

        for domain in SOURCE_DOMAINS:
            print(
                f"{domain}:",
                f"accuracy={source_results[domain]['accuracy']:.4f},",
                f"macro_f1={source_results[domain]['macro_f1']:.4f}"
            )

        print("mean source macro F1:", round(mean_f1,4))

        epoch_record= {
            "epoch": epoch + 1,
            "train_loss": average_loss,
            "classification_loss": average_classification_loss,
            "mmd_loss": average_mmd_loss,
            "source_validation": source_results,
            "mean_source_macro_f1": mean_f1
        }

        history.append(epoch_record)

        with open(history_path,"w") as f:
            json.dump(history,f,indent=2)

        if mean_f1 > best_f1:
            best_f1= mean_f1
            patience_count= 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "backbone": backbone.state_dict(),
                    "classifier": classifier.state_dict(),
                    "best_f1": best_f1,
                    "lambda_dg": args.lambda_dg
                },
                checkpoint_path
            )

        else:
            patience_count+= 1
            print(f"no improvement ({patience_count}/5)")

        if patience_count >= 5:
            print("stopping early")
            break

    print("\nBest source macro F1:", round(best_f1,4))
    print("Saved checkpoint:", checkpoint_path)
    print("Saved history:", history_path)


def main(args):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)
    print("Method:", args.method)

    os.makedirs(args.output_dir,exist_ok=True)

    source_train, source_val= make_source_loaders(
        pacs_root=args.pacs_root,
        split_path=args.split_path
    )

    if args.method == "dan_dg":
        print("lambda_dg:", args.lambda_dg)
        train_dan_dg(args,source_train,source_val,device)

    else:
        raise ValueError("Unknown method: " + args.method)


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument("--method",type=str,required=True,choices=["dan_dg"])
    parser.add_argument("--lambda_dg",type=float,default=1.0)
    parser.add_argument("--pacs_root",type=str,default="datasets/PACS/kfold")
    parser.add_argument("--split_path",type=str,default="splits/pacs_sketch_seed6304.json")
    parser.add_argument("--output_dir",type=str,default="task3/results")

    args= parser.parse_args()

    main(args)