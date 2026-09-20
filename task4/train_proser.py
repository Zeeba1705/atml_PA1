import os
import json
import random
import argparse
import numpy as np
import torch

from task4.data.cifar10 import cifar10_loaders
from task4.methods.proser import (
    PROSERModel,
    load_vanilla_weights,
    classifier_placeloss,
    data_placeholder_loss
)


SEED= 6304
GAMMA= 0.1


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def evaluate(model,loader,device):
    model.eval()

    correct= 0
    total= 0
    total_loss= 0

    criterion= torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images,labels in loader:
            images= images.to(device)
            labels= labels.to(device)

            logits= model(images)

            known_logits= logits[:,:10]

            loss= criterion(
                known_logits,
                labels
            )

            predictions= known_logits.argmax(dim=1)

            correct+= (predictions == labels).sum().item()
            total+= labels.size(0)
            total_loss+= loss.item() * labels.size(0)

    avg_loss= total_loss / total
    accuracy= correct / total

    return avg_loss,accuracy


def train_proser(data_root,split_path,vanilla_checkpoint,output_dir):
    set_seed()

    device= torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:",device)

    train_loader,val_loader,test_loader= cifar10_loaders(
        data_root=data_root,
        split_path=split_path,
        method="vanilla"
    )

    print("Train:",len(train_loader.dataset))
    print("Val:",len(val_loader.dataset))
    print("Test:",len(test_loader.dataset))

    model= PROSERModel().to(device)

    optimizer= torch.optim.SGD(
        model.parameters(),
        lr=1e-3,
        momentum=0.9,
        weight_decay=5e-4
    )

    scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=50
    )

    os.makedirs(output_dir,exist_ok=True)

    best_path= os.path.join(
        output_dir,
        "proser_best.pt"
    )

    latest_path= os.path.join(
        output_dir,
        "proser_latest.pt"
    )

    history_path= os.path.join(
        output_dir,
        "proser_history.json"
    )

    results_path= os.path.join(
        output_dir,
        "proser_results.json"
    )

    best_val_acc= 0
    start_epoch= 1
    history= []


    if os.path.exists(latest_path):
        print("Found PROSER checkpoint. Resuming...")

        checkpoint= torch.load(
            latest_path,
            map_location=device
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        best_val_acc= checkpoint["best_val_acc"]
        start_epoch= checkpoint["epoch"] + 1

        if os.path.exists(history_path):
            with open(history_path,"r") as f:
                history= json.load(f)

        print("Resuming from epoch:",start_epoch)
        print("Best val accuracy so far:",best_val_acc)

    else:
        print("Initializing PROSER from Vanilla checkpoint...")

        model= load_vanilla_weights(
            model,
            vanilla_checkpoint,
            device
        )


    for epoch in range(start_epoch,51):
        model.train()

        total_loss_sum= 0
        classifier_loss_sum= 0
        known_loss_sum= 0
        dummy_loss_sum= 0
        mixup_loss_sum= 0

        correct= 0
        total= 0

        for images,labels in train_loader:
            images= images.to(device)
            labels= labels.to(device)

            midpoint= images.size(0) // 2

            classifier_images= images[:midpoint]
            classifier_labels= labels[:midpoint]

            mixup_images= images[midpoint:]
            mixup_labels= labels[midpoint:]

            optimizer.zero_grad()

            classifier_logits= model(
                classifier_images
            )

            classifier_loss,known_loss,dummy_loss= classifier_placeloss(
                classifier_logits,
                classifier_labels,
                beta=1.0
            )

            mixup_loss= data_placeholder_loss(
                model,
                mixup_images,
                mixup_labels,
                alpha=2.0
            )

            loss= classifier_loss + GAMMA * mixup_loss

            loss.backward()
            optimizer.step()

            total_loss_sum+= loss.item()
            classifier_loss_sum+= classifier_loss.item()
            known_loss_sum+= known_loss.item()
            dummy_loss_sum+= dummy_loss.item()
            mixup_loss_sum+= mixup_loss.item()

            known_logits= classifier_logits[:,:10]
            predictions= known_logits.argmax(dim=1)

            correct+= (
                predictions == classifier_labels
            ).sum().item()

            total+= classifier_labels.size(0)


        num_batches= len(train_loader)

        train_loss= total_loss_sum / num_batches
        train_classifier_loss= classifier_loss_sum / num_batches
        train_known_loss= known_loss_sum / num_batches
        train_dummy_loss= dummy_loss_sum / num_batches
        train_mixup_loss= mixup_loss_sum / num_batches

        train_acc= correct / total

        val_loss,val_acc= evaluate(
            model,
            val_loader,
            device
        )

        current_lr= optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | "
            f"LR {current_lr:.6f} | "
            f"Total {train_loss:.4f} | "
            f"L1 {train_classifier_loss:.4f} | "
            f"Known {train_known_loss:.4f} | "
            f"Dummy {train_dummy_loss:.4f} | "
            f"L2 {train_mixup_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Val Acc {val_acc:.4f}"
        )

        history.append({
            "epoch": epoch,
            "lr": current_lr,
            "total_loss": train_loss,
            "classifier_placeholder_loss": train_classifier_loss,
            "known_loss": train_known_loss,
            "dummy_loss": train_dummy_loss,
            "data_placeholder_loss": train_mixup_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })


        if val_acc > best_val_acc:
            best_val_acc= val_acc

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc
            },best_path)

            print("Saved new best PROSER checkpoint.")


        scheduler.step()


        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc
        },latest_path)


        with open(history_path,"w") as f:
            json.dump(
                history,
                f,
                indent=2
            )


    print("\nBest validation accuracy:",best_val_acc)

    checkpoint= torch.load(
        best_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test_loss,test_acc= evaluate(
        model,
        test_loader,
        device
    )

    print("Best epoch:",checkpoint["epoch"])
    print("CIFAR-10 test accuracy:",test_acc)


    results= {
        "best_epoch": checkpoint["epoch"],
        "best_val_acc": checkpoint["val_acc"],
        "test_loss": test_loss,
        "test_acc": test_acc,
        "gamma": GAMMA,
        "beta": 1.0,
        "num_dummy": 5
    }

    with open(results_path,"w") as f:
        json.dump(
            results,
            f,
            indent=2
        )


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        default="/content/drive/MyDrive/atml_PA1/datasets"
    )

    parser.add_argument(
        "--split_path",
        default="/content/atml_PA1/splits/cifar10_seed6304.json"
    )

    parser.add_argument(
        "--vanilla_checkpoint",
        default="/content/drive/MyDrive/atml_PA1/task4_results/vanilla_best.pt"
    )

    parser.add_argument(
        "--output_dir",
        default="/content/drive/MyDrive/atml_PA1/task4_results"
    )

    args= parser.parse_args()

    train_proser(
        data_root=args.data_root,
        split_path=args.split_path,
        vanilla_checkpoint=args.vanilla_checkpoint,
        output_dir=args.output_dir
    )