import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn

from task4.data.cifar10 import cifar10_loaders
from task4.models.resnet_cifar import CIFARResNet18


SEED= 6304


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

    criterion= nn.CrossEntropyLoss()

    with torch.no_grad():
        for images,labels in loader:
            images= images.to(device)
            labels= labels.to(device)

            logits= model(images)
            loss= criterion(logits,labels)

            predictions= logits.argmax(dim=1)

            correct+= (predictions == labels).sum().item()
            total+= labels.size(0)
            total_loss+= loss.item() * labels.size(0)

    accuracy= correct / total
    avg_loss= total_loss / total

    return avg_loss,accuracy


def train_model(method,data_root,split_path,output_dir):
    set_seed()

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device)
    print("Method:",method)

    train_loader,val_loader,test_loader= cifar10_loaders(
        data_root=data_root,
        split_path=split_path,
        method=method
    )

    print("Train:",len(train_loader.dataset))
    print("Val:",len(val_loader.dataset))
    print("Test:",len(test_loader.dataset))

    model= CIFARResNet18(num_classes=10).to(device)

    criterion= nn.CrossEntropyLoss()

    optimizer= torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )

    scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100
    )

    os.makedirs(output_dir,exist_ok=True)

    checkpoint_path= os.path.join(output_dir,f"{method}_best.pt")
    latest_checkpoint_path= os.path.join(output_dir,f"{method}_latest.pt")
    history_path= os.path.join(output_dir,f"{method}_history.json")
    results_path= os.path.join(output_dir,f"{method}_results.json")

    best_val_acc= 0
    history= []
    start_epoch= 1

    if os.path.exists(latest_checkpoint_path):
        print("Found latest checkpoint. Resuming training...")

        checkpoint= torch.load(latest_checkpoint_path,map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        best_val_acc= checkpoint["best_val_acc"]
        start_epoch= checkpoint["epoch"] + 1

        if os.path.exists(history_path):
            with open(history_path,"r") as f:
                history= json.load(f)

        print("Resuming from epoch:",start_epoch)
        print("Best validation accuracy so far:",best_val_acc)

    for epoch in range(start_epoch,101):
        model.train()

        running_loss= 0
        correct= 0
        total= 0

        for images,labels in train_loader:
            images= images.to(device)
            labels= labels.to(device)

            optimizer.zero_grad()

            logits= model(images)
            loss= criterion(logits,labels)

            loss.backward()
            optimizer.step()

            running_loss+= loss.item() * labels.size(0)

            predictions= logits.argmax(dim=1)

            correct+= (predictions == labels).sum().item()
            total+= labels.size(0)

        train_loss= running_loss / total
        train_acc= correct / total

        val_loss,val_acc= evaluate(model,val_loader,device)

        current_lr= optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | "
            f"LR {current_lr:.6f} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.4f}"
        )

        history.append({
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
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
            },checkpoint_path)

            print("Saved new best checkpoint.")

        scheduler.step()

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc
        },latest_checkpoint_path)

        with open(history_path,"w") as f:
            json.dump(history,f,indent=2)

    print("\nBest validation accuracy:",best_val_acc)

    checkpoint= torch.load(checkpoint_path,map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss,test_acc= evaluate(model,test_loader,device)

    print("Best epoch:",checkpoint["epoch"])
    print("CIFAR-10 test accuracy:",test_acc)

    final_results= {
        "method": method,
        "best_epoch": checkpoint["epoch"],
        "best_val_acc": checkpoint["val_acc"],
        "test_loss": test_loss,
        "test_acc": test_acc
    }

    with open(results_path,"w") as f:
        json.dump(final_results,f,indent=2)


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        choices=["vanilla","gcsc"],
        required=True
    )

    parser.add_argument(
        "--data_root",
        default="/content/data"
    )

    parser.add_argument(
        "--split_path",
        default="/content/atml_PA1/splits/cifar10_seed6304.json"
    )

    parser.add_argument(
        "--output_dir",
        default="/content/drive/MyDrive/atml_PA1/task4_results"
    )

    args= parser.parse_args()

    train_model(
        method=args.method,
        data_root=args.data_root,
        split_path=args.split_path,
        output_dir=args.output_dir
    )