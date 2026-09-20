import os
import json
import argparse
import torch

from torch.utils.data import DataLoader,Subset
from torchvision import datasets

from task4.data.cifar10 import get_eval_transform
from task4.data.cifar100_unknowns import cifar100_unknown_loaders
from task4.models.resnet_cifar import CIFARResNet18
from task4.methods.proser import PROSERModel


def extract(model,loader,device):
    model.eval()

    all_logits= []
    all_features= []
    all_labels= []

    with torch.no_grad():
        for images,labels in loader:
            images= images.to(device)

            logits,features= model(
                images,
                return_features=True
            )

            all_logits.append(logits.cpu())
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    return {
        "logits":torch.cat(all_logits,dim=0),
        "features":torch.cat(all_features,dim=0),
        "labels":torch.cat(all_labels,dim=0)
    }


def load_model(method,checkpoint_path,device):
    if method == "proser":
        model= PROSERModel().to(device)

    else:
        model= CIFARResNet18(
            num_classes=10
        ).to(device)

    checkpoint= torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return model


def main(method,data_root,split_path,checkpoint_path,output_dir):
    device= torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:",device)
    print("Method:",method)

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    model= load_model(
        method,
        checkpoint_path,
        device
    )

    eval_transform= get_eval_transform()

    with open(split_path,"r") as f:
        split_data= json.load(f)

    train_dataset= datasets.CIFAR10(
        root=data_root,
        train=True,
        transform=eval_transform,
        download=True
    )

    test_dataset= datasets.CIFAR10(
        root=data_root,
        train=False,
        transform=eval_transform,
        download=True
    )

    train_subset= Subset(
        train_dataset,
        split_data["train_idx"]
    )

    val_subset= Subset(
        train_dataset,
        split_data["val_idx"]
    )

    train_loader= DataLoader(
        train_subset,
        batch_size=128,
        shuffle=False,
        num_workers=2
    )

    val_loader= DataLoader(
        val_subset,
        batch_size=128,
        shuffle=False,
        num_workers=2
    )

    test_loader= DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2
    )

    near_loader,far_loader,cifar100_classes, near_indices, far_indices= cifar100_unknown_loaders(
        data_root=data_root,
        transform=eval_transform
    )

    outputs= {
        "train":extract(
            model,
            train_loader,
            device
        ),

        "val":extract(
            model,
            val_loader,
            device
        ),

        "test":extract(
            model,
            test_loader,
            device
        ),

        "near":extract(
            model,
            near_loader,
            device
        ),

        "far":extract(
            model,
            far_loader,
            device
        )
    }

    for split_name,data in outputs.items():
        path= os.path.join(
            output_dir,
            f"{method}_{split_name}_outputs.pt"
        )

        torch.save(
            data,
            path
        )

        print(
            split_name,
            data["logits"].shape,
            data["features"].shape
        )

    class_path= os.path.join(
        output_dir,
        "cifar100_classes.json"
    )

    with open(class_path,"w") as f:
        json.dump(
            cifar100_classes,
            f,
            indent=2
        )

    with open(
    os.path.join(
        output_dir,
        "cifar100_unknown_indices.json"
    ),
    "w") as f:
        json.dump({
        "near":near_indices,
        "far":far_indices
    },f,indent=2)


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        choices=["vanilla","gcsc","proser"],
        required=True
    )

    parser.add_argument(
        "--data_root",
        required=True
    )

    parser.add_argument(
        "--split_path",
        required=True
    )

    parser.add_argument(
        "--checkpoint_path",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        required=True
    )

    args= parser.parse_args()

    main(
        method=args.method,
        data_root=args.data_root,
        split_path=args.split_path,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir
    )