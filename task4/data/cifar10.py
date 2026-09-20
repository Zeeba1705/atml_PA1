import os
import json
import numpy as np

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader,Subset
from torchvision import datasets,transforms


SEED= 6304

def get_train_transform(method="vanilla"):
    trasnfomrations= [
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip()
    ]

    if method == "gcsc":
        trasnfomrations.append(transforms.RandAugment(num_ops=2,magnitude=9))

    trasnfomrations.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616))
    ])

    return transforms.Compose(trasnfomrations)


def get_eval_transform():
    return transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616))])


def create_split(dataset,split_path):
    labels= np.array(dataset.targets)
    indices= np.arange(len(dataset))

    train_idx,val_idx= train_test_split(indices, test_size=0.1, random_state=SEED, stratify=labels)

    split_data= {
        "seed": SEED,
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist()
    }

    os.makedirs(os.path.dirname(split_path),exist_ok=True)

    with open(split_path,"w") as f:
        json.dump(split_data,f,indent=2)

    return split_data


def cifar10_loaders(data_root="datasets",split_path="splits/cifar10_seed6304.json",method="vanilla"):
   
    base_dataset= datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True
    )

    if os.path.exists(split_path):
        with open(split_path,"r") as f:
            split_data= json.load(f)
    else:
        split_data= create_split(base_dataset,split_path)

    train_transform= get_train_transform(method)
    eval_transform= get_eval_transform()

    train_dataset= datasets.CIFAR10(
        root=data_root,
        train=True,
        transform=train_transform,
        download=True
    )

    val_dataset= datasets.CIFAR10(
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
        val_dataset,
        split_data["val_idx"]
    )

    train_loader= DataLoader(
        train_subset,
        batch_size=128,
        shuffle=True,
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

    return train_loader,val_loader,test_loader