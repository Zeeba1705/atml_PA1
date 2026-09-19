import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from shared.pacs import PACSDomain
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms


SOURCE_DOMAINS = ["photo","art_painting", "cartoon",]
TARGET_DOMAIN = "sketch"

SEED= 6304
IMAGENET_MEAN= [0.485, 0.456, 0.406]
IMAGENET_STD= [0.229, 0.224, 0.225]

def make_source_splits(
    pacs_root="datasets/PACS/kfold",
    save_path="splits/pacs_sketch_seed6304.json",
):
    pacs_root= Path(pacs_root)
    save_path= Path(save_path)

    split_data = {
        "seed": SEED,
        "source_domains": SOURCE_DOMAINS,
        "target_domain": TARGET_DOMAIN,
        "splits": {},
    }

    for domain in SOURCE_DOMAINS:
        dataset= PACSDomain(
            root=pacs_root,
            domain=domain,
            transform=None,
        )

        indices= list(range(len(dataset)))

        labels= [
            dataset.samples[i][1]
            for i in indices
        ]

        train_idx, val_idx = train_test_split(indices,test_size=0.20,random_state=SEED,stratify=labels,)

        split_data["splits"][domain] = {"train_idx": sorted(train_idx),"val_idx": sorted(val_idx),}

        print(
            f"{domain}: "
            f"{len(train_idx)} train, "
            f"{len(val_idx)} val"
        )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(save_path, "w") as f:
        json.dump(
            split_data,
            f,
            indent=2
        )

    print(f"\nSaved splits to: {save_path}")

    return split_data


def load_source_splits(
    split_path="splits/pacs_sketch_seed6304.json",
):
    with open(split_path, "r") as f:
        split_data = json.load(f)

    return split_data


if __name__ == "__main__":
    make_source_splits()


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def make_pacs_loaders(pacs_root="datasets/PACS/kfold", split_path="splits/pacs_sketch_seed6304.json"):

    split_data= load_source_splits(split_path)

    train_transform= get_train_transform()
    eval_transform= get_eval_transform()

    source_train_loaders= {}
    source_val_loaders= {}

    for domain in SOURCE_DOMAINS:

        train_dataset = PACSDomain(
            root=pacs_root,
            domain=domain,
            transform=train_transform,
        )

        val_dataset = PACSDomain(
            root=pacs_root,
            domain=domain,
            transform=eval_transform,
        )

        train_idx= split_data["splits"][domain]["train_idx"]
        val_idx= split_data["splits"][domain]["val_idx"]

        train_subset= Subset(train_dataset,train_idx)

        val_subset= Subset(val_dataset,val_idx)

        source_train_loaders[domain] = DataLoader(
            train_subset,
            batch_size=8,
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )

        source_val_loaders[domain] = DataLoader(
            val_subset,
            batch_size=64,
            shuffle=False,
            num_workers=2,
        )


    target_train_dataset = PACSDomain(
    root=pacs_root,
    domain=TARGET_DOMAIN,
    transform=train_transform,
    )

    target_train_loader = DataLoader(
        target_train_dataset,
        batch_size=24,
        shuffle=True,
        num_workers=2,
        drop_last=True,

    )

    target_eval_dataset = PACSDomain(
        root=pacs_root,
        domain=TARGET_DOMAIN,
        transform=eval_transform,
    )

    target_eval_loader = DataLoader(
        target_eval_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
    )

    return (source_train_loaders,source_val_loaders,target_train_loader,target_eval_loader)

if __name__ == "__main__":
    source_train, source_val, target_train, target_eval = make_pacs_loaders()

    for domain in SOURCE_DOMAINS:
        images, labels = next(iter(source_train[domain]))
        print(domain, images.shape, labels.shape)

    sketch_images, _ = next(iter(target_train))
    print("sketch", sketch_images.shape)

def make_source_loaders(pacs_root="datasets/PACS/kfold", split_path="splits/pacs_sketch_seed6304.json"):
    split_data= load_source_splits(split_path)

    train_transform= get_train_transform()
    eval_transform= get_eval_transform()

    source_train_loaders= {}
    source_val_loaders= {}

    for domain in SOURCE_DOMAINS:
        train_dataset= PACSDomain(root=pacs_root,domain=domain,transform=train_transform)
        val_dataset= PACSDomain(root=pacs_root,domain=domain,transform=eval_transform)

        train_idx= split_data["splits"][domain]["train_idx"]
        val_idx= split_data["splits"][domain]["val_idx"]

        train_subset= Subset(train_dataset,train_idx)
        val_subset= Subset(val_dataset,val_idx)

        source_train_loaders[domain]= DataLoader(
            train_subset,
            batch_size=8,
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )

        source_val_loaders[domain]= DataLoader(
            val_subset,
            batch_size=64,
            shuffle=False,
            num_workers=2,
        )

    return source_train_loaders, source_val_loaders