from torch.utils.data import DataLoader,Subset
from torchvision import datasets


NEAR_CLASSES= [
    "bus",
    "pickup_truck",
    "motorcycle",
    "tractor",
    "wolf",
    "fox",
    "leopard",
    "camel"
]

FAR_CLASSES= [
    "bottle",
    "bowl",
    "chair",
    "clock",
    "keyboard",
    "mushroom",
    "sunflower",
    "wardrobe"
]


def cifar100_unknown_loaders(data_root="datasets",transform=None):
    dataset= datasets.CIFAR100(root=data_root, train=False, transform=transform,download=True)

    class_to_idx= {name:i for i,name in enumerate(dataset.classes)}

    near_ids= {class_to_idx[name] for name in NEAR_CLASSES}
    far_ids= {class_to_idx[name] for name in FAR_CLASSES}

    near_indices= []
    far_indices= []

    for i,label in enumerate(dataset.targets):
        if label in near_ids:
            near_indices.append(i)
        elif label in far_ids:
            far_indices.append(i)

    near_dataset= Subset(dataset,near_indices)
    far_dataset= Subset(dataset,far_indices)

    near_loader= DataLoader(near_dataset,batch_size=128,shuffle=False,num_workers=2)

    far_loader= DataLoader(far_dataset,batch_size=128,shuffle=False,num_workers=2)

    return near_loader,far_loader