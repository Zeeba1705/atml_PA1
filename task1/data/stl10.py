import json
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision.datasets import STL10
from torchvision import transforms

SEED= 6304

IMAGENET_NORMALIZE= transforms.Normalize(
    mean=[0.485,0.456,0.406],
    std=[0.229,0.224,0.225]
)

CLIP_NORMALIZE= transforms.Normalize(
    mean=[0.48145466,0.4578275,0.40821073],
    std=[0.26862954,0.26130258,0.27577711]
)


def load_stl10(data_root,download=False):
    train_dataset= STL10(root=data_root,split="train",download=download,transform=None)
    test_dataset= STL10(root=data_root,split="test",download=download,transform=None)
    return train_dataset,test_dataset


def load_split(split_path):
    with open(split_path,"r") as f:
        split= json.load(f)

    return (
        np.array(split["train_idx"]),
        np.array(split["val_idx"]),
        np.array(split["test_subset_idx"])
    )


class IndexedSTL10(Dataset):
    def __init__(self,base_dataset,indices,normalize):
        self.base_dataset= base_dataset
        self.indices= np.asarray(indices)
        self.normalize= normalize

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,i):
        dataset_idx= int(self.indices[i])
        image,label= self.base_dataset[dataset_idx]
        image= transforms.functional.resize(image,[224,224])
        image= transforms.functional.to_tensor(image)
        image= self.normalize(image)
        return image,label


def make_test_loaders(test_dataset,test_indices,batch_size=64,num_workers=2):
    imagenet_dataset= IndexedSTL10(test_dataset,test_indices,IMAGENET_NORMALIZE)
    clip_dataset= IndexedSTL10(test_dataset,test_indices,CLIP_NORMALIZE)

    imagenet_loader= DataLoader(imagenet_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)
    clip_loader= DataLoader(clip_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    return imagenet_loader,clip_loader
