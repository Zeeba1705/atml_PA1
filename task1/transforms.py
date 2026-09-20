import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


def common_224_tensor(image):
    image= TF.resize(image,[224,224])
    return TF.to_tensor(image)


def translate_reflect_tensor(image,dx=0,dy=0):
    image= common_224_tensor(image)
    pad= max(abs(dx),abs(dy))

    if pad == 0:
        return image

    image= F.pad(image.unsqueeze(0),(pad,pad,pad,pad),mode="reflect").squeeze(0)
    left= pad - dx
    top= pad - dy
    return image[:,top:top + 224,left:left + 224]


class TranslationDataset(Dataset):
    def __init__(self,base_dataset,indices,dx,dy,normalize):
        self.base_dataset= base_dataset
        self.indices= indices
        self.dx= dx
        self.dy= dy
        self.normalize= normalize

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,i):
        idx= int(self.indices[i])
        image,label= self.base_dataset[idx]
        image= translate_reflect_tensor(image,self.dx,self.dy)
        image= self.normalize(image)
        return image,label
