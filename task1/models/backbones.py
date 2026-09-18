import torch.nn as nn
import torch
import torchvision
import open_clip

from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    vit_b_16,
    ViT_B_16_Weights
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#ResNet-50
resnet_weights = ResNet50_Weights.IMAGENET1K_V2
resnet= resnet50(weights=resnet_weights)

#ViT-B/16
vit_weights = ViT_B_16_Weights.IMAGENET1K_V1
vit= vit_b_16(weights=vit_weights)

#CLIP ViT-B/32
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32",pretrained="openai")


for param in resnet.parameters():
    param.requires_grad = False

for param in vit.parameters():
    param.requires_grad = False

for param in clip_model.parameters():
    param.requires_grad = False

resnet= resnet.to(device)
vit= vit.to(device)
clip_model= clip_model.to(device)
resnet.fc= nn.Identity()
resnet.eval()

vit.heads= nn.Identity()
vit.eval()

clip_model.eval()

#testin 
dummy_image= torch.randn(1, 3, 224, 224, device=device)

with torch.no_grad():
    resnet_features= resnet(dummy_image)
    vit_features= vit(dummy_image)
    clip_features= clip_model.encode_image(dummy_image)

    clip_features= clip_features / clip_features.norm(dim=-1,keepdim=True)
