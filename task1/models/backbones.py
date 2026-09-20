import torch
import torch.nn as nn
import open_clip
from torchvision.models import resnet50,ResNet50_Weights,vit_b_16,ViT_B_16_Weights


def load_backbones(device):
    resnet= resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    resnet.fc= nn.Identity()

    vit= vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    vit.heads= nn.Identity()

    clip_model,_,_= open_clip.create_model_and_transforms("ViT-B-32",pretrained="openai")

    for model in [resnet,vit,clip_model]:
        for param in model.parameters():
            param.requires_grad= False
        model.to(device)
        model.eval()

    return resnet,vit,clip_model


def build_clip_text_features(clip_model,class_names,device):
    prompts= [f"a photo of a {class_name}." for class_name in class_names]
    tokenizer= open_clip.get_tokenizer("ViT-B-32")
    tokens= tokenizer(prompts).to(device)

    with torch.no_grad():
        text_features= clip_model.encode_text(tokens)
        text_features= text_features / text_features.norm(dim=-1,keepdim=True)

    return text_features


def extract_features(model,loader,device,is_clip=False):
    all_features= []
    all_labels= []
    model.eval()

    with torch.no_grad():
        for images,labels in loader:
            images= images.to(device)

            if is_clip:
                features= model.encode_image(images)
                features= features / features.norm(dim=-1,keepdim=True)
            else:
                features= model(images)

            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_features,dim=0),torch.cat(all_labels,dim=0)
