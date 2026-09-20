import torch.nn as nn
from torchvision.models import resnet18


class CIFARResNet18(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()

        model= resnet18(weights=None)

        model.conv1= nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        model.maxpool= nn.Identity()

        feature_dim= model.fc.in_features
        model.fc= nn.Identity()

        self.backbone= model
        self.classifier= nn.Linear(feature_dim,num_classes)

    def forward(self,x,return_features=False):
        features= self.backbone(x)
        logits= self.classifier(features)

        if return_features:
            return logits,features

        return logits