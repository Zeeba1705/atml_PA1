import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.model= resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc= nn.Identity()

    def forward(self, x):
        return self.model(x)
    
def freeze_batchnorm_stats(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()