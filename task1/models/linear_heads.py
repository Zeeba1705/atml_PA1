import torch
import torch.nn as nn

HEAD_DIMS= {
    "resnet":2048,
    "vit":768,
    "clip":512
}


def make_linear_head(model_name,num_classes=10):
    return nn.Linear(HEAD_DIMS[model_name],num_classes)


def load_linear_head(model_name,checkpoint_path,device,num_classes=10):
    head= make_linear_head(model_name,num_classes=num_classes).to(device)
    state= torch.load(checkpoint_path,map_location=device)
    head.load_state_dict(state)
    head.eval()
    return head
