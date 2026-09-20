import torch
import torch.nn as nn
import torch.nn.functional as F

from task4.models.resnet_cifar import CIFARResNet18


N_KNOWN= 10
N_DUMMY= 5


class PROSERModel(nn.Module):
    def __init__(self):
        super().__init__()

        base_model= CIFARResNet18(num_classes=N_KNOWN + N_DUMMY)

        self.backbone= base_model.backbone
        self.classifier= base_model.classifier


    def forward(self,x,return_features=False):
        features= self.backbone(x)
        logits= self.classifier(features)

        if return_features:
            return logits,features

        return logits


    def forward_to_layer2(self,x):
        x= self.backbone.conv1(x)
        x= self.backbone.bn1(x)
        x= self.backbone.relu(x)
        x= self.backbone.maxpool(x)

        x= self.backbone.layer1(x)
        x= self.backbone.layer2(x)

        return x


    def forward_from_layer2(self,x,return_features=False):
        x= self.backbone.layer3(x)
        x= self.backbone.layer4(x)

        x= self.backbone.avgpool(x)
        features= torch.flatten(x,1)

        logits= self.classifier(features)

        if return_features:
            return logits,features

        return logits


def load_vanilla_weights(model,checkpoint_path,device):
    vanilla= CIFARResNet18(num_classes=N_KNOWN).to(device)

    checkpoint= torch.load(
        checkpoint_path,
        map_location=device
    )

    vanilla.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.backbone.load_state_dict(
        vanilla.backbone.state_dict()
    )

    with torch.no_grad():
        model.classifier.weight[:N_KNOWN].copy_(
            vanilla.classifier.weight
        )

        model.classifier.bias[:N_KNOWN].copy_(
            vanilla.classifier.bias
        )

    return model


def classifier_placeloss(logits,labels,beta=1.0):
    known_logits= logits[:,:N_KNOWN]
    dummy_logits= logits[:,N_KNOWN:]

    best_dummy= dummy_logits.max(
        dim=1,
        keepdim=True
    ).values

    combi= torch.cat(
        [known_logits,best_dummy],
        dim=1
    )

    known_loss= F.cross_entropy(
        combi,
        labels
    )

    masked_logits= combi.clone()

    batch_idx= torch.arange(
        labels.size(0),
        device=labels.device
    )

    masked_logits[batch_idx,labels]= -1e9

    dummy_labels= torch.full(
        (labels.size(0),),
        N_KNOWN,
        dtype=torch.long,
        device=labels.device
    )

    dummy_loss= F.cross_entropy(
        masked_logits,
        dummy_labels
    )

    total_loss= known_loss + beta * dummy_loss

    return total_loss,known_loss,dummy_loss


def different_class_pairs(labels):
    pair_indices= []

    for i in range(labels.size(0)):
        candidates= torch.where(
            labels != labels[i]
        )[0]

        if len(candidates) == 0:
            pair_indices.append(i)
            continue

        random_index= torch.randint(
            len(candidates),
            (1,),
            device=labels.device
        )

        pair_indices.append(
            candidates[random_index].item()
        )

    return torch.tensor(
        pair_indices,
        device=labels.device
    )


def data_placeholder_loss(model,images,labels,alpha=2.0):
    hidden= model.forward_to_layer2(images)

    pair_indices= different_class_pairs(labels)
    paired_hidden= hidden[pair_indices]

    beta_distribution= torch.distributions.Beta(
        alpha,
        alpha
    )

    lambdas= beta_distribution.sample(
        (images.size(0),)
    ).to(images.device)

    lambdas= lambdas.view(
        -1,
        1,
        1,
        1
    )

    mixed_hidden= (
        lambdas * hidden
        +
        (1 - lambdas) * paired_hidden
    )

    mixed_logits= model.forward_from_layer2(
        mixed_hidden
    )

    known_logits= mixed_logits[:,:N_KNOWN]
    dummy_logits= mixed_logits[:,N_KNOWN:]

    best_dummy= dummy_logits.max(
        dim=1,
        keepdim=True
    ).values

    combi= torch.cat(
        [known_logits,best_dummy],
        dim=1
    )

    dummy_labels= torch.full(
        (images.size(0),),
        N_KNOWN,
        dtype=torch.long,
        device=images.device
    )

    mixup_loss= F.cross_entropy(
        combi,
        dummy_labels
    )

    return mixup_loss