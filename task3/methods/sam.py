import torch
import torch.nn.functional as F

from shared.pacs_protocol import SOURCE_DOMAINS
from task2.models.backbone import freeze_batchnorm_stats

def source_loss(backbone, classifier, batches, device):
    losses=[]

    for domain in SOURCE_DOMAINS:
        images,labels =batches[domain]

        images= images.to(device)
        labels= labels.to(device)

        features= backbone(images)
        logits= classifier(features)

        losses.append(F.cross_entropy(logits,labels))

    return sum(losses) / len(losses)

def get_perturbs(parameters, rho):
    grad_norm= torch.norm(torch.stack([p.grad.norm(p=2) for p in parameters if p.grad is not None]),
        p=2
    )

    scale= rho / (grad_norm + 1e-12)

    perturbations= {}

    with torch.no_grad():
        for p in parameters:
            if p.grad is not None:
                e_w= p.grad * scale
                p.add_(e_w)
                perturbations[p]= e_w

    return perturbations

def sam_train(backbone, classifier, batches, optimizer, device, rho=0.05):
    backbone.train()
    classifier.train()

    freeze_batchnorm_stats(backbone)

    optimizer.zero_grad()

    first_loss= source_loss(backbone,classifier,batches,device)
    first_loss.backward()

    parameters= list(backbone.parameters()) + list(classifier.parameters())

    perturbations= get_perturbs(parameters,rho)

    optimizer.zero_grad()

    freeze_batchnorm_stats(backbone)

    second_loss= source_loss(backbone,classifier,batches,device)
    second_loss.backward()

    with torch.no_grad():
        for p, e_w in perturbations.items():
            p.sub_(e_w)

    optimizer.step()

    return first_loss.item(), second_loss.item()