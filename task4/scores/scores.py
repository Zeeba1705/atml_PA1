import torch
import torch.nn.functional as F


def msp_score(logits):
    probabilities= F.softmax(logits,dim=1)
    return 1 - probabilities.max(dim=1).values


def mls_score(logits):
    return -logits.max(dim=1).values


def energy_score(logits):
    return -torch.logsumexp(logits,dim=1)


def fit_mahalanobis(features,labels,num_classes=10):
    class_means= []

    for class_idx in range(num_classes):
        class_features= features[labels == class_idx]
        class_means.append(class_features.mean(dim=0))

    class_means= torch.stack(class_means)

    residuals= features - class_means[labels]
    variance= (residuals ** 2).mean(dim=0)
    variance= variance + 1e-6

    return class_means,variance


def mahalanobis_score(features,class_means,variance):
    differences= features.unsqueeze(1) - class_means.unsqueeze(0)
    distances= (differences ** 2 / variance).sum(dim=2)

    return distances.min(dim=1).values


def proser_placeholder_score(logits,bias=0.0,temperature=1024.0):
    known_logits= logits[:,:10]
    dummy_logits= logits[:,10:]

    strongest_dummy= dummy_logits.max(dim=1,keepdim=True).values
    strongest_dummy= strongest_dummy + bias

    effective_logits= torch.cat(
        [known_logits,strongest_dummy],
        dim=1
    )

    probabilities= F.softmax(
        effective_logits / temperature,
        dim=1
    )

    dummy_probability= probabilities[:,-1]
    max_known_probability= probabilities[:,:10].max(dim=1).values

    return dummy_probability - max_known_probability


def fit_proser_bias(val_logits):
    known_max= val_logits[:,:10].max(dim=1).values
    dummy_max= val_logits[:,10:].max(dim=1).values

    margins= known_max - dummy_max

    bias= torch.quantile(
        margins,
        0.05
    )

    return bias.item()