import torch
import torch.nn.functional as F


def msp_score(logits):
    probabilities= F.softmax(logits,dim=1)
    max_probabilities= probabilities.max(dim=1).values
    return 1 - max_probabilities


def mls_score(logits):
    max_logits= logits.max(dim=1).values
    return -max_logits


def energy_score(logits):
    return -torch.logsumexp(logits,dim=1)


def fit_mahalanobis(features,labels,num_classes=10):
    class_means= []

    for class_idx in range(num_classes):
        class_features= features[labels == class_idx]
        class_mean= class_features.mean(dim=0)
        class_means.append(class_mean)

    class_means= torch.stack(class_means)

    residuals= features - class_means[labels]

    variance= (residuals ** 2).mean(dim=0)
    variance= variance + 1e-6

    return class_means,variance


def mahalanobis_score(features,class_means,variance):
    differences= features.unsqueeze(1) - class_means.unsqueeze(0)

    distances= (differences ** 2 / variance).sum(dim=2)

    return distances.min(dim=1).values