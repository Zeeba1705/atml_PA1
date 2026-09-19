import torch
import torch.nn.functional as F

from task2.methods.dan import mmd_loss


def dan_dg_loss(backbone, classifier, batches, device, lambda_dg=1.0):
    features = {}
    class_losses = []

    for domain in ["photo", "art_painting", "cartoon"]:
        images, labels= batches[domain]

        images= images.to(device)
        labels= labels.to(device)

        feats= backbone(images)
        logits= classifier(feats)

        features[domain]= feats
        class_losses.append(F.cross_entropy(logits, labels))

    classification_loss = sum(class_losses) / len(class_losses)

    photo_art_mmd = mmd_loss(features["photo"], features["art_painting"])
    photo_cartoon_mmd = mmd_loss(features["photo"], features["cartoon"])
    art_cartoon_mmd = mmd_loss(features["art_painting"], features["cartoon"])

    alignment_loss= (photo_art_mmd + photo_cartoon_mmd + art_cartoon_mmd) / 3

    total_loss= classification_loss + lambda_dg * alignment_loss

    return total_loss, classification_loss, alignment_loss