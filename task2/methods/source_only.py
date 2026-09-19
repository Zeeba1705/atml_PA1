import torch
from shared.pacs_protocol import SOURCE_DOMAINS
from task2.models.backbone import freeze_batchnorm_stats

def src_train(backbone, classifier, src_iter, src_loader, optimizer, criterion, device):
    backbone.train()
    classifier.train()

    freeze_batchnorm_stats(backbone)

    src_imgs=[]
    src_labels=[]
    
    for domain in SOURCE_DOMAINS:
        try:
            images, labels= next(src_iter[domain])
        except StopIteration:
            src_iter[domain] = iter(src_loader[domain])
            images, labels = next(src_iter[domain])

        src_imgs.append(images)
        src_labels.append(labels)

    src_imgs= torch.cat(src_imgs, dim=0)
    src_labels=torch.cat(src_labels, dim=0)

    src_imgs= src_imgs.to(device)
    src_labels= src_labels.to(device)

    optimizer.zero_grad()

    features= backbone(src_imgs)
    logits= classifier(features)

    loss= criterion(logits, src_labels)
    loss.backward()
    optimizer.step()

    return loss.item()

    

