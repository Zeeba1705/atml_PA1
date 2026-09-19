import torch
from shared.pacs_protocol import SOURCE_DOMAINS
from task2.models.backbone import freeze_batchnorm_stats

def rbf_kernel(x,y, median):
    kernel=0

    x_2= (x**2).sum(dim=1, keepdim=True)
    y_2= (y**2).sum(dim=1, keepdim=True)

    distances = (x_2+ y_2.T - 2 * x @ y.T )

    for strength in [0.5, 1.0, 2.0]:
        bw= strength* median
        rbf= torch.exp(-distances / (2 * bw))

        kernel+= rbf
    
    return kernel

def median_dist(src_feats, target_feats):
    combined = torch.cat([src_feats, target_feats],dim=0)
    x_2= (combined**2).sum(dim=1, keepdim=True)
    y_2= (combined**2).sum(dim=1, keepdim=True)

    distances= (x_2+ y_2.T - 2 * combined @ combined.T).clamp(min=0)

    mask= ~torch.eye(
        distances.size(0),
        dtype=torch.bool,
        device=distances.device
    )

    values= distances[mask]

    return values.median().detach()

def mmd_loss(source_features, target_features):

    median = median_dist(
        source_features,
        target_features
    )

    median= median.clamp(min=1e-6).clamp(min=0)

    k_ss =rbf_kernel(
        source_features,
        source_features,
        median
    )

    k_tt =rbf_kernel(
        target_features,
        target_features,
        median
    )

    k_st =rbf_kernel(
        source_features,
        target_features,
        median
    )

    return (k_ss.mean()+ k_tt.mean()- 2 * k_st.mean())
   

def dan_train(backbone, classifier, src_iter, src_loader,target_imgs, optimizer, criterion, device, lambda_mmd=1.0):
    backbone.train()
    classifier.train()

    freeze_batchnorm_stats(backbone)

    src_imgs = []
    src_labels = []

    for domain in SOURCE_DOMAINS:

        try:
            images, labels = next(src_iter[domain])

        except StopIteration:
            src_iter[domain] = iter(src_loader[domain])
            images, labels = next(src_iter[domain])

        src_imgs.append(images)
        src_labels.append(labels)

    src_imgs= torch.cat(src_imgs, dim=0)
    src_labels= torch.cat(src_labels, dim=0)

    src_imgs= src_imgs.to(device)
    src_labels= src_labels.to(device)
    target_imgs= target_imgs.to(device)

    optimizer.zero_grad()

    src_feats= backbone(src_imgs)
    target_feats=backbone(target_imgs)

    source_logits= classifier(src_feats)

    cls_loss= criterion(source_logits,src_labels)
    alignment_loss= mmd_loss(src_feats, target_feats)

    tot_loss= cls_loss+ lambda_mmd*alignment_loss

    tot_loss.backward()
    optimizer.step()

    return (tot_loss.item(), cls_loss.item(), alignment_loss.item())
