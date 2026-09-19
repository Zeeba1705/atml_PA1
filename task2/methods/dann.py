import torch

from shared.pacs_protocol import SOURCE_DOMAINS
from task2.models.backbone import freeze_batchnorm_stats


class GradientReversal(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha):
    return GradientReversal.apply(x, alpha)


def dann_train(backbone,classifier,discriminator,src_iter,src_loader,target_imgs,optimizer,criterion,
    device,
    alpha
):

    backbone.train()
    classifier.train()
    discriminator.train()

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

    src_imgs = torch.cat(src_imgs, dim=0)
    src_labels = torch.cat(src_labels, dim=0)

    src_imgs = src_imgs.to(device)
    src_labels = src_labels.to(device)
    target_imgs = target_imgs.to(device)

    optimizer.zero_grad()

    src_feats = backbone(src_imgs)
    target_feats = backbone(target_imgs)

    source_logits = classifier(src_feats)

    cls_loss = criterion(
        source_logits,
        src_labels
    )

    all_feats = torch.cat(
        [src_feats, target_feats],
        dim=0
    )

    feature_var = all_feats.var(
        dim=0
    ).mean()

    reversed_feats = grad_reverse(
        all_feats,
        alpha
    )

    source_domain_labels = torch.zeros(
        src_feats.size(0),
        dtype=torch.long,
        device=device
    )

    target_domain_labels = torch.ones(
        target_feats.size(0),
        dtype=torch.long,
        device=device
    )

    domain_labels = torch.cat(
        [
            source_domain_labels,
            target_domain_labels
        ]
    )

    domain_logits = discriminator(
        reversed_feats
    )

    domain_loss = criterion(
        domain_logits,
        domain_labels
    )

    domain_preds = domain_logits.argmax(dim=1)

    domain_acc = (
        domain_preds == domain_labels
    ).float().mean()

    total_loss = (
        cls_loss
        + domain_loss
    )

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
    list(backbone.parameters())
    + list(classifier.parameters())
    + list(discriminator.parameters()),
    max_norm=1.0
)
    optimizer.step()

    return (
        total_loss.item(),
        cls_loss.item(),
        domain_loss.item(),
        domain_acc.item(),
        feature_var.item()
    )