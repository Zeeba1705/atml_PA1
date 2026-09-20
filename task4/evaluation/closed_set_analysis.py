import torch


CIFAR10_CLASSES= [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


def per_class_accuracy(logits,labels):
    predictions= logits[:,:10].argmax(dim=1)

    results= {}

    for class_id,class_name in enumerate(CIFAR10_CLASSES):
        mask= labels == class_id

        accuracy= (
            predictions[mask] == labels[mask]
        ).float().mean().item()

        results[class_name]= float(accuracy)

    return results