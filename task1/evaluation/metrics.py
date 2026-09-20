import torch
from sklearn.metrics import f1_score


def accuracy(predictions,labels):
    return (predictions == labels).float().mean().item()


def macro_f1(predictions,labels):
    return f1_score(labels.numpy(),predictions.numpy(),average="macro")


def prediction_consistency(clean_predictions,transformed_predictions):
    return (clean_predictions == transformed_predictions).float().mean().item()


def summarize_predictions(predictions,labels,confidences=None):
    result= {
        "accuracy":accuracy(predictions,labels),
        "macro_f1":macro_f1(predictions,labels)
    }

    if confidences is not None:
        result["mean_max_confidence"]= confidences.float().mean().item()

    return result
