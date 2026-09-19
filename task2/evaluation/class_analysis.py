import torch
import numpy as np

from sklearn.metrics import confusion_matrix


def class_analysis(backbone,classifier,target_loader,class_names,device):
    backbone.eval()
    classifier.eval()

    labels_all= []
    preds_all= []

    with torch.no_grad():
        for images, labels in target_loader:
            images = images.to(device)

            features= backbone(images)
            logits= classifier(features)
            preds= logits.argmax(dim=1)

            labels_all.extend(labels.tolist())
            preds_all.extend(preds.cpu().tolist())

    labels_all = np.array(labels_all)
    preds_all = np.array(preds_all)

    matrix= confusion_matrix(
        labels_all,
        preds_all,
        labels=list(range(len(class_names)))
    )

    class_acc= {}
    confusions= {}

    for class_id, class_name in enumerate(class_names):

        mask= labels_all == class_id
        total= mask.sum()

        if total > 0:
            correct = (preds_all[mask] == class_id).sum()

            accuracy = correct / total
        else:
            accuracy = 0.0

        class_acc[class_name] = float(accuracy)

        row = matrix[class_id].copy()
        row[class_id] = 0

        if row.max() > 0:
            confused_with= row.argmax()
            
            confusions[class_name]= {
        "predicted_as": class_names[confused_with],
        "count": int(row[confused_with])
         }

        else:
            confusions[class_name]= {"predicted_as": None,
        "count": 0
        }

    return {
        "per_class_accuracy": class_acc,
        "dominant_confusions": confusions,
        "confusion_matrix": matrix.tolist()
    }