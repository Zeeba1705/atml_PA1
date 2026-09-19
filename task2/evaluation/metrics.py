from sklearn.metrics import accuracy_score, f1_score
import torch

def classific_metrics(labels, predictions):
    
    accuracy= accuracy_score(labels, predictions)
    macro_f1= f1_score(labels,predictions,average="macro")

    return {"accuracy": accuracy,"macro_f1": macro_f1}

def evaluate_model(backbone, classifier, loader, device):
    backbone.eval()
    classifier.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            features = backbone(images)
            logits = classifier(features)

            preds = logits.argmax(dim=1)

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())

    return classific_metrics(all_labels,all_preds)