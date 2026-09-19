import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from shared.pacs_protocol import make_source_loaders, make_target_eval_loader, SOURCE_DOMAINS
from task2.models.backbone import ResNetBackbone
from task2.models.classifier_head import ClassifierHead
from task2.evaluation.metrics import evaluate_model
from task2.evaluation.class_analysis import class_analysis
from task3.evaluation.source_domain_separability import source_domain_separability


SEED= 6304

CLASS_NAMES= [
    "dog",
    "elephant",
    "giraffe",
    "guitar",
    "horse",
    "house",
    "person"
]


def load_model(checkpoint_path,device):
    backbone= ResNetBackbone().to(device)
    classifier= ClassifierHead().to(device)

    checkpoint= torch.load(
        checkpoint_path,
        map_location=device
    )

    backbone.load_state_dict(checkpoint["backbone"])
    classifier.load_state_dict(checkpoint["classifier"])

    backbone.eval()
    classifier.eval()

    return backbone, classifier


def sharpness_proxy(backbone,classifier,source_val,device,rho=0.05):
    backbone.eval()
    classifier.eval()

    rng= np.random.default_rng(SEED)

    images_all= []
    labels_all= []

    for domain in SOURCE_DOMAINS:
        dataset= source_val[domain].dataset

        indices= rng.choice(
            len(dataset),
            size=32,
            replace=False
        )

        domain_images= []
        domain_labels= []

        for i in indices:
            image, label= dataset[int(i)]
            domain_images.append(image)
            domain_labels.append(label)

        images_all.append(torch.stack(domain_images))
        labels_all.append(torch.tensor(domain_labels))

    images= torch.cat(images_all,dim=0).to(device)
    labels= torch.cat(labels_all,dim=0).to(device)

    parameters= list(backbone.parameters()) + list(classifier.parameters())

    for p in parameters:
        p.grad= None

    logits= classifier(backbone(images))
    original_loss= F.cross_entropy(logits,labels)

    original_loss.backward()

    grad_norm= torch.norm(
        torch.stack([
            p.grad.norm(p=2)
            for p in parameters
            if p.grad is not None
        ]),
        p=2
    )

    scale= rho / (grad_norm + 1e-12)

    changes= []

    with torch.no_grad():
        for p in parameters:
            if p.grad is not None:
                delta= p.grad * scale
                p.add_(delta)
                changes.append((p,delta))

    with torch.no_grad():
        perturbed_logits= classifier(backbone(images))
        perturbed_loss= F.cross_entropy(perturbed_logits,labels)

    with torch.no_grad():
        for p, delta in changes:
            p.sub_(delta)

    for p in parameters:
        p.grad= None

    return {
        "original_loss": original_loss.item(),
        "perturbed_loss": perturbed_loss.item(),
        "sharpness": perturbed_loss.item() - original_loss.item()
    }


def evaluate_checkpoint(checkpoint_path,source_val,target_eval,device):
    backbone, classifier= load_model(checkpoint_path,device)

    source_results= {}

    for domain in SOURCE_DOMAINS:
        source_results[domain]= evaluate_model(
            backbone,
            classifier,
            source_val[domain],
            device
        )

    mean_accuracy= sum(
        source_results[d]["accuracy"]
        for d in SOURCE_DOMAINS
    ) / len(SOURCE_DOMAINS)

    mean_f1= sum(
        source_results[d]["macro_f1"]
        for d in SOURCE_DOMAINS
    ) / len(SOURCE_DOMAINS)

    worst_accuracy= min(
        source_results[d]["accuracy"]
        for d in SOURCE_DOMAINS
    )

    worst_f1= min(
        source_results[d]["macro_f1"]
        for d in SOURCE_DOMAINS
    )

    target_results= evaluate_model(
        backbone,
        classifier,
        target_eval,
        device
    )

    separability= source_domain_separability(
        backbone,
        source_val,
        device
    )

    sharpness= sharpness_proxy(
        backbone,
        classifier,
        source_val,
        device
    )

    target_class_analysis= class_analysis(
        backbone,
        classifier,
        target_eval,
        CLASS_NAMES,
        device
    )

    return {
        "source_validation": source_results,
        "mean_source_accuracy": mean_accuracy,
        "mean_source_macro_f1": mean_f1,
        "worst_source_accuracy": worst_accuracy,
        "worst_source_macro_f1": worst_f1,
        "sketch": target_results,
        "source_domain_separability": separability,
        "sharpness": sharpness,
        "class_analysis": target_class_analysis
    }


def main(args):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    os.makedirs(args.output_dir,exist_ok=True)

    _, source_val= make_source_loaders(
        pacs_root=args.pacs_root,
        split_path=args.split_path
    )

    print("Source validation loaders created")

    target_eval= make_target_eval_loader(
        pacs_root=args.pacs_root
    )

    print("Sketch loaded for final evaluation")

    checkpoints= {
        "erm": args.erm_checkpoint,
        "dan_dg": args.dan_dg_checkpoint,
        "sam": args.sam_checkpoint
    }

    results= {}

    for method, checkpoint_path in checkpoints.items():
        print(f"\n--- Evaluating {method} ---")

        results[method]= evaluate_checkpoint(
            checkpoint_path,
            source_val,
            target_eval,
            device
        )

    erm_accuracy= results["erm"]["sketch"]["accuracy"]
    erm_class_acc= results["erm"]["class_analysis"]["per_class_accuracy"]

    for method in results:
        results[method]["sketch_accuracy_change"]= (
            results[method]["sketch"]["accuracy"] - erm_accuracy
        )

        class_changes= {}

        method_class_acc= results[method]["class_analysis"]["per_class_accuracy"]

        for class_name in CLASS_NAMES:
            class_changes[class_name]= (
                method_class_acc[class_name] - erm_class_acc[class_name]
            )

        results[method]["per_class_accuracy_change"]= class_changes

    save_path= os.path.join(
        args.output_dir,
        "final_metrics.json"
    )

    with open(save_path,"w") as f:
        json.dump(results,f,indent=2)

    print("\n=== Main Results ===")

    for method in results:
        print(f"\n{method}")

        for domain in SOURCE_DOMAINS:
            print(
                domain,
                f"accuracy={results[method]['source_validation'][domain]['accuracy']:.4f}",
                f"macro_f1={results[method]['source_validation'][domain]['macro_f1']:.4f}"
            )

        print("mean source accuracy:", round(results[method]["mean_source_accuracy"],4))
        print("mean source F1:", round(results[method]["mean_source_macro_f1"],4))
        print("worst source accuracy:", round(results[method]["worst_source_accuracy"],4))
        print("worst source F1:", round(results[method]["worst_source_macro_f1"],4))
        print("Sketch accuracy:", round(results[method]["sketch"]["accuracy"],4))
        print("Sketch macro F1:", round(results[method]["sketch"]["macro_f1"],4))
        print("Sketch accuracy change:", round(results[method]["sketch_accuracy_change"],4))
        print("source separability:", round(results[method]["source_domain_separability"],4))
        print("sharpness:", round(results[method]["sharpness"]["sharpness"],4))

    print("\nSaved:", save_path)

    controlled_checkpoints= {
        "0.1": args.dan_dg_01_checkpoint,
        "1.0": args.dan_dg_checkpoint,
        "10.0": args.dan_dg_10_checkpoint
    }

    controlled_results= {}

    print("\n=== DAN-DG Controlled Study ===")

    for lambda_value, checkpoint_path in controlled_checkpoints.items():
        print(f"\nEvaluating lambda={lambda_value}")

        model_results= evaluate_checkpoint(
            checkpoint_path,
            source_val,
            target_eval,
            device
        )

        model_results["lambda_dg"]= float(lambda_value)

        model_results["sketch_accuracy_change"]= (
            model_results["sketch"]["accuracy"] - erm_accuracy
        )

        controlled_results[lambda_value]= model_results

        print("mean source F1:", round(model_results["mean_source_macro_f1"],4))
        print("Sketch accuracy:", round(model_results["sketch"]["accuracy"],4))
        print("Sketch macro F1:", round(model_results["sketch"]["macro_f1"],4))
        print("source separability:", round(model_results["source_domain_separability"],4))

    controlled_path= os.path.join(
        args.output_dir,
        "dan_dg_controlled_study.json"
    )

    with open(controlled_path,"w") as f:
        json.dump(controlled_results,f,indent=2)

    print("\nSaved:", controlled_path)


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument("--pacs_root",type=str,required=True)
    parser.add_argument("--split_path",type=str,default="splits/pacs_sketch_seed6304.json")

    parser.add_argument("--erm_checkpoint",type=str,required=True)
    parser.add_argument("--dan_dg_checkpoint",type=str,required=True)
    parser.add_argument("--sam_checkpoint",type=str,required=True)

    parser.add_argument("--dan_dg_01_checkpoint",type=str,required=True)
    parser.add_argument("--dan_dg_10_checkpoint",type=str,required=True)

    parser.add_argument("--output_dir",type=str,default="task3/results")

    args= parser.parse_args()

    main(args)
