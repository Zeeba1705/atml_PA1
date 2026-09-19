import os
import json
import argparse
import torch

from shared.pacs_protocol import make_pacs_loaders, SOURCE_DOMAINS
from task2.models.backbone import ResNetBackbone
from task2.models.classifier_head import ClassifierHead
from task2.evaluation.metrics import evaluate_model
from task2.evaluation.domain_separability import domain_separability
from task2.evaluation.class_analysis import class_analysis

def load_model(checkpoint_path, device):
    backbone= ResNetBackbone().to(device)
    classifier= ClassifierHead().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    backbone.load_state_dict(
        checkpoint["backbone"]
    )

    classifier.load_state_dict(
        checkpoint["classifier"]
    )

    return backbone, classifier

def evaluate_checkpoint(checkpoint_path,source_val,target_eval,device):
    backbone, classifier= load_model(checkpoint_path,device)

    results= {}
    source_results= {}

    for domain in SOURCE_DOMAINS:
        scores= evaluate_model(
            backbone,
            classifier,
            source_val[domain],
            device
        )

        source_results[domain] = scores

    results["source_validation"]= source_results

    mean_acc= sum(
        source_results[d]["accuracy"]
        for d in SOURCE_DOMAINS
    ) / len(SOURCE_DOMAINS)

    mean_f1= sum(
        source_results[d]["macro_f1"]
        for d in SOURCE_DOMAINS
    ) / len(SOURCE_DOMAINS)

    results["mean_source_accuracy"] = mean_acc
    results["mean_source_macro_f1"] = mean_f1

    target_scores= evaluate_model(
        backbone,
        classifier,
        target_eval,
        device
    )

    results["target"] = target_scores

    sep_score = domain_separability(
    backbone,
    source_val,
    target_eval,
    device
)

    results["domain_separability"] = sep_score

    class_names = target_eval.dataset.classes
    class_results = class_analysis(
    backbone,
    classifier,
    target_eval,
    class_names,
    device)

    results["class_analysis"]= class_results

    return results

def main(args):

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    _, source_val, _, target_eval = make_pacs_loaders(pacs_root=args.pacs_root)
    
    checkpoints = {
        "source_only": os.path.join(
            args.checkpoint_dir,
            "source_only_best.pt"
        ),

        "dan": os.path.join(
            args.checkpoint_dir,
            "dan_lambda_1.0_best.pt"
        ),

        "dann": os.path.join(
            args.checkpoint_dir,
            "dann_best.pt"
        ),

        "cdan": os.path.join(
            args.checkpoint_dir,
            "cdan_best.pt"
        ),
    }

    all_results = {}

    for method in checkpoints:
        checkpoint_path= checkpoints[method]

        print("\nEvaluating", method)

        method_result = evaluate_checkpoint(
            checkpoint_path,
            source_val,
            target_eval,
            device
        )

        all_results[method]= method_result

        print(
       method,"target acc:", round(method_result["target"]["accuracy"], 4), "target f1:",
       round(method_result["target"]["macro_f1"], 4), "domain sep:", round(method_result["domain_separability"],4))

    baseline_acc= all_results["source_only"]["target"]["accuracy"]
    for method in all_results:
        target_acc= all_results[method]["target"]["accuracy"]

        all_results[method]["target_accuracy_change"]= target_acc-baseline_acc

    baseline_classes = all_results[ "source_only"]["class_analysis"]["per_class_accuracy"]

    for method in all_results:

        method_classes = all_results[method]["class_analysis"]["per_class_accuracy"]

        class_changes = {}

        for class_name in baseline_classes:
            class_changes[class_name] = (
                method_classes[class_name]
                - baseline_classes[class_name]
            )

        all_results[method]["per_class_accuracy_change"]= class_changes
    
    output_path = os.path.join(
        args.checkpoint_dir,
        "final_metrics.json"
    )

    with open(output_path, "w") as f:
        json.dump(
            all_results,
            f,
            indent=2
        )

    print("\nSaved final metrics to:")
    print(output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pacs_root",
        type=str,
        default="datasets/PACS/kfold"
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="task2/results"
    )

    args = parser.parse_args()

    main(args)