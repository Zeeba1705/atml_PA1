import os
import json
import argparse
import numpy as np
import torch

from task4.scores.scores import (
    msp_score,
    mls_score,
    energy_score,
    fit_mahalanobis,
    mahalanobis_score,
    fit_proser_bias,
    proser_placeholder_score
)

from task4.evaluation.metrics import evaluate_score
from task4.evaluation.failure_analysis import (
    per_unknown_class_analysis,
    accepted_failures
)


def load_outputs(cache_dir,method,split):
    path= os.path.join(
        cache_dir,
        f"{method}_{split}_outputs.pt"
    )

    return torch.load(
        path,
        map_location="cpu"
    )


def numpy_scores(scores):
    return scores.detach().cpu().numpy()


def closed_set_accuracy(outputs):
    logits= outputs["logits"][:,:10]
    labels= outputs["labels"]

    predictions= logits.argmax(dim=1)

    return (
        predictions == labels
    ).float().mean().item()


def main(cache_dir,output_dir):
    os.makedirs(
        output_dir,
        exist_ok=True
    )

    with open(
        os.path.join(
            cache_dir,
            "cifar100_classes.json"
        ),
        "r"
    ) as f:
        cifar100_classes= json.load(f)


    # =====================================
    # VANILLA
    # =====================================

    vanilla_train= load_outputs(
        cache_dir,
        "vanilla",
        "train"
    )

    vanilla_val= load_outputs(
        cache_dir,
        "vanilla",
        "val"
    )

    vanilla_test= load_outputs(
        cache_dir,
        "vanilla",
        "test"
    )

    vanilla_near= load_outputs(
        cache_dir,
        "vanilla",
        "near"
    )

    vanilla_far= load_outputs(
        cache_dir,
        "vanilla",
        "far"
    )


    score_functions= {
        "msp":msp_score,
        "mls":mls_score,
        "energy":energy_score
    }


    vanilla_results= {}


    for score_name,score_function in score_functions.items():
        val_scores= numpy_scores(
            score_function(
                vanilla_val["logits"]
            )
        )

        test_scores= numpy_scores(
            score_function(
                vanilla_test["logits"]
            )
        )

        near_scores= numpy_scores(
            score_function(
                vanilla_near["logits"]
            )
        )

        far_scores= numpy_scores(
            score_function(
                vanilla_far["logits"]
            )
        )

        vanilla_results[score_name]= evaluate_score(
            val_scores,
            test_scores,
            near_scores,
            far_scores
        )


    class_means,variance= fit_mahalanobis(
        vanilla_train["features"],
        vanilla_train["labels"]
    )


    mah_val= mahalanobis_score(
        vanilla_val["features"],
        class_means,
        variance
    )

    mah_test= mahalanobis_score(
        vanilla_test["features"],
        class_means,
        variance
    )

    mah_near= mahalanobis_score(
        vanilla_near["features"],
        class_means,
        variance
    )

    mah_far= mahalanobis_score(
        vanilla_far["features"],
        class_means,
        variance
    )


    vanilla_results["mahalanobis"]= evaluate_score(
        numpy_scores(mah_val),
        numpy_scores(mah_test),
        numpy_scores(mah_near),
        numpy_scores(mah_far)
    )


    # =====================================
    # MODEL COMPARISON USING MLS
    # =====================================

    model_results= {}


    for method in [
        "vanilla",
        "gcsc",
        "proser"
    ]:
        val= load_outputs(
            cache_dir,
            method,
            "val"
        )

        test= load_outputs(
            cache_dir,
            method,
            "test"
        )

        near= load_outputs(
            cache_dir,
            method,
            "near"
        )

        far= load_outputs(
            cache_dir,
            method,
            "far"
        )


        val_logits= val["logits"][:,:10]
        test_logits= test["logits"][:,:10]
        near_logits= near["logits"][:,:10]
        far_logits= far["logits"][:,:10]


        results= evaluate_score(
            numpy_scores(
                mls_score(val_logits)
            ),
            numpy_scores(
                mls_score(test_logits)
            ),
            numpy_scores(
                mls_score(near_logits)
            ),
            numpy_scores(
                mls_score(far_logits)
            )
        )


        results["closed_set_accuracy"]= float(
            closed_set_accuracy(test)
        )


        model_results[
            method + "_mls"
        ]= results


    # =====================================
    # PROSER PLACEHOLDER SCORE
    # =====================================

    proser_val= load_outputs(
        cache_dir,
        "proser",
        "val"
    )

    proser_test= load_outputs(
        cache_dir,
        "proser",
        "test"
    )

    proser_near= load_outputs(
        cache_dir,
        "proser",
        "near"
    )

    proser_far= load_outputs(
        cache_dir,
        "proser",
        "far"
    )


    proser_bias= fit_proser_bias(
        proser_val["logits"]
    )


    placeholder_val= proser_placeholder_score(
        proser_val["logits"],
        bias=proser_bias
    )

    placeholder_test= proser_placeholder_score(
        proser_test["logits"],
        bias=proser_bias
    )

    placeholder_near= proser_placeholder_score(
        proser_near["logits"],
        bias=proser_bias
    )

    placeholder_far= proser_placeholder_score(
        proser_far["logits"],
        bias=proser_bias
    )


    proser_placeholder_results= evaluate_score(
        numpy_scores(placeholder_val),
        numpy_scores(placeholder_test),
        numpy_scores(placeholder_near),
        numpy_scores(placeholder_far)
    )


    proser_placeholder_results[
        "closed_set_accuracy"
    ]= float(
        closed_set_accuracy(
            proser_test
        )
    )


    proser_placeholder_results[
        "calibration_bias"
    ]= float(
        proser_bias
    )


    model_results[
        "proser_placeholder"
    ]= proser_placeholder_results


    # =====================================
    # EXTRA CLASS-LEVEL ANALYSIS
    # vanilla MLS
    # =====================================

    vanilla_mls_threshold= vanilla_results[
        "mls"
    ]["threshold"]


    near_mls_tensor= mls_score(
        vanilla_near["logits"]
    )

    far_mls_tensor= mls_score(
        vanilla_far["logits"]
    )


    per_class_near= per_unknown_class_analysis(
        vanilla_near["logits"],
        vanilla_near["labels"],
        near_mls_tensor,
        vanilla_mls_threshold,
        cifar100_classes
    )


    per_class_far= per_unknown_class_analysis(
        vanilla_far["logits"],
        vanilla_far["labels"],
        far_mls_tensor,
        vanilla_mls_threshold,
        cifar100_classes
    )


    # =====================================
    # REQUIRED FAILURE EXAMPLES
    # =====================================

    near_failures= accepted_failures(
        vanilla_near["logits"],
        vanilla_near["labels"],
        near_mls_tensor,
        vanilla_mls_threshold,
        cifar100_classes,
        n=10
    )


    far_failures= accepted_failures(
        vanilla_far["logits"],
        vanilla_far["labels"],
        far_mls_tensor,
        vanilla_mls_threshold,
        cifar100_classes,
        n=10
    )


    # =====================================
    # SCORE SUMMARIES
    # =====================================

    score_summary= {}


    for score_name,score_function in score_functions.items():
        score_summary[score_name]= {}

        for split_name,data in [
            ("known",vanilla_test),
            ("near",vanilla_near),
            ("far",vanilla_far)
        ]:
            scores= score_function(
                data["logits"]
            )

            score_summary[
                score_name
            ][split_name]= {
                "mean":float(
                    scores.mean().item()
                ),

                "median":float(
                    scores.median().item()
                ),

                "q25":float(
                    torch.quantile(
                        scores,
                        0.25
                    ).item()
                ),

                "q75":float(
                    torch.quantile(
                        scores,
                        0.75
                    ).item()
                )
            }


    # =====================================
    # SAVE EVERYTHING
    # =====================================

    final_results= {
        "vanilla_scores":
            vanilla_results,

        "model_comparison":
            model_results,

        "per_class_near":
            per_class_near,

        "per_class_far":
            per_class_far,

        "near_failures":
            near_failures,

        "far_failures":
            far_failures,

        "score_summary":
            score_summary
    }


    output_path= os.path.join(
        output_dir,
        "task4_results.json"
    )


    with open(output_path,"w") as f:
        json.dump(
            final_results,
            f,
            indent=2
        )


    print(
        json.dumps(
            final_results,
            indent=2
        )
    )


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--cache_dir",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        required=True
    )

    args= parser.parse_args()

    main(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir
    )