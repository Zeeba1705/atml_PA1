import os
import argparse
import torch
import matplotlib.pyplot as plt

from task4.scores.scores import (
    msp_score,
    mls_score,
    fit_mahalanobis,
    mahalanobis_score
)


def load_outputs(cache_dir,split):
    return torch.load(
        os.path.join(
            cache_dir,
            f"vanilla_{split}_outputs.pt"
        ),
        map_location="cpu"
    )


def main(cache_dir,output_dir):
    os.makedirs(
        output_dir,
        exist_ok=True
    )

    train= load_outputs(
        cache_dir,
        "train"
    )

    test= load_outputs(
        cache_dir,
        "test"
    )

    near= load_outputs(
        cache_dir,
        "near"
    )

    far= load_outputs(
        cache_dir,
        "far"
    )


    means,variance= fit_mahalanobis(
        train["features"],
        train["labels"]
    )


    score_sets= {
        "MSP":{
            "Known":msp_score(
                test["logits"]
            ),
            "Near":msp_score(
                near["logits"]
            ),
            "Far":msp_score(
                far["logits"]
            )
        },

        "MLS":{
            "Known":mls_score(
                test["logits"]
            ),
            "Near":mls_score(
                near["logits"]
            ),
            "Far":mls_score(
                far["logits"]
            )
        },

        "Mahalanobis":{
            "Known":mahalanobis_score(
                test["features"],
                means,
                variance
            ),
            "Near":mahalanobis_score(
                near["features"],
                means,
                variance
            ),
            "Far":mahalanobis_score(
                far["features"],
                means,
                variance
            )
        }
    }


    figure,axes= plt.subplots(
        1,
        3,
        figsize=(12,3.5)
    )


    for axis,(name,scores) in zip(
        axes,
        score_sets.items()
    ):
        for group,values in scores.items():
            axis.hist(
                values.numpy(),
                bins=40,
                density=True,
                alpha=0.45,
                label=group
            )

        axis.set_title(name)
        axis.set_xlabel(
            "Unknownness score"
        )
        axis.set_ylabel(
            "Density"
        )

        axis.legend()


    figure.tight_layout()

    path= os.path.join(
        output_dir,
        "vanilla_score_distributions.png"
    )

    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )

    print(
        "Saved:",
        path
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
        args.cache_dir,
        args.output_dir
    )