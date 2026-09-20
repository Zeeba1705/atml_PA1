import os
import json
import argparse
import matplotlib.pyplot as plt


def plot_group(data,title,path):
    classes= list(
        data.keys()
    )

    rejection_rates= [
        data[class_name][
            "rejection_rate"
        ]
        for class_name in classes
    ]

    plt.figure(
        figsize=(9,4)
    )

    plt.bar(
        classes,
        rejection_rates
    )

    plt.axhline(
        0.5,
        linestyle="--"
    )

    plt.ylabel(
        "Rejection rate"
    )

    plt.title(
        title
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.ylim(
        0,
        1
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def main(results_path,output_dir):
    os.makedirs(
        output_dir,
        exist_ok=True
    )

    with open(results_path,"r") as f:
        results= json.load(f)

    plot_group(
        results["per_class_near"],
        "Vanilla MLS: Near Unknown Rejection by Class",
        os.path.join(
            output_dir,
            "near_class_rejection.png"
        )
    )

    plot_group(
        results["per_class_far"],
        "Vanilla MLS: Far Unknown Rejection by Class",
        os.path.join(
            output_dir,
            "far_class_rejection.png"
        )
    )


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--results_path",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        required=True
    )

    args= parser.parse_args()

    main(
        args.results_path,
        args.output_dir
    )