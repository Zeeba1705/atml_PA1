import os
import json
import argparse
import matplotlib.pyplot as plt

from torchvision import datasets


def plot_failures(dataset,failures,original_indices,title,path,n=6):
    selected= failures[:n]

    figure,axes= plt.subplots(
        2,
        3,
        figsize=(9,6)
    )

    axes= axes.flatten()

    for axis,failure in zip(axes,selected):
        subset_index= failure["index"]

        original_index= original_indices[
            subset_index
        ]

        image,_= dataset[
            original_index
        ]

        axis.imshow(image)

        axis.set_title(
            f'{failure["unknown_class"]}\n'
            f'→ {failure["predicted_known_class"]}\n'
            f'score={failure["score"]:.3f}\n'
            f'thresh={failure["threshold"]:.3f}'
        )

        axis.axis("off")

    for axis in axes[len(selected):]:
        axis.axis("off")

    figure.suptitle(title)

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)


def main(data_root,results_path,indices_path,output_dir):
    os.makedirs(
        output_dir,
        exist_ok=True
    )

    with open(results_path,"r") as f:
        results= json.load(f)

    with open(indices_path,"r") as f:
        unknown_indices= json.load(f)

    cifar100= datasets.CIFAR100(
        root=data_root,
        train=False,
        download=True
    )

    plot_failures(
        cifar100,
        results["near_failures"],
        unknown_indices["near"],
        "Accepted Near Unknowns",
        os.path.join(
            output_dir,
            "near_failures.png"
        )
    )

    plot_failures(
        cifar100,
        results["far_failures"],
        unknown_indices["far"],
        "Accepted Far Unknowns",
        os.path.join(
            output_dir,
            "far_failures.png"
        )
    )


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        required=True
    )

    parser.add_argument(
        "--results_path",
        required=True
    )

    parser.add_argument(
        "--indices_path",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        required=True
    )

    args= parser.parse_args()

    main(
        data_root=args.data_root,
        results_path=args.results_path,
        indices_path=args.indices_path,
        output_dir=args.output_dir
    )