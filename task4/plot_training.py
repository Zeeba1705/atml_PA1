import os
import json
import argparse
import matplotlib.pyplot as plt


def load_history(path):
    with open(path,"r") as f:
        return json.load(f)


def plot_standard_training(method,history,output_dir):
    epochs= [row["epoch"] for row in history]
    train_acc= [row["train_acc"] for row in history]
    val_acc= [row["val_acc"] for row in history]

    plt.figure(figsize=(6,4))

    plt.plot(epochs,train_acc,label="Train accuracy")
    plt.plot(epochs,val_acc,label="Validation accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{method.upper()} CIFAR-10 Training")
    plt.legend()
    plt.tight_layout()

    path= os.path.join(
        output_dir,
        f"{method}_training_accuracy.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:",path)


def plot_proser_training(history,output_dir):
    epochs= [row["epoch"] for row in history]

    train_acc= [
        row["train_acc"]
        for row in history
    ]

    val_acc= [
        row["val_acc"]
        for row in history
    ]

    classifier_loss= [
        row["classifier_placeholder_loss"]
        for row in history
    ]

    known_loss= [
        row["known_loss"]
        for row in history
    ]

    dummy_loss= [
        row["dummy_loss"]
        for row in history
    ]

    mixup_loss= [
        row["data_placeholder_loss"]
        for row in history
    ]


    # PROSER accuracy
    plt.figure(figsize=(6,4))

    plt.plot(
        epochs,
        train_acc,
        label="Train accuracy"
    )

    plt.plot(
        epochs,
        val_acc,
        label="Validation accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("PROSER CIFAR-10 Training")
    plt.legend()
    plt.tight_layout()

    path= os.path.join(
        output_dir,
        "proser_training_accuracy.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:",path)


    # PROSER placeholder losses
    plt.figure(figsize=(6,4))

    plt.plot(
        epochs,
        classifier_loss,
        label="Classifier placeholder (L1)"
    )

    plt.plot(
        epochs,
        mixup_loss,
        label="Data placeholder (L2)"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PROSER Placeholder Losses")
    plt.legend()
    plt.tight_layout()

    path= os.path.join(
        output_dir,
        "proser_placeholder_losses.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:",path)


    # Breakdown of classifier-placeholder loss
    plt.figure(figsize=(6,4))

    plt.plot(
        epochs,
        known_loss,
        label="Known-class loss"
    )

    plt.plot(
        epochs,
        dummy_loss,
        label="Dummy loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PROSER Classifier-Placeholder Components")
    plt.legend()
    plt.tight_layout()

    path= os.path.join(
        output_dir,
        "proser_classifier_loss_components.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:",path)


def main(results_dir,output_dir):
    os.makedirs(
        output_dir,
        exist_ok=True
    )


    for method in ["vanilla","gcsc"]:
        history_path= os.path.join(
            results_dir,
            f"{method}_history.json"
        )

        if not os.path.exists(history_path):
            print("Missing:",history_path)
            continue

        history= load_history(
            history_path
        )

        plot_standard_training(
            method,
            history,
            output_dir
        )


    proser_path= os.path.join(
        results_dir,
        "proser_history.json"
    )

    if os.path.exists(proser_path):
        proser_history= load_history(
            proser_path
        )

        plot_proser_training(
            proser_history,
            output_dir
        )

    else:
        print(
            "PROSER history not found yet."
        )


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--results_dir",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        required=True
    )

    args= parser.parse_args()

    main(
        results_dir=args.results_dir,
        output_dir=args.output_dir
    )