import os
import json
import argparse
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_history_list(history):
    if isinstance(history, list):
        return history

    for key in ["history", "epochs", "records"]:
        if key in history:
            return history[key]

    raise ValueError("Could not find epoch history.")


def get_value(row, possible_keys):
    for key in possible_keys:
        if key in row:
            return row[key]

    return None


def plot_classification_loss(checkpoint_dir, output_dir):
    files = {
        "Source-only": "source_only_history.json",
        "DAN": "dan_lambda_1.0_history.json",
        "DANN": "dann_history.json",
        "CDAN": "cdan_history.json"
    }

    plt.figure(figsize=(7, 5))

    for method, filename in files.items():
        path = os.path.join(checkpoint_dir, filename)
        history = get_history_list(load_json(path))

        epochs = []
        losses = []

        for i, row in enumerate(history):
            loss = get_value(row, ["classification_loss", "class_loss", "cls_loss"])

            if loss is not None:
                epoch = row.get("epoch", i + 1)
                epochs.append(epoch)
                losses.append(loss)

        if len(losses) > 0:
            plt.plot(epochs, losses, marker="o", label=method)

    plt.xlabel("Epoch")
    plt.ylabel("Classification loss")
    plt.title("Task 2 Classification Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "classification_loss.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Saved:", output_path)


def plot_alignment_losses(checkpoint_dir, output_dir):
    files = {
        "DAN MMD": ("dan_lambda_1.0_history.json", ["mmd_loss", "alignment_loss"]),
        "DANN domain": ("dann_history.json", ["domain_loss"]),
        "CDAN domain": ("cdan_history.json", ["domain_loss"])
    }

    plt.figure(figsize=(7, 5))

    for label, (filename, possible_keys) in files.items():
        path = os.path.join(checkpoint_dir, filename)
        history = get_history_list(load_json(path))

        epochs = []
        losses = []

        for i, row in enumerate(history):
            loss = get_value(row, possible_keys)

            if loss is not None:
                epoch = row.get("epoch", i + 1)
                epochs.append(epoch)
                losses.append(loss)

        if len(losses) > 0:
            plt.plot(epochs, losses, marker="o", label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Alignment / domain loss")
    plt.title("Task 2 Alignment Losses")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "alignment_losses.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Saved:", output_path)


def plot_source_f1(checkpoint_dir, output_dir):
    files = {
        "Source-only": "source_only_history.json",
        "DAN": "dan_lambda_1.0_history.json",
        "DANN": "dann_history.json",
        "CDAN": "cdan_history.json"
    }

    plt.figure(figsize=(7, 5))

    for method, filename in files.items():
        path = os.path.join(checkpoint_dir, filename)
        history = get_history_list(load_json(path))

        epochs = []
        scores = []

        for i, row in enumerate(history):
            score = get_value(row, ["mean_source_macro_f1", "mean_source_f1", "mean_f1"])

            if score is not None:
                epoch = row.get("epoch", i + 1)
                epochs.append(epoch)
                scores.append(score)

        if len(scores) > 0:
            plt.plot(epochs, scores, marker="o", label=method)

    plt.xlabel("Epoch")
    plt.ylabel("Mean source validation macro-F1")
    plt.title("Task 2 Source Validation Performance")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "source_validation_f1.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Saved:", output_path)


def plot_controlled_study(checkpoint_dir, output_dir):
    path = os.path.join(checkpoint_dir, "dan_controlled_study.json")
    results = load_json(path)

    lambdas = ["0.1", "1.0", "10.0"]

    source_f1 = [results[x]["mean_source_macro_f1"] for x in lambdas]
    target_acc = [results[x]["target"]["accuracy"] for x in lambdas]
    domain_sep = [results[x]["domain_separability"] for x in lambdas]

    x = range(len(lambdas))

    plt.figure(figsize=(7, 5))

    plt.plot(x, source_f1, marker="o", label="Mean source macro-F1")
    plt.plot(x, target_acc, marker="o", label="Sketch accuracy")
    plt.plot(x, domain_sep, marker="o", label="Domain separability")

    plt.xticks(x, lambdas)
    plt.xlabel("DAN λ")
    plt.ylabel("Score")
    plt.title("DAN Alignment-Strength Study")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "dan_controlled_study.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Saved:", output_path)


def main(args):
    output_dir = os.path.join(args.checkpoint_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    plot_classification_loss(args.checkpoint_dir, output_dir)
    plot_alignment_losses(args.checkpoint_dir, output_dir)
    plot_source_f1(args.checkpoint_dir, output_dir)
    plot_controlled_study(args.checkpoint_dir, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="task2/results")
    args = parser.parse_args()

    main(args)