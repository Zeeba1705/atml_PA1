import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def load_json(path):
    with open(path,"r") as f:
        return json.load(f)


def main(args):
    os.makedirs(args.figure_dir,exist_ok=True)

    results= load_json(
        os.path.join(args.task3_dir,"final_metrics.json")
    )

    controlled= load_json(
        os.path.join(args.task3_dir,"dan_dg_controlled_study.json")
    )

    rows= []

    for method in ["erm","dan_dg","sam"]:
        r= results[method]

        rows.append({
            "Method": method.upper(),
            "Photo F1": r["source_validation"]["photo"]["macro_f1"],
            "Art F1": r["source_validation"]["art_painting"]["macro_f1"],
            "Cartoon F1": r["source_validation"]["cartoon"]["macro_f1"],
            "Mean Source F1": r["mean_source_macro_f1"],
            "Worst Source F1": r["worst_source_macro_f1"],
            "Sketch Acc": r["sketch"]["accuracy"],
            "Sketch F1": r["sketch"]["macro_f1"],
            "Sketch Acc Change": r["sketch_accuracy_change"],
            "Source Separability": r["source_domain_separability"],
            "Sharpness": r["sharpness"]["sharpness"]
        })

    main_df= pd.DataFrame(rows)

    print("\n=== TASK 3 MAIN RESULTS ===")
    print(main_df.round(4).to_string(index=False))

    main_df.to_csv(
        os.path.join(args.task3_dir,"task3_main_table.csv"),
        index=False
    )

   
    dan_history= load_json(
        os.path.join(args.task3_dir,"dan_dg_lambda_1.0_history.json")
    )

    sam_history= load_json(
        os.path.join(args.task3_dir,"sam_rho_0.05_history.json")
    )

    plt.figure(figsize=(7,5))

    plt.plot(
        [row["epoch"] for row in dan_history],
        [row["classification_loss"] for row in dan_history],
        marker="o",
        label="DAN-DG"
    )

    plt.plot(
        [row["epoch"] for row in sam_history],
        [row["train_loss"] for row in sam_history],
        marker="o",
        label="SAM"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Classification Loss")
    plt.title("Task 3 Classification Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        os.path.join(args.figure_dir,"classification_loss.png"),
        dpi=200
    )

    plt.show()

    plt.figure(figsize=(7,5))

    plt.plot(
        [row["epoch"] for row in dan_history],
        [row["mmd_loss"] for row in dan_history],
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Pairwise MMD")
    plt.title("DAN-DG Alignment Loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        os.path.join(args.figure_dir,"dan_dg_mmd.png"),
        dpi=200
    )

    plt.show()

    plt.figure(figsize=(7,5))

    plt.plot(
        [row["epoch"] for row in dan_history],
        [row["mean_source_macro_f1"] for row in dan_history],
        marker="o",
        label="DAN-DG"
    )

    plt.plot(
        [row["epoch"] for row in sam_history],
        [row["mean_source_macro_f1"] for row in sam_history],
        marker="o",
        label="SAM"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Mean Source Macro-F1")
    plt.title("Source Validation Performance")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        os.path.join(args.figure_dir,"source_validation_f1.png"),
        dpi=200
    )

    plt.show()

    plt.figure(figsize=(7,5))

    epochs= [row["epoch"] for row in sam_history]

    plt.plot(
        epochs,
        [row["train_loss"] for row in sam_history],
        marker="o",
        label="Clean Loss"
    )

    plt.plot(
        epochs,
        [row["perturbed_loss"] for row in sam_history],
        marker="o",
        label="Perturbed Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SAM Clean vs Perturbed Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        os.path.join(args.figure_dir,"sam_losses.png"),
        dpi=200
    )

    plt.show()


    controlled_rows= []

    for value in ["0.1","1.0","10.0"]:
        r= controlled[value]

        controlled_rows.append({
            "Lambda": float(value),
            "Mean Source F1": r["mean_source_macro_f1"],
            "Worst Source F1": r["worst_source_macro_f1"],
            "Sketch Acc": r["sketch"]["accuracy"],
            "Sketch F1": r["sketch"]["macro_f1"],
            "Sketch Acc Change": r["sketch_accuracy_change"],
            "Source Separability": r["source_domain_separability"]
        })

    controlled_df= pd.DataFrame(controlled_rows)

    print("\n=== DAN-DG CONTROLLED STUDY ===")
    print(controlled_df.round(4).to_string(index=False))

    controlled_df.to_csv(
        os.path.join(args.task3_dir,"dan_dg_controlled_table.csv"),
        index=False
    )

  
    plt.figure(figsize=(7,5))

    plt.plot(
        controlled_df["Lambda"],
        controlled_df["Mean Source F1"],
        marker="o",
        label="Mean Source F1"
    )

    plt.plot(
        controlled_df["Lambda"],
        controlled_df["Sketch Acc"],
        marker="o",
        label="Sketch Accuracy"
    )

    plt.plot(
        controlled_df["Lambda"],
        controlled_df["Source Separability"],
        marker="o",
        label="Source Separability"
    )

    plt.xscale("log")

    plt.xlabel("DAN-DG λ")
    plt.ylabel("Score")
    plt.title("DAN-DG Alignment Strength Study")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        os.path.join(args.figure_dir,"dan_dg_controlled_study.png"),
        dpi=200
    )

    plt.show()

  
    class_names= list(
        results["erm"]["class_analysis"]["per_class_accuracy"].keys()
    )

    class_rows= []

    for class_name in class_names:
        class_rows.append({
            "Class": class_name,
            "ERM": results["erm"]["class_analysis"]["per_class_accuracy"][class_name],
            "DAN-DG": results["dan_dg"]["class_analysis"]["per_class_accuracy"][class_name],
            "SAM": results["sam"]["class_analysis"]["per_class_accuracy"][class_name],
            "DAN-DG Change": results["dan_dg"]["per_class_accuracy_change"][class_name],
            "SAM Change": results["sam"]["per_class_accuracy_change"][class_name]
        })

    class_df= pd.DataFrame(class_rows)

    print("\n=== PER-CLASS SKETCH RESULTS ===")
    print(class_df.round(4).to_string(index=False))

    class_df.to_csv(
        os.path.join(args.task3_dir,"per_class_sketch_table.csv"),
        index=False
    )

    x= range(len(class_df))
    width= 0.35

    plt.figure(figsize=(9,5))

    plt.bar(
        [i - width/2 for i in x],
        class_df["DAN-DG Change"],
        width=width,
        label="DAN-DG"
    )

    plt.bar(
        [i + width/2 for i in x],
        class_df["SAM Change"],
        width=width,
        label="SAM"
    )

    plt.axhline(0,linewidth=1)

    plt.xticks(
        list(x),
        class_df["Class"],
        rotation=30
    )

    plt.ylabel("Sketch Accuracy Change vs ERM")
    plt.title("Per-Class Sketch Accuracy Change")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(args.figure_dir,"per_class_sketch_change.png"),
        dpi=200
    )

    plt.show()


    confusion_rows= []

    for method in ["erm","dan_dg","sam"]:
        confusions= results[method]["class_analysis"]["dominant_confusions"]

        for class_name in class_names:
            confusion_rows.append({
                "Method": method.upper(),
                "True Class": class_name,
                "Predicted As": confusions[class_name]["predicted_as"],
                "Count": confusions[class_name]["count"]
            })

    confusion_df= pd.DataFrame(confusion_rows)

    print("\n=== DOMINANT SKETCH CONFUSIONS ===")
    print(confusion_df.to_string(index=False))

    confusion_df.to_csv(
        os.path.join(args.task3_dir,"dominant_confusions.csv"),
        index=False
    )

    print("\nSaved figures to:", args.figure_dir)


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument("--task3_dir",type=str,required=True)
    parser.add_argument("--figure_dir",type=str,required=True)

    args= parser.parse_args()

    main(args)