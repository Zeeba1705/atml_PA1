import torch


CIFAR10_CLASSES= [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


def per_unknown_class_analysis(logits,labels,scores,threshold,class_names):
    predictions= logits.argmax(dim=1)

    results= {}

    unique_labels= torch.unique(labels)

    for label in unique_labels:
        label_id= label.item()
        mask= labels == label

        class_scores= scores[mask]
        class_predictions= predictions[mask]

        rejected_mask= class_scores > threshold
        accepted_mask= class_scores <= threshold

        rejection_rate= rejected_mask.float().mean().item()
        acceptance_rate= accepted_mask.float().mean().item()

        all_prediction_counts= torch.bincount(
            class_predictions,
            minlength=10
        )

        most_common_all= all_prediction_counts.argmax().item()

        if accepted_mask.sum().item() > 0:
            accepted_predictions= class_predictions[accepted_mask]

            accepted_prediction_counts= torch.bincount(
                accepted_predictions,
                minlength=10
            )

            most_common_accepted= accepted_prediction_counts.argmax().item()

            most_common_accepted_name= CIFAR10_CLASSES[
                most_common_accepted
            ]

            most_common_accepted_count= int(
                accepted_prediction_counts[
                    most_common_accepted
                ].item()
            )

        else:
            most_common_accepted_name= None
            most_common_accepted_count= 0

        results[class_names[label_id]]= {
            "num_examples": int(mask.sum().item()),

            "rejection_rate": rejection_rate,
            "acceptance_rate": acceptance_rate,

            "most_common_prediction_all": CIFAR10_CLASSES[
                most_common_all
            ],

            "most_common_prediction_all_count": int(
                all_prediction_counts[
                    most_common_all
                ].item()
            ),

            "most_common_prediction_accepted": most_common_accepted_name,

            "most_common_prediction_accepted_count": most_common_accepted_count
        }

    return results