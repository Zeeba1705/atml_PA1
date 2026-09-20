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
    predictions= logits[:,:10].argmax(dim=1)

    results= {}

    for label in torch.unique(labels):
        class_id= label.item()

        mask= labels == label

        class_scores= scores[mask]
        class_predictions= predictions[mask]

        rejected= class_scores > threshold
        accepted= class_scores <= threshold

        prediction_counts= torch.bincount(
            class_predictions,
            minlength=10
        )

        most_common= prediction_counts.argmax().item()

        if accepted.sum().item() > 0:
            accepted_predictions= class_predictions[
                accepted
            ]

            accepted_counts= torch.bincount(
                accepted_predictions,
                minlength=10
            )

            accepted_common= accepted_counts.argmax().item()

            accepted_name= CIFAR10_CLASSES[
                accepted_common
            ]

            accepted_count= int(
                accepted_counts[
                    accepted_common
                ].item()
            )

        else:
            accepted_name= None
            accepted_count= 0

        results[class_names[class_id]]= {
            "num_examples":int(
                mask.sum().item()
            ),

            "rejection_rate":float(
                rejected.float().mean().item()
            ),

            "acceptance_rate":float(
                accepted.float().mean().item()
            ),

            "mean_score":float(
                class_scores.float().mean().item()
            ),

            "median_score":float(
                class_scores.float().median().item()
            ),

            "most_common_prediction_all":
                CIFAR10_CLASSES[most_common],

            "most_common_prediction_all_count":
                int(
                    prediction_counts[
                        most_common
                    ].item()
                ),

            "most_common_prediction_accepted":
                accepted_name,

            "most_common_prediction_accepted_count":
                accepted_count
        }

    return results


def accepted_failures(logits,labels,scores,threshold,class_names,n=10):
    predictions= logits[:,:10].argmax(dim=1)

    accepted_indices= torch.where(
        scores <= threshold
    )[0]

    accepted_scores= scores[
        accepted_indices
    ]

    order= torch.argsort(
        accepted_scores
    )

    accepted_indices= accepted_indices[
        order
    ]

    failures= []

    for idx in accepted_indices[:n]:
        idx= idx.item()

        failures.append({
            "index":idx,
            "unknown_class":
                class_names[
                    labels[idx].item()
                ],
            "predicted_known_class":
                CIFAR10_CLASSES[
                    predictions[idx].item()
                ],
            "score":float(
                scores[idx].item()
            ),
            "threshold":float(
                threshold
            )
        })

    return failures