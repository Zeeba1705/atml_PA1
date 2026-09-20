import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auroc(known_scores,unknown_scores):
    labels= np.concatenate([
        np.zeros(len(known_scores)),
        np.ones(len(unknown_scores))
    ])

    scores= np.concatenate([
        known_scores,
        unknown_scores
    ])

    return roc_auc_score(
        labels,
        scores
    )


def threshold_from_validation(val_scores):
    return np.percentile(
        val_scores,
        95
    )


def known_acceptance_rate(scores,threshold):
    return np.mean(
        scores <= threshold
    )


def unknown_rejection_rate(scores,threshold):
    return np.mean(
        scores > threshold
    )


def evaluate_score(val_scores,test_scores,near_scores,far_scores):
    threshold= threshold_from_validation(
        val_scores
    )

    all_unknown= np.concatenate([
        near_scores,
        far_scores
    ])

    return {
        "threshold":float(threshold),

        "auroc_near":float(
            compute_auroc(
                test_scores,
                near_scores
            )
        ),

        "auroc_far":float(
            compute_auroc(
                test_scores,
                far_scores
            )
        ),

        "auroc_all":float(
            compute_auroc(
                test_scores,
                all_unknown
            )
        ),

        "known_acceptance":float(
            known_acceptance_rate(
                test_scores,
                threshold
            )
        ),

        "near_rejection":float(
            unknown_rejection_rate(
                near_scores,
                threshold
            )
        ),

        "far_rejection":float(
            unknown_rejection_rate(
                far_scores,
                threshold
            )
        )
    }