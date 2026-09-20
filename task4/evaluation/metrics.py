import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auroc(known_scores,unknown_scores):
    labels= np.concatenate([np.zeros(len(known_scores)),np.ones(len(unknown_scores))
    ])

    scores= np.concatenate([known_scores,unknown_scores])

    return roc_auc_score(labels,scores)


def threshold_from_validation(val_scores):
    return np.percentile(val_scores,95)


def known_acceptance_rate(known_scores,threshold):
    accepted= known_scores <= threshold
    return accepted.mean()


def unknown_rejection_rate(unknown_scores,threshold):
    rejected= unknown_scores > threshold
    return rejected.mean()


def evaluate_score(val_scores,test_known_scores,near_scores,far_scores):
    threshold= threshold_from_validation(val_scores)

    all_unknown_scores= np.concatenate([near_scores,far_scores])

    results= {
        "threshold": float(threshold),

        "auroc_near": float(
            compute_auroc(test_known_scores,near_scores)
        ),

        "auroc_far": float(
            compute_auroc(test_known_scores,far_scores)
        ),

        "auroc_all": float(
            compute_auroc(test_known_scores,all_unknown_scores)
        ),

        "known_acceptance": float(
            known_acceptance_rate(test_known_scores,threshold)
        ),

        "near_rejection": float(
            unknown_rejection_rate(near_scores,threshold)
        ),

        "far_rejection": float(
            unknown_rejection_rate(far_scores,threshold)
        )
    }

    return results