import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

SEED= 6304


def fit_tsne(clean_features,transformed_features,perplexity=30,seed=SEED):
    clean= clean_features.float().cpu().numpy()
    transformed= transformed_features.float().cpu().numpy()
    combined= np.concatenate([clean,transformed],axis=0)

    tsne= TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto"
    )

    projected= tsne.fit_transform(combined)
    n= len(clean)
    return projected[:n],projected[n:]


def plot_clean_vs_transformed(clean_2d,transformed_2d,labels,title,output_path):
    labels= np.asarray(labels)

    plt.figure(figsize=(7,6))

    for class_id in np.unique(labels):
        mask= labels == class_id
        plt.scatter(clean_2d[mask,0],clean_2d[mask,1],s=16,alpha=0.55,label=f"Class {class_id} clean")
        plt.scatter(transformed_2d[mask,0],transformed_2d[mask,1],s=16,alpha=0.55,marker="x",label=f"Class {class_id} transformed")

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    plt.savefig(output_path,dpi=300,bbox_inches="tight")
    plt.close()
