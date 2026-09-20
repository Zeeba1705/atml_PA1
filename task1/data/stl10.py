import json
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import STL10

SEED = 6304


def load_stl10(root, download=False):
    train = STL10(root=root, split="train", download=download, transform=None)
    test = STL10(root=root, split="test", download=download, transform=None)
    return train, test


def make_split(train_dataset, test_dataset, seed=SEED, test_per_class=50):
    from sklearn.model_selection import train_test_split

    train_labels = np.array(train_dataset.labels)
    train_idx, val_idx = train_test_split(
        np.arange(len(train_dataset)),
        test_size=0.2,
        stratify=train_labels,
        random_state=seed,
    )

    test_labels = np.array(test_dataset.labels)
    rng = np.random.default_rng(seed)
    selected_test_idx = []

    for class_id in range(10):
        class_idx = np.where(test_labels == class_id)[0]
        n = min(test_per_class, len(class_idx))
        selected_test_idx.extend(rng.choice(class_idx, size=n, replace=False))

    return {
        "train_idx": np.asarray(train_idx).tolist(),
        "val_idx": np.asarray(val_idx).tolist(),
        "test_subset_idx": np.asarray(selected_test_idx).tolist(),
    }


def save_split(split, path):
    with open(path, "w") as f:
        json.dump(split, f, indent=2)


def load_split(path):
    with open(path, "r") as f:
        return json.load(f)


class IndexedTransformDataset(Dataset):
    def __init__(self, base_dataset, indices, image_transform, normalize):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.image_transform = image_transform
        self.normalize = normalize

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        image, label = self.base_dataset[idx]
        image = self.image_transform(image)
        image = self.normalize(image)
        return image, label
