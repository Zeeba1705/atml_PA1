from torchvision.datasets import STL10
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split
import os
import json

SEED= 6304


transform = transforms.ToTensor()


train_dataset = STL10(
    root="datasets",
    split="train",
    download=True,
    transform=transform
)

test_dataset = STL10(
    root="datasets",
    split="test",
    download=True,
    transform=transform
)


print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))
print("Classes:", train_dataset.classes)

#creating the stratified train/test split

train_labels= np.array(train_dataset.labels)

train_idx, val_idx = train_test_split(np.arange(len(train_dataset)), test_size=0.2,
    stratify=train_labels,random_state=SEED)

print("Train split:", len(train_idx))
print("Val split:", len(val_idx))

test_labels= np.array(test_dataset.labels)

rando= np.random.default_rng(SEED)

selected_test_idx = []

#taking 10 classses, so that we get a balance across classes - 50 per class to make 500
for class_id in range(10):
    class_idx= np.where(test_labels== class_id)[0]
    chosen= rando.choice(class_idx, size=50, replace=False)
    selected_test_idx.extend(chosen)

selected_test_idx= np.array(selected_test_idx)

os.makedirs("splits", exist_ok=True)

da_data = {"train_idx": train_idx.tolist(), "val_idx": val_idx.tolist(),"test_subset_idx": selected_test_idx.tolist()}

with open("splits/stl10_seed6304.json", "w") as f:
    json.dump(da_data, f)

print("Saved.")