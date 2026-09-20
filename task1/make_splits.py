import argparse
import os
from task1.data.stl10 import load_stl10, make_split, save_split


def main(data_root, output_path, download=False):
    train, test = load_stl10(data_root, download=download)
    split = make_split(train, test)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_split(split, output_path)
    print("Saved:", output_path)
    print("train:", len(split["train_idx"]), "val:", len(split["val_idx"]), "test:", len(split["test_subset_idx"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_path", default="splits/stl10_seed6304.json")
    p.add_argument("--download", action="store_true")
    a = p.parse_args()
    main(a.data_root, a.output_path, a.download)
