from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class PACSDomain(Dataset):
    def __init__(self, root, domain, transform=None):
        self.root= Path(root)
        self.domain= domain
        self.transform= transform

        domain_path= self.root / domain

        if not domain_path.exists():
            raise FileNotFoundError(
                f"Could not find PACS domain at: {domain_path}"
            )

        self.classes = sorted(
            folder.name
            for folder in domain_path.iterdir()
            if folder.is_dir()
        )

        self.class_to_idx = {
            class_name: i
            for i, class_name in enumerate(self.classes)
        }

        self.samples= []

        for class_name in self.classes:
            class_path = domain_path / class_name
            label = self.class_to_idx[class_name]

            for image_path in sorted(class_path.iterdir()):
                if image_path.suffix.lower() in {
                    ".jpg", ".jpeg", ".png", ".bmp"
                }:
                    self.samples.append(
                        (image_path, label)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
