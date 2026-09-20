import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from skimage.color import rgb2lab, lab2rgb

IMAGENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
CLIP_NORMALIZE = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
)

COMMON_TRANSFORM = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
GRAYSCALE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])


def make_lab_palette_transfer(reference_img):
    ref_rgb = np.array(reference_img).astype(np.float32) / 255.0
    ref_lab = rgb2lab(ref_rgb)
    ref_a_mean, ref_a_std = ref_lab[:, :, 1].mean(), ref_lab[:, :, 1].std()
    ref_b_mean, ref_b_std = ref_lab[:, :, 2].mean(), ref_lab[:, :, 2].std()

    def transfer(img_tensor):
        img_rgb = img_tensor.permute(1, 2, 0).numpy()
        img_lab = rgb2lab(img_rgb)
        L, a, b = img_lab[:, :, 0], img_lab[:, :, 1], img_lab[:, :, 2]

        new_a = ((a - a.mean()) / (a.std() + 1e-6)) * ref_a_std + ref_a_mean
        new_b = ((b - b.mean()) / (b.std() + 1e-6)) * ref_b_std + ref_b_mean

        transferred = np.stack([L, new_a, new_b], axis=2)
        rgb = np.clip(lab2rgb(transferred), 0, 1)
        return torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1)

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(transfer),
    ])


def translate_reflect(img, dx=0, dy=0):
    img = TF.resize(img, [224, 224])
    pad = max(abs(dx), abs(dy))
    if pad == 0:
        return img
    img = TF.pad(img, [pad, pad, pad, pad], padding_mode="reflect")
    return TF.crop(img, top=pad - dy, left=pad - dx, height=224, width=224)


def make_patch_orders(indices, seed=6304):
    rng = np.random.default_rng(seed)
    orders = {}
    for idx in indices:
        order = rng.permutation(16)
        while np.array_equal(order, np.arange(16)):
            order = rng.permutation(16)
        orders[int(idx)] = order
    return orders


def patch_shuffle(img, order):
    img = TF.to_tensor(TF.resize(img, [224, 224]))
    patch_size = 56
    patches = []
    for r in range(4):
        for c in range(4):
            y, x = r * patch_size, c * patch_size
            patches.append(img[:, y:y + patch_size, x:x + patch_size])

    shuffled = torch.empty_like(img)
    for pos, old_pos in enumerate(order):
        r, c = divmod(pos, 4)
        y, x = r * patch_size, c * patch_size
        shuffled[:, y:y + patch_size, x:x + patch_size] = patches[old_pos]
    return shuffled
