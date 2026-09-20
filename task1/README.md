# Task 1 — Inductive Biases and Feature Representations

This folder is the script-based version of the original Colab notebook. It preserves the final experimental choices used there:

- STL-10
- seed 6304
- stratified 80/20 train/validation split
- balanced 500-image test subset (50/class)
- frozen ResNet-50 `IMAGENET1K_V2`
- frozen ViT-B/16 `IMAGENET1K_V1`
- frozen OpenCLIP ViT-B/32 `pretrained="openai"`
- linear heads trained with AdamW, lr `1e-3`, weight decay `1e-4`, max 50 epochs, patience 5
- CLIP zero-shot prompt `a photo of a {class}.`
- grayscale + LAB color-statistics transfer
- AdaIN cue conflicts with the five class pairs and manually retained candidate IDs from the notebook
- translation at 0/8/16/32 px, four cardinal directions, reflection padding
- fixed 4x4 patch shuffle per test image, seed 6304
- cosine representation stability
- t-SNE and UMAP fitted jointly to clean + transformed features

## Main scripts

`make_splits.py` — create the fixed split JSON once.

`extract_features.py` — GPU/accelerated extraction of frozen train/val features.

`train_heads.py` — train and save the three linear heads.

`run_clean.py` — clean baseline metrics for ResNet, ViT, CLIP linear, and CLIP zero-shot.

`run_color.py` — grayscale and the notebook's LAB palette/color-statistics transfer.

`make_cue_conflicts.py` — generate AdaIN candidates and save the notebook's manually retained valid set.

`run_cue_conflicts.py` — shape/texture/other counts, shape bias, coverage, and per-example predictions.

`run_translation.py` — translation accuracy and consistency; saves incrementally and resumes from existing JSON.

`run_patch_shuffle.py` — patch-shuffle accuracy drop and prediction consistency.

`run_representation.py` — cosine stability for grayscale, cue conflict, translation, and patch shuffle; also caches paired features for t-SNE/UMAP.

`plot_translation.py`, `plot_interventions.py`, `plot_cue_conflicts.py`, `plot_representation.py`, `plot_training.py` — CPU plotting scripts.

## AdaIN dependency

The original notebook used `naoto0804/pytorch-AdaIN`. Clone it separately and download its `vgg_normalised.pth` and `decoder.pth` weights into `<adain_repo>/models/`. Pass that path to `make_cue_conflicts.py --adain_repo ...`.

## Notes

Raw datasets, `.pt` checkpoints/features, and generated cue-conflict images should stay out of Git. Commit scripts, split JSON, small machine-readable result JSON files, and final figures as appropriate.
