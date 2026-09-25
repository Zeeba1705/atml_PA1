# ATML Programming Assignment 1 — Beyond IID Learning

EE-5102 / CS-6304 Advanced Topics in Machine Learning, Fall 2026

This repository contains the code and experiment outputs for Programming Assignment 1. The assignment studies robustness beyond the IID closed-set setting through four tasks:

1. Inductive biases and representations
2. Domain adaptation
3. Domain generalization
4. Open-set recognition

## Repository Structure

```text
├── common/              # shared utilities
├── task1/               # inductive biases and representation analysis
├── task2/               # domain adaptation
├── task3/               # domain generalization
├── task4/               # open-set recognition
├── results/             # small machine-readable result files, if shared globally
├── figures/             # report figures, if shared globally
├── requirements.txt
├── .gitignore
└── README.md
```

Each task directory contains its own training/evaluation code, configurations, results, and plotting utilities.

## Environment

Python 3.x and PyTorch were used for all experiments.

Install dependencies with:

```bash
pip install -r requirements.txt
```

Main libraries include:
- PyTorch / torchvision
- NumPy
- scikit-learn
- matplotlib
- pandas
- OpenCLIP
- UMAP, where applicable
- scikit-image, where applicable

Seed `6304` is used throughout the assignment where a fixed random seed is required.

## Task 1 — Inductive Biases and Representations

### Dataset and Splits

Dataset: **STL-10**

The official STL-10 training partition contains 5,000 images. A stratified train/validation split was created using seed `6304`:

- Training: 80% = 4,000 images
- Validation: 20% = 1,000 images
- 400 training images per class
- 100 validation images per class

For final intervention evaluation, a class-balanced subset of 500 images was sampled from the official STL-10 test partition using seed `6304`:

- 50 images per class
- The same 500 images were used for every model and intervention

The split indices are saved so that the exact same images can be reused across runs.

### Models

- ResNet-50: `ResNet50_Weights.IMAGENET1K_V2`
- ViT-B/16: `ViT_B_16_Weights.IMAGENET1K_V1`
- CLIP ViT-B/32: OpenCLIP with `pretrained="openai"`

All pretrained backbones were frozen.

Linear classifier dimensions:
- ResNet: 2048 → 10
- ViT: 768 → 10
- CLIP Linear: 512 → 10

CLIP Zero-shot used the fixed prompt:

```text
a photo of a {class}
```

### Linear-Head Training Hyperparameters

- Optimizer: AdamW
- Learning rate: `1e-3`
- Weight decay: `1e-4`
- Batch size: `128`
- Maximum epochs: `50`
- Early stopping patience: `5`
- Selection metric: validation accuracy
- Seed: `6304`

### Interventions

#### Color
- Grayscale
- LAB color transfer using a fixed STL-10 reference image

The LAB transfer preserves luminance while matching the chromatic `a` and `b` statistics of the reference image.

#### Shape vs. Texture
Cue-conflict images were created using AdaIN.

Main settings:
- AdaIN style strength: `alpha = 1.0`
- 5 class-pair combinations
- Both directions generated where applicable
- 250 candidate images generated
- 200 accepted after the predefined visual rejection procedure

Shape bias and coverage were evaluated only on the accepted images.

#### Translation
Displacements: `0, 8, 16, 32 pixels`

Each non-zero displacement was evaluated in four directions: up, down, left, and right. Reflection padding followed by shifted cropping was used, and results were averaged over directions.

#### Patch Shuffle
Each image was divided into a `4 × 4` grid, giving 16 image patches, which were shuffled using a non-identity permutation.

### Representation Analysis

For each intervention, transformed features were compared with clean features using cosine similarity. UMAP was used for representation visualization where applicable.

Machine-readable outputs are stored under `task1/results/` and figures under `task1/figures/`.

## Task 2 — Unsupervised Domain Adaptation

### Dataset and Splits

Dataset: **PACS**

Source domains:
- Photo
- Art Painting
- Cartoon

Target domain:
- Sketch

Within each source domain, a stratified train/validation split was created using seed `6304`:

- Training: 80%
- Validation: 20%

The same source splits are reused in Task 3.

Sketch images are available without class labels during Task 2 adaptation. Sketch labels are used only after all models, checkpoints, and hyperparameters have been fixed.

### Model

Backbone: `torchvision ResNet-18` with `ResNet18_Weights.IMAGENET1K_V1`.

The ImageNet classifier was replaced by a 7-class linear classifier and the complete network was fine-tuned.

### Preprocessing

Training:
- Resize to `256 × 256`
- Random `224 × 224` crop
- Random horizontal flip
- ImageNet normalization

Validation/evaluation:
- Resize to `256 × 256`
- Center `224 × 224` crop
- ImageNet normalization

### Common Training Hyperparameters

- Optimizer: AdamW
- Learning rate: `1e-4`
- Weight decay: `1e-4`
- Maximum epochs: `30`
- Early stopping patience: `5`
- Checkpoint selection: mean source-validation macro-F1
- Seed: `6304`

BatchNorm policy:
- Running means/variances frozen at ImageNet-pretrained values
- BatchNorm scale and bias parameters remain trainable

Adaptation batches contain:
- 8 Photo examples
- 8 Art Painting examples
- 8 Cartoon examples
- 24 Sketch target examples

Total: `24 source + 24 target examples per adaptation update`

### Methods

#### Source-only ERM
Cross-entropy training using only labeled Photo, Art Painting, and Cartoon examples.

#### DAN
MMD is applied to the 512-dimensional pre-classifier feature.

Main setting: `lambda_MMD = 1`

Controlled study: `lambda_MMD ∈ {0.1, 1, 10}`

MMD uses a sum of three RBF kernels with bandwidth multipliers `0.5, 1, 2` relative to the median pairwise squared feature distance in the current batch.

#### DANN
Domain discriminator:
- Input: 512-dimensional feature
- Hidden layer: 256 units
- Activation: ReLU
- Dropout: `0.5`
- Output: 2 classes, source vs. target

Gradient-reversal schedule:

```text
alpha(p) = 2 / (1 + exp(-10p)) - 1
```

Domain-loss weight: `1`

Only source examples contribute to classification loss. Both source and target examples contribute to domain loss.

The implementation also uses gradient clipping and L2 normalization in the adversarial/domain branch.

#### CDAN
Uses the same adversarial discriminator structure and GRL schedule as DANN.

The discriminator input is class-conditioned using `vec(f ⊗ p)`, where `f` is the backbone feature and `p` is the classifier probability vector.

Gradient clipping is used. No equivalent feature normalization is applied in the CDAN branch.

### Domain-Separability Diagnostic

After training, frozen representations are used to train a logistic-regression domain classifier. Task 2 uses a balanced source-vs-target diagnostic.

Target labels are not used for model selection.

Results are stored under `task2/results/`.

## Task 3 — Domain Generalization

### Dataset and Splits

Task 3 reuses the exact PACS source splits from Task 2:
- Photo
- Art Painting
- Cartoon

The Source-only Task 2 checkpoint is reused unchanged as the ERM baseline.

Sketch is completely unavailable during training, diagnostics, checkpoint selection, and hyperparameter selection. Sketch is loaded only for final evaluation.

### Common Training Setup

Task 3 uses the same preprocessing, optimizer, BatchNorm policy, source splits, and training budget as Task 2.

- Optimizer: AdamW
- Learning rate: `1e-4`
- Weight decay: `1e-4`
- Maximum epochs: `30`
- Early stopping patience: `5`
- Checkpoint selection: mean source-validation macro-F1
- Seed: `6304`

Each training batch contains:
- 8 examples from Photo
- 8 examples from Art Painting
- 8 examples from Cartoon

### Methods

#### ERM
Reuses the Task 2 Source-only checkpoint without retraining.

#### DAN-DG
Uses the same MMD implementation as Task 2 but aligns the three source domains pairwise:
- Photo ↔ Art Painting
- Photo ↔ Cartoon
- Art Painting ↔ Cartoon

MMD is applied to the 512-dimensional pre-classifier feature.

Main setting: `lambda_DG = 1`

Controlled study: `lambda_DG ∈ {0.1, 1, 10}`

The same RBF-kernel bandwidth multipliers are used: `0.5, 1, 2`.

#### SAM
Sharpness-Aware Minimization uses `rho = 0.05`.

Each update uses two forward/backward passes:
1. compute the normalized ascent perturbation
2. update the original parameters using loss at the perturbed point

The same frozen BatchNorm-running-statistics policy is used during both passes.

### Source-Domain Separability

Frozen representations are collected from the three source validation sets.

A balanced multinomial logistic-regression classifier predicts Photo, Art Painting, or Cartoon.

Settings:
- Train/test split: `70/30`
- Logistic regression: `C = 1`
- Seed: `6304`
- Chance accuracy: `33.3%`

### Sharpness Diagnostic

A fixed validation batch is selected using seed `6304`:
- 32 examples per source domain

The same batch is used for ERM, DAN-DG, and SAM.

Perturbation radius: `0.05`

The reported sharpness proxy is `L(theta + epsilon) - L(theta)` after one normalized gradient-ascent perturbation.

Results are stored under `task3/results/`.

## Task 4 — Open-Set Recognition

### Dataset and Splits

Known classes: **CIFAR-10**

A stratified split of the official CIFAR-10 training partition was created using seed `6304`:
- Training: 90%
- Validation: 10%

The complete CIFAR-10 test set is used for final known-class evaluation.

Unknown evaluation examples come only from the CIFAR-100 test set.

Near-semantic unknowns:
- bus
- pickup_truck
- motorcycle
- tractor
- wolf
- fox
- leopard
- camel

Far-semantic unknowns:
- bottle
- bowl
- chair
- clock
- keyboard
- mushroom
- sunflower
- wardrobe

Each group contains 800 images, with 100 images per class.

CIFAR-100 training images are never used. Unknown examples do not influence model training, checkpoint selection, score construction, hyperparameter selection, or rejection-threshold calibration.

### Model Architecture

A CIFAR-adapted ResNet-18 is used throughout.

Changes from standard ImageNet ResNet-18:
- First convolution: `3 × 3`, stride `1`
- Initial max-pooling layer removed
- Input resolution: `32 × 32`

### Vanilla and GCSC Training Hyperparameters

- Optimizer: SGD
- Learning rate: `0.1`
- Momentum: `0.9`
- Weight decay: `5e-4`
- Learning-rate schedule: cosine decay
- Batch size: `128`
- Epochs: `100`
- Seed: `6304`
- Checkpoint selection: highest CIFAR-10 validation accuracy

Training augmentation:
- Random crop to `32 × 32`
- Padding: `4`
- Random horizontal flip

### Vanilla

Ten-class ResNet-18 trained from random initialization using cross-entropy.

After training, the selected checkpoint is frozen and used to extract logits and penultimate features.

Post-hoc novelty scores:
- MSP
- MLS
- Energy
- Mahalanobis

For Mahalanobis scoring:
- class means are estimated from unaugmented CIFAR-10 training features
- one shared diagonal covariance matrix is used
- covariance diagonal stabilization: `1e-6`

### GCSC

Uses the same training setup as Vanilla, with one additional augmentation:

```text
RandAugment(num_ops=2, magnitude=9)
```

GCSC is evaluated using MLS.

### PROSER

Initialization: selected Vanilla checkpoint

Classifier placeholders:
- 5 dummy classifiers
- `beta = 1`

Data placeholders:
- manifold mixup after `layer2`
- different-class CIFAR-10 examples only
- mixing coefficient: `lambda ~ Beta(2, 2)`
- data-placeholder weight: `gamma = 0.1`

Each batch is split equally:
- one half for classifier-placeholder training
- one half for data-placeholder / manifold-mixup training

PROSER fine-tuning:
- Optimizer: SGD
- Learning rate: `1e-3`
- Momentum: `0.9`
- Weight decay: `5e-4`
- Learning-rate schedule: cosine decay
- Batch size: `128`
- Epochs: `50`
- Seed: `6304`

PROSER checkpoint selection uses CIFAR-10 validation accuracy only.

### Rejection Thresholds

For every model/score, the rejection threshold is the 95th percentile of unknownness on CIFAR-10 validation data.

An input is accepted as known when `unknownness <= threshold`, targeting approximately 95% known validation acceptance.

Reported metrics include:
- CIFAR-10 closed-set accuracy
- known test acceptance
- near-unknown rejection
- far-unknown rejection
- near AUROC
- far AUROC
- all-unknown AUROC

Results are stored under `task4_results/`.

Final machine-readable summary: `task4_results/task4_results.json`

## Reproducing the Experiments

### Task 1

Run the Task 1 scripts/notebook in the following order:
1. dataset/model setup
2. train/validate linear heads
3. clean baseline evaluation
4. grayscale and LAB color-transfer evaluation
5. cue-conflict generation/evaluation
6. translation evaluation
7. patch-shuffle evaluation
8. representation-stability analysis
9. UMAP/visualization generation

### Task 2

Methods:
- Source-only
- DAN
- DANN
- CDAN

Example commands:

```bash
python -m task2.train --method source
python -m task2.train --method dan
python -m task2.train --method dann
python -m task2.train --method cdan
python -m task2.evaluate
```

Controlled DAN settings: `lambda_MMD ∈ {0.1, 1, 10}`

### Task 3

Methods:
- ERM
- DAN-DG
- SAM

ERM reuses the selected Task 2 Source-only checkpoint.

Controlled DAN-DG settings: `lambda_DG ∈ {0.1, 1, 10}`

Run the Task 3 training scripts before the final Sketch evaluation script.

### Task 4

Workflow:
1. Train Vanilla
2. Train GCSC
3. Initialize and fine-tune PROSER from Vanilla
4. Extract logits/features
5. Compute novelty scores
6. Calibrate thresholds using CIFAR-10 validation data
7. Evaluate CIFAR-10 test acceptance
8. Evaluate near/far CIFAR-100 unknowns
9. Generate failure analysis and plots

## Reproducibility Notes

- Seed `6304` is used where specified for split construction, training comparisons, diagnostics, and evaluation sampling.
- Exact split indices are saved where applicable.
- Reported hyperparameters are preserved in configuration files or experiment scripts.
- Small JSON/CSV result files used to generate the report are committed.
- Raw datasets and unnecessary large checkpoints are excluded from Git.
- Final reported values can be traced to saved result files or reproducible evaluation commands.
- Target labels are never used for Task 2 model selection.
- Sketch is not used during Task 3 training, diagnostics, hyperparameter selection, or model selection.
- Task 2 target results are not used to choose Task 3 settings.
- CIFAR-100 unknown examples do not influence Task 4 training, checkpoint selection, score construction, or threshold calibration.

## External Code and Tools

The implementation primarily uses standard functionality from PyTorch, torchvision, scikit-learn, NumPy, matplotlib, OpenCLIP, UMAP, and scikit-image.

Cue-conflict generation uses AdaIN style transfer based on:

```text
https://github.com/naoto0804/pytorch-AdaIN
```

PROSER implementation details were based on:

Zhou et al., *Learning Placeholders for Open-Set Recognition*, CVPR 2021.

Any materially reused external implementation code should be identified in the relevant source file and/or task README.

## AI Coding Assistance

ChatGPT was used for coding assistance, including debugging, code organization, plotting utilities, and implementation guidance. All submitted code was reviewed and understood by the author.

Generative AI was not used to write the submitted PDF report.

## Results

Machine-readable results used in the report are retained within the corresponding task directories so that reported values can be traced back to saved experiment outputs.

## Author

Zainab Amir  
28100166
