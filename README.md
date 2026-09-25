# ATML Programming Assignment 1 — Beyond IID Learning
EE-5102 / CS-6304 Advanced Topics in Machine Learning, Fall 2026

This repository contains the code and experiment outputs for Programming Assignment 1. The assignment studies robustness beyond the IID closed-set setting through four tasks:

1. Inductive biases and representations
2. Domain adaptation
3. Domain generalization
4. Open-set recognition

## Repository Structure
├── common/ # shared utilities
├── task1/ # inductive biases and representation analysis
├── task2/ # domain adaptation
├── task3/ # domain generalization
├── task4/ # open-set recognition
├── results/ # small machine-readable result files, if shared globally
├── figures/ # report figures, if shared globally
├── requirements.txt
├── .gitignore
└── README.md

Each task directory contains its own training/evaluation code, configurations, results, and plotting utilities.

## Environment

Python 3.x and PyTorch were used for all experiments.

Install dependencies with:

pip install -r requirements.txt


Main libraries include:
- PyTorch / torchvision
- NumPy
- scikit-learn
- matplotlib
- pandas
- OpenCLIP
- UMAP, where applicable

All experiments use random seed 6304 where applicable.

## Datasets

Raw datasets are not committed to this repository.

**Task 1**
Dataset: STL-10

Models:
- ImageNet-pretrained ResNet-50
- ImageNet-pretrained ViT-B/16
- CLIP ViT-B/32

The task evaluates clean performance and controlled interventions involving color, cue conflict, translation, and patch shuffling, along with representation-level diagnostics.

**Tasks 2 and 3**
Dataset: PACS

Domains:
- Photo
- Art Painting
- Cartoon
- Sketch

Tasks 2 and 3 use the same source-domain split logic and fixed seed.

Task 2 treats Sketch as an unlabeled target domain during adaptation.

Task 3 does not use Sketch during training, hyperparameter selection, or model selection; Sketch is used only for final evaluation.

**Task 4**
Known classes: CIFAR-10

Unknown evaluation classes: selected CIFAR-100 classes.

Near-semantic unknowns:
bus, pickup_truck, motorcycle, tractor, wolf, fox, leopard, camel

Far-semantic unknowns:
bottle, bowl, chair, clock, keyboard, mushroom, sunflower, wardrobe

CIFAR-100 examples are used only for final evaluation and do not influence training, checkpoint selection, score construction, or rejection-threshold selection.

## Reproducing the Experiments

### Task 1 — Inductive Biases and Representations

Task 1 evaluates ResNet-50, ViT-B/16, and CLIP under controlled visual interventions.

Run the task scripts/notebook in `task1/` in the following order:

1. dataset/model setup
2. clean baseline evaluation
3. grayscale and palette interventions
4. cue-conflict evaluation
5. translation evaluation
6. patch-shuffle evaluation
7. representation-stability analysis
8. UMAP/visualization generation

Machine-readable outputs are stored under `task1/results/`.
Figures are stored under `task1/figures/`.

### Task 2 — Domain Adaptation

Methods:
- Source-only
- DAN
- DANN
- CDAN

The source domains are Photo, Art Painting, and Cartoon. Sketch is used as an unlabeled target domain during adaptation.

Run the source baseline first, followed by the adaptation methods and evaluation scripts.

Example structure:

python -m task2.train --method source
python -m task2.train --method dan
python -m task2.train --method dann
python -m task2.train --method cdan
python -m task2.evaluate


The controlled DAN alignment-strength experiment uses:
`lambda = {0.1, 1, 10}`

Results are stored under `task2/results/`.

### Task 3 — Domain Generalization

Methods:
- ERM
- DAN-DG
- SAM

The ERM baseline reuses the Task 2 Source-only checkpoint.

No Sketch examples are used during Task 3 training or model selection.

Main settings:
- DAN-DG lambda = 1
- SAM rho = 0.05

Controlled DAN-DG settings:
`lambda = {0.1, 1, 10}`

Run the training and final Sketch evaluation scripts from `task3/`.

Results are stored under `task3/results/`.

### Task 4 — Open-Set Recognition

Models:
- Vanilla
- GCSC
- PROSER

Post-hoc scores evaluated on the frozen Vanilla model:
- MSP
- MLS
- Energy
- Mahalanobis

The common model comparison uses MLS, with an additional PROSER placeholder-based score.

Example workflow:

1. Train Vanilla
2. Train GCSC
3. Initialize and fine-tune PROSER from Vanilla
4. Extract logits/features
5. Compute novelty scores
6. Calibrate thresholds using CIFAR-10 validation data
7. Evaluate near/far CIFAR-100 unknowns
8. Generate failure analysis and plots

Results are stored under `task4_results/`.

The final machine-readable summary is: `task4_results/task4_results.json`

## Reproducibility Notes

- Random seeds are fixed where practical.
- Reported hyperparameters are preserved in configuration files or experiment scripts.
- Small JSON/CSV result files used to generate the report are committed.
- Raw datasets and large checkpoints are excluded from Git.
- Final reported values can be traced to saved result files or reproducible evaluation commands.
- Target labels are never used for Task 2 model selection.
- Sketch is not used during Task 3 training or model selection.
- Task 2 target results are not used to choose Task 3 settings.
- CIFAR-100 unknown examples do not influence Task 4 training or threshold calibration.

## External Code and Tools

The implementation primarily uses standard functionality from PyTorch, torchvision, scikit-learn, NumPy, matplotlib, and OpenCLIP.

PROSER implementation details were based on:
Zhou et al., *Learning Placeholders for Open-Set Recognition*, CVPR 2021.

## AI Coding Assistance

ChatGPT was used for coding assistance, including debugging, code organization, plotting utilities, and implementation guidance. All submitted code was reviewed and understood by the author.
Generative AI was not used to write the submitted PDF report.

## Results

Machine-readable results used in the report are retained within the corresponding task directories so that reported values can be traced back to saved experiment outputs.

## Author

[Zainab Amir]
[28100166]
