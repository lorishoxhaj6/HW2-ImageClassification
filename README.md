# HW2-ImageClassification

Computer Vision, Homework 2. Compares three transfer-learning approaches to an ImageNet-pre-trained
MobileNetV3-Small to classify the 37 breeds of the Oxford-IIIT Pet dataset:
full fine-tuning, classifier-head training, and frozen features + logistic regression.

## Setup

```bash
# Clone the repository
git clone https://github.com/username/my-project.git

# Enter the project directory
cd my-project

# Create a virtual environment
python -m venv venv

# Activate the virtual environment (Windows)
venv\Scripts\activate

# Activate the virtual environment (Linux/macOS)
source venv/bin/activate

# Install project dependencies
pip install -r requirements.txt
```
# Environment

## Environment

- Python: 3.10.2
- PyTorch: 2.14.1
- Torchvision: 0.29.2
- numpy: 2.2.6
- matplotlib: 3.10.9
- Pillow: 12.3.0
- scikit-learn: 1.7.2

## Dataset

- Oxford-IIIT Pet Dataset, 37 cat and dog breeds: https://www.robots.ox.ac.uk/~vgg/data/pets/
- License: CC BY-SA 4.0. The images are not redistributed in this repository.
- The script downloads it automatically into `./data` (`torchvision.datasets.OxfordIIITPet`).
- Labels: the 37 breed categories.

Citation: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar. *Cats and Dogs*. CVPR 2012.

## Run everything

```bash
python main.py
```

This runs the three approaches and writes all metrics, figures and checkpoints to `results/`.

## Protocol

- **Splits:** the official `trainval` partition is split 80/20 into train/validation, stratified by
  breed (seed 24; indices in `results/split.json`). 
- **Preprocessing:** the transforms of `MobileNet_V3_Small_Weights.DEFAULT`
  (resize 256, center-crop 224, ImageNet mean/std). **Augmentation:** random horizontal flip,
  training set only. **Class balancing:** none (per-breed counts in `results/class_counts.csv`).
- **Pre-trained weights:** `MobileNet_V3_Small_Weights.DEFAULT` (ImageNet-1K). The last layer is
  replaced by `Linear(1024, 37)` in every approach.

| Approach | Trainable parameters | Training setup |
|---|---|---|
| Full fine-tuning | whole network | Adam, lr 1e-4, batch 32, 10 epochs, cross-entropy |
| Classifier-head training | `classifier.0` and `classifier.3` (`features` frozen) | Adam, lr 1e-3, batch 32, 10 epochs, cross-entropy |
| Frozen features + logistic regression | classifier only, on the fixed feature vector | scaler fitted on train features, `C` chosen from {0.01, 0.1, 1} |

Model selection uses the validation set only: the epoch with the highest validation accuracy for the
networks, the value of `C` with the highest validation accuracy for logistic regression.

## Reproducibility

- Seed 24 for Python, NumPy and PyTorch; the data loaders use a seeded generator.
- Deterministic operations enabled (`cudnn.deterministic`, `cudnn.benchmark=False`,
  `use_deterministic_algorithms(warn_only=True)`).
- Remaining sources of variation: some CUDA kernels have no deterministic version (GPU runs can
  differ slightly), training times depend on the hardware, and results can change with library versions.


## Outputs (`results/`)

`results_table.md`, `results.json` (config, metrics), `class_counts.csv`, `split.json`,
`{full,head}_best.pt` (selected checkpoints), and per approach `*_confusion.png` and `*_examples.png`.