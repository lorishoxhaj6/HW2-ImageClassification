import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

# Setup: seeds, determinism, versions
ROOT = "./data"
OUT = Path("results")
OUT.mkdir(exist_ok=True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


SEED = 42
NUM_CLASSES = 37
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = {"full": 1e-4, "head": 1e-3}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed():
    """Seeds Python, NumPy and PyTorch (CPU and GPU) with SEED so that runs are reproducible."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)   # a few CUDA ops have no deterministic version



set_seed()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Data preparation
# Preprocessing comes from the pre-trained weights: resize 256 -> center-crop 224 -> ImageNet mean/std.
# Augmentation (random horizontal flip) is applied to the training set only.
# Reuse the preprocessing expected by the pretrained MobileNetV3 weights.
weights = models.MobileNet_V3_Small_Weights.DEFAULT
eval_transform = weights.transforms()
train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), eval_transform])

trainval_aug = datasets.OxfordIIITPet(ROOT, split="trainval", target_types="category",
                                      transform=train_transform, download=True)
trainval_plain = datasets.OxfordIIITPet(ROOT, split="trainval", target_types="category",
                                        transform=eval_transform, download=True)
test_set = datasets.OxfordIIITPet(ROOT, split="test", target_types="category",
                                  transform=eval_transform, download=True)
class_names = trainval_plain.classes

# Stratified train/validation split of the official trainval partition
labels = np.array(trainval_plain._labels)
train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=0.2,
                                      stratify=labels, random_state=SEED)

# save a JSON audit file of train and val indices for reproducibility
with open(OUT / "split.json", "w") as f:
    json.dump({"seed": SEED, "train": train_idx.tolist(), "val": val_idx.tolist()}, f)

# Training uses augmented images; evaluation uses the same images without augmentation.
train_set = Subset(trainval_aug, train_idx.tolist())
train_eval_set = Subset(trainval_plain, train_idx.tolist())  # same images, no augmentation (for train metrics)
val_set = Subset(trainval_plain, val_idx.tolist())

# Images per breed in each split (no class balancing is used)
# Count how many images of each breed are present in each split.
train_counts = np.bincount(labels[train_idx], minlength=NUM_CLASSES)
val_counts = np.bincount(labels[val_idx], minlength=NUM_CLASSES)
test_counts = np.bincount(test_set._labels, minlength=NUM_CLASSES)

with open(OUT / "class_counts.csv", "w") as f:
    f.write("breed,train,val,test\n")
    for i, breed in enumerate(class_names):
        f.write(f"{breed},{train_counts[i]},{val_counts[i]},{test_counts[i]}\n")
print(f"Images per breed (min-max): train {train_counts.min()}-{train_counts.max()}, "
      f"val {val_counts.min()}-{val_counts.max()}, test {test_counts.min()}-{test_counts.max()}")
print(f"Train/val overlap: {len(set(train_idx) & set(val_idx))} images")


def make_loader(dataset, shuffle=False):
    """Returns a DataLoader with its own seeded generator, so the batch order is reproducible."""
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, generator=generator)


loaders = {"train": make_loader(train_eval_set), "val": make_loader(val_set)}  

# Shared model and evaluation code
def build_model(strategy):
    """MobileNetV3-Small with a new 37-output layer. 'head' freezes the convolutional features."""
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    if strategy == "head":
        for p in model.features.parameters():
            p.requires_grad = False
    return model.to(device)


def count_trainable(model):
    """Counts the parameters of the model that are updated during training."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def now():
    """Returns the current time in seconds, after waiting for any pending GPU work to finish."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


@torch.no_grad()
def predict(model, loader):
    """Runs the model in evaluation mode over a loader and returns (true labels, predicted labels)."""
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        y_pred.append(model(x.to(device)).argmax(1).cpu())
        y_true.append(y)
    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()


def scores(y_true, y_pred):
    """Returns accuracy and macro-averaged F1 (unweighted mean over the 37 breeds)."""
    return {"accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", labels=range(NUM_CLASSES), zero_division=0)}



# Train both fine-tuning strategies and select the best epoch using validation accuracy.
def train(strategy):
    """Trains with a fixed learning rate and keeps the epoch with the best validation accuracy."""
    set_seed()
    model = build_model(strategy)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE[strategy])
    loss_fn = nn.CrossEntropyLoss()
    loader = make_loader(train_set, shuffle=True)

    best_acc, best_state, best_epoch, train_time = -1, None, 0, 0.0
    for epoch in range(1, EPOCHS + 1):
        start = now()
        model.train()
        if strategy == "head":
            model.features.eval()  # keep BatchNorm statistics of the frozen part fixed
        total_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        train_time += now() - start

        val_acc = accuracy_score(*predict(model, loaders["val"]))
        print(f"[{strategy}] epoch {epoch}/{EPOCHS} | train loss {total_loss / len(train_set):.4f} | val acc {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(best_state, OUT / f"{strategy}_best.pt")
    return model, train_time, best_epoch


selected = {}  # approach -> everything needed for the final evaluation

for strategy, name in [("full", "Full fine-tuning"), ("head", "Classifier-head training")]:
    model, train_time, best_epoch = train(strategy)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"{name}: best epoch {best_epoch}, trainable parameters {count_trainable(model):,}\n")
    selected[strategy] = {
        "name": name, "trainable_params": count_trainable(model), "train_time": train_time,
        "predict": lambda split, m=model: predict(m, loaders[split]),
        "config": {"loss": "CrossEntropyLoss", "optimizer": "Adam", "learning_rate": LEARNING_RATE[strategy],
                   "batch_size": BATCH_SIZE, "epochs": EPOCHS, "selected_epoch": best_epoch,
                   "selection_rule": "epoch with highest validation accuracy",
                   "trainable_layers": "all" if strategy == "full" else sorted({n.rsplit(".", 1)[0] for n in trainable})},
    }

# Third approach: use the pretrained network as a fixed feature extractor, then train logistic regression.
extractor = models.mobilenet_v3_small(weights=weights)
# Remove the classifier so the network outputs image features instead of class scores.
extractor.classifier = nn.Identity()
extractor = extractor.to(device).eval()
for p in extractor.parameters():
    p.requires_grad = False


@torch.no_grad()
def extract(loader):
    """Passes the images through the frozen network and returns (576-d feature vectors, labels)."""
    feats, ys = [], []
    for x, y in loader:
        feats.append(extractor(x.to(device)).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(feats), np.concatenate(ys)

# Extract features once; the same representations are reused for all C values.
features = {}
start = now()
features["train"] = extract(loaders["train"])
extract_time = now() - start
features["val"] = extract(loaders["val"])

# Try a small set of regularization strengths and select the best one on validation data.
best_acc = -1
for C in (0.01, 0.1, 1.0):
    # the scaler is part of the pipeline, so it is fitted on the training features only
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000, random_state=SEED))
    start = now()
    clf.fit(*features["train"])
    fit_time = now() - start
    val_acc = clf.score(*features["val"])
    print(f"[features] C={C} | val acc {val_acc:.4f}")
    if val_acc > best_acc:
        best_acc, best_clf, best_C, best_fit_time = val_acc, clf, C, fit_time

logreg = best_clf[-1]
selected["features"] = {
    "name": "Frozen features + Logistic Regression",
    "trainable_params": logreg.coef_.size + logreg.intercept_.size,
    "train_time": extract_time + best_fit_time,
    "predict": lambda split: (features[split][1], best_clf.predict(features[split][0])),
    "config": {"loss": "log-loss (multinomial)", "optimizer": "lbfgs", "C": best_C, "scaling": "StandardScaler",
               "dimensionality_reduction": "none", "selection_rule": "C with highest validation accuracy"},
}
print(f"Logistic regression: best C {best_C}\n")

# Final evaluation: every selected model is evaluated once on the test set
loaders["test"] = make_loader(test_set)
features["test"] = extract(loaders["test"])

# Save a row-normalized confusion matrix so each row shows the prediction distribution for one breed.
def plot_confusion(cm, title, path):
    """Saves the confusion matrix, normalised by row, with the breed names on both axes."""
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES), class_names, rotation=90, fontsize=6)
    ax.set_yticks(range(NUM_CLASSES), class_names, fontsize=6)
    ax.set(xlabel="Predicted breed", ylabel="True breed", title=title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

# Save a few correct and incorrect test predictions for a visual inspection
def plot_examples(y_true, y_pred, title, path, n=4):
    """Randomly chosen correct (top row) and incorrect (bottom row) test predictions."""
    rng = np.random.default_rng(SEED)
    rows = [np.flatnonzero(y_true == y_pred), np.flatnonzero(y_true != y_pred)]
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.4), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for r, candidates in enumerate(rows):
        for c, i in enumerate(rng.choice(candidates, min(n, len(candidates)), replace=False)):
            axes[r, c].imshow(Image.open(test_set._images[i]).convert("RGB"))
            axes[r, c].set_title(f"True: {class_names[y_true[i]]}\nPred: {class_names[y_pred[i]]}", fontsize=8,
                                 color="green" if r == 0 else "red")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

# Evaluate each selected approach on train, validation, and test splits, then save its metrics.
results = {}
for key, s in selected.items():
    y_train, p_train = s["predict"]("train")
    y_val, p_val = s["predict"]("val")
    y_test, p_test = s["predict"]("test")
    results[key] = {"name": s["name"], "trainable_params": int(s["trainable_params"]),
                    "train_time": s["train_time"], "config": s["config"],
                    "train": scores(y_train, p_train), "val": scores(y_val, p_val), "test": scores(y_test, p_test)}

    cm = confusion_matrix(y_test, p_test, labels=range(NUM_CLASSES))
    plot_confusion(cm, f"{s['name']}: test confusion matrix", OUT / f"{key}_confusion.png")
    plot_examples(y_test, p_test, s["name"], OUT / f"{key}_examples.png")

    # most frequent mistakes (true breed -> predicted breed)
    errors = cm.copy()
    np.fill_diagonal(errors, 0)
    print(f"{s['name']} - most confused breeds:")
    for flat in np.argsort(errors, axis=None)[::-1][:5]:
        t, p = np.unravel_index(flat, errors.shape)
        if errors[t, p] > 0:
            print(f"   {class_names[t]} -> {class_names[p]}: {errors[t, p]} images")

# Build and save a Markdown table with the main results for all approaches.
lines = ["| Approach | Trainable params | Train time (s) | Train acc | Val acc | Test acc | Test macro-F1 | Train-test gap |",
         "|---|---|---|---|---|---|---|---|"]
for r in results.values():
    gap = r["train"]["accuracy"] - r["test"]["accuracy"]
    lines.append(f"| {r['name']} | {r['trainable_params']:,} | {r['train_time']:.1f} | "
                 f"{r['train']['accuracy']:.4f} | {r['val']['accuracy']:.4f} | {r['test']['accuracy']:.4f} | "
                 f"{r['test']['macro_f1']:.4f} | {gap:.4f} |")
table = "\n".join(lines)
print("\n" + table)
(OUT / "results_table.md").write_text(table + "\n")
with open(OUT / "results.json", "w") as f:
    json.dump({"results": results}, f, indent=2, default=float)