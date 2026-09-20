import copy
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

SEED = 6304
MODEL_DIMS = {"resnet": 2048, "vit": 768, "clip": 512}


def build_head(model_name, num_classes=10, device="cpu"):
    return nn.Linear(MODEL_DIMS[model_name], num_classes).to(device)


def load_heads(checkpoint_dir, device):
    heads = {}
    for name in ["resnet", "vit", "clip"]:
        head = build_head(name, device=device)
        path = os.path.join(checkpoint_dir, f"{name}_linear_head.pt")
        head.load_state_dict(torch.load(path, map_location=device))
        head.eval()
        heads[name] = head
    return heads


def _run_epoch(head, loader, optimizer, criterion, device, train=True):
    head.train(train)
    total_loss = total_correct = total_examples = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            if train:
                optimizer.zero_grad()

            logits = head(features)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_examples += labels.size(0)

    return total_loss / total_examples, total_correct / total_examples


def train_linear_head(train_features, train_labels, val_features, val_labels,
                      input_dim, device, batch_size=128, max_epochs=50,
                      lr=1e-3, weight_decay=1e-4, patience=5):
    train_ds = TensorDataset(train_features.float(), train_labels.long())
    val_ds = TensorDataset(val_features.float(), val_labels.long())

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    head = nn.Linear(input_dim, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val = -1.0
    best_state = None
    stale = 0

    for epoch in range(max_epochs):
        train_loss, train_acc = _run_epoch(head, train_loader, optimizer, criterion, device, True)
        val_loss, val_acc = _run_epoch(head, val_loader, optimizer, criterion, device, False)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1:02d} | Train loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}"
        )

        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(head.state_dict())
            stale = 0
        else:
            stale += 1

        if stale >= patience:
            print("Early stopping.")
            break

    head.load_state_dict(best_state)
    return head, history


def save_head_and_history(head, history, name, checkpoint_dir, results_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    torch.save(head.state_dict(), os.path.join(checkpoint_dir, f"{name}_linear_head.pt"))
    with open(os.path.join(results_dir, f"{name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)
