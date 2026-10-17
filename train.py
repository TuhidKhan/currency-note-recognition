import os
import json
import time
import sys
import torch
import torch.nn as nn
from torchvision import models
from utils.dataset_loader import get_data_loaders


def build_or_load_model(num_classes, device, weights_path="models/model.pth"):
    """
    Builds ResNet-50. If previous fine-tuned weights exist, loads them
    to rapidly adapt to newly added variations.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, num_classes)
    )

    if os.path.exists(weights_path):
        print(f"Loading existing checkpoint from {weights_path} for continuous fine-tuning...", flush=True)
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
            print("Successfully loaded existing weights!", flush=True)
        except Exception as e:
            print(f"Starting fresh from ImageNet weights due to: {e}", flush=True)

    # Unfreeze layer4 + fc
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model.to(device)


def train_model(epochs=12, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # 1. Load Data
    print("Preparing dataset loaders...", flush=True)
    train_loader, val_loader, full_dataset = get_data_loaders(
        root_dir="dataset", batch_size=batch_size
    )
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Dataset has {num_classes} classes: {class_names}", flush=True)

    os.makedirs("models", exist_ok=True)
    with open("models/class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    # 2. Balanced class weights
    class_counts = [0] * num_classes
    for _, class_idx in full_dataset.samples:
        class_counts[class_idx] += 1
    total_samples = len(full_dataset.samples)
    class_weights = [total_samples / (num_classes * max(1, c)) for c in class_counts]
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Total samples: {total_samples}, samples per class: {dict(zip(class_names, class_counts))}", flush=True)

    # 3. Model
    model = build_or_load_model(num_classes, device)

    # 4. Loss & Differential Optimizer
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.AdamW([
        {"params": model.layer4.parameters(), "lr": 5e-5, "weight_decay": 1e-4},
        {"params": model.fc.parameters(), "lr": 5e-4, "weight_decay": 1e-4}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    start_time = time.time()

    print("\n" + "=" * 55, flush=True)
    print("STARTING RAPID CONTINUOUS FINE-TUNING", flush=True)
    print("=" * 55, flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        scheduler.step()
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100

        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / total_val if total_val > 0 else 0
        epoch_val_acc = (correct_val / total_val) * 100 if total_val > 0 else 0

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"| Train Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:.1f}% "
            f"| Val Loss: {epoch_val_loss:.4f} - Acc: {epoch_val_acc:.1f}%",
            flush=True
        )

        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), "models/model.pth")
            os.makedirs("hf_space_backend", exist_ok=True)
            torch.save(model.state_dict(), "hf_space_backend/model.pth")
            with open("hf_space_backend/class_names.json", "w") as f:
                json.dump(class_names, f, indent=2)

    total_time = time.time() - start_time
    print("=" * 55, flush=True)
    print(f"Training completed in {total_time:.1f} seconds!", flush=True)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%", flush=True)
    print("=" * 55, flush=True)


if __name__ == "__main__":
    train_model(epochs=12, batch_size=16)
