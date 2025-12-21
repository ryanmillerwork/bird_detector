#!/usr/bin/env python3
"""
Train ConvNeXt-Small classifier for bird species identification.
Stage 2 of the detection pipeline.

Configuration via .env file:
    DATA_DIR: Path to hand_sorted images (default: ./hand_sorted)
    OUTPUT_DIR: Where to save models (default: ./models)  
    BATCH_SIZE: Training batch size (default: 16)
    NUM_WORKERS: Data loader workers (default: 4)
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import os
import json
import random
import time
import hashlib
from pathlib import Path
from datetime import datetime


def load_env():
    """Load config from .env file."""
    env = {}
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().strip().split("\n"):
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                env[key.strip()] = val.strip()
    return env


_env = load_env()

# Set temp directory before importing torch (fixes temp dir errors)
local_tmp = Path("./tmp")
local_tmp.mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(local_tmp.absolute())
os.environ["TEMP"] = str(local_tmp.absolute())
os.environ["TMP"] = str(local_tmp.absolute())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import timm

# Configuration (from .env file)
DATA_DIR = Path(_env.get("DATA_DIR", "hand_sorted"))
OUTPUT_DIR = Path(_env.get("OUTPUT_DIR", "models"))
MIN_SAMPLES = 5  # Ignore classes with fewer samples
INPUT_SIZE = 320
BATCH_SIZE = int(_env.get("BATCH_SIZE", "16"))
NUM_WORKERS = int(_env.get("NUM_WORKERS", "4"))
EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BirdDataset(Dataset):
    """Dataset for bird classification."""
    
    def __init__(self, samples, class_to_idx, transform=None):
        self.samples = samples  # List of (path, class_idx)
        self.class_to_idx = class_to_idx
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_samples_and_classes(data_dir, min_samples=5):
    """Load samples and filter classes with too few samples."""
    data_dir = Path(data_dir)
    
    # Count samples per class
    class_counts = {}
    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
            class_counts[class_dir.name] = count
    
    # Filter classes
    valid_classes = [c for c, n in class_counts.items() if n >= min_samples]
    print(f"Found {len(class_counts)} classes, keeping {len(valid_classes)} with >= {min_samples} samples")
    
    # Show what we're dropping
    dropped = [f"{c}({n})" for c, n in class_counts.items() if n < min_samples]
    if dropped:
        print(f"Dropping: {', '.join(dropped)}")
    
    # Build class mapping
    class_to_idx = {c: i for i, c in enumerate(sorted(valid_classes))}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    
    # Collect all samples
    samples = []
    for class_name in valid_classes:
        class_dir = data_dir / class_name
        for img_path in class_dir.glob("*.jpg"):
            samples.append((str(img_path), class_to_idx[class_name]))
        for img_path in class_dir.glob("*.png"):
            samples.append((str(img_path), class_to_idx[class_name]))
    
    random.shuffle(samples)
    
    print(f"Total samples: {len(samples)}")
    for class_name in sorted(valid_classes):
        count = class_counts[class_name]
        print(f"  {class_name}: {count}")
    
    return samples, class_to_idx, idx_to_class


def create_model(num_classes):
    """Create ConvNeXt-Small model with pretrained weights."""
    model = timm.create_model(
        "convnext_small.fb_in22k_ft_in1k",
        pretrained=True,
        num_classes=num_classes,
    )
    return model


def class_mapping_fingerprint(class_to_idx: dict) -> str:
    """Stable fingerprint for a class mapping (order-independent)."""
    payload = json.dumps(class_to_idx, sort_keys=True, separators=(",", ":")).encode("utf-8")
    # Short fingerprint is enough to detect mismatches without being noisy.
    return hashlib.sha256(payload).hexdigest()[:12]


def load_checkpoint_safely(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    *,
    num_classes: int,
    class_to_idx: dict,
):
    """
    Load a training checkpoint.

    If the checkpoint was trained with a different class mapping (count or mapping),
    we "warm start" by only loading tensors that match by name *and* shape, and we
    reset optimizer/scheduler/epoch tracking. This avoids head size mismatches.
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    ckpt_num_classes = checkpoint.get("num_classes")
    if ckpt_num_classes is None and isinstance(checkpoint.get("class_to_idx"), dict):
        ckpt_num_classes = len(checkpoint["class_to_idx"])

    current_fp = class_mapping_fingerprint(class_to_idx)
    ckpt_fp = checkpoint.get("class_mapping_fingerprint")
    if ckpt_fp is None and isinstance(checkpoint.get("class_to_idx"), dict):
        ckpt_fp = class_mapping_fingerprint(checkpoint["class_to_idx"])

    mapping_mismatch = (ckpt_num_classes is not None and ckpt_num_classes != num_classes) or (
        ckpt_fp is not None and ckpt_fp != current_fp
    )

    if mapping_mismatch:
        print(
            "\nCheckpoint class mapping differs from current training run; "
            "loading matching weights only and restarting optimizer/scheduler."
        )
        if ckpt_num_classes is not None:
            print(f"  Checkpoint num_classes: {ckpt_num_classes}, current: {num_classes}")
        if ckpt_fp is not None:
            print(f"  Checkpoint mapping fp: {ckpt_fp}, current: {current_fp}")

        ckpt_state = checkpoint.get("model_state_dict", checkpoint)
        model_state = model.state_dict()
        filtered_state = {}
        skipped = 0
        for k, v in ckpt_state.items():
            if k in model_state and hasattr(v, "shape") and model_state[k].shape == v.shape:
                filtered_state[k] = v
            else:
                skipped += 1

        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        if skipped:
            print(f"  Skipped {skipped} tensor(s) due to shape/name mismatch (e.g., classifier head).")
        if missing:
            print(f"  Missing keys (expected with new head): {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

        return {
            "start_epoch": 0,
            "best_val_acc": 0.0,
            "resumed": False,
        }

    # Exact match: full resume
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    best_val_acc = checkpoint.get("val_acc", 0.0)

    # Advance scheduler to correct position (suppress harmless warning)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(start_epoch):
            scheduler.step()

    return {
        "start_epoch": start_epoch,
        "best_val_acc": best_val_acc,
        "resumed": True,
    }


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return running_loss / len(dataloader), 100.0 * correct / total


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(dataloader), 100.0 * correct / total


def export_to_onnx(model, num_classes, input_size, output_path):
    """Export model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    torch.onnx.export(
        model.cpu(),
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"Exported ONNX model to {output_path}")


def main():
    print(f"Training on: {DEVICE}")
    print(f"Input size: {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load data
    samples, class_to_idx, idx_to_class = get_samples_and_classes(DATA_DIR, MIN_SAMPLES)
    num_classes = len(class_to_idx)
    print()
    
    # Save class mapping
    class_map_path = OUTPUT_DIR / "class_mapping.json"
    with open(class_map_path, "w") as f:
        json.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f, indent=2)
    print(f"Saved class mapping to {class_map_path}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE + 32, INPUT_SIZE + 32)),
        transforms.RandomCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Split data 80/20
    train_size = int(0.8 * len(samples))
    val_size = len(samples) - train_size
    
    train_samples = samples[:train_size]
    val_samples = samples[train_size:]
    
    train_dataset = BirdDataset(train_samples, class_to_idx, train_transform)
    val_dataset = BirdDataset(val_samples, class_to_idx, val_transform)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )
    
    # Create model
    print("\nLoading ConvNeXt-Small pretrained model...")
    model = create_model(num_classes)
    model = model.to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Resume from checkpoint if exists
    start_epoch = 0
    best_val_acc = 0.0
    checkpoint_path = OUTPUT_DIR / "best_model.pt"
    
    if checkpoint_path.exists():
        print(f"\nFound existing checkpoint, resuming training...")
        resume = load_checkpoint_safely(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            num_classes=num_classes,
            class_to_idx=class_to_idx,
        )
        start_epoch = resume["start_epoch"]
        best_val_acc = resume["best_val_acc"]
        if resume["resumed"]:
            print(f"  Resumed from epoch {start_epoch}, best val_acc: {best_val_acc:.2f}%")
        else:
            print("  Warm-started from checkpoint weights (new classifier head).")
    
    # Training loop
    print("\nStarting training...\n")
    
    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()
        
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.1f}s, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = OUTPUT_DIR / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
                "num_classes": num_classes,
                "class_mapping_fingerprint": class_mapping_fingerprint(class_to_idx),
                "input_size": INPUT_SIZE,
            }, checkpoint_path)
            print(f"  Saved best model (val_acc: {val_acc:.2f}%)")
            
            # Also export ONNX so we have it even if training is interrupted
            onnx_path = OUTPUT_DIR / "bird_classifier.onnx"
            export_to_onnx(model, num_classes, INPUT_SIZE, str(onnx_path))
            model = model.to(DEVICE)  # Move back to device after export
        
        print()
    
    # Save final model
    final_path = OUTPUT_DIR / "final_model.pt"
    torch.save({
        "epoch": EPOCHS,
        "model_state_dict": model.state_dict(),
        "val_acc": val_acc,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "num_classes": num_classes,
        "class_mapping_fingerprint": class_mapping_fingerprint(class_to_idx),
        "input_size": INPUT_SIZE,
    }, final_path)
    print(f"Saved final model to {final_path}")
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    
    # Load best model for export
    checkpoint = torch.load(OUTPUT_DIR / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    onnx_path = OUTPUT_DIR / "bird_classifier.onnx"
    export_to_onnx(model, num_classes, INPUT_SIZE, str(onnx_path))
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()

