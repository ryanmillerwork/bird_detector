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
import tempfile
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

# Set temp directory before importing torch.
# IMPORTANT: default to a *local* filesystem (e.g. /tmp). On many clusters, the repo
# lives on NFS; using ./tmp can create noisy `.nfs*` cleanup errors with multiprocessing.
_tmp_root = Path(os.environ.get("BIRD_DETECTOR_TMPDIR", "/tmp/bird_detector_tmp"))
_tmp_root.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(_tmp_root)
os.environ["TEMP"] = str(_tmp_root)
os.environ["TMP"] = str(_tmp_root)
# Ensure Python's tempfile module uses the same location.
tempfile.tempdir = str(_tmp_root)

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
_requested_workers = int(_env.get("NUM_WORKERS", "4"))
try:
    _suggested_max_workers = len(os.sched_getaffinity(0))
except Exception:
    _suggested_max_workers = os.cpu_count() or 4

# PyTorch warns based on total workers across all DataLoaders (train + val).
# Keep the total <= suggested max to avoid oversubscription in cgroups/SLURM allocations.
TRAIN_NUM_WORKERS = min(_requested_workers, _suggested_max_workers)
VAL_NUM_WORKERS = min(_requested_workers, max(0, _suggested_max_workers - TRAIN_NUM_WORKERS))
NUM_WORKERS = TRAIN_NUM_WORKERS  # Back-compat: used for printing only
EPOCHS = 30
LEARNING_RATE = 1e-4
RESUME = _env.get("RESUME", "ask").strip().lower()  # ask, resume, reset, fresh
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


def class_mappings_match(ckpt_mapping, current_mapping: dict) -> bool:
    """True only if both mappings are identical (same keys and indices)."""
    if not isinstance(ckpt_mapping, dict):
        return False
    if len(ckpt_mapping) != len(current_mapping):
        return False
    return ckpt_mapping == current_mapping


def describe_class_mapping_diff(ckpt_mapping, current_mapping: dict) -> tuple[list[str], list[str]]:
    """Return (added, removed) class names relative to the checkpoint."""
    ckpt_classes = set(ckpt_mapping.keys()) if isinstance(ckpt_mapping, dict) else set()
    current_classes = set(current_mapping.keys())
    added = sorted(current_classes - ckpt_classes)
    removed = sorted(ckpt_classes - current_classes)
    return added, removed


def backup_artifacts(output_dir: Path) -> None:
    """Rename existing deploy artifacts so a reset/fresh run cannot overwrite them."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for name in ("best_model.pt", "bird_classifier.onnx"):
        path = output_dir / name
        if path.exists():
            bak = output_dir / f"{name}.{ts}.bak"
            path.rename(bak)
            print(f"  Backed up {path} -> {bak}")


def prompt_resume_mode(checkpoint_epoch: int, val_acc: float) -> str:
    """Interactive choice: resume, reset, or fresh. Defaults to resume."""
    display_epoch = checkpoint_epoch + 1
    print(f"\nFound existing checkpoint (epoch {display_epoch}, best val_acc {val_acc:.2f}%).")
    print(f"  1) Resume - continue from epoch {display_epoch}, keep {val_acc:.2f}% as the bar to beat")
    print("  2) Reset  - keep the trained weights, restart at epoch 1 with the bar at 0")
    print("  3) Fresh  - discard and retrain from ImageNet-pretrained weights")
    while True:
        try:
            choice = input("Choice [1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return "resume"
        if choice in ("", "1"):
            return "resume"
        if choice == "2":
            return "reset"
        if choice == "3":
            return "fresh"
        print("Invalid choice. Enter 1, 2, or 3.")


def resolve_resume_mode(checkpoint_epoch: int, val_acc: float) -> str:
    """Pick resume/reset/fresh from RESUME env and optional interactive prompt."""
    if RESUME not in ("ask", "resume", "reset", "fresh"):
        print(f"Warning: unknown RESUME={RESUME!r}; treating as 'ask'")
        mode = "ask"
    else:
        mode = RESUME

    if mode != "ask":
        return mode

    if sys.stdin.isatty():
        return prompt_resume_mode(checkpoint_epoch, val_acc)

    print("Non-interactive stdin; resuming from checkpoint (set RESUME=reset|fresh to override).")
    return "resume"


def apply_checkpoint_resume(checkpoint: dict, model, optimizer, scheduler) -> tuple[int, float]:
    """Full resume: weights, optimizer, epoch counter, and scheduler position."""
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    best_val_acc = checkpoint.get("val_acc", 0.0)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(start_epoch):
            scheduler.step()

    return start_epoch, best_val_acc


def apply_checkpoint_reset(checkpoint: dict, model) -> None:
    """Load trained weights only; caller keeps start_epoch=0 and best_val_acc=0."""
    model.load_state_dict(checkpoint["model_state_dict"])


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
    print(f"DataLoader workers (train/val): {TRAIN_NUM_WORKERS}/{VAL_NUM_WORKERS} (requested: {_requested_workers}, suggested max total: {_suggested_max_workers})")
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
        num_workers=TRAIN_NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=VAL_NUM_WORKERS,
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
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        ckpt_epoch = checkpoint.get("epoch", 0)
        ckpt_val_acc = checkpoint.get("val_acc", 0.0)
        ckpt_class_to_idx = checkpoint.get("class_to_idx")

        if not class_mappings_match(ckpt_class_to_idx, class_to_idx):
            added, removed = describe_class_mapping_diff(ckpt_class_to_idx, class_to_idx)
            print("\nCheckpoint class mapping does not match current data; starting from pretrained weights.")
            if removed:
                print(f"  Removed classes: {', '.join(removed)}")
            if added:
                print(f"  Added classes: {', '.join(added)}")
            if not removed and not added:
                print("  Class names match but index assignments differ (sample counts may have shifted).")
            print("  Backing up existing artifacts before fresh training...")
            backup_artifacts(OUTPUT_DIR)
        else:
            mode = resolve_resume_mode(ckpt_epoch, ckpt_val_acc)

            if mode == "resume":
                start_epoch, best_val_acc = apply_checkpoint_resume(
                    checkpoint, model, optimizer, scheduler
                )
                print(f"  Resuming from epoch {start_epoch}, best val_acc: {best_val_acc:.2f}%")
                if start_epoch >= EPOCHS:
                    print(
                        f"\nCheckpoint already completed all {EPOCHS} epochs. "
                        "Use reset or fresh (RESUME=reset|fresh) to train again."
                    )
                    sys.exit(0)
            elif mode == "reset":
                print("\nReset: keeping trained weights, restarting at epoch 1 with val bar at 0.")
                backup_artifacts(OUTPUT_DIR)
                apply_checkpoint_reset(checkpoint, model)
            else:  # fresh
                print("\nFresh: retraining from ImageNet-pretrained weights.")
                backup_artifacts(OUTPUT_DIR)

    print("\nStarting training...\n")

    val_acc = best_val_acc
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

