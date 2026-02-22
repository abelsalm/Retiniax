import torch
import wandb
import monai 
import scipy
import torch.nn as nn
import torch.optim as optim
import timm
from earlystopping import EarlyStopping
from losses import AsymmetricLossMultiLabel, AsymmetricLossBinary, CoherenceFactorLoss
from losses import true_negative_accuracy, true_positive_accuracy, simple_accuracy
from losses import TrueNegativeBCELoss, TruePositiveBCELoss, weights, CombinedBCELoss
from model import DeepClassifier, DeepClassifierBinary, DeepClassifierMultiHead
from ocular_dataset import OcularDataset
from tuning_testing import get_outputs, optimize_per_class_factor_f1, evaluate_with_factors
from training_utils import train_model, evaluate_model, plot_losses, CLASSES, train_epoch
from torch.utils.data import DataLoader
from transforms import monai_transform_sequence, val_transform_sequence
from ocular_dataset import OcularDataset
from transforms import image_size


## train and val with different transforms 
data_dir = "/workspace/data_15"

# binary for now
train_dataset = OcularDataset(
        csv_file="/workspace/Retiniax/training_data/train_dataset.csv",
        data_dir=data_dir,
        transform=monai_transform_sequence,
    )

val_dataset = OcularDataset(
        csv_file="/workspace/Retiniax/training_data/val_dataset.csv",
        data_dir=data_dir,
        transform=val_transform_sequence,
    )

BS = 32

train_loader = DataLoader(
        train_dataset,
        batch_size=BS, # SET HERE BATCH SIZE
        shuffle=True, 
        num_workers=19, # SET HERE Workers
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,   # ← KEEP workers alive between epochs (avoids respawn cost)
        prefetch_factor=4,         # ← prefetch 4 batches per worker (hides I/O latency)
    )

val_loader = DataLoader(
        val_dataset,
        batch_size=32, # SET HERE BATCH SIZE
        shuffle=False, # ← no need to shuffle validation
        num_workers=19, # SET HERE Workers
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,   # ← KEEP workers alive between epochs
        prefetch_factor=4,         # ← prefetch 4 batches per worker
    )

model = "inception_next_tiny.sail_in1k"
drop_rate = 0

# backbone
backbone = timm.create_model(model, in_chans=3, pretrained=False, num_classes=0, drop_path_rate=drop_rate)

# wrapper
model = DeepClassifier(encoder=backbone, n_classes=14)

# ── Performance: enable cuDNN autotuner for fixed input size (384×384) ──
torch.backends.cudnn.benchmark = True

# ── Criterion ──
criterion = CombinedBCELoss(w_tp=4.0, w_tn=0.5, class_weight_tp=weights, class_weight_tn=None)

# ── Hyper-parameters ──
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
FROZEN_EPOCHS =  4     # phase 1: encoder frozen, only head learns
EPOCHS        = 100     # phase 2: everything unfrozen
LR_FROZEN     = 1e-4   # higher LR is fine when only head trains (fewer params)
LR            = 2e-5
WD            = 5e-5

model.to(DEVICE)

train_losses, val_losses = [], []
scaler = None  # GradScaler will be auto-created on first epoch

wandb.init(
    project="retiniax-training",
    name= "inception_next_tiny_5",
    config={
        "backbone":        "inception_next_tiny.sail_in1k",
        "drop_rate":       drop_rate,
        "n_classes":   14,
        "batch_size":  BS,
        "Frozen epochs":   FROZEN_EPOCHS,
        "lr_frozen":       LR_FROZEN,
        "epochs":          EPOCHS,
        "lr":              LR,
        "weight_decay":    WD,
        "criterion":       "CombinedBCELoss, w_tp=4.0, w_tn=0.5, class_weight_tp=weights, class_weight_tn=None",
        "optimizer":       "AdamW",
        "scheduler":       "ReduceLROnPlateau",
        "mixed_precision": True,
    },
)


# FROZEN PHASE
print("=" * 60)
print(f" Encoder FROZEN training head only for {FROZEN_EPOCHS} epochs")
print("=" * 60)

# Freeze encoder
for param in model.encoder.parameters():
    param.requires_grad = False

# Optimizer only on trainable (head) parameters
head_params = [p for p in model.parameters() if p.requires_grad]
optimizer_frozen = optim.AdamW(head_params, lr=LR_FROZEN, weight_decay=WD)
scheduler_frozen = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_frozen, mode="min", factor=0.5, patience=2
)

for epoch in range(FROZEN_EPOCHS):
    train_loss, scaler = train_epoch(
        model, train_loader, criterion, optimizer_frozen,
        device=DEVICE, multi_h=False, scaler=scaler, use_amp=True)
    train_losses.append(train_loss)

    # Benchmark evaluation
    all_probs, all_targets, all_logits = get_outputs(model, val_loader, device=DEVICE)
    factors, f1s = optimize_per_class_factor_f1(all_probs, all_targets, num_classes=14)
    val_loss, tp_acc, tn_acc, tp_bce, tn_bce, tp_bce_rsc, tn_bce_rsc = evaluate_with_factors(
        all_logits, all_targets, num_classes=14, factors=factors, training_loss=criterion
    )
    val_losses.append(val_loss)
    mean_f1 = f1s.mean()
    dot_bce = tp_bce*tn_bce
    dot_rescaled_bce = tp_bce_rsc*tn_bce_rsc
    dot_acc = tp_acc*tn_acc

    scheduler_frozen.step(val_loss)
    current_lr = optimizer_frozen.param_groups[0]["lr"]

    print(
        f"[FROZEN] Epoch {epoch+1}/{FROZEN_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
        f"TP-Acc: {tp_acc:.4f} | TN-Acc: {tn_acc:.4f} | "
        f"TP-BCE: {tp_bce:.4f} | TN-BCE: {tn_bce:.4f} | "
        f"TP-BCE-rsc: {tp_bce_rsc:.4f} | TN-BCE-rsc: {tn_bce_rsc:.4f} | "
        f"BCE-dot: {dot_bce:.4f} | "
        f"BCE-dot-rescaled: {dot_rescaled_bce:.4f} | "
        f"F1: {mean_f1:.4f} | "
        f"LR: {current_lr:.2e}"
        f"Image size: {image_size}"
    )

    # ── Log everything to wandb ──
    wandb.log({
        "epoch":               epoch + 1,
        "train/loss":          train_loss,
        "val/loss":            val_loss,
        "mean_f1":             mean_f1,
        "dot_acc":             dot_acc,
        "tp_acc":              tp_acc,
        "tn_acc":              tn_acc,
        "tp_bce":              tp_bce,
        "tn_bce":              tn_bce,
        "tp_bce_rescaled":     tp_bce_rsc,
        "tn_bce_rescaled":     tn_bce_rsc,
        "dot_bce":     dot_bce,
        "dot_bce_rescaled":     dot_rescaled_bce,
    })


# UNFROZEN PHASE
print("\n" + "=" * 60)
print(f"Encoder UNFROZEN fine-tuning everything for {EPOCHS} epochs")
print("=" * 60)

# Unfreeze encoder
for param in model.encoder.parameters():
    param.requires_grad = True

# New optimizer & scheduler for ALL parameters
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)
# Reset scaler (fresh start for the new optimizer)
scaler = None

for epoch in range(EPOCHS):
    train_loss, scaler = train_epoch(
        model, train_loader, criterion, optimizer,
        device=DEVICE, multi_h=False,
        scaler=scaler, use_amp=True,
    )
    train_losses.append(train_loss)

    # Benchmark evaluation
    all_probs, all_targets, all_logits = get_outputs(model, val_loader, device=DEVICE)
    factors, f1s = optimize_per_class_factor_f1(all_probs, all_targets, num_classes=14)
    val_loss, tp_acc, tn_acc, tp_bce, tn_bce, tp_bce_rsc, tn_bce_rsc = evaluate_with_factors(
        all_logits, all_targets, num_classes=14, factors=factors, training_loss=criterion
    )
    val_losses.append(val_loss)
    mean_f1 = f1s.mean()
    dot_bce = tp_bce*tn_bce
    dot_rescaled_bce = tp_bce_rsc*tn_bce_rsc
    dot_acc = tp_acc*tn_acc
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
        f"TP-Acc: {tp_acc:.4f} | TN-Acc: {tn_acc:.4f} | "
        f"TP-BCE: {tp_bce:.4f} | TN-BCE: {tn_bce:.4f} | "
        f"TP-BCE-rsc: {tp_bce_rsc:.4f} | TN-BCE-rsc: {tn_bce_rsc:.4f} | "
        f"BCE-dot: {dot_bce:.4f} | "
        f"BCE-dot-rescaled: {dot_rescaled_bce:.4f} | "
        f"F1: {mean_f1:.4f} | "
        f"LR: {current_lr:.2e}"
    )

    # ── Log everything to wandb ──
    wandb.log({
        "epoch":               epoch + 1,
        "train/loss":          train_loss,
        "val/loss":            val_loss,
        "mean_f1":             mean_f1,
        "dot_acc":             dot_acc,
        "tp_acc":              tp_acc,
        "tn_acc":              tn_acc,
        "tp_bce":              tp_bce,
        "tn_bce":              tn_bce,
        "tp_bce_rescaled":     tp_bce_rsc,
        "tn_bce_rescaled":     tn_bce_rsc,
        "dot_bce":     dot_bce,
        "dot_bce_rescaled":     dot_rescaled_bce,
    })




wandb.finish()
print("Training complete.")