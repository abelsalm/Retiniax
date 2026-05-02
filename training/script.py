import os
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
import shutil


if __name__ == '__main__':

    ## train and val with different transforms 
    data_dir = "/workspace/data_15"
    csv_train = "/workspace/Retiniax/training_data/train_dataset.csv"
    csv_val = "/workspace/Retiniax/training_data/val_dataset.csv"
    save_dir = "/workspace/data_15"

    # for local tests
    '''data_dir = "/Users/abelsalmona/Documents/Retinax/Data/Data Clean/dataset"
    csv_train = "/Users/abelsalmona/Documents/Retinax/Retiniax/training_data/train_dataset_cropped.csv"
    csv_val = "/Users/abelsalmona/Documents/Retinax/Retiniax/training_data/val_dataset_cropped.csv"'''

    train_dataset = OcularDataset(
            csv_file=csv_train,
            data_dir=data_dir,
            transform=monai_transform_sequence,
        )

    val_dataset = OcularDataset(
            csv_file=csv_val,
            data_dir=data_dir,
            transform=val_transform_sequence,
        )

    BS = 8

    train_loader = DataLoader(
            train_dataset,
            batch_size=BS,
            shuffle=True, 
            num_workers=19,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            prefetch_factor=4,
        )

    val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=19,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            prefetch_factor=4,
        )

    model_name = "inception_next_small.sail_in1k"
    drop_rate = 0

    backbone = timm.create_model(model_name, in_chans=3, pretrained=False, num_classes=0, drop_path_rate=drop_rate)
    model = DeepClassifier(encoder=backbone, n_classes=14)

    torch.backends.cudnn.benchmark = True

    criterion = CombinedBCELoss(w_tp=4.0, w_tn=0.5, class_weight_tp=weights, class_weight_tn=None)

    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
    FROZEN_EPOCHS =  4
    EPOCHS        = 128
    CHECKPOINT_EVERY = 10  # unfrozen phase: save best dot_acc model per window
    CHECKPOINT_DIR = "/workspace/Retiniax/training/checkpoints"
    LR_FROZEN     = 1e-4
    LR            = 5e-5
    WD            = 1e-5

    model.to(DEVICE)

    train_losses, val_losses = [], []
    scaler = None

    use_wandb = True

    if use_wandb:
        wandb.init(
            project="retiniax-training",
            name= "inception_small_base_10",
            config={
                "backbone":        'inception_small',
                "drop_rate":       drop_rate,
                "n_classes":       14,
                "batch_size":      BS,
                "Frozen epochs":   FROZEN_EPOCHS,
                "lr_frozen":       LR_FROZEN,
                "epochs":          EPOCHS,
                "lr":              LR,
                "weight_decay":    WD,
                "criterion":       "CombinedBCELoss, w_tp=4.0, w_tn=0.5, class_weight_tp=weights, class_weight_tn=None",
                "optimizer":       "AdamW",
                "scheduler":       "SequentialLR: CosineAnnealingLR (first 2/3 epochs, eta_min=LR/10), then CosineAnnealingWarmRestarts (SGDR, periodic LR bumps)",
                "mixed_precision": True,
                "checkpoint_every": CHECKPOINT_EVERY,
                "checkpoint_dir": CHECKPOINT_DIR,
            },
        )

    # ══════════════════════════════════════════════════════════
    # FROZEN PHASE
    # ══════════════════════════════════════════════════════════
    print("=" * 60)
    print(f" Encoder FROZEN training head only for {FROZEN_EPOCHS} epochs")
    print("=" * 60)

    for param in model.encoder.parameters():
        param.requires_grad = False

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
            f"LR: {current_lr:.2e} | "
            f"Image size: {image_size}"
        )

        if use_wandb:
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
                "dot_bce":             dot_bce,
                "dot_bce_rescaled":    dot_rescaled_bce,
                "lr":                  current_lr,
            })

    # ══════════════════════════════════════════════════════════
    # UNFROZEN PHASE
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"Encoder UNFROZEN fine-tuning everything for {EPOCHS} epochs")
    print("=" * 60)

    for param in model.encoder.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    milestone = int(2 * EPOCHS / 3)
    phase2_epochs = EPOCHS - milestone
    # SGDR: cosine decay from LR down to eta_min, then restart to LR — repeats to help escape plateaus.
    t0_restart = max(1, phase2_epochs // 6)
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=milestone, eta_min=LR / 10
    )
    scheduler2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=t0_restart, T_mult=1, eta_min=LR / 10
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [scheduler1, scheduler2], [milestone]
    )
    scaler = None

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_dot_acc_window = -1.0
    best_state_window = None
    best_epoch_in_window = None
    best_tp_acc_window = None
    best_tn_acc_window = None
    window_start_epoch = 1  # 1-based unfrozen epoch index at start of current window

    for epoch in range(EPOCHS):
        train_loss, scaler = train_epoch(
            model, train_loader, criterion, optimizer,
            device=DEVICE, multi_h=False,
            scaler=scaler, use_amp=True,
        )
        train_losses.append(train_loss)

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
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        unfrozen_epoch = epoch + 1
        if dot_acc > best_dot_acc_window:
            best_dot_acc_window = dot_acc
            best_epoch_in_window = unfrozen_epoch
            best_tp_acc_window = tp_acc
            best_tn_acc_window = tn_acc
            best_state_window = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        end_of_full_window = unfrozen_epoch % CHECKPOINT_EVERY == 0
        if end_of_full_window and best_state_window is not None:
            window_end = unfrozen_epoch
            ckpt_name = (
                f"best_dot_acc_{window_start_epoch:04d}_{window_end:04d}_"
                f"ep{best_epoch_in_window:04d}_{best_dot_acc_window:.6f}.pt"
            )
            ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
            torch.save(
                {
                    "window_start_epoch": window_start_epoch,
                    "window_end_epoch": window_end,
                    "best_epoch_in_window": best_epoch_in_window,
                    "dot_acc": best_dot_acc_window,
                    "tp_acc": best_tp_acc_window,
                    "tn_acc": best_tn_acc_window,
                    "model_state_dict": best_state_window,
                },
                ckpt_path,
            )
            print(f"  Saved checkpoint (best in window {window_start_epoch}-{window_end}): {ckpt_path}")
            best_dot_acc_window = -1.0
            best_state_window = None
            best_epoch_in_window = None
            best_tp_acc_window = None
            best_tn_acc_window = None
            window_start_epoch = window_end + 1

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

        if use_wandb:
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
                "dot_bce":             dot_bce,
                "dot_bce_rescaled":    dot_rescaled_bce,
                "lr":                  current_lr,
            })

    if EPOCHS % CHECKPOINT_EVERY != 0 and best_state_window is not None:
        window_end = EPOCHS
        ckpt_name = (
            f"best_dot_acc_{window_start_epoch:04d}_{window_end:04d}_"
            f"ep{best_epoch_in_window:04d}_{best_dot_acc_window:.6f}.pt"
        )
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
        torch.save(
            {
                "window_start_epoch": window_start_epoch,
                "window_end_epoch": window_end,
                "best_epoch_in_window": best_epoch_in_window,
                "dot_acc": best_dot_acc_window,
                "tp_acc": best_tp_acc_window,
                "tn_acc": best_tn_acc_window,
                "model_state_dict": best_state_window,
            },
            ckpt_path,
        )
        print(f"  Saved checkpoint (best in final partial window {window_start_epoch}-{window_end}): {ckpt_path}")

    # save all checkpoints in the save dir 
    for file in os.listdir(CHECKPOINT_DIR):
        if file.endswith(".pt"):
            shutil.copy(os.path.join(CHECKPOINT_DIR, file), os.path.join(save_dir, file))


    if use_wandb:
        wandb.finish()
    print("Training complete.")
