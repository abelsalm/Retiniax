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


if __name__ == '__main__':

    ## train and val with different transforms 
    data_dir = "/workspace/data_15"
    csv_train = "/workspace/Retiniax/training_data/train_dataset.csv"
    csv_val = "/workspace/Retiniax/training_data/val_dataset.csv"

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

    BS = 32

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
            batch_size=BS,
            shuffle=False,
            num_workers=19,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            prefetch_factor=4,
        )

    model_name = "inception_next_base.sail_in1k_384"
    drop_rate = 0

    backbone = timm.create_model(model_name, in_chans=3, pretrained=False, num_classes=0, drop_path_rate=drop_rate)
    model = DeepClassifier(encoder=backbone, n_classes=14)

    torch.backends.cudnn.benchmark = True

    criterion = CombinedBCELoss(w_tp=4.0, w_tn=0.5, class_weight_tp=weights, class_weight_tn=None)

    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
    FROZEN_EPOCHS =  2
    EPOCHS        = 100
    PHASE2_EPOCHS =  2*EPOCHS // 5
    PHASE3_EPOCHS = EPOCHS - PHASE2_EPOCHS
    WARMUP_EPOCHS = 4
    LR_FROZEN     = 1e-4
    LR            = 2e-5
    WD            = 8e-5

    model.to(DEVICE)

    train_losses, val_losses = [], []
    scaler = None
    global_epoch = 0

    use_wandb = True

    if use_wandb:
        wandb.init(
            project="retiniax-training",
            name= "inception_next_base_loss_switch_1",
            config={
                "backbone":        "inception_next_base.sail_in1k_384, pretrained=False",
                "drop_rate":       drop_rate,
                "n_classes":       14,
                "batch_size":      BS,
                "frozen_epochs":   FROZEN_EPOCHS,
                "lr_frozen":       LR_FROZEN,
                "phase2_epochs":   PHASE2_EPOCHS,
                "phase3_epochs":   PHASE3_EPOCHS,
                "warmup_epochs":   WARMUP_EPOCHS,
                "lr":              LR,
                "weight_decay":    WD,
                "criterion_phase2": "CombinedBCELoss, w_tp=4.0, w_tn=0.5, class_weight_tp=weights",
                "criterion_phase3": "AsymmetricLossMultiLabel, gamma_neg=4, gamma_pos=0, clip=0.1",
                "optimizer":       "AdamW",
                "mixed_precision": True,
            },
        )

    # ══════════════════════════════════════════════════════════
    # PHASE 1: FROZEN
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
        global_epoch += 1

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
                "epoch":               global_epoch,
                "phase":               "frozen",
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
    # PHASE 2: UNFROZEN — CombinedBCELoss
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"Encoder UNFROZEN fine-tuning with CombinedBCELoss for {PHASE2_EPOCHS} epochs")
    print("=" * 60)

    for param in model.encoder.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS, eta_min=LR/5)
    scaler = None

    for epoch in range(PHASE2_EPOCHS):
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
        global_epoch += 1

        print(
            f"[BCE] Epoch {epoch+1}/{PHASE2_EPOCHS} (global {global_epoch}) | "
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
                "epoch":               global_epoch,
                "phase":               "bce_unfrozen",
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
    # PHASE 3: UNFROZEN — AsymmetricLossMultiLabel
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"Encoder UNFROZEN fine-tuning with AsymmetricLossMultiLabel for {PHASE3_EPOCHS} epochs")
    print("=" * 60)

    criterion2 = AsymmetricLossMultiLabel(clip=0.05, gamma_neg=3, gamma_pos=1)
    optimizer2 = optim.AdamW(model.parameters(), lr=LR/5, weight_decay=WD)
    scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer2, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=PHASE3_EPOCHS - WARMUP_EPOCHS, eta_min=LR/5)
    scheduler2 = torch.optim.lr_scheduler.SequentialLR(optimizer2, [scheduler_warmup, scheduler_cosine], [WARMUP_EPOCHS])
    scaler = None

    for epoch in range(PHASE3_EPOCHS):
        train_loss, scaler = train_epoch(
            model, train_loader, criterion2, optimizer2,
            device=DEVICE, multi_h=False,
            scaler=scaler, use_amp=True,
        )
        train_losses.append(train_loss)

        all_probs, all_targets, all_logits = get_outputs(model, val_loader, device=DEVICE)
        factors, f1s = optimize_per_class_factor_f1(all_probs, all_targets, num_classes=14)
        val_loss, tp_acc, tn_acc, tp_bce, tn_bce, tp_bce_rsc, tn_bce_rsc = evaluate_with_factors(
            all_logits, all_targets, num_classes=14, factors=factors, training_loss=criterion2
        )
        val_losses.append(val_loss)
        mean_f1 = f1s.mean()
        dot_bce = tp_bce*tn_bce
        dot_rescaled_bce = tp_bce_rsc*tn_bce_rsc
        dot_acc = tp_acc*tn_acc
        scheduler2.step()
        current_lr = optimizer2.param_groups[0]["lr"]
        global_epoch += 1

        print(
            f"[ASL] Epoch {epoch+1}/{PHASE3_EPOCHS} (global {global_epoch}) | "
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
                "epoch":               global_epoch,
                "phase":               "asl_unfrozen",
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

    if use_wandb:
        wandb.finish()
    print("Training complete.")
