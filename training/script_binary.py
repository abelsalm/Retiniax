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
from ocular_dataset import OcularDatasetBinary
from tuning_testing import get_outputs, optimize_per_class_factor_f1, evaluate_with_factors
from training_utils import train_model, evaluate_model, plot_losses, CLASSES, train_epoch
from torch.utils.data import DataLoader
from transforms import monai_transform_sequence, val_transform_sequence
from transforms import image_size


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

    train_dataset = OcularDatasetBinary(
            csv_file=csv_train,
            data_dir=data_dir,
            transform=monai_transform_sequence,
        )

    val_dataset = OcularDatasetBinary(
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
    model = DeepClassifierBinary(encoder=backbone)

    torch.backends.cudnn.benchmark = True

    criterion = AsymmetricLossBinary() 

    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
    FROZEN_EPOCHS =  4
    EPOCHS        = 120
    LR_FROZEN     = 1e-4
    LR            = 5e-5
    WD            = 1e-5

    model.to(DEVICE)

    train_losses, val_losses = [], []
    scaler = None

    use_wandb = True

    if use_wandb:
        wandb.init(
            project="retiniax-training-binary",
            name= "inception_small_binary_2",
            config={
                "backbone":        'inception_small',
                "drop_rate":       drop_rate,
                "n_classes":       2,
                "batch_size":      BS,
                "Frozen epochs":   FROZEN_EPOCHS,
                "lr_frozen":       LR_FROZEN,
                "epochs":          EPOCHS,
                "lr":              LR,
                "weight_decay":    WD,
                "criterion":       "CombinedBCELoss, w_tp=1.0, w_tn=1.0, no class weights",
                "optimizer":       "AdamW",
                "scheduler":       "CosineAnnealingLR, ConstantLR, SequentialLR with 2/3 of the epochs 2/3 on cosine set to decrease to 1/5 of the initial LR",
                "mixed_precision": True,
                "task":            "binary (healthy vs pathological)",
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
        factors, f1s = optimize_per_class_factor_f1(all_probs, all_targets, num_classes=2)
        val_loss, tp_acc, tn_acc, tp_bce, tn_bce, tp_bce_rsc, tn_bce_rsc = evaluate_with_factors(
            all_logits, all_targets, num_classes=2, factors=factors, training_loss=criterion
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
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(2*EPOCHS/3), eta_min=LR/5)
    scheduler2 = optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=EPOCHS-int(2*EPOCHS/3))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], [int(2*EPOCHS/3)])
    scaler = None

    for epoch in range(EPOCHS):
        train_loss, scaler = train_epoch(
            model, train_loader, criterion, optimizer,
            device=DEVICE, multi_h=False,
            scaler=scaler, use_amp=True,
        )
        train_losses.append(train_loss)

        all_probs, all_targets, all_logits = get_outputs(model, val_loader, device=DEVICE)
        factors, f1s = optimize_per_class_factor_f1(all_probs, all_targets, num_classes=2)
        val_loss, tp_acc, tn_acc, tp_bce, tn_bce, tp_bce_rsc, tn_bce_rsc = evaluate_with_factors(
            all_logits, all_targets, num_classes=2, factors=factors, training_loss=criterion
        )
        val_losses.append(val_loss)
        mean_f1 = f1s.mean()
        dot_bce = tp_bce*tn_bce
        dot_rescaled_bce = tp_bce_rsc*tn_bce_rsc
        dot_acc = tp_acc*tn_acc
        scheduler.step()
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

    if use_wandb:
        wandb.finish()
    print("Training complete.")
