''' FILE TO DEFINE THE FUNCTION THAT SCALES THE TEMPERATURE FACTORS FOR A GIVEN MODEL ON 
    A VALIDATION SET TO OPTIMIZE THE PREDICTIONS AND RANK THE MODELS ON THE SAME BENCHMARK '''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import true_negative_accuracy, true_positive_accuracy, simple_accuracy
from losses import TrueNegativeBCELoss, TruePositiveBCELoss, CrossEntropyLoss

# ── Find optimal per-class probability factor to maximize F1 ──────────────
def optimize_per_class_factor_f1(
    model,
    dataloader,
    num_classes: int,
    device=None,
    min_factor: float = 0.5,
    max_factor: float = 10.0,
    n_grid_points: int = 200,
):
    """
    For each class independently, find the multiplicative factor f* such that

        preds = (sigmoid(logit) * f  >=  0.5)

    maximises the per-class F1 score on the given dataset.

    This is strictly equivalent to finding the optimal decision threshold
    t* = 0.5 / f*  per class, but keeps the threshold fixed at 0.5 so
    you only need to store one scalar per class.

    Args:
        model: PyTorch model returning logits of shape (N, num_classes).
        dataloader: DataLoader yielding dicts with 'image' and 'label'.
        num_classes: Number of output classes.
        device: 'cuda', 'cpu', or None (auto).
        min_factor: Smallest factor to try  (0.5 → effective threshold 1.0).
        max_factor: Largest  factor to try  (10  → effective threshold 0.05).
        n_grid_points: Granularity of the 1-D grid search.

    Returns:
        best_factors (torch.Tensor): shape (num_classes,)
        best_f1s     (torch.Tensor): shape (num_classes,)
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.eval()

    # ── 1. Collect all probabilities and targets ──
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch['image'], batch['label']
            inputs = inputs.to(device)
            logits = model(inputs)
            if logits.dim() == 1:
                logits = logits.view(-1, 1)
            all_probs.append(torch.sigmoid(logits).detach().cpu())
            all_targets.append(targets.detach().cpu())

    all_probs = torch.cat(all_probs, dim=0)      # (N, C)
    all_targets = torch.cat(all_targets, dim=0)   # (N, C)

    if all_probs.shape[1] != num_classes:
        raise ValueError(
            f"Model output dim ({all_probs.shape[1]}) != num_classes ({num_classes})."
        )

    # ── 2. Grid search per class ──
    factor_grid = torch.linspace(min_factor, max_factor, steps=n_grid_points)

    best_factors = torch.ones(num_classes, dtype=torch.float32)
    best_f1s = torch.zeros(num_classes, dtype=torch.float32)

    for c in range(num_classes):
        probs_c = all_probs[:, c]
        targets_c = (all_targets[:, c] > 0.5).int()

        best_f1 = -1.0
        best_factor = 1.0

        for factor in factor_grid:
            preds = (probs_c * factor >= 0.5).int()

            tp = ((preds == 1) & (targets_c == 1)).sum().item()
            fp = ((preds == 1) & (targets_c == 0)).sum().item()
            fn = ((preds == 0) & (targets_c == 1)).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_factor = factor.item()

        best_factors[c] = best_factor
        best_f1s[c] = best_f1

    # ── 3. Print results ──
    for c in range(num_classes):
        eff_threshold = 0.5 / best_factors[c].item()
        print(f"Class {c:2d}: F1 = {best_f1s[c]:.4f},  factor = {best_factors[c]:.4f}  (≡ threshold {eff_threshold:.4f})")

    print("\nOptimal per-class factor vector:")
    print(best_factors)
    return best_factors, best_f1s


# ── Evaluate with per-class probability factors ──────────────────────────
def evaluate_with_factors(
    model,
    dataloader,
    num_classes: int,
    factors: torch.Tensor,
    device=None,
    training_loss: nn.Module = None,
):
    """
    Evaluate a model applying per-class multiplicative factors on the
    probabilities, then thresholding at 0.5.

    Computes:
      - TP / TN accuracy  (using factored predictions)
      - TP / TN BCE       (on raw logits, no factor — pure model quality)
      - training_loss      (on raw logits, if provided)

    Args:
        model: PyTorch model returning logits (N, num_classes).
        dataloader: DataLoader yielding dicts with 'image' and 'label'.
        num_classes: Number of output classes.
        factors: Tensor of shape (num_classes,) — per-class probability multipliers.
        device: 'cuda', 'cpu', or None (auto).
        training_loss: Optional loss module to compute on raw logits.

    Returns:
        training_loss_value, tp_acc, tn_acc, tp_bce, tn_bce
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    factors = factors.detach().float().view(1, -1).to(device)
    if factors.shape[1] != num_classes:
        raise ValueError(
            f"factors has {factors.shape[1]} elements but num_classes={num_classes}."
        )

    model.eval()

    all_preds = []
    all_targets = []
    all_logits = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch['image'], batch['label']
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            if logits.dim() == 1:
                logits = logits.view(-1, 1)

            # Factor-scaled predictions (proba * factor >= 0.5)
            probs = torch.sigmoid(logits)
            preds = (probs * factors >= 0.5).int()

            all_preds.append(preds.detach().cpu())
            all_targets.append(targets.detach().cpu())
            all_logits.append(logits.detach().cpu())

    preds_np = torch.cat(all_preds, dim=0).numpy().astype(np.int32)
    targets_np = torch.cat(all_targets, dim=0).numpy().astype(np.int32)
    all_logits = torch.cat(all_logits, dim=0)
    targets_torch = torch.cat(all_targets, dim=0).float()

    # ── Accuracy metrics (on factored predictions) ──
    tp_acc = float(true_positive_accuracy(preds_np, targets_np))
    tn_acc = float(true_negative_accuracy(preds_np, targets_np))

    # ── BCE metrics (on raw logits, no factor — measures pure model quality) ──
    tp_loss_fn = TruePositiveBCELoss()
    tn_loss_fn = TrueNegativeBCELoss()
    tp_bce = float(tp_loss_fn(all_logits, targets_torch).item())
    tn_bce = float(tn_loss_fn(all_logits, targets_torch).item())

    # ── Training loss (on raw logits, if provided) ──
    if training_loss is not None:
        training_loss_value = float(training_loss(all_logits, targets_torch).item())
    else:
        training_loss_value = 0.0

    print(f"Training loss on val: {training_loss_value:.4f}")
    print(f"TP accuracy:  {tp_acc:.4f}")
    print(f"TN accuracy:  {tn_acc:.4f}")
    print(f"TP BCE (raw): {tp_bce:.4f}")
    print(f"TN BCE (raw): {tn_bce:.4f}")

    return training_loss_value, tp_acc, tn_acc, tp_bce, tn_bce
