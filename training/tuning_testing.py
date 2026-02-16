''' FILE TO DEFINE THE FUNCTION THAT SCALES THE TEMPERATURE FACTORS FOR A GIVEN MODEL ON 
    A VALIDATION SET TO OPTIMIZE THE PREDICTIONS AND RANK THE MODELS ON THE SAME BENCHMARK '''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import true_negative_accuracy, true_positive_accuracy, simple_accuracy
from losses import TrueNegativeBCELoss, TruePositiveBCELoss, CrossEntropyLoss
from tqdm import tqdm
import time

# get the outputs of the model on the validation set only once 
def get_outputs(model, dataloader, device='cuda'):
    model.eval()

    # all tensors, logits, probs and targets
    all_probs = []
    all_targets = []
    all_logits = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, targets = batch['image'], batch['label']
            inputs = inputs.to(device)
            logits = model(inputs)
            if logits.dim() == 1:
                logits = logits.view(-1, 1)
            all_logits.append(logits)
            all_probs.append(torch.sigmoid(logits).detach().cpu())
            all_targets.append(targets.detach().cpu())

    all_probs = torch.cat(all_probs, dim=0)      # (N, C)
    all_targets = torch.cat(all_targets, dim=0)   # (N, C)
    all_logits = torch.cat(all_logits, dim=0)

    return all_probs, all_targets, all_logits

# find optimal per-class probability factor to maximize F1 
def optimize_per_class_factor_f1(
    all_probs,
    all_targets,
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
        all_probs: Tensor of shape (N, num_classes) containing the probabilities for each class.
        all_targets: Tensor of shape (N, num_classes) containing the targets for each class.
        num_classes: Number of output classes.
        device: 'cuda', 'cpu', or None (auto).
        min_factor: Smallest factor to try  (0.5 → effective threshold 1.0).
        max_factor: Largest  factor to try  (10  → effective threshold 0.05).
        n_grid_points: Granularity of the 1-D grid search.

    Returns:
        best_factors (torch.Tensor): shape (num_classes,)
        best_f1s     (torch.Tensor): shape (num_classes,)
    """

    if all_probs.shape[1] != num_classes:
        raise ValueError(
            f"Model output dim ({all_probs.shape[1]}) != num_classes ({num_classes})."
        )

    # grid search per class
    factor_grid = torch.linspace(min_factor, max_factor, steps=n_grid_points)

    best_factors = torch.ones(num_classes, dtype=torch.float32)
    best_f1s = torch.zeros(num_classes, dtype=torch.float32)

    for c in tqdm(range(num_classes), desc="Optimizing per-class F1"):
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

    # print results
    for c in range(num_classes):
        eff_threshold = 0.5 / best_factors[c].item()
        print(f"Class {c:2d}: F1 = {best_f1s[c]:.4f},  factor = {best_factors[c]:.4f}  (≡ threshold {eff_threshold:.4f})")

    print("\nOptimal per-class factor vector:")
    print(best_factors)
    return best_factors, best_f1s


# evaluate with per-class probability factors directly from logits
def evaluate_with_factors(
    all_logits: torch.Tensor,
    all_targets: torch.Tensor,
    num_classes: int,
    factors: torch.Tensor,
    training_loss: nn.Module = None,
):
    """
    Evaluate a model by applying per-class multiplicative factors on the
    probabilities (sigmoid(logit)), then thresholding at 0.5 after scaling.

    Computes:
      - TP / TN accuracy  (using factored predictions)
      - TP / TN BCE       (on raw logits, no factor — pure model quality)
      - TP / TN BCE       (on logits after factor rescaling — to compare effect of rescaling)
      - training_loss     (on raw logits, if provided)

    Args:
        all_logits: Tensor (N, num_classes), raw model logits on device.
        all_targets: Tensor (N, num_classes), ground-truth binary.
        num_classes: Number of output classes.
        factors: Tensor (num_classes,), scaling factors for probability per class.
        training_loss: Optional loss module to compute on raw logits.

    Returns:
        training_loss_value, tp_acc, tn_acc, tp_bce, tn_bce, tp_bce_rescaled, tn_bce_rescaled
    """

    start_time = time.time()

    # Ensure tensors are on same device
    device = all_logits.device
    all_targets = all_targets.to(device)
    factors = factors.to(device)

    # Compute probabilities from logits
    probs = torch.sigmoid(all_logits)           # (N, C)
    prob_factors = factors.view(1, -1)          # (1, C)
    probs_scaled = probs * prob_factors         # (N, C)
    preds_factored = (probs_scaled >= 0.5).long()

    targets_bin = (all_targets > 0.5).long()
    
    # Convert to numpy for accuracy metrics
    preds_np = preds_factored.cpu().numpy()
    targets_np = targets_bin.cpu().numpy()

    # ── Accuracy metrics (on factored predictions) ──
    tp_acc = float(true_positive_accuracy(preds_np, targets_np))
    tn_acc = float(true_negative_accuracy(preds_np, targets_np))

    # ── BCE metrics (on raw logits, no factor — pure model quality) ──
    tp_loss_fn = TruePositiveBCELoss()
    tn_loss_fn = TrueNegativeBCELoss()
    tp_bce = float(tp_loss_fn(all_logits, all_targets).item())
    tn_bce = float(tn_loss_fn(all_logits, all_targets).item())

    # ── BCE metrics (on logits after factor rescaling to compare with pure) ──
    # To apply factor scaling to probabilities, transform threshold so: sigmoid(f*logit)=0.5 ⇒ logit=logit_thr, f*sigmoid(logit_thr)=0.5
    # But for BCE, we simply use logits' effect through scaling logit before sigmoid
    logits_scaled = all_logits + factors.log().view(1, -1)
    tp_bce_rescaled = float(tp_loss_fn(logits_scaled, all_targets).item())
    tn_bce_rescaled = float(tn_loss_fn(logits_scaled, all_targets).item())

    # ── Training loss (on raw logits, if provided) ──
    if training_loss is not None:
        training_loss_value = float(training_loss(all_logits, all_targets).item())
    else:
        training_loss_value = 0.0

    print(f"Training loss on val: {training_loss_value:.4f}")
    print(f"TP accuracy (factored):  {tp_acc:.4f}")
    print(f"TN accuracy (factored):  {tn_acc:.4f}")
    print(f"TP BCE (raw logits):     {tp_bce:.4f}")
    print(f"TN BCE (raw logits):     {tn_bce:.4f}")
    print(f"TP BCE (factored logit): {tp_bce_rescaled:.4f}")
    print(f"TN BCE (factored logit): {tn_bce_rescaled:.4f}")

    end_time = time.time()
    print(f"Time taken to compute all metrics: {end_time - start_time:.2f} seconds")

    return training_loss_value, tp_acc, tn_acc, tp_bce, tn_bce, tp_bce_rescaled, tn_bce_rescaled

