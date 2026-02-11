''' FILE TO DEFINE THE FUNCTION THAT SCALES THE TEMPERATURE FACTORS FOR A GIVEN MODEL ON 
    A VALIDATION SET TO OPTIMIZE THE PREDICTIONS AND RANK THE MODELS ON THE SAME BENCHMARK '''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import true_negative_accuracy, true_positive_accuracy, simple_accuracy
from losses import TrueNegativeBCELoss, TruePositiveBCELoss, CrossEntropyLoss

# optimize the temperature factors for each class to maximize the F1 score
def optimize_per_class_temperature_f1(
    model,
    dataloader,
    num_classes: int,
    device=None,
    min_temp: float = 0.1,
    max_temp: float = 10.0,
    n_grid_points: int = 32,
):
    """
    Optimize per-class temperature scaling factors to maximize the F1 score of
    each class independently on a validation set.

    Each class c has an associated raw parameter t_raw[c] which is mapped to
    a temperature T[c] via a scaled sigmoid:

        T[c] = min_temp + (max_temp - min_temp) * sigmoid(t_raw[c])

    For each class, we perform a 1D grid search over t_raw in [-5, 5] and
    select the temperature that yields the best F1 score for that class
    (treating it as a binary classification problem with threshold 0.5).

    Args:
        model: PyTorch model returning logits of shape (N, num_classes).
        dataloader: DataLoader providing (inputs, targets) where
                    targets has shape (N, num_classes) with {0,1} labels.
        num_classes (int): Number of non-exclusive classes.
        device: Device to run the model on (e.g. 'cuda', 'cpu').
                If None, inferred from model parameters.
        min_temp (float): Minimum allowed temperature in the scaled sigmoid.
        max_temp (float): Maximum allowed temperature in the scaled sigmoid.
        n_grid_points (int): Number of grid points for the 1D search
                             in the raw parameter space [-5, 5].

    Returns:
        best_temps (torch.Tensor): Tensor of shape (num_classes,) containing
                                   the optimal temperature for each class.
        best_f1s (torch.Tensor): Tensor of shape (num_classes,) with the
                                 corresponding F1 scores.
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.eval()

    # Collect logits and targets over the whole validation set
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch['image'], batch['label']

            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            # Ensure correct shape
            if logits.dim() == 1:
                logits = logits.view(-1, 1)

            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    if all_logits.shape[1] != num_classes:
        raise ValueError(
            f"Model output dimension ({all_logits.shape[1]}) does not match "
            f"num_classes ({num_classes})."
        )

    # Prepare grid over raw temperature parameters
    raw_grid = torch.linspace(-5.0, 5.0, steps=n_grid_points)

    best_temps = torch.zeros(num_classes, dtype=torch.float32)
    best_f1s = torch.zeros(num_classes, dtype=torch.float32)

    for c in range(num_classes):
        logits_c = all_logits[:, c]
        targets_c = all_targets[:, c]

        # Ensure binary labels
        targets_c = (targets_c > 0.5).int()

        best_f1 = -1.0
        best_temp = 1.0

        for raw in raw_grid:
            # Scaled sigmoid to get temperature in [min_temp, max_temp]
            temp = min_temp + (max_temp - min_temp) * torch.sigmoid(raw)

            # Apply temperature scaling on this class only
            scaled_logits = logits_c / temp
            probs = torch.sigmoid(scaled_logits)
            preds = (probs >= 0.5).int()

            # Compute F1 for this class
            tp = ((preds == 1) & (targets_c == 1)).sum().item()
            fp = ((preds == 1) & (targets_c == 0)).sum().item()
            fn = ((preds == 0) & (targets_c == 1)).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            if f1 > best_f1:
                best_f1 = f1
                best_temp = temp.item()

        best_temps[c] = best_temp
        best_f1s[c] = best_f1

    # Print results
    for c in range(num_classes):
        print(f"Class {c}: best F1 = {best_f1s[c]:.4f}, temperature = {best_temps[c]:.4f}")

    print("Optimal per-class temperature vector:")
    print(best_temps)

    return best_temps, best_f1s


# function to evaluate the model with the temperature factors on the benchmark losses 
def evaluate_with_temperatures(
    model,
    dataloader,
    num_classes: int,
    temperatures: torch.Tensor,
    device=None,
    threshold: float = 0.5,
    training_loss: nn.Module = None,
):
    """
    Evaluate a model with per-class temperature scaling for accuracies, while
    computing TP/TN BCE losses on the *untempered* probabilities.

    Returns:
        tp_acc (float): true_positive_accuracy computed on tempered predictions
        tn_acc (float): true_negative_accuracy computed on tempered predictions
        tp_bce (float): TruePositiveBCELoss on untempered probabilities
        tn_bce (float): TrueNegativeBCELoss on untempered probabilities
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    temperatures = temperatures.detach().to(device).float().view(1, -1)
    if temperatures.shape[1] != num_classes:
        raise ValueError(
            f"temperatures has shape {tuple(temperatures.shape)} but expected (num_classes,) "
            f"or (1, num_classes) with num_classes={num_classes}."
        )

    model.eval()

    all_preds_temp = []
    all_targets = []
    all_probs_no_temp = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch['image'], batch['label']

            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            if logits.dim() == 1:
                logits = logits.view(-1, 1)

            if logits.shape[1] != num_classes:
                raise ValueError(
                    f"Model output dimension ({logits.shape[1]}) does not match "
                    f"num_classes ({num_classes})."
                )

            # Tempered predictions for accuracy metrics
            scaled_logits = logits / temperatures
            probs_temp = torch.sigmoid(scaled_logits)
            preds_temp = (probs_temp >= threshold).int()

            # Untempered probabilities for BCE losses
            probs_no_temp = torch.sigmoid(logits)

            all_preds_temp.append(preds_temp.detach().cpu())
            all_targets.append(targets.detach().cpu())
            all_probs_no_temp.append(probs_no_temp.detach().cpu())

    preds_temp = torch.cat(all_preds_temp, dim=0).numpy().astype(np.int32)
    targets_np = torch.cat(all_targets, dim=0).numpy().astype(np.int32)

    tp_acc = float(true_positive_accuracy(preds_temp, targets_np))
    tn_acc = float(true_negative_accuracy(preds_temp, targets_np))

    probs_no_temp = torch.cat(all_probs_no_temp, dim=0)
    targets_torch = torch.cat(all_targets, dim=0).float()

    tp_loss_fn = TruePositiveBCELoss()
    tn_loss_fn = TrueNegativeBCELoss()

    tp_bce = float(tp_loss_fn(probs_no_temp, targets_torch).item())
    tn_bce = float(tn_loss_fn(probs_no_temp, targets_torch).item())

    training_loss_value = float(training_loss(probs_no_temp, targets_torch).item())
    
    print(f"Training loss on validation set: {training_loss_value}")
    print(f"Tempered TP accuracy: {tp_acc}")
    print(f"Tempered TN accuracy: {tn_acc}")
    print(f"Un-tempered TP BCE:   {tp_bce}")
    print(f"Un-tempered TN BCE:   {tn_bce}")

    return training_loss_value, tp_acc, tn_acc, tp_bce, tn_bce
