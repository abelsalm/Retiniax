''' FILE TO DEFINE THE LOSS FUNCTIONS WITH TWO TYPES :
    FIRST THE BENCHMARK LOSSES TO ENABLE A STRICT COMPARISON BETWEEN MODELS OR HYPERPARAMETERS
    SECOND THE DIFFERENT LOSSES USED AS TRAINING LOSSES FOR BACKPROPAGATION '''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

### BENCHMARK LOSSES ###--------------------------------------------------------------------------------

# accuracy on the true positives among our pathologies 
# NUMPY VERSION
def true_positive_accuracy(predictions, targets):
    # only consider the accuracy on the true positive cases
    true_positives = (predictions == targets) & (targets == 1)
    return true_positives.mean()/np.max(targets.sum(), 1)

# accuracy on the true negatives among our pathologies 
# NUMPY VERSION
def true_negative_accuracy(predictions, targets):
    # only consider the accuracy on the true negative cases
    true_negatives = (predictions == targets) & (targets == 0)
    length_of_targets = targets.shape[1]
    n_tn = np.sum(np.ones(length_of_targets) - targets)
    return true_negatives.mean()/n_tn

# simple accuracy, for the binary classification task
# NUMPY VERSION
def simple_accuracy(predictions, targets):
    return (predictions == targets).mean()

# Cross entropy for the binary classification task as a nn.Module
# TORCH VERSION
class CrossEntropyLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(*args, **kwargs)
    def forward(self, predictions, targets):
        return self.loss_fn(predictions, targets)

# BCE on the pathologies for true positives as a nn.Module
# TORCH VERSION
class TruePositiveBCELoss(nn.Module):
    def __init__(self, class_weight=None, *args, **kwargs):
        """
        Args:
            class_weight (torch.Tensor, optional): Per-class weights for the
                non-exclusive pathology classes. Should be broadcastable to
                the shape of `targets`.
        """
        super().__init__()
        self.class_weight = class_weight

    def forward(self, predictions, targets):
        targets = targets.float()
        sum_of_targets = torch.sum(targets)

        if self.class_weight is None:
            bce = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')  # [N, C]
        else:
            cw = self.class_weight.to(predictions.device)
            print(f"predictions: {predictions.shape}")
            print(f"targets: {targets.shape}")
            print(f"predictions values: {predictions}")
            print(f"targets values: {targets}")
            bce = F.binary_cross_entropy_with_logits(predictions, targets,
                                         weight=cw, reduction='none')  # [N, C]
        positive_mask = (targets == 1).float()
        bce = bce * positive_mask
        print(f"bce: {bce}")

        # By zeroing predictions for TN, targets==0 entries will have zero loss (BCE(0,0)=0)
        # Average over ALL (so only TPs matter in the mean)
        return bce.sum()/sum_of_targets

# BCE on the pathologies for true negatives as a nn.Module
# TORCH VERSION
class TrueNegativeBCELoss(nn.Module):
    def __init__(self, class_weight=None, *args, **kwargs):
        """
        Args:
            class_weight (torch.Tensor, optional): Per-class weights for the
                non-exclusive pathology classes. Should be broadcastable to
                the shape of `targets`.
        """
        super().__init__()
        self.class_weight = class_weight

    def forward(self, predictions, targets):
        targets = targets.float()
        sum_of_targets = torch.sum(1 - targets)

        if self.class_weight is None:
            bce = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')  # [N, C]
        else:
            cw = self.class_weight.to(predictions.device)
            bce = F.binary_cross_entropy_with_logits(predictions, targets,
                                         weight=cw, reduction='none')  # [N, C]
        negative_mask = (targets == 0).float()
        bce = bce * negative_mask
        print(f"tn bce: {bce}")

        # By zeroing predictions for TN, targets==0 entries will have zero loss (BCE(0,0)=0)
        # Average over ALL (so only TPs matter in the mean)
        return bce.sum()/sum_of_targets



### Other TRAINING LOSSES ###--------------------------------------------------------------------------------

# Focal loss for the binary classification task
# TORCH VERSION
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    
    Focal Loss addresses class imbalance by down-weighting easy examples
    and focusing training on hard examples.
    
    FL(p_t) = -α * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: Weighting factor for rare classes (float or tensor of shape [num_classes])
        gamma: Focusing parameter (gamma=0 is equivalent to BCE)
        reduction: 'mean' or 'sum'
        class_weight: Optional per-class weight tensor of shape [num_classes]
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', class_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        # Optional per-class weighting for multi-label case: tensor of shape (num_classes,)
        self.class_weight = class_weight
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Raw logits from model [batch_size, num_classes]
            targets: Binary targets [batch_size, num_classes]
        
        Returns:
            Focal loss value
        """
        # Ensure class weights, if any, are on the right device
        if self.class_weight is not None:
            self.class_weight = self.class_weight.to(logits.device).view(1, -1)

        # Compute probabilities
        probs = torch.sigmoid(logits)
        
        # Compute p_t: probability of true class
        # For positive examples: p_t = p
        # For negative examples: p_t = 1 - p
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute BCE component
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Apply alpha weighting
        if isinstance(self.alpha, (float, int)):
            # Uniform alpha for all classes
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        else:
            # Per-class alpha (tensor)
            alpha_t = self.alpha.unsqueeze(0) * targets + (1 - self.alpha.unsqueeze(0)) * (1 - targets)
        
        # Compute focal loss
        focal_loss = alpha_t * focal_weight * bce
        
        # Apply optional per-class weights
        if self.class_weight is not None:
            focal_loss = focal_loss * self.class_weight
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Coherence factor between multiple pathologies and the binary classification task
# TORCH VERSION
class CoherenceFactorLoss(nn.Module):
    '''Here we want to ensure that if some pathologies are detected, then the binary classification task should also detect them
    This is done by ensuring that the probability of the binary classification task is close to the probability of the most probable pathology
    Takes in input the the logit for the binary logit and the 14 other logits for the pathologies.
    The max of the 14 logits for the pathologies is used to compute the coherence factor.'''
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.loss_fn = nn.MSELoss(*args, **kwargs)
    def forward(self, binary_logit, pathologies_logits):
        max_pathology_logit = torch.max(pathologies_logits, dim=1)
        return self.loss_fn(binary_logit, max_pathology_logit)

# ASSYMETRIC LOSS FOR THE BINARY CLASSIFICATION TASK
class AsymmetricLossMultiLabel(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification (Non-Exclusive).
    
    Paper: Asymmetric Loss For Multi-Label Classification
    Source: https://arxiv.org/abs/2009.14119
    
    Implementation follows Equation (7) and (5) in the paper.
    """
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8,
                 disable_torch_grad_focal_loss=True, class_weight=None):
        """
        Args:
            gamma_neg (float): Focusing parameter for negative samples (gamma-). 
                               Paper recommends 4.
            gamma_pos (float): Focusing parameter for positive samples (gamma+). 
                               Paper recommends 0.
            clip (float): Probability margin (m) for hard thresholding. 
                          Paper recommends 0.05.
            eps (float): Small value for numerical stability.
            disable_torch_grad_focal_loss (bool): If True, disables gradient calculation for the 
                                                  focal term itself (sometimes used for stability).
            class_weight (torch.Tensor, optional): Per-class weights of shape (C,) applied
                                                  element-wise to the loss.
        """
        super(AsymmetricLossMultiLabel, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        # Optional per-class weights for non-exclusive multi-label classes
        self.class_weight = class_weight

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): Logits (before sigmoid) of shape (N, C).
            y (torch.Tensor): Targets (0 or 1) of shape (N, C).
        """
        # Calculating probabilities
        x_sigmoid = torch.sigmoid(x)
        x_sigmoid_pos = x_sigmoid
        x_sigmoid_neg = 1 - x_sigmoid

        # Asymmetric Clipping (Probability Shifting) - Equation (5)
        if self.clip > 0:
            x_sigmoid_neg = (x_sigmoid_neg + self.clip).clamp(max=1)

        # Basic Cross Entropy Terms
        # L+ part: log(p)
        pos_term = torch.log(x_sigmoid_pos.clamp(min=self.eps))
        # L- part: log(1-p_m) -> which is log(shifted_prob_neg)
        neg_term = torch.log(x_sigmoid_neg.clamp(min=self.eps))

        # Asymmetric Focusing - Equation (7)
        # Calculate modulating factors (1-p)^gamma_pos and (p_m)^gamma_neg
        if self.disable_torch_grad_focal_loss:
            torch.set_grad_enabled(False)
        
        # Positive focusing factor: (1-p)^gamma_pos
        pos_weight = (1 - x_sigmoid_pos) ** self.gamma_pos
        # Negative focusing factor: (p_m)^gamma_neg -> here p_m is actually (1 - x_sigmoid_neg_shifted)
        # Note: In the paper p_m = max(p-m, 0). 
        # So the weight is (p_m)^gamma_neg.
        # Our x_sigmoid_neg is (1-p). The shifted version effectively handles the margin logic.
        # We use the probability of being negative (1-p) for easier implementation.
        # Let's align strictly with Eq (5): p_m = max(p-m, 0).
        # We need p (probability of class).
        p = x_sigmoid
        p_m = (p - self.clip).clamp(min=0)
        neg_weight = p_m ** self.gamma_neg

        if self.disable_torch_grad_focal_loss:
            torch.set_grad_enabled(True)

        # Final Loss Calculation
        # L = - y * PosWeight * PosTerm - (1-y) * NegWeight * NegTerm
        # Note: NegTerm is log(1-p_m). 
        # If p < m, p_m=0, 1-p_m=1, log(1)=0. Correctly hard-thresholds.
        
        # Re-calculating NegTerm strictly based on p_m for clarity with Eq 7
        neg_term_strict = torch.log((1 - p_m).clamp(min=self.eps))
        
        loss = - y * pos_weight * pos_term - (1 - y) * neg_weight * neg_term_strict
        
        # Apply optional per-class weights: shape (N, C) * (1, C)
        if self.class_weight is not None:
            class_weight = self.class_weight.to(x.device).view(1, -1)
            loss = loss * class_weight
        
        return loss.sum() # or .mean() depending on required reduction

class AsymmetricLossBinary(nn.Module):
    """
    Asymmetric Loss for Single-Label Binary Classification.
    
    A specialized wrapper for binary cases (N, 1) or (N,).
    """
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8):
        super(AsymmetricLossBinary, self).__init__()
        self.asl = AsymmetricLossMultiLabel(gamma_neg, gamma_pos, clip, eps)

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): Logits of shape (N, 1) or (N,).
            y (torch.Tensor): Targets of shape (N, 1) or (N,).
        """
        # Unify shapes to (N, 1)
        if x.dim() == 1:
            x = x.view(-1, 1)
        if y.dim() == 1:
            y = y.view(-1, 1)
            
        return self.asl(x, y)

### WEIGHTS FOR THE DIFFERENT CLASSES--------------------------------------------------------------------------------
# weight with square root of the inverse of the frequency of the class (for ODIR and RDS datasets)
'following the following results from the stats_on_a_file.py file'
'''Total positive labels across all samples: 27661
Average labels per sample: 1.13
================================================================================
output NCS                                  11254
AUTRES/ DIVERS                        1484
CICATRICE                              466
DIABETE                               3284
DMLA                                   693
DRUSEN - AEP - dépots - matériel      3261
GLAUCOME                              3869
INFLAMMATION UVEITE                    137
MYOPIE                                 882
OEDEME PAPILLAIRE                      254
PATHOLOGIE VASCULAIRE RETINIENNE       314
RETINE                                 454
TROUBLES DES MILIEUX                  1179
TUMEUR                                 130
dtype: int64



================================================================================
INVERSE SQRT WEIGHTS - 1/sqrt(percentage)
================================================================================

Total samples in dataset: 24447

Column Name                                Percentage   Weight (1/√%)
--------------------------------------------------------------------------------
turn is : 0
ncs_weight 1.4738705151273124
NCS                                           0.4603          1.4739
AUTRES/ DIVERS                                0.0607          4.0588
CICATRICE                                     0.0191          7.2430
DIABETE                                       0.1343          2.7284
DMLA                                          0.0283          5.9394
DRUSEN - AEP - dépots - matériel              0.1334          2.7380
GLAUCOME                                      0.1583          2.5137
INFLAMMATION UVEITE                           0.0056         13.3583
MYOPIE                                        0.0361          5.2648
OEDEME PAPILLAIRE                             0.0104          9.8106
PATHOLOGIE VASCULAIRE RETINIENNE              0.0128          8.8236
RETINE                                        0.0186          7.3381
TROUBLES DES MILIEUX                          0.0482          4.5536
TUMEUR                                        0.0053         13.7133
================================================================================

weights NCS                                  1.000000
AUTRES/ DIVERS                       2.753826
CICATRICE                            4.914287
DIABETE                              1.851194
DMLA                                 4.029831
DRUSEN - AEP - dépots - matériel     1.857711
GLAUCOME                             1.705509
INFLAMMATION UVEITE                  9.063442
MYOPIE                               3.572063
OEDEME PAPILLAIRE                    6.656357
PATHOLOGIE VASCULAIRE RETINIENNE     5.986716
RETINE                               4.978810
TROUBLES DES MILIEUX                 3.089559
TUMEUR                               9.304259'''
weights = torch.tensor([1.0, 2.753826, 4.914287, 1.851194, 4.029831, 1.857711, 1.705509, 9.063442, 3.572063, 6.656357, 5.986716, 4.978810, 3.089559, 9.304259])

### Logical combined BCE ###--------------------------------------------------------------------------------
class CombinedBCELoss(nn.Module):
    """
    Weighted combination of TruePositiveBCELoss and TrueNegativeBCELoss.

    total = w_tp * TruePositiveBCE + w_tn * TrueNegativeBCE

    Args:
        w_tp (float): Weighting factor for the true-positive BCE term.
        w_tn (float): Weighting factor for the true-negative BCE term.
        class_weight (torch.Tensor, optional): Per-class weight tensor of shape (C,)
            forwarded to both TruePositiveBCELoss and TrueNegativeBCELoss.
    """
    def __init__(self, w_tp=1.0, w_tn=1.0, class_weight_tp=None, class_weight_tn=None):
        super().__init__()
        self.w_tp = w_tp
        self.w_tn = w_tn
        self.tp_loss = TruePositiveBCELoss(class_weight=class_weight_tp)
        self.tn_loss = TrueNegativeBCELoss(class_weight=class_weight_tn)

    def forward(self, predictions, targets):
        """
        Args:
            predictions (torch.Tensor): Predicted probabilities (after sigmoid), shape (N, C).
            targets (torch.Tensor): Ground-truth binary labels, shape (N, C).

        Returns:
            torch.Tensor: Weighted sum of TP-BCE and TN-BCE.
        """
        loss_tp = self.tp_loss(predictions, targets)
        loss_tn = self.tn_loss(predictions, targets)
        return self.w_tp * loss_tp + self.w_tn * loss_tn


### COMBINED COHERENCE LOSS (multi-head criterion) ###----------------------------------------------------------
class CombinedCoherenceLoss(nn.Module):
    """
    Combined loss for the multi-head architecture (pathology head + binary head).
    
    Combines:
      1. Focal (or BCE) loss on the pathology logits
      2. Focal (or BCE) loss on the binary logit
      3. Coherence term: MSE between sigmoid(binary) and max(sigmoid(patho))
    
    Args:
        w_patho:           weight for the pathology loss term
        w_binary:          weight for the binary loss term
        w_coherence:       weight for the coherence loss term
        pos_weight_value:  positive-class weight (used to derive focal alpha when > 1)
        use_focal_loss:    if True use FocalLoss, else plain BCEWithLogitsLoss
        focal_alpha:       default focal alpha
        focal_gamma:       focal gamma
        focal_alpha_patho: optional per-class alpha tensor for pathology focal loss
    """
    def __init__(
        self,
        w_patho=1.0,
        w_binary=1.0,
        w_coherence=0.5,
        pos_weight_value=1.0,
        use_focal_loss=True,
        focal_alpha=0.25,
        focal_gamma=2.0,
        focal_alpha_patho=None,
    ):
        super().__init__()
        self.w_patho = w_patho
        self.w_binary = w_binary
        self.w_coherence = w_coherence
        self.pos_weight_value = pos_weight_value
        self.use_focal_loss = use_focal_loss

        if use_focal_loss:
            # Pathology focal loss
            if focal_alpha_patho is None:
                alpha_patho = (
                    pos_weight_value / (1 + pos_weight_value)
                    if pos_weight_value > 1.0
                    else focal_alpha
                )
            else:
                alpha_patho = focal_alpha_patho
            self.focal_loss_patho = FocalLoss(alpha=alpha_patho, gamma=focal_gamma)

            # Binary focal loss
            alpha_binary = (
                pos_weight_value / (1 + pos_weight_value)
                if pos_weight_value > 1.0
                else focal_alpha
            )
            self.focal_loss_binary = FocalLoss(alpha=alpha_binary, gamma=focal_gamma)
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits_patho, logits_binary, targets_patho, targets_binary):
        device = logits_patho.device

        # 1. Pathology loss
        if self.use_focal_loss:
            loss_patho = self.focal_loss_patho(logits_patho, targets_patho)
        else:
            pos_weight = torch.full(
                (logits_patho.size(1),), self.pos_weight_value, device=device
            )
            loss_patho = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(
                logits_patho, targets_patho
            )

        # 2. Binary loss
        if self.use_focal_loss:
            loss_binary = self.focal_loss_binary(logits_binary, targets_binary)
        else:
            loss_binary = self.bce_loss(logits_binary, targets_binary)

        # 3. Coherence loss (MSE between binary prob and max pathology prob)
        probs_patho = torch.sigmoid(logits_patho)
        probs_binary = torch.sigmoid(logits_binary)
        max_patho_prob, _ = torch.max(probs_patho, dim=1, keepdim=True)
        loss_coherence = torch.mean((probs_binary - max_patho_prob) ** 2)

        return (
            self.w_patho * loss_patho
            + self.w_binary * loss_binary
            + self.w_coherence * loss_coherence
        )


if __name__ == "__main__":
    print(f"Weight for the classes: {weights}")
