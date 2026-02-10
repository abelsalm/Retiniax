import torch
import numpy as np
import torch.nn.functional as F
from torchmetrics.classification import BinaryCalibrationError
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
from earlystopping import EarlyStopping
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision
import torch.optim as optim
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from torch.amp import autocast, GradScaler


# classes and functions--------------------------------------------------------------------------------------------------
CLASSES = ['NCS', 'AUTRES', 'CICATRICE', 'DIABETE', 'DMLA', 
           'DRUSEN', 'GLAUCOME', 'UVEITE', 'MYOPIE', 'OEDEME', 
           'VASCULAIRE', 'RETINE', 'TROUBLES', 'TUMEUR']


# class to store training configuration--------------------------------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """
    Comprehensive training configuration dataclass.
    All training parameters should be specified here.
    """

    # multi_head or not 
    multi_head: bool = True
    
    # Training hyperparameters
    epochs: int = 10
    device: str = 'cuda'
    
    # Frozen encoder phase
    freezed_epochs: Optional[int] = None
    use_frozen_phase: bool = False
    
    # Optimizer configuration
    optimizer_type: str = 'AdamW'  # 'AdamW', 'Adam', 'SGD'
    lr: float = 1e-4
    weight_decay: float = 1e-5
    use_differential_lr: bool = True  # Use different LRs for encoder vs heads
    
    # Frozen phase optimizer (if different from main optimizer)
    optimizer_frozen_type: str = 'AdamW'
    lr_frozen: float = 1e-4
    weight_decay_frozen: float = 1e-5
    
    # Scheduler configuration
    use_scheduler: bool = False
    scheduler_type: str = 'ReduceLROnPlateau'  # 'ReduceLROnPlateau', 'CosineAnnealingLR', 'StepLR', 'LambdaLR'
    scheduler_params: Dict[str, Any] = field(default_factory=lambda: {
        'mode': 'min',
        'factor': 0.5,
        'patience': 3
    })
    
    # Early stopping
    use_early_stopping: bool = False
    early_stopping_patience: Optional[int] = None
    checkpoint_path: Optional[str] = None
    
    # Gradient configuration
    gradient_clip_norm: Optional[float] = None  # If None, no clipping. Otherwise, max_norm value
    gradient_accumulation_steps: int = 1  # Accumulate gradients over N steps
    
    # Loss configuration
    loss : str = 'bce'
    
    # Visualization and logging
    visualize_batches: bool = True  # Show batch visualizations
    print_every_n_batches: int = 0  # Print metrics every N batches (0 = only first batch)
    print_weights: bool = True  # Print sample weights each epoch to verify training
    
    # Validation configuration
    validate_every_n_epochs: int = 1  # Validate every N epochs
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create TrainingConfig from a dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'TrainingConfig':
        """Load TrainingConfig from a JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert TrainingConfig to dictionary."""
        return {
            'multi': self.multi,
            'multi_h': self.multi_h,
            'epochs': self.epochs,
            'device': self.device,
            'freezed_epochs': self.freezed_epochs,
            'use_frozen_phase': self.use_frozen_phase,
            'optimizer_type': self.optimizer_type,
            'lr_encoder': self.lr_encoder,
            'lr_heads': self.lr_heads,
            'lr_unified': self.lr_unified,
            'weight_decay': self.weight_decay,
            'use_differential_lr': self.use_differential_lr,
            'optimizer_frozen_type': self.optimizer_frozen_type,
            'lr_frozen': self.lr_frozen,
            'weight_decay_frozen': self.weight_decay_frozen,
            'use_scheduler': self.use_scheduler,
            'scheduler_type': self.scheduler_type,
            'scheduler_params': self.scheduler_params,
            'use_early_stopping': self.use_early_stopping,
            'early_stopping_patience': self.early_stopping_patience,
            'checkpoint_path': self.checkpoint_path,
            'gradient_clip_norm': self.gradient_clip_norm,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'loss_weights': self.loss_weights,
            'visualize_batches': self.visualize_batches,
            'print_every_n_batches': self.print_every_n_batches,
            'print_weights': self.print_weights,
            'validate_every_n_epochs': self.validate_every_n_epochs,
            'mixed_precision': self.mixed_precision,
            'compile_model': self.compile_model,
        }
    
    def to_json(self, json_path: str):
        """Save TrainingConfig to a JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)


# function to create optimizer--------------------------------------------------------------------------------------------------
def _create_optimizer(model, config: TrainingConfig, frozen_phase: bool = False):
    """Create optimizer based on config."""
    if frozen_phase:
        opt_type = config.optimizer_frozen_type
        lr = config.lr_frozen
        wd = config.weight_decay_frozen
    else:
        opt_type = config.optimizer_type
        wd = config.weight_decay
        
        if config.use_differential_lr and hasattr(model, 'head_pathology') and hasattr(model, 'head_binary'):
            # Multi-head with differential LR
            encoder_params = list(model.encoder.parameters())
            head_params = list(model.head_pathology.parameters()) + list(model.head_binary.parameters())
            if hasattr(model, 'bn'):
                head_params += list(model.bn.parameters())
            if hasattr(model, 'dropout'):
                head_params += list(model.dropout.parameters())
            
            param_groups = [
                {'params': encoder_params, 'lr': config.lr_encoder, 'weight_decay': wd},
                {'params': head_params, 'lr': config.lr_heads, 'weight_decay': wd}
            ]
            
            if opt_type == 'AdamW':
                return optim.AdamW(param_groups)
            elif opt_type == 'Adam':
                return optim.Adam(param_groups)
            elif opt_type == 'SGD':
                return optim.SGD(param_groups, momentum=0.9)
            else:
                raise ValueError(f"Unknown optimizer type: {opt_type}")
        else:
            # Unified LR
            lr = config.lr_unified
    
    # Create optimizer with unified LR
    if opt_type == 'AdamW':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == 'Adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == 'SGD':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")


def _create_scheduler(optimizer, config: TrainingConfig, num_epochs: int):
    """Create learning rate scheduler based on config."""
    if not config.use_scheduler:
        return None
    
    scheduler_type = config.scheduler_type
    params = config.scheduler_params.copy()
    
    if scheduler_type == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
    elif scheduler_type == 'CosineAnnealingLR':
        T_max = params.pop('T_max', num_epochs)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, **params)
    elif scheduler_type == 'StepLR':
        step_size = params.pop('step_size', 10)
        gamma = params.pop('gamma', 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, **params)
    elif scheduler_type == 'LambdaLR':
        # Lambda function should be provided in params
        lr_lambda = params.pop('lr_lambda', lambda epoch: 1.0)
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, **params)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# function to train the model for an epoch given a dataloader, a criterion, a config, and a device--------------------------------------------------
def train_epoch(model, dataloader, criterion, optimizer, device='cuda', multi_h=True,
                scaler=None, use_amp=True, visualize_first_batch=False):
    """
    Fonction d'entraînement pour une époque.
    
    Args:
        model: Modèle PyTorch à entraîner
        dataloader: DataLoader pour les données d'entraînement
        criterion: Fonction de loss
        optimizer: Optimiseur PyTorch
        device: Device sur lequel entraîner ('cuda' ou 'cpu')
        multi_h: Si True, architecture multi-head (défaut: True)
        scaler: GradScaler for mixed precision (created automatically if None and use_amp=True)
        use_amp: If True, use automatic mixed precision (float16) for faster GPU training
        visualize_first_batch: If True, visualize first batch (causes GPU sync, slow)
    
    Returns:
        float: Loss moyenne d'entraînement sur l'époque
        GradScaler: the scaler (pass it back on subsequent calls)
    """
    # ── cuDNN benchmark: cache best conv algorithm for fixed input sizes ──
    if device != 'cpu':
        torch.backends.cudnn.benchmark = True
    
    # ── Create AMP scaler on first call ──
    amp_enabled = use_amp and (device != 'cpu')
    if amp_enabled and scaler is None:
        scaler = GradScaler()
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training", leave=True)
    for i, batch in pbar:
        # Extraire images et labels du batch
        if isinstance(batch, dict):
            images = batch['image']
            labels = batch['label']
        elif isinstance(batch, (list, tuple)):
            images, labels = batch[0], batch[1]
        else:
            raise ValueError("Batch doit être un dict avec 'image' et 'label' ou un tuple (images, labels)")
        
        # non_blocking=True overlaps CPU→GPU transfer with computation when pin_memory is used
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if visualize_first_batch and i == 0:
            labels_for_viz = labels.clone()
            visualize_batch(images, labels_for_viz)
        
        # Zero gradients — set_to_none=True is faster (avoids memset to 0)
        optimizer.zero_grad(set_to_none=True)
        
        # ── Forward pass with AMP (float16 for matmuls/convs, float32 for reductions) ──
        if amp_enabled:
            with autocast(device_type='cuda', dtype=torch.float16):
                loss = _compute_loss(model, images, labels, criterion, multi_h)
            # Backward with scaled loss to avoid float16 underflow
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = _compute_loss(model, images, labels, criterion, multi_h)
            loss.backward()
            optimizer.step()
        
        # Accumuler la loss
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar with running average loss
        pbar.set_postfix(loss=total_loss / num_batches)
    
    # Retourner la loss moyenne
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, scaler


# function to compute loss and potentially clip logits----------------------------------------------------------------------------------------------
def _compute_loss(model, images, labels, criterion, multi_h, clip_logits=True, max_logit_value=10.0):
    """
    Helper function to compute loss based on model type.
    
    Args:
        clip_logits: If True, clip logits to prevent extreme values during training
        max_logit_value: Maximum absolute value for logits clipping
    """

    if not multi_h:
        outputs = model(images)
        if clip_logits:
            outputs = torch.clamp(outputs, min=-max_logit_value, max=max_logit_value)
        labels_for_loss = labels
        loss = criterion(outputs, labels_for_loss)
    else:
        label_b = labels[:, 0:1]
        labels_patho = labels[:, 1:]
        output_patho, output_bin = model(images)
        
        # Clip logits to prevent extreme values that cause numerical instability
        if clip_logits:
            output_patho = torch.clamp(output_patho, min=-max_logit_value, max=max_logit_value)
            output_bin = torch.clamp(output_bin, min=-max_logit_value, max=max_logit_value)
        
        loss = criterion(
            logits_patho=output_patho, 
            logits_binary=output_bin, 
            targets_patho=labels_patho, 
            targets_binary=label_b
        )
    return loss


# function to train model--------------------------------------------------------------------------------------------------
def train_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    config: Optional[TrainingConfig] = None,
    # Legacy parameters for backward compatibility
    optimizer_=None,
    multi_h=None,
    freezed_epochs=None,
    optimizer_freezed=None,
    scheduler=None,
    device=None,
    epochs=None,
    early_stopping_patience=None,
    checkpoint_path=None,
    multi=None
):
    """
    Train a model with comprehensive configuration support.
    
    Args:
        model: PyTorch model to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        criterion: Loss function
        config: TrainingConfig object (preferred method)
        optimizer_, multi_h, freezed_epochs, etc.: Legacy parameters for backward compatibility
    
    Returns:
        tuple: (train_losses, val_losses, val_has, val_f1s)
    """
    # Handle backward compatibility: create config from legacy parameters if needed
    if config is None:
        config = TrainingConfig()
        # Override with legacy parameters if provided
        if multi is not None:
            config.multi = multi
        if multi_h is not None:
            config.multi_h = multi_h
        if device is not None:
            config.device = device
        if epochs is not None:
            config.epochs = epochs
        if freezed_epochs is not None:
            config.freezed_epochs = freezed_epochs
            config.use_frozen_phase = True
        if early_stopping_patience is not None:
            config.early_stopping_patience = early_stopping_patience
            config.use_early_stopping = True
        if checkpoint_path is not None:
            config.checkpoint_path = checkpoint_path
    
    train_losses, val_losses, val_has, val_f1s = [], [], [], []
    model.to(config.device)
    
    # Compile model if requested (PyTorch 2.0+)
    if config.compile_model:
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"Warning: Could not compile model: {e}")

    # Early stopping setup
    if config.use_early_stopping and config.early_stopping_patience is not None:
        early_stopper = EarlyStopping(
            patience=config.early_stopping_patience, 
            verbose=True, 
            path=config.checkpoint_path or 'best_model.pth'
        )
    else:
        early_stopper = None

    # Initial validation
    val_loss, val_fa, val_ha, val_f1 = evaluate_loss(
        model, val_loader, criterion, 
        device=config.device, 
        multi=config.multi, 
        multi_h=config.multi_h
    )
    print(f"At initialisation : Val Loss = {val_loss:.4f}, Val First Accuracy = {val_fa:.4f}, Accuracy  = {val_ha:.4f}, F1 Loss = {val_f1:.4f}")

    # Frozen encoder phase
    if config.use_frozen_phase and config.freezed_epochs is not None:
        # Use provided optimizer or create from config
        if optimizer_freezed is not None:
            optimizer_frozen = optimizer_freezed
        else:
            optimizer_frozen = _create_optimizer(model, config, frozen_phase=True)
        
        # Freeze encoder
        for param in model.encoder.parameters():
            param.requires_grad = False
        
        for epoch in range(config.freezed_epochs):
            model.train()
            total_loss = 0
            optimizer_frozen.zero_grad()  # Zero gradients at start of epoch
            
            for i, batch in enumerate(train_loader):
                images, labels = batch['image'], batch['label']
                images, labels = images.to(config.device), labels.to(config.device)

                # Store original labels for visualization
                labels_for_viz = labels.clone()
                
                # Forward pass with mixed precision if enabled
                loss = _compute_loss(
                    model, images, labels, criterion, 
                    config.multi, config.multi_h
                )
                
                # Scale loss for gradient accumulation
                loss = loss / config.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation: only step every N batches
                if (i + 1) % config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if config.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 
                            max_norm=config.gradient_clip_norm
                        )
                    
                    # Optimizer step
                    optimizer_frozen.step()
                    
                    optimizer_frozen.zero_grad()
                
                total_loss += loss.item() * config.gradient_accumulation_steps

                # Visualization
                if config.visualize_batches and i == 0:
                    visualize_batch(images, labels_for_viz)
                
                # Print metrics during training
                if config.print_every_n_batches > 0 and (i + 1) % config.print_every_n_batches == 0:
                    print(f"  Batch {i+1}/{len(train_loader)}, Loss: {loss.item() * config.gradient_accumulation_steps:.4f}")
            
            # Handle remaining gradients if batch count is not divisible by accumulation_steps
            if len(train_loader) % config.gradient_accumulation_steps != 0:
                # Gradient clipping
                if config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        max_norm=config.gradient_clip_norm
                    )
                
                # Optimizer step
                optimizer_frozen.step()
                
                optimizer_frozen.zero_grad()

            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation
            if (epoch + 1) % config.validate_every_n_epochs == 0:
                val_loss, val_fa, val_ha, val_f1 = evaluate_loss(
                    model, val_loader, criterion, 
                    device=config.device, 
                    multi=config.multi, 
                    multi_h=config.multi_h
                )
                val_losses.append(val_loss)
                val_has.append(val_ha)
                val_f1s.append(val_f1)
            else:
                # If not validating, append last values
                val_losses.append(val_losses[-1] if val_losses else 0)
                val_has.append(val_has[-1] if val_has else 0)
                val_f1s.append(val_f1s[-1] if val_f1s else 0)
                val_loss, val_ha, val_f1 = val_losses[-1], val_has[-1], val_f1s[-1]

            print(f"Still freezing encoder : Epoch {epoch+1}/{config.freezed_epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}, Val FA = {val_fa:.4f}, Accuracy  = {val_ha:.4f}, F1 Loss = {val_f1:.4f}")

    # CRITICAL FIX: UNFREEZE encoder for the unfrozen training phase
    # Previously this was freezing it again, preventing learning!
    for param in model.encoder.parameters():
        param.requires_grad = True  # Changed from False to True
    
    # Create optimizer for unfrozen phase
    if optimizer_ is not None:
        optimizer = optimizer_
    else:
        optimizer = _create_optimizer(model, config, frozen_phase=False)
        
        # Print optimizer configuration for debugging
        print(f"Optimizer created for unfrozen phase:")
        for idx, group in enumerate(optimizer.param_groups):
            print(f"  Group {idx}: lr={group['lr']}, weight_decay={group.get('weight_decay', 0)}")
    
    # Create scheduler
    if scheduler is not None:
        train_scheduler = scheduler
    else:
        num_unfrozen_epochs = config.epochs - (config.freezed_epochs if config.use_frozen_phase else 0)
        train_scheduler = _create_scheduler(optimizer, config, num_unfrozen_epochs)

    # Unfrozen training phase
    num_unfrozen_epochs = config.epochs - (config.freezed_epochs if config.use_frozen_phase else 0)
    for epoch in range(num_unfrozen_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"LR starting the epoch : {current_lr}")
        model.train()
        total_loss = 0
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        for i, batch in enumerate(train_loader):
            images, labels = batch['image'], batch['label']
            images, labels = images.to(config.device), labels.to(config.device)

            # Store original labels for visualization
            labels_for_viz = labels.clone()
            
            loss = _compute_loss(
                model, images, labels, criterion, 
                config.multi, config.multi_h
            )
            
            # Scale loss for gradient accumulation
            loss = loss / config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation: only step every N batches
            if (i + 1) % config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        max_norm=config.gradient_clip_norm
                    )
                
                # Optimizer step
                optimizer.step()
                
                optimizer.zero_grad()
            
            total_loss += loss.item() * config.gradient_accumulation_steps

            # Visualization
            if config.visualize_batches and i == 0:
                visualize_batch(images, labels_for_viz)
            
            # Print metrics during training
            if config.print_every_n_batches > 0 and (i + 1) % config.print_every_n_batches == 0:
                print(f"  Batch {i+1}/{len(train_loader)}, Loss: {loss.item() * config.gradient_accumulation_steps:.4f}")
        
        # Handle remaining gradients if batch count is not divisible by accumulation_steps
        if len(train_loader) % config.gradient_accumulation_steps != 0:
            # Gradient clipping
            if config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=config.gradient_clip_norm
                )
            
            # Optimizer step
            optimizer.step()
            
            optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        if (epoch + 1) % config.validate_every_n_epochs == 0:
            val_loss, val_fa, val_ha, val_f1 = evaluate_loss(
                model, val_loader, criterion, 
                device=config.device, 
                multi=config.multi, 
                multi_h=config.multi_h
            )
            val_losses.append(val_loss)
            val_has.append(val_ha)
            val_f1s.append(val_f1)
        else:
            # If not validating, append last values
            val_losses.append(val_losses[-1] if val_losses else 0)
            val_has.append(val_has[-1] if val_has else 0)
            val_f1s.append(val_f1s[-1] if val_f1s else 0)
            val_loss, val_ha, val_f1 = val_losses[-1], val_has[-1], val_f1s[-1]

        # Scheduler step
        if train_scheduler is not None:
            if isinstance(train_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                train_scheduler.step(val_loss)
            else:
                train_scheduler.step()

        print(f"Epoch {epoch+1}/{num_unfrozen_epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}, Val FA = {val_fa:.4f}, Accuracy  = {val_ha:.4f}, F1 Loss = {val_f1:.4f}")

        # Early stopping
        if early_stopper is not None:
            early_stopper(1/val_fa, model)
            if early_stopper.early_stop:
                print("Early stopping triggered.")
                break
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"LR ending the epoch : {current_lr}")

    # Reload best model
    if early_stopper is not None and early_stopper.best_loss is not None:
        model.load_state_dict(torch.load(early_stopper.path))
        print(f"Reloaded best model from {early_stopper.path}")

    return train_losses, val_losses, val_has, val_f1s


# function to visualize batch--------------------------------------------------------------------------------------------------
def visualize_batch(images, labels, class_names=CLASSES, num_images=4):
    """
    Affiche un échantillon d'un batch d'images et leurs labels.
    
    Args:
        images (Tensor): Le tenseur d'images (Batch, C, H, W)
        labels (Tensor): Le tenseur de labels (Batch, 15)
        class_names (list, optional): Liste des noms des 15 classes pour décoder.
        num_images (int): Nombre d'images à afficher (pour ne pas spammer).
    """
    
    # 1. Détacher du graphe de calcul et envoyer sur CPU
    # .detach() coupe le lien avec les gradients (économie mémoire)
    # .cpu() ramène les données de la VRAM vers la RAM
    images = images.detach().cpu()
    labels = labels.detach().cpu()
    
    # On limite le nombre d'images à afficher
    batch_size = images.shape[0]
    limit = min(batch_size, num_images)
    
    # 2. Dé-normalisation (Inverse de ImageNet Normalization)
    # C'est CRUCIAL pour voir les vraies couleurs
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # On applique l'inverse : image = (image * std) + mean
    images_denorm = images[:limit] * std + mean
    
    # On s'assure que les valeurs restent entre 0 et 1 pour matplotlib
    images_denorm = torch.clamp(images_denorm, 0, 1)

    # 3. Création d'une grille d'images
    grid = torchvision.utils.make_grid(images_denorm, nrow=limit, padding=2)
    
    # 4. Affichage avec Matplotlib
    plt.figure(figsize=(15, 5))
    # Permute : PyTorch est (C, H, W), Matplotlib veut (H, W, C)
    plt.imshow(grid.permute(1, 2, 0)) 
    plt.axis('off')
    plt.title(f"Aperçu du batch (x{limit})")
    plt.show()

    # 5. Affichage des Labels (Texte)
    print(f"--- Labels associés aux {limit} premières images ---")
    for i in range(limit):
        current_label = labels[i]
        
        # Récupération des indices où il y a un '1' (maladie présente)
        active_indices = torch.where(current_label == 1)[0]
        
        if class_names:
            # On convertit les indices en noms de maladies
            active_names = [class_names[idx] for idx in active_indices]
            label_str = ", ".join(active_names) if active_names else "Normal / Rien"
        else:
            # Sinon on affiche juste les indices bruts
            label_str = f"Indices actifs: {active_indices.numpy()}"
            
        print(f"Image {i+1}: {label_str}")
    print("-" * 50)


# function to calculate metrics--------------------------------------------------------------------------------------------------
def calculate_metrics(pred_logits, target_labels, threshold=0.5, multi=True):
    """
    pred_logits : sorties brutes du modèle (avant sigmoid)
    target_labels : labels 0/1
    """
    if multi:
        # 1. Sigmoid → probabilités
        probs = torch.sigmoid(pred_logits)

        # 2. Seuil → prédictions binaires
        preds = (probs > threshold).float()

        # Conversion CPU/numpy
        preds_np = preds.detach().cpu().numpy()
        preds_np_float = probs.detach().cpu().numpy()
        targets_np = target_labels.detach().cpu().numpy()

        # get argmax : 
        one_hot_pred = np.zeros_like(preds_np_float)
        one_hot_pred[np.arange(preds_np_float.shape[0]), np.argmax(preds_np_float, axis=-1)] = 1

        # main class acc :
        first_acc = np.sum(one_hot_pred*targets_np)

        # --- Métrique 1 : Hamming Accuracy ---
        acc = np.mean(preds_np == targets_np)

        # --- Métrique 2 : F1 macro ---
        f1 = f1_score(targets_np, preds_np, average='macro', zero_division=0)

        return first_acc, acc, f1

    else:
        # prédictions
        probs = torch.softmax(pred_logits, dim=1)
        preds = torch.argmax(probs, dim=1)              # [B]

        # labels : conversion one-hot -> classe
        if target_labels.dim() == 2 and target_labels.size(1) == 2:
            targets = torch.argmax(target_labels, dim=1)    # [B]
        else:
            targets = target_labels.view(-1)                # [B]

        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        acc = accuracy_score(targets_np, preds_np)
        f1 = f1_score(targets_np, preds_np, average='binary', zero_division=0)
        return acc, f1

# function to evaluate loss--------------------------------------------------------------------------------------------------
def evaluate_loss(model, loader, criterion, device='cuda', multi=True, multi_h=True):
    model.eval()
    total_loss = 0
    total_f1 = 0 
    total_fa = 0
    total_ha = 0
    total_samples = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            images, labels = batch['image'], batch['label']
            images, labels = images.to(device), labels.to(device)
            
            # Get model outputs based on architecture type
            if not multi:
                # Single output for binary classification
                outputs = model(images)
                # For binary classification, convert one-hot to class indices
                if labels.dim() == 2 and labels.size(1) == 2:
                    labels_for_loss = labels.argmax(dim=1)
                else:
                    labels_for_loss = labels.view(-1)
                loss = criterion(outputs, labels_for_loss)
            else:
                if not multi_h:
                    # Single output for multi-label
                    outputs = model(images)
                    labels_for_loss = labels
                    loss = criterion(outputs, labels_for_loss)
                else:
                    # Multi-head: get both outputs
                    output_patho, output_bin = model(images)
                    
                    # Check for extreme values (NaN, Inf, or very large logits)
                    if torch.isnan(output_patho).any() or torch.isinf(output_patho).any():
                        print(f"WARNING: NaN or Inf detected in output_patho at batch {i}")
                        print(f"  output_patho stats: min={output_patho.min().item():.4f}, max={output_patho.max().item():.4f}, mean={output_patho.mean().item():.4f}")
                    
                    if torch.isnan(output_bin).any() or torch.isinf(output_bin).any():
                        print(f"WARNING: NaN or Inf detected in output_bin at batch {i}")
                        print(f"  output_bin stats: min={output_bin.min().item():.4f}, max={output_bin.max().item():.4f}, mean={output_bin.mean().item():.4f}")
                    
                    # Clip extreme logits to prevent numerical instability
                    output_patho = torch.clamp(output_patho, min=-50, max=50)
                    output_bin = torch.clamp(output_bin, min=-50, max=50)
                    
                    # Concatenate for metrics calculation: [binary, patho1, patho2, ..., patho14]
                    outputs = torch.cat((output_bin, output_patho), dim=1)
                    
                    # Prepare labels for loss
                    label_b = labels[:, 0:1]
                    labels_patho = labels[:, 1:]
                    
                    # Compute loss
                    loss = criterion(
                        logits_patho=output_patho, 
                        logits_binary=output_bin, 
                        targets_patho=labels_patho, 
                        targets_binary=label_b
                    )
                    
                    # Check for extreme loss values
                    if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100:
                        print(f"WARNING: Extreme loss value at batch {i}: {loss.item():.4f}")
                        print(f"  output_patho range: [{output_patho.min().item():.2f}, {output_patho.max().item():.2f}]")
                        print(f"  output_bin range: [{output_bin.min().item():.2f}, {output_bin.max().item():.2f}]")
            
            # Calculate metrics
            b_fa, b_ha, b_f1 = calculate_metrics(outputs, labels, multi=multi)
            batch_size = labels.size(0)
            
            # Handle extreme loss values
            loss_value = loss.item()
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"ERROR: Loss is NaN or Inf at batch {i}, skipping this batch")
                continue
            
            total_loss += loss_value * batch_size
            total_fa += b_fa * batch_size
            total_f1 += b_f1 * batch_size
            total_ha += b_ha * batch_size
            total_samples += batch_size
            
            # Debug printing for first few batches
            if i < 2:
                # For binary classification with CrossEntropyLoss, use softmax not sigmoid
                if not multi:
                    probs = torch.softmax(outputs, dim=1)
                else:
                    probs = torch.sigmoid(outputs)
                
                # Check for extreme probabilities (should be between 0 and 1)
                if (probs < 0).any() or (probs > 1).any():
                    print(f"WARNING: Probabilities outside [0,1] range at batch {i}")
                
                print("Moyenne des probabilités :", probs.mean().item())
                print("Exemple de probas brutes :", probs[0].detach().cpu().numpy())
                print("Exemple de targets :", labels[0].detach().cpu().numpy())
                
                # Also print logits to see if they're extreme
                if multi and multi_h:
                    print(f"Exemple de logits patho (min/max): [{output_patho[0].min().item():.2f}, {output_patho[0].max().item():.2f}]")
                    print(f"Exemple de logits binary: {output_bin[0].item():.2f}")
    
    if total_samples == 0:
        print("ERROR: No valid samples processed!")
        return 0.0, 0.0, 0.0
    
    return total_loss / total_samples, total_fa / total_samples, total_ha / total_samples, total_f1 / total_samples


# function to evaluate model--------------------------------------------------------------------------------------------------
def evaluate_model(model, loader, device='cuda', threshold=0.5):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)[:, 1]
            preds = (probs > threshold).int()

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["Non-Referable", "Referable"])
    auc = roc_auc_score(all_labels, all_probs)
    ece = BinaryCalibrationError(n_bins=5, norm='l1')(torch.tensor(all_probs), torch.tensor(all_labels))

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if tp + fn > 0 else 0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0

    print(report)
    print(f"Confusion Matrix:\n{cm}")
    print(f"AUC: {auc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"ECE: {ece:.4f}")

    return {
        'report': report,
        'confusion_matrix': cm,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ece': ece
    }

# function to plot losses--------------------------------------------------------------------------------------------------
def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# function to print model wieghts to ensure training--------------------------------------------------------------------------------------------------
def _print_model_weights(model, epoch, multi_h=True, num_samples=3):
    """
    Print sample weights from different parts of the model to verify training.
    
    Args:
        model: The model to inspect
        epoch: Current epoch number
        multi_h: Whether it's a multi-head model
        num_samples: Number of weight values to print per layer
    """
    print(f"\n--- Weight Inspection (Epoch {epoch}) ---")
    
    # Store previous weights for comparison (if available)
    if not hasattr(model, '_prev_weights'):
        model._prev_weights = {}
    
    with torch.no_grad():
        # 1. Encoder weights (first and last layers if available)
        encoder_params = list(model.encoder.named_parameters())
        if len(encoder_params) > 0:
            # First layer of encoder
            first_name, first_param = encoder_params[0]
            if first_param.numel() > 0:
                first_weights = first_param.data.flatten()[:num_samples]
                prev_first = model._prev_weights.get('encoder_first', None)
                
                print(f"Encoder - {first_name}:")
                print(f"  Current: {first_weights.cpu().numpy()}")
                if prev_first is not None:
                    change = (first_weights - prev_first).abs().mean().item()
                    print(f"  Change from last epoch: {change:.6f}")
                else:
                    print(f"  (First epoch - no comparison)")
                
                model._prev_weights['encoder_first'] = first_weights.clone()
            
            # Last layer of encoder (if different from first)
            if len(encoder_params) > 1:
                last_name, last_param = encoder_params[-1]
                if last_param.numel() > 0:
                    last_weights = last_param.data.flatten()[:num_samples]
                    prev_last = model._prev_weights.get('encoder_last', None)
                    
                    print(f"Encoder - {last_name}:")
                    print(f"  Current: {last_weights.cpu().numpy()}")
                    if prev_last is not None:
                        change = (last_weights - prev_last).abs().mean().item()
                        print(f"  Change from last epoch: {change:.6f}")
                    else:
                        print(f"  (First epoch - no comparison)")
                    
                    model._prev_weights['encoder_last'] = last_weights.clone()
        
        # 2. Head weights (if multi-head)
        if multi_h and hasattr(model, 'head_pathology'):
            patho_weights = model.head_pathology.weight.data.flatten()[:num_samples]
            prev_patho = model._prev_weights.get('head_pathology', None)
            
            print(f"Head Pathology:")
            print(f"  Current: {patho_weights.cpu().numpy()}")
            if prev_patho is not None:
                change = (patho_weights - prev_patho).abs().mean().item()
                print(f"  Change from last epoch: {change:.6f}")
            else:
                print(f"  (First epoch - no comparison)")
            
            model._prev_weights['head_pathology'] = patho_weights.clone()
            
            if hasattr(model, 'head_binary'):
                binary_weights = model.head_binary.weight.data.flatten()[:num_samples]
                prev_binary = model._prev_weights.get('head_binary', None)
                
                print(f"Head Binary:")
                print(f"  Current: {binary_weights.cpu().numpy()}")
                if prev_binary is not None:
                    change = (binary_weights - prev_binary).abs().mean().item()
                    print(f"  Change from last epoch: {change:.6f}")
                else:
                    print(f"  (First epoch - no comparison)")
                
                model._prev_weights['head_binary'] = binary_weights.clone()
        
        # 3. BatchNorm/Dropout parameters (if available)
        if hasattr(model, 'bn'):
            bn_weight = model.bn.weight.data.flatten()[:num_samples]
            prev_bn = model._prev_weights.get('bn', None)
            
            print(f"BatchNorm:")
            print(f"  Current: {bn_weight.cpu().numpy()}")
            if prev_bn is not None:
                change = (bn_weight - prev_bn).abs().mean().item()
                print(f"  Change from last epoch: {change:.6f}")
            else:
                print(f"  (First epoch - no comparison)")
            
            model._prev_weights['bn'] = bn_weight.clone()
    
    print("--- End Weight Inspection ---\n")