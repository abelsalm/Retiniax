import numpy as np
import torch
from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotated,
    RandGaussianNoised,
    RandGaussianSharpend,
    RandHistogramShiftd,
    RandZoomd,
    Rand2DElasticd,
    Resized,
    ToTensorD,
    EnsureChannelFirstD,
    MapTransform,
    LoadImaged,
    RandLambdad,
    Transform,
    ScaleIntensityRanged,
    Lambdad,
)
from monai.config import KeysCollection
from monai.utils import set_determinism
import scipy.ndimage as ndi
from typing import Dict, Any

# Fix for OverflowError: Python integer 4294967296 out of bounds for uint32
# This is a known issue with MONAI and NumPy version incompatibility
# Monkey-patch the get_seed function to return a safe value
import monai.transforms.compose as monai_compose
'''_original_get_seed = monai_compose.get_seed

def _safe_get_seed():
    """Safe version of get_seed that ensures values are within uint32 bounds."""
    seed = _original_get_seed()
    # Ensure seed is within valid uint32 range [0, 2^32 - 1]
    MAX_SEED = np.iinfo(np.uint32).max  # 4294967295
    if seed >= MAX_SEED:
        seed = seed % MAX_SEED
    return int(seed)

monai_compose.get_seed = _safe_get_seed

# Set MONAI determinism and numpy random seed
np.random.seed(42)
torch.manual_seed(42)
set_determinism(seed=42)'''

img_key = 'image'


# custom transform to mask the circular region outside the central eye
class MaskCircularRegiond(MapTransform):
    """
    Transform MONAI personnalisée pour masquer la zone hors du cercle central dans les images de fond d'œil.
    
    Le cercle est centré sur l'image et son rayon est la moitié de la largeur de l'image.
    Tous les pixels en dehors du cercle sont mis à zéro (noir).
    
    Args:
        keys: Les clés du dictionnaire à transformer (généralement ["image"])
    """
    
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
    
    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self._mask_circular_region(d[key])
        return d
    
    def _mask_circular_region(self, img: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """
        Applique un masque circulaire à l'image.
        
        Args:
            img: Image au format (C, H, W) ou (H, W, C) ou (H, W)
        
        Returns:
            Image avec les pixels hors du cercle mis à zéro
        """
        # Convertir en numpy si c'est un tensor
        is_tensor = torch.is_tensor(img)
        if is_tensor:
            img_np = img.detach().cpu().numpy()
        else:
            img_np = np.asarray(img)
        
        # Déterminer les dimensions spatiales
        if img_np.ndim == 3:
            # Format (C, H, W) ou (H, W, C)
            if img_np.shape[0] in (1, 3):  # Channel-first (C, H, W)
                h, w = img_np.shape[1], img_np.shape[2]
                channel_first = True
            else:  # Channel-last (H, W, C)
                h, w = img_np.shape[0], img_np.shape[1]
                channel_first = False
        else:  # 2D (H, W)
            h, w = img_np.shape
            channel_first = None
        
        # Calculer le centre et le rayon
        center_x, center_y = w / 2.0, h / 2.0
        radius = w / 2.0  # Rayon = demi-largeur de l'image
        
        # Créer une grille de coordonnées
        y_coords, x_coords = np.ogrid[:h, :w]
        
        # Calculer la distance de chaque pixel au centre
        distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        
        # Créer le masque binaire (1.0 dans le cercle, 0.0 en dehors)
        # Utiliser float32 pour garantir des valeurs exactes 0.0 et 1.0
        mask = (distances <= radius).astype(np.float32)
        
        # Appliquer le masque
        if img_np.ndim == 3:
            if channel_first:
                # (C, H, W) - étendre le masque pour toutes les canaux
                mask_expanded = np.stack([mask, mask, mask], axis=0)  
                img_np_masked = img_np * mask_expanded

                # --- Quick plot of the masked image (only for debug, single image) ---
                '''try:
                    import matplotlib.pyplot as plt
                    # show the first 3-channel image, shape (C, H, W) -> (H, W, C)
                    img_to_show = np.transpose(img_np_masked, (1, 2, 0))
                    plt.imshow(img_to_show.astype(np.float32) / (img_to_show.max() if img_to_show.max() != 0 else 1))
                    plt.title("Masked (C,H,W->HWC)")
                    plt.axis('off')
                    plt.show()
                except Exception as e:
                    print("Plotting masked image failed:", e)'''
            else:
                # (H, W, C) - étendre le masque pour toutes les canaux
                mask_expanded = np.stack([mask, mask, mask], axis=-1)  # shape (H, W, 3)
                img_np_masked = img_np * mask_expanded
        else:
            # 2D (H, W)
            img_np_masked = img_np * mask
        
        # Reconvertir en tensor si nécessaire
        if is_tensor:
            result = torch.from_numpy(img_np_masked).to(img.device if hasattr(img, 'device') else torch.device('cpu'))
            return result
        
        return img_np_masked

# custom transform to apply a random 2d elastic deformation to a centered square sub-region
class Rand2DElasticCenteredSquared(Rand2DElasticd):
    """
    Subclass of Rand2DElasticd that applies elastic deformation only to a centered square sub-region.
    
    The square is centered in the image and has a width that is a fraction of the original image width.
    The elastic deformation is applied only to this square region, leaving the rest of the image unchanged.
    
    Args:
        keys: Keys to transform
        square_width_ratio: Ratio of the square width to the image width (default: 0.8, i.e., 80% of image width)
        spacing: Spacing for the elastic deformation grid
        magnitude_range: Range of magnitudes for the elastic deformation
        prob: Probability of applying the transform
        **kwargs: Additional arguments passed to Rand2DElasticd
    """
    
    def __init__(
        self,
        keys: KeysCollection,
        square_width_ratio: float = 0.8,
        spacing: tuple[int, int] | int = (20, 30),
        magnitude_range: tuple[int, int] = (1, 2),
        prob: float = 0.1,
        allow_missing_keys: bool = False,
        **kwargs
    ):
        # Initialize parent
        super().__init__(
            keys=keys,
            spacing=spacing,
            magnitude_range=magnitude_range,
            prob=prob,
            allow_missing_keys=allow_missing_keys,
            **kwargs
        )
        self.square_width_ratio = square_width_ratio
        # Store parameters for easy access
        self._spacing = spacing
        self._magnitude_range = magnitude_range
        self._prob = prob
    
    def __call__(self, data: dict) -> dict:
        """
        Apply elastic deformation only to a centered square sub-region.
        """
        d = dict(data)
        
        # Check if we should apply the transform
        if self._prob < 1.0 and np.random.random() >= self._prob:
            return d
        
        for key in self.key_iterator(d):
            d[key] = self._apply_elastic_to_centered_square(d[key], key)
        
        return d
    
    def _apply_elastic_to_centered_square(
        self, img: np.ndarray | torch.Tensor, key: str
    ) -> np.ndarray | torch.Tensor:
        """
        Apply elastic deformation only to a centered square sub-region.
        
        Args:
            img: Image in (C, H, W) format
            key: The key name for creating temporary dict
        
        Returns:
            Image with elastic deformation applied only to the centered square
        """
        # Convert to numpy if tensor
        is_tensor = torch.is_tensor(img)
        if is_tensor:
            img_np = img.detach().cpu().numpy()
        else:
            img_np = np.asarray(img)
        
        # Ensure channel-first format (C, H, W)
        if img_np.ndim != 3 or img_np.shape[0] not in (1, 3):
            raise ValueError(f"Expected (C, H, W) format, got shape {img_np.shape}")
        
        c, h, w = img_np.shape
        
        # Calculate the centered square region
        square_size = int(min(h, w) * self.square_width_ratio)
        start_h = (h - square_size) // 2
        start_w = (w - square_size) // 2
        end_h = start_h + square_size
        end_w = start_w + square_size
        
        # Extract the centered square sub-region
        square_region = img_np[:, start_h:end_h, start_w:end_w].copy()  # (C, square_size, square_size)
        
        # Create a temporary dict with just the square region
        temp_data = {key: square_region}
        
        # Create a new Rand2DElasticd instance with adjusted spacing for the smaller region
        # Scale spacing proportionally to maintain similar deformation density
        if isinstance(self._spacing, tuple):
            spacing_scale = square_size / max(h, w)
            adjusted_spacing = (
                max(1, int(self._spacing[0] * spacing_scale)),
                max(1, int(self._spacing[1] * spacing_scale))
            )
        else:
            spacing_scale = square_size / max(h, w)
            adjusted_spacing = max(1, int(self._spacing * spacing_scale))
        
        # Create a temporary elastic transform for the square region
        temp_elastic = Rand2DElasticd(
            keys=[key],
            spacing=adjusted_spacing,
            magnitude_range=self._magnitude_range,
            prob=1.0,  # Always apply since we already checked prob
            allow_missing_keys=self.allow_missing_keys,
        )
        
        # Apply the elastic deformation to the square region
        deformed_square = temp_elastic(temp_data)[key]
        
        # Put the deformed square back into the original image
        result = img_np.copy()
        result[:, start_h:end_h, start_w:end_w] = deformed_square
        
        # Convert back to tensor if needed
        if is_tensor:
            result = torch.from_numpy(result).to(img.device if hasattr(img, 'device') else torch.device('cpu'))
        
        return result


# functions for custom transforms using lambdad
def random_sqrt_transform(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applique la racine carrée (sqrt) à l'image 'image' avec une probabilité gérée par RandLambda.
    Le dictionnaire 'data' est l'entrée standard pour les transforms MONAI basés sur un dictionnaire.
    """
    img = data['image']
    
    # S'assurer que les valeurs sont positives avant la racine carrée
    # et que le type est float pour le calcul
    img = np.sqrt(np.maximum(0.0, img)).astype(np.float32)
    
    data['image'] = img
    return data

# custom transform to apply a random laplace filter to the image
def random_laplace_filter(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applique le filtre Laplacien (sharpening) à l'image 'image' avec une probabilité.
    Nous utilisons ici un sharpening par soustraction du Laplacien pour conserver l'image originale.
    """
    img = data['image']
    
    # Définir le facteur d'accentuation (hyperparamètre à ajuster)
    alpha = 0.5 
    
    C, H, W = img.shape
    laplace_result = np.zeros_like(img, dtype=np.float32)
    
    # Appliquer le filtre Laplacien canal par canal (R, G, B)
    for c in range(C):
        # ndi.laplace calcule le laplacien
        laplace_result[c, :, :] = ndi.laplace(img[c, :, :])
        
    # Sharpening: I_sharp = I_original - alpha * laplacien
    # (Note: Le Laplacien pur est souvent utilisé comme fonction, pas le sharpening)
    # Si vous voulez le Laplacien pur, commentez la ligne ci-dessous et retournez laplace_result.
    
    img_sharpened = img - alpha * laplace_result
    
    data['image'] = img_sharpened.astype(np.float32)
    return data

# custom transform to crop the image to a square for BRSET images
class CropToSquareD(MapTransform):
    """
    Croppe l'image pour qu'elle soit un carré centré, de la plus grande taille possible qui tienne dans l'image d'origine.

    Args:
        keys: Les clés du dictionnaire à transformer (généralement ["image"])
    """
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self._crop_to_square(d[key])
        return d

    def _crop_to_square(self, img: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        # Handle both tensor and numpy input, with shape (C, H, W)
        is_tensor = isinstance(img, torch.Tensor)
        if is_tensor:
            h, w = img.shape[-2:]
        else:
            h, w = img.shape[-2:]

        side = min(h, w)
        h_start = (h - side) // 2
        w_start = (w - side) // 2

        if is_tensor:
            cropped = img[..., h_start:h_start+side, w_start:w_start+side]
        else:
            cropped = img[..., h_start:h_start+side, w_start:w_start+side]

        return cropped

# training transform sequence with data augmentation
monai_transform_sequence = Compose(
    [
        # load image
        LoadImaged(keys=[img_key]),

        # --- Pre-processing/Required Format Steps (Often placed first) ---
        EnsureChannelFirstD(keys=[img_key], channel_dim=-1),

        # normalize pixel values from 0-255 to 0-1
        ScaleIntensityRanged(keys=[img_key], a_min=0, a_max=255, b_min=0, b_max=1),

        # crop to square
        CropToSquareD(keys=[img_key]),
        
        # horizontal flip
        RandFlipd(keys=[img_key], prob=0.5, spatial_axis=1), # 0 for H-flip in (C, H, W)

        # vertical flip
        RandFlipd(keys=[img_key], prob=0.5, spatial_axis=0), # 0 for H-flip in (C, H, W)

        # random rotation
        RandRotated(keys=[img_key],range_x = 0.5, prob=1.0, padding_mode = 'border'),

        # random sqrt transform
        #RandLambdad(keys=[img_key], func=random_sqrt_transform, prob=0.6 ),

        # random laplace filter
        #RandLambdad(keys=[img_key], func=random_laplace_filter, prob=0.4 ),
        
        # rand 2d elastic deformation
        Rand2DElasticd(keys=[img_key], spacing=(20, 80), magnitude_range=(2, 5), prob=0.7),

        # rand 2d elastic deformation
        Rand2DElasticCenteredSquared(keys=[img_key], square_width_ratio=((2**0.5)/3.14159), spacing=(100, 200), magnitude_range=(10, 15), prob=0.7),
        
        # gaussian noise
        # RandGaussianNoised(keys=[img_key], prob=0.4, mean=0.0, std=0.15),

        # gaussian sharpen
        RandGaussianSharpend(keys=[img_key], prob=0.6, alpha=(0.1, 0.5)), # Alpha controls the strength
        
        # histogram nonlinear shift
        # RandHistogramShiftd(keys=[img_key], prob=0.5, num_control_points=40),
        
        # randzoom (Randomly zooms the image)
        RandZoomd(keys=[img_key], min_zoom=1.0, max_zoom=1.2, prob=0.5), 

        # Masquer la zone hors du cercle central (fond d'œil)
        MaskCircularRegiond(keys=[img_key]),

        # resize
        Resized(keys=[img_key], spatial_size=(384, 384), mode="bilinear"),
        
        # Ensure the final output is a PyTorch Tensor
        ToTensorD(keys=[img_key])
    ]
)

# for validation only apply necessary transforms 
val_transform_sequence = Compose(
    [
        LoadImaged(keys=[img_key]),
        
        # --- Pre-processing/Required Format Steps (Often placed first) ---
        EnsureChannelFirstD(keys=[img_key], channel_dim=-1),
        
        # Normalize pixel values from 0-255 to 0-1 (CRITICAL: must match training)
        ScaleIntensityRanged(keys=[img_key], a_min=0, a_max=255, b_min=0, b_max=1),

        # crop to square
        CropToSquareD(keys=[img_key]),
        
        # Masquer la zone hors du cercle central (fond d'œil)
        MaskCircularRegiond(keys=[img_key]),

        # resize
        Resized(keys=[img_key], spatial_size=(384, 384), mode="bilinear"),
        
        # Ensure the final output is a PyTorch Tensor
        ToTensorD(keys=[img_key])
    ]
)