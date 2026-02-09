import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset 
from typing import Dict
from monai.transforms import Compose

class OcularDatasetBinary(Dataset):
    """
    Dataset personnalisé pour les rétinographies, gérant les labels multi-labels
    à partir d'un fichier CSV.
    """
    # Common image extensions to try when the CSV filename has no extension
    _IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    def __init__(self, csv_file: str, data_dir: str, transform: Compose):
        """
        :param csv_file: Chemin vers le fichier CSV contenant les noms de fichiers et les labels.
        :param data_dir: Dossier racine où se trouvent toutes les images.
        :param transform: La composition des transformations MONAI (Compose).
        """
        self.data_frame = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform
        
        # Identifier les colonnes de pathologies (Multi-labels)
        # On suppose que toutes les colonnes après 'file' sont des pathologies
        self.pathology_keys = self.data_frame.columns[1:].tolist()
        self.num_classes = len(self.pathology_keys)

    def __len__(self) -> int:
        return len(self.data_frame)

    @staticmethod
    def _resolve_image_path(base_path: str, extensions=_IMG_EXTENSIONS) -> str:
        """Return *base_path* as-is if it exists, otherwise try appending
        common image extensions and return the first match."""
        if os.path.isfile(base_path):
            return base_path
        for ext in extensions:
            candidate = base_path + ext
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            f"Image not found: {base_path} (also tried extensions {extensions})"
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        
        # --- 1. Chemin de l'Image ---
        file_name = self.data_frame.iloc[idx, 0] # La première colonne est 'file'
        image_path = self._resolve_image_path(os.path.join(self.data_dir, file_name))
        
        # --- 2. Tenseur Label Multi-Label ---
        # On extrait la ligne de labels (de la colonne 1 jusqu'à la fin)
        # Le format doit être un tenseur binaire (0 ou 1) de taille (N_classes,)
        labels = self.data_frame.iloc[idx, 1:].values.astype(np.float32)
        if np.mean(labels[1:]) > 1e-3 :
            label = np.array([0,1])
        else :
            label = np.array([1,0])
        
        # --- 3. Création du Dictionnaire d'Entrée MONAI ---
        data = {
            "image": image_path, # Le path est l'entrée pour LoadImaged
            "label": label,     # Tenseur de labels multi-label
            "image_path": image_path, # Le path est l'entrée pour LoadImaged
        }

        # --- 4. Application des Transforms ---
        # LoadImaged(keys=['file']) va charger l'image à partir de image_path
        # Toutes les transforms sont appliquées, y compris les RandLambda et ToTensord
        transformed_data = self.transform(data)
        
        return transformed_data

class OcularDataset(Dataset):
    """
    Dataset personnalisé pour les rétinographies, gérant les labels multi-labels
    à partir d'un fichier CSV.
    """
    # Common image extensions to try when the CSV filename has no extension
    _IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    def __init__(self, csv_file: str, data_dir: str, transform: Compose):
        """
        :param csv_file: Chemin vers le fichier CSV contenant les noms de fichiers et les labels.
        :param data_dir: Dossier racine où se trouvent toutes les images.
        :param transform: La composition des transformations MONAI (Compose).
        """
        self.data_frame = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform
        
        # Identifier les colonnes de pathologies (Multi-labels)
        # On suppose que toutes les colonnes après 'file' sont des pathologies
        self.pathology_keys = self.data_frame.columns[1:].tolist()
        self.num_classes = len(self.pathology_keys)

    def __len__(self) -> int:
        return len(self.data_frame)

    @staticmethod
    def _resolve_image_path(base_path: str, extensions=_IMG_EXTENSIONS) -> str:
        """Return *base_path* as-is if it already exists (or already has a
        recognised extension).  Otherwise try appending each extension in
        *extensions* and return the first match.  Raises FileNotFoundError
        when nothing is found."""
        if os.path.isfile(base_path):
            return base_path
        for ext in extensions:
            candidate = base_path + ext
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            f"Image not found: {base_path} (also tried extensions {extensions})"
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        
        # --- 1. Chemin de l'Image ---
        file_name = self.data_frame.iloc[idx, 0] # La première colonne est 'file'
        image_path = self._resolve_image_path(os.path.join(self.data_dir, file_name))
        
        # --- 2. Tenseur Label Multi-Label ---
        # On extrait la ligne de labels (de la colonne 1 jusqu'à la fin)
        # Le format doit être un tenseur binaire (0 ou 1) de taille (N_classes,)
        labels = self.data_frame.iloc[idx, 1:].values.astype(np.float32)
        
        # --- 3. Création du Dictionnaire d'Entrée MONAI ---
        data = {
            "image": image_path, # Le path est l'entrée pour LoadImaged
            "label": labels,     # Tenseur de labels multi-label
            "image_path": image_path, # Le path est l'entrée pour LoadImaged
        }

        # --- 4. Application des Transforms ---
        # LoadImaged(keys=['file']) va charger l'image à partir de image_path
        # Toutes les transforms sont appliquées, y compris les RandLambda et ToTensord
        transformed_data = self.transform(data)
        
        return transformed_data