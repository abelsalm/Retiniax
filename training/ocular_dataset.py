import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset 
from typing import Dict, List
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
            "detailed_labels": labels, # Tenseur de labels multi-label
        }

        # --- 4. Application des Transforms ---
        # LoadImaged(keys=['file']) va charger l'image à partir de image_path
        # Toutes les transforms sont appliquées, y compris les RandLambda et ToTensord
        transformed_data = self.transform(data)
        
        return transformed_data

class ClassBalancedBatchSampler:
    """
    Custom batch sampler for OcularDatasetBinary that fights class imbalance.

    For each batch of size ``batch_size``:
      - ``int(proportion * batch_size)`` samples are drawn with *uniform*
        probability across all detailed-label classes (each pathology column,
        plus a "healthy" class, is equally likely to be picked, then a random
        sample from that class is selected).
      - The remaining samples are drawn from the natural data distribution
        (plain random sampling over the whole dataset).

    Usage::

        sampler = ClassBalancedBatchSampler(dataset, batch_size=32, proportion=0.5)
        loader  = DataLoader(dataset, batch_sampler=sampler)
    """

    def __init__(
        self,
        dataset: OcularDatasetBinary,
        batch_size: int,
        proportion: float,
        drop_last: bool = False,
    ):
        assert 0.0 <= proportion <= 1.0, "proportion must be in [0, 1]"

        self.batch_size = batch_size
        self.proportion = proportion
        self.drop_last = drop_last
        self.num_samples = len(dataset)

        df = dataset.data_frame
        pathology_cols = df.columns[1:].tolist()

        self.class_to_indices: Dict[str, List[int]] = {}
        for col_name in pathology_cols:
            members = df.index[df[col_name] > 0].tolist()
            if members:
                self.class_to_indices[col_name] = members

        healthy_mask = df[pathology_cols].sum(axis=1) == 0
        healthy_indices = df.index[healthy_mask].tolist()
        if healthy_indices:
            self.class_to_indices["__healthy__"] = healthy_indices

        self.class_keys = list(self.class_to_indices.keys())
        self.num_classes = len(self.class_keys)

    def __iter__(self):
        n_uniform = int(self.proportion * self.batch_size)
        n_natural = self.batch_size - n_uniform

        all_indices = list(range(self.num_samples))
        random.shuffle(all_indices)
        ptr = 0

        for _ in range(len(self)):
            batch: List[int] = []

            for _ in range(n_uniform):
                cls_key = random.choice(self.class_keys)
                batch.append(random.choice(self.class_to_indices[cls_key]))

            for _ in range(n_natural):
                if ptr >= self.num_samples:
                    random.shuffle(all_indices)
                    ptr = 0
                batch.append(all_indices[ptr])
                ptr += 1

            random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size


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


# Other datasets to regroup pathologies in different groups 
# AUTRES/ DIVERS,CICATRICE ,DIABETE,DMLA,DRUSEN - AEP - dépots - matériel ,GLAUCOME,INFLAMMATION UVEITE ,MYOPIE,OEDEME PAPILLAIRE,PATHOLOGIE VASCULAIRE RETINIENNE,RETINE,TROUBLES DES MILIEUX,TUMEUR
# indexes for the groups of pathologies
central = [5, 6]
vascular = [4, 11]
disc = [7, 10, 9]
cataract = [13]
others = [2, 3, 8, 12, 14]

class OcularDatasetSpecific(Dataset):
    """
    Dataset personnalisé pour les rétinographies, gérant les labels multi-labels
    à partir d'un fichier CSV.
    """
    # Common image extensions to try when the CSV filename has no extension
    _IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    def __init__(self, csv_file: str, data_dir: str, classes: list, transform: Compose):
        """
        :param csv_file: Chemin vers le fichier CSV contenant les noms de fichiers et les labels.
        :param data_dir: Dossier racine où se trouvent toutes les images.
        :param transform: La composition des transformations MONAI (Compose).
        """
        self.data_frame = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform
        self.classes = classes
        
        # Identifier les colonnes de pathologies (Multi-labels)
        # On suppose que toutes les colonnes après 'file' sont des pathologies
        self.pathology_keys = self.data_frame.columns[classes].tolist()
        self.num_classes = len(classes)

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
        labels = self.data_frame.iloc[idx, self.classes].values.astype(np.float32)
        
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

