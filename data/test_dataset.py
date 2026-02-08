from typing import Any
import os
import torch 
from torch.utils.data import DataLoader

# Import des librairies nécessaires pour la visualisation
import numpy as np
import matplotlib.pyplot as plt

# Les imports de vos fichiers locaux (supposés corrects)
from ocular_dataset import OcularDatasetBinary
from transforms import monai_transform_sequence


# --- PARAMÈTRES ---
CSV_FILE = "/workspace/data_15/data_patho_matrix.csv" 
DATA_ROOT_DIR = "/workspace/data_15/" 
BATCH_SIZE = 4 # Taille du batch pour la visualisation
NUM_WORKERS = 4 

# 1. Instanciation des Transforms (le Compose de MONAI)
# On suppose que 'monai_transform_sequence' est l'objet Compose importé
train_transforms = monai_transform_sequence 


if __name__ == '__main__':
    # Le garde __main__ résout les problèmes de multiprocessing (spawn)
    
    # 2. Instanciation du Dataset
    train_dataset = OcularDatasetBinary(
        csv_file=CSV_FILE,
        data_dir=DATA_ROOT_DIR,
        transform=train_transforms
    )

    # 3. Instanciation du DataLoader
    train_loader = DataLoader[Any](
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(), 
    )

    # Récupérer la liste des noms des pathologies pour l'affichage des labels
    pathology_keys = train_dataset.pathology_keys

    # --- Test du DataLoader et Affichage ---
    print(f"Nombre total d'échantillons d'entraînement : {len(train_dataset)}")
    print(f"Nombre total de classes de pathologie : {train_dataset.num_classes}")
    print("OKAY OKAY GAMBERGE")
    
    # Itérer sur un batch pour la vérification et la visualisation
    for batch in train_loader:
        
        print("\n--- Exemple de sortie d'un Batch ---")
        
        # 1. Extraction des tenseurs du batch
        images_tensor = batch['image']
        labels_tensor = batch['label']

        # 2. Conversion des tenseurs PyTorch en NumPy pour Matplotlib
        # .cpu() est nécessaire si vous travaillez sur GPU
        # .numpy() effectue la conversion
        # .permute(0, 2, 3, 1) change le format de (B, C, H, W) à (B, H, W, C)
        # qui est le format attendu par Matplotlib pour les images RGB
        images_np = images_tensor.cpu().numpy().transpose(0, 2, 3, 1)
        labels_np = labels_tensor.cpu().numpy()
        
        # 3. Affichage des informations de forme
        print(f"Taille du tenseur Image (Batch x C x H x W) : {images_tensor.shape}")
        print(f"Taille du tenseur Label (Batch x N_classes) : {labels_tensor.shape}")
        print(f"Type de données Image : {images_tensor.dtype}")
        print(f"Type de données Label : {labels_tensor.dtype}")
        
        # 4. Création de la figure Matplotlib
        num_images = images_np.shape[0]
        
        fig, axes = plt.subplots(
            nrows=num_images * 2, 
            ncols=1, 
            figsize=(12, num_images * 6), # Hauteur généreuse pour laisser de la place
            gridspec_kw={
                'height_ratios': [10, 1] * num_images, # L'image a une grande case réservée
                'hspace': 0.2 # Espace standard entre les cases de la grille
            }
        )
        
        plt.suptitle("Batch d'Images et Labels", fontsize=20, weight='bold', y=0.95)

        for i in range(num_images):
            # --- IMAGE (On ne touche pas à sa position) ---
            ax_img = axes[i * 2]
            img = images_np[i]
            
            ax_img.imshow(img)
            ax_img.set_title(f"Échantillon {i} - Image (Format HWC: {img.shape})")
            ax_img.axis('off')
            
            # --- GRAPHE DES LABELS ---
            ax_label = axes[i * 2 + 1]
            label = labels_np[i]

            '''ax_label.bar(
                pathology_keys, 
                label, 
                color=['green' if val == 1 else 'lightgray' for val in label], 
                width=0.6
            )'''
            
            # Nettoyage visuel du graphe
            #ax_label.set_xticks(range(len(pathology_keys)))
            #ax_label.set_xticklabels(pathology_keys, rotation=90, ha="center", fontsize=10)
            ax_label.set_ylim(0, 1.1)
            ax_label.set_yticks([]) 
            ax_label.spines['top'].set_visible(False)
            ax_label.spines['right'].set_visible(False)
            ax_label.spines['left'].set_visible(False)
            
            for j, val in enumerate(label):
                if val == 1:
                    ax_label.text(j, 0.5, "1", ha='center', va='center', fontweight='bold')

            # ============================================================
            # === TRANSLATION DU GRAPHE UNIQUEMENT ===
            # ============================================================
            # 1. Obtenir la position actuelle du graphe [left, bottom, width, height]
            pos = ax_label.get_position()
            
            # 2. Définir le décalage vers le haut
            # Comme l'image est carrée dans une case rectangulaire, il y a du vide sous l'image.
            # On remonte le graphe DANS ce vide.
            shift_up = 0.08  # <--- AJUSTEZ CE CHIFFRE (0.04 à 0.08)
            
            # 3. Appliquer la nouvelle position SEULEMENT au graphe
            # On garde le même x, width, height. On change juste y (bottom).
            ax_label.set_position([pos.x0, pos.y0 + shift_up, pos.width, pos.height])
            # ============================================================

        # IMPORTANT : Pas de tight_layout, sinon il annule le décalage manuel
        plt.subplots_adjust(top=0.92, bottom=0.05, left=0.05, right=0.95)
        
        save_path = "/workspace/Retiniax/data_loaded_binary.png"
        fig.savefig(save_path)
        print(f"Figure sauvegardée : {save_path}")
        plt.show()
        
        break