# file to run the inference on the val dataset for the trained model
# save the predictions in all classes inside a 
import timm
from model import DeepClassifierMultiHead
import torch
from ocular_dataset import OcularDataset
from torch.utils.data import DataLoader
from transforms import val_transform_sequence
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


# function to run the inference on the val dataset for the trained model
def main():
    # backbone
    backbone = timm.create_model(
        'convnext_tiny.fb_in22k_ft_in1k_384',
        in_chans=3,
        pretrained=False,
        num_classes=0
    )
    # wrapper
    model = DeepClassifierMultiHead(encoder=backbone, n_pathologies=14)

    # weights
    checkpoint_path = '/workspace/Retiniax/trained models/config1.pth'

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model (robust to CPU-only or GPU)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # load val dataset
    val_dataset = OcularDataset(
        csv_file="/workspace/Retiniax/data/validation_dataset.csv",
        data_dir="/workspace/data_15",
        transform=val_transform_sequence,
    )

    # On macOS / Windows, use num_workers=0 to avoid multiprocessing spawn issues
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Get the list of column/class names from your dataset (excluding the image path column 'file')
    class_names = [
        'NCS', 'AUTRES/ DIVERS', 'CATARACTE', 'CICATRICE ', 'DIABETE', 'DMLA',
        'DRUSEN - AEP - dépots - matériel ', 'GLAUCOME', 'INFLAMMATION UVEITE ',
        'MYOPIE', 'OEDEME PAPILLAIRE', 'PATHOLOGIE VASCULAIRE RETINIENNE', 'RETINE',
        'TROUBLES DES MILIEUX', 'TUMEUR'
    ]

    # We'll accumulate predictions in a list of dicts for efficiency
    all_predictions = []

    # run inference with tqdm progress bar
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Inference", total=len(val_loader))):
            # if i == 0: for testing
            images, labels = batch['image'], batch['label']
            images = images.to(device)
            output_patho, output_bin = model(images)
            outputs = torch.cat((output_bin, output_patho), dim=1)
            # Apply sigmoid since it's multi-label, to get probabilities in [0, 1]
            probs = torch.sigmoid(outputs).cpu().numpy()
            image_paths = batch['image_path'] if 'image_path' in batch else batch['file']
            # Loop over batch and collect probabilities per image
            for path, prob in zip(image_paths, probs):
                row = {'image_path': path}
                for name, value in zip(class_names, prob):
                    row[name] = value
                all_predictions.append(row)
            '''else:
                break'''

    # Convert to DataFrame
    predictions_df = pd.DataFrame(all_predictions, columns=['image_path'] + class_names)

    # save the predictions to a csv file
    predictions_df.to_csv(
        '/workspace/Retiniax/trained models/config1_val.csv',
        index=False
    )


if __name__ == "__main__":
    main()
