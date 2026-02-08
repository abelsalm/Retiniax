import torch 
import timm
import numpy 
import torch.nn as nn
from timm.models import ConvNeXt

# example of an encoder, num_classes = 0 to add a final layer 
convnext_small = timm.create_model('convnext_tiny.fb_in22k_ft_in1k_384',
                                   in_chans=3, pretrained=True, num_classes=0)

# define a class for the classification models with multiple possible classes
class DeepClassifier(nn.Module):

    def __init__(self, encoder, n_classes):
        super(DeepClassifier, self).__init__()
        # encoder
        self.encoder = encoder
        # flattening layer, applying pooling and flattening
        # note here that we could try different pooling methods 
        # --> combine avg and max 
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)
        self.flatten = nn.Flatten(1)


        # size of the features after encoding
        # encoder outputs n feature maps 
        # that are flattened into a vector of size n
        # doubled because we avg and max ppol
        self.feature_size = self.encoder.num_features *2
        
        # classifier output
        # we use a final simple linear layer to output the final prediction 
        self.classifier = nn.Linear(self.feature_size, n_classes)

    def forward(self, x):
        # Pass through encoder
        x = self.encoder.forward_features(x)  # shape: (batch_size , c, h', w')
        
        # Apply pooling and flatten
        x_avg = self.flatten(self.pool_avg(x))  # shape: (batch_size, feature_size)
        x_max = self.flatten(self.pool_max(x)) # shape: (batch_size, feature_size)

        x_out = torch.cat([x_avg, x_max], dim=1)

        # classification output
        output = self.classifier(x_out)  # shape: (batch_size, 15)
        
        return output

# define a class for the classification models with multiple possible classes
class DeepClassifierBinary(nn.Module):

    def __init__(self, encoder):
        super(DeepClassifierBinary, self).__init__()
        # encoder
        self.encoder = encoder
        # flattening layer, applying pooling and flattening
        # note here that we could try different pooling methods 
        # --> combine avg and max 
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)
        self.flatten = nn.Flatten(1)


        # size of the features after encoding
        # encoder outputs n feature maps 
        # that are flattened into a vector of size n
        # doubled because we avg and max ppol
        self.feature_size = self.encoder.num_features *2
        
        # classifier output
        # we use a final simple linear layer to output the final prediction 
        self.classifier = nn.Linear(self.feature_size, 2)

    def forward(self, x):
        # Pass through encoder
        x = self.encoder.forward_features(x)  # shape: (batch_size , c, h', w')
        
        # Apply pooling and flatten
        x_avg = self.flatten(self.pool_avg(x))  # shape: (batch_size, feature_size)
        x_max = self.flatten(self.pool_max(x)) # shape: (batch_size, feature_size)

        x_out = torch.cat([x_avg, x_max], dim=1)

        # classification output
        output = self.classifier(x_out)  # shape: (batch_size, 15)
        
        return output

class DeepClassifierMultiHead(nn.Module):
    # On suppose que n_pathologies = 14 (les maladies)
    def __init__(self, encoder, n_pathologies=14, dropout_rate=0.2):
        super(DeepClassifierMultiHead, self).__init__()
        
        # 1. ENCODER
        self.encoder = encoder
        
        # 2. POOLING & FLATTEN
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)
        self.flatten = nn.Flatten(1)

        # Taille des features (x2 car on concatène Avg et Max)
        self.feature_size = self.encoder.num_features * 2
        
        # 3. REGULARISATION BLOCK (Avant les têtes)
        # BatchNorm aide à standardiser les valeurs venant du pooling mixte
        self.bn = nn.BatchNorm1d(self.feature_size)
        # Dropout empêche l'overfitting en "bruitant" les activations
        self.dropout = nn.Dropout(p=dropout_rate)

        # 4. LES DEUX TÊTES (Multi-Task)
        
        # Tête A : Prédit les 14 pathologies spécifiques
        # Sortie : 14 valeurs (une par maladie)
        self.head_pathology = nn.Linear(self.feature_size, n_pathologies)
        
        # Tête B : Prédit binaire "Il y a quelque chose" vs "Sain"
        # Sortie : 1 valeur (0 = Sain, 1 = Pathologique ou inversement selon votre label)
        self.head_binary = nn.Linear(self.feature_size, 1)

    def forward(self, x):
        # --- Feature Extraction ---
        features = self.encoder.forward_features(x)  # (batch, c, h, w)
        
        # --- Pooling Mixte ---
        x_avg = self.flatten(self.pool_avg(features))
        x_max = self.flatten(self.pool_max(features))
        x_concat = torch.cat([x_avg, x_max], dim=1) # (batch, feature_size)

        # --- Regularisation ---
        # On normalise et on applique le dropout sur les features partagées
        x_reg = self.bn(x_concat)
        x_reg = self.dropout(x_reg)

        # --- Prédictions ---
        # Les deux têtes lisent le même vecteur nettoyé
        out_patho = self.head_pathology(x_reg)  # Logits pour les 14 maladies
        out_binary = self.head_binary(x_reg)    # Logit pour Sain/Pas Sain
        
        return out_patho, out_binary