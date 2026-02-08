import pandas as pd
import numpy as np

# Function to count the occurrences of the labels in the dataset
def count_occurrences_labels_brset():
    """
    Count occurrences of specific pathology columns in labels_brset.csv,
    making sure all elements are converted to numbers (they may be strings).
    """
    # Load the dataset
    df = pd.read_csv('data_patho_matrix_full.csv')
    
    # Define the columns to analyze
    columns_to_count = ['file', 'NCS', 'AUTRES/ DIVERS', 'CICATRICE ', 'DIABETE', 'DMLA', 'DRUSEN - AEP - dépots - matériel ', 'GLAUCOME', 'INFLAMMATION UVEITE ', 'MYOPIE', 'OEDEME PAPILLAIRE', 'PATHOLOGIE VASCULAIRE RETINIENNE', 'RETINE', 'TROUBLES DES MILIEUX', 'TUMEUR']

    # Ensure columns exist for numeric conversion (skip 'file' which is not numeric)
    label_columns = [col for col in columns_to_count if col != 'file' and col in df.columns]

    # Convert label columns to numeric (in-place), coerce errors to NaN then fill with 0 and cast to int
    df[label_columns] = df[label_columns].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    
    print("=" * 80)
    print("OCCURRENCES COUNT - labels_brset.csv")
    print("=" * 80)
    print(f"\nTotal samples in dataset: {len(df)}\n")
    
    # Count occurrences for each column
    print(f"{'Column Name':<30} {'Count':>10} {'Percentage':>12}")
    print("-" * 80)
    
    for col in columns_to_count:
        if col in df.columns:
            if col == 'file':
                print(f"{col:<30} {'-':>10} {'-':>12}")
            else:
                count = df[col].sum()
                percentage = (count / len(df)) * 100
                print(f"{col:<30} {int(count):>10} {percentage:>11.2f}%")
        else:
            print(f"{col:<30} {'NOT FOUND':>10} {'N/A':>12}")
    
    print("=" * 80)

    # Additional statistics
    total_positives = df[label_columns].sum(axis=1).sum()
    avg_labels_per_sample = df[label_columns].sum(axis=1).mean()

    print(f"\nTotal positive labels across all samples: {int(total_positives)}")
    print(f"Average labels per sample: {avg_labels_per_sample:.2f}")
    print("=" * 80)

    # Return only the sum of label columns, skip 'file'
    return df[label_columns].sum()

# Function to calculate the inverse sqrt weights for the labels
def calculate_inverse_sqrt_weights():
    """
    Calculate 1/sqrt(percentage) for each pathology column,
    where percentage = sum(column) / number_of_rows.
    This is useful for calculating class weights for imbalanced datasets.
    """
    # Load the dataset
    df = pd.read_csv('data_patho_matrix_full.csv')
    
    # Define the label columns (excluding 'file')
    label_columns = ['NCS', 'AUTRES/ DIVERS', 'CICATRICE ', 'DIABETE', 'DMLA', 
                     'DRUSEN - AEP - dépots - matériel ', 'GLAUCOME', 'INFLAMMATION UVEITE ', 
                     'MYOPIE', 'OEDEME PAPILLAIRE', 'PATHOLOGIE VASCULAIRE RETINIENNE', 
                     'RETINE', 'TROUBLES DES MILIEUX', 'TUMEUR']
    
    # Keep only columns that exist in the dataframe
    label_columns = [col for col in label_columns if col in df.columns]
    
    # Convert label columns to numeric (in-place), coerce errors to NaN then fill with 0
    df[label_columns] = df[label_columns].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    
    # Get the number of rows
    num_rows = len(df)
    
    # Calculate weights for each column
    weights = {}
    
    print("=" * 80)
    print("INVERSE SQRT WEIGHTS - 1/sqrt(percentage)")
    print("=" * 80)
    print(f"\nTotal samples in dataset: {num_rows}\n")
    print(f"{'Column Name':<40} {'Percentage':>12} {'Weight (1/√%)':>15}")
    print("-" * 80)

    ncs_weight = 1
    turn = 0

    for col in label_columns:
        col_sum = df[col].sum()
        percentage = col_sum / num_rows
        
        # Calculate 1/sqrt(percentage)
        # Handle edge case where percentage is 0
        if percentage > 0:
            weight = 1 / np.sqrt(percentage)
        else:
            weight = 0.0
        
        if col == 'NCS':
            ncs_weight = weight
            print('turn is :', turn)
            print('ncs_weight', ncs_weight)

        weights[col] = weight/ncs_weight
        print(f"{col:<40} {percentage:>11.4f} {weight:>15.4f}")

        turn += 1
    
    print("=" * 80)
    

    ## Returns the weights normalized following the NCS weight since its the most common class
    ## following the 1/sqrt(percentage) formula
    return pd.Series(weights)


if __name__ == "__main__":
    output = count_occurrences_labels_brset()
    print('output', output)
    
    print("\n\n")
    
    weights = calculate_inverse_sqrt_weights()
    print('\nweights', weights)