import pandas as pd
import numpy as np

# Load the files
labels_df = pd.read_csv('labels_brset.csv')
data_patho_original = pd.read_csv('data_patho_matrix.csv')

# Create correspondence mapping - hardcoded based on provided mapping
correspondence_map = {
    'diabetic_retinopathy': 'DIABETE',
    'macular_edema': 'DIABETE',
    'scar': 'CICATRICE ',
    'nevus': 'TUMEUR',
    'amd': 'DMLA',
    'vascular_occlusion': 'PATHOLOGIE VASCULAIRE RETINIENNE',
    'hypertensive_retinopathy': 'AUTRES/ DIVERS',
    'drusens': 'DRUSEN - AEP - dépots - matériel ',
    'hemorrhage': 'AUTRES/ DIVERS',
    'myopic_fundus': 'MYOPIE',
    'retinal_detachment': 'RETINE',
    'increased_cup_disc': 'GLAUCOME',
    'other': 'AUTRES/ DIVERS'
}

print("Correspondence mapping:")
print(correspondence_map)

# Initialize the new data rows
new_rows = []

# Process each row in labels_brset
for _, row in labels_df.iterrows():
    new_row = {}
    
    # Fill the 'file' column with 'image_id' value
    new_row['file'] = row['image_id']
    
    # Initialize all pathology columns to 0
    new_row['NCS'] = 0
    new_row['AUTRES/ DIVERS'] = 0
    new_row['CATARACTE'] = 0  # Always 0 as per requirement
    new_row['CICATRICE '] = 0
    new_row['DIABETE'] = 0
    new_row['DMLA'] = 0
    new_row['DRUSEN - AEP - dépots - matériel '] = 0
    new_row['GLAUCOME'] = 0
    new_row['INFLAMMATION UVEITE '] = 0
    new_row['MYOPIE'] = 0
    new_row['OEDEME PAPILLAIRE'] = 0
    new_row['PATHOLOGIE VASCULAIRE RETINIENNE'] = 0
    new_row['RETINE'] = 0
    new_row['TROUBLES DES MILIEUX'] = 0
    new_row['TUMEUR'] = 0
    
    # Map columns based on correspondence
    for source_col, target_col in correspondence_map.items():
        if source_col in row.index and pd.notna(row[source_col]):
            value = int(row[source_col]) if row[source_col] != '' else 0
            if value > 0:  # If the source column has a positive value
                # Handle columns that map to the same target (logical OR)
                if target_col in new_row:
                    new_row[target_col] = max(new_row[target_col], 1)
    
    # Special rule for OEDEME PAPILLAIRE
    # Put 1 if optic_disc=2 AND increased_cup_disc=0
    if 'optic_disc' in row.index and 'increased_cup_disc' in row.index:
        try:
            optic_disc_val = int(row['optic_disc']) if pd.notna(row['optic_disc']) else -1
            increased_cup_disc_val = int(row['increased_cup_disc']) if pd.notna(row['increased_cup_disc']) else -1
            if optic_disc_val == 2 and increased_cup_disc_val == 0:
                new_row['OEDEME PAPILLAIRE'] = 1
        except (ValueError, TypeError):
            pass
    
    # Special rule for TROUBLES DES MILIEUX
    # Put 1 if focus=2
    if 'focus' in row.index:
        try:
            focus_val = int(row['focus']) if pd.notna(row['focus']) else -1
            if focus_val == 2:
                new_row['TROUBLES DES MILIEUX'] = 1
        except (ValueError, TypeError):
            pass
    
    # Check if all pathology columns are 0, if so set NCS to 1 (Normal)
    pathology_cols = [col for col in new_row.keys() if col != 'file' and col != 'NCS']
    if all(new_row[col] == 0 for col in pathology_cols):
        new_row['NCS'] = 1
    
    new_rows.append(new_row)

# Create new dataframe with all rows
result_df = pd.DataFrame(new_rows)

# Ensure column order matches the original data_patho_matrix.csv
column_order = ['file', 'NCS', 'AUTRES/ DIVERS', 'CATARACTE', 'CICATRICE ', 'DIABETE', 'DMLA', 
                'DRUSEN - AEP - dépots - matériel ', 'GLAUCOME', 'INFLAMMATION UVEITE ', 
                'MYOPIE', 'OEDEME PAPILLAIRE', 'PATHOLOGIE VASCULAIRE RETINIENNE', 
                'RETINE', 'TROUBLES DES MILIEUX', 'TUMEUR']

result_df = result_df[column_order]

# Transfer CATARACTE values to TROUBLES DES MILIEUX before deleting CATARACTE column
# If CATARACTE = 1, set TROUBLES DES MILIEUX = 1
result_df.loc[result_df['CATARACTE'] == 1, 'TROUBLES DES MILIEUX'] = 1

# Remove CATARACTE column from new rows
result_df = result_df.drop(columns=['CATARACTE'])

# Update column order without CATARACTE
column_order_final = ['file', 'NCS', 'AUTRES/ DIVERS', 'CICATRICE ', 'DIABETE', 'DMLA', 
                      'DRUSEN - AEP - dépots - matériel ', 'GLAUCOME', 'INFLAMMATION UVEITE ', 
                      'MYOPIE', 'OEDEME PAPILLAIRE', 'PATHOLOGIE VASCULAIRE RETINIENNE', 
                      'RETINE', 'TROUBLES DES MILIEUX', 'TUMEUR']

# Process original data_patho_matrix.csv to match new format
if 'CATARACTE' in data_patho_original.columns:
    # Transfer CATARACTE to TROUBLES DES MILIEUX for original data too
    data_patho_original.loc[data_patho_original['CATARACTE'] == 1, 'TROUBLES DES MILIEUX'] = 1
    # Remove CATARACTE column
    data_patho_original = data_patho_original.drop(columns=['CATARACTE'])

# Ensure both dataframes have the same columns in the same order
data_patho_original = data_patho_original[column_order_final]
result_df = result_df[column_order_final]

# Concatenate original data with new data
final_df = pd.concat([data_patho_original, result_df], ignore_index=True)

# Save to CSV
final_df.to_csv('data_patho_matrix_full.csv', index=False)

print(f"\nProcessing complete!")
print(f"Original rows: {len(data_patho_original)}")
print(f"New rows added: {len(result_df)}")
print(f"Total rows in final file: {len(final_df)}")
print(f"\nFirst 5 rows of the result:")
print(final_df.head())
print(f"\nSummary of pathologies:")
for col in column_order_final[1:]:  # Skip 'file' column
    count = final_df[col].sum()
    print(f"{col}: {count} cases")
