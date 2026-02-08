import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the datasets
train_df = pd.read_csv('train_dataset.csv')
val_df = pd.read_csv('val_dataset.csv')

# Get class columns (all columns except 'file')
class_columns = [col for col in train_df.columns if col != 'file']

# Calculate class distributions (sum of each class)
train_dist = train_df[class_columns].sum()
val_dist = val_df[class_columns].sum()

# Create a single figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# Plot training distribution
x_pos = np.arange(len(class_columns))
width = 0.35

bars1 = ax1.bar(x_pos - width/2, train_dist.values, width, label='Train', alpha=0.8)
bars2 = ax1.bar(x_pos + width/2, val_dist.values, width, label='Validation', alpha=0.8)

ax1.set_xlabel('Classes', fontsize=12)
ax1.set_ylabel('Number of Samples', fontsize=12)
ax1.set_title('Class Distribution: Train vs Validation', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(class_columns, rotation=45, ha='right', fontsize=9)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=7)

# Plot normalized percentages
train_percent = (train_dist / len(train_df)) * 100
val_percent = (val_dist / len(val_df)) * 100

bars3 = ax2.bar(x_pos - width/2, train_percent.values, width, label='Train %', alpha=0.8)
bars4 = ax2.bar(x_pos + width/2, val_percent.values, width, label='Validation %', alpha=0.8)

ax2.set_xlabel('Classes', fontsize=12)
ax2.set_ylabel('Percentage of Samples (%)', fontsize=12)
ax2.set_title('Class Distribution (Normalized): Train vs Validation', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(class_columns, rotation=45, ha='right', fontsize=9)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig('class_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# Print statistics
print("=" * 80)
print("DATASET STATISTICS")
print("=" * 80)
print(f"\nTotal samples - Train: {len(train_df)}, Validation: {len(val_df)}")
print(f"\nClass distributions (Train):")
for class_name, count in train_dist.items():
    percentage = (count / len(train_df)) * 100
    print(f"  {class_name}: {count} ({percentage:.2f}%)")

print(f"\nClass distributions (Validation):")
for class_name, count in val_dist.items():
    percentage = (count / len(val_df)) * 100
    print(f"  {class_name}: {count} ({percentage:.2f}%)")

# Count samples with NCS and at least one other label
print("\n" + "=" * 80)
print("MULTI-LABEL SAMPLES WITH NCS")
print("=" * 80)

# Get other class columns (excluding NCS)
other_classes = [col for col in class_columns if col != 'NCS']

# For train dataset
train_ncs_mask = train_df['NCS'] == 1
train_other_labels_sum = train_df[other_classes].sum(axis=1)
train_ncs_with_others = ((train_ncs_mask) & (train_other_labels_sum > 0)).sum()

# For val dataset
val_ncs_mask = val_df['NCS'] == 1
val_other_labels_sum = val_df[other_classes].sum(axis=1)
val_ncs_with_others = ((val_ncs_mask) & (val_other_labels_sum > 0)).sum()

print(f"\nTrain dataset:")
print(f"  Total samples with NCS: {train_ncs_mask.sum()}")
print(f"  Samples with NCS AND at least one other label: {train_ncs_with_others}")
if train_ncs_mask.sum() > 0:
    print(f"  Percentage of NCS samples that are multi-label: {(train_ncs_with_others / train_ncs_mask.sum()) * 100:.2f}%")

print(f"\nValidation dataset:")
print(f"  Total samples with NCS: {val_ncs_mask.sum()}")
print(f"  Samples with NCS AND at least one other label: {val_ncs_with_others}")
if val_ncs_mask.sum() > 0:
    print(f"  Percentage of NCS samples that are multi-label: {(val_ncs_with_others / val_ncs_mask.sum()) * 100:.2f}%")

print(f"\nTotal (Train + Validation):")
total_ncs = train_ncs_mask.sum() + val_ncs_mask.sum()
total_ncs_with_others = train_ncs_with_others + val_ncs_with_others
print(f"  Total samples with NCS: {total_ncs}")
print(f"  Total samples with NCS AND at least one other label: {total_ncs_with_others}")
if total_ncs > 0:
    print(f"  Percentage of NCS samples that are multi-label: {(total_ncs_with_others / total_ncs) * 100:.2f}%")

print("=" * 80)

