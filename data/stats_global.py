import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def stats_global(file_path, save_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Get class columns (all columns except 'file')
    class_columns = [col for col in df.columns if col != 'file']

    # Calculate class distributions (sum of each class)
    class_dist = df[class_columns].sum()

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # Plot absolute counts
    x_pos = np.arange(len(class_columns))
    bars1 = ax1.bar(x_pos, class_dist.values, alpha=0.8, color='steelblue')

    ax1.set_xlabel('Classes', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_title('Global Class Distribution (Absolute Counts)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(class_columns, rotation=45, ha='right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=8)

    # Plot normalized percentages
    class_percent = (class_dist / len(df)) * 100
    bars2 = ax2.bar(x_pos, class_percent.values, alpha=0.8, color='coral')

    ax2.set_xlabel('Classes', fontsize=12)
    ax2.set_ylabel('Percentage of Samples (%)', fontsize=12)
    ax2.set_title('Global Class Distribution (Percentages)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(class_columns, rotation=45, ha='right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Print statistics
    print("=" * 80)
    print("GLOBAL DATASET STATISTICS - data_patho_matrix_full.csv")
    print("=" * 80)
    print(f"\nTotal samples: {len(df)}")
    print(f"\nClass distributions:")
    for class_name, count in class_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  {class_name}: {count} ({percentage:.2f}%)")

    # Count samples with NCS and at least one other label
    print("\n" + "=" * 80)
    print("MULTI-LABEL SAMPLES WITH NCS")
    print("=" * 80)

    # Get other class columns (excluding NCS)
    other_classes = [col for col in class_columns if col != 'NCS']

    ncs_mask = df['NCS'] == 1
    other_labels_sum = df[other_classes].sum(axis=1)
    ncs_with_others = ((ncs_mask) & (other_labels_sum > 0)).sum()

    print(f"\nTotal samples with NCS: {ncs_mask.sum()}")
    print(f"Samples with NCS AND at least one other label: {ncs_with_others}")
    if ncs_mask.sum() > 0:
        print(f"Percentage of NCS samples that are multi-label: {(ncs_with_others / ncs_mask.sum()) * 100:.2f}%")

    # Calculate average number of labels per sample
    total_labels_per_sample = df[class_columns].sum(axis=1)
    print(f"\n" + "=" * 80)
    print("MULTI-LABEL STATISTICS")
    print("=" * 80)
    print(f"\nAverage number of labels per sample: {total_labels_per_sample.mean():.2f}")
    print(f"Min labels per sample: {total_labels_per_sample.min()}")
    print(f"Max labels per sample: {total_labels_per_sample.max()}")
    print(f"\nDistribution of number of labels per sample:")
    label_counts = total_labels_per_sample.value_counts().sort_index()
    for num_labels, count in label_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {int(num_labels)} label(s): {count} samples ({percentage:.2f}%)")

    print("=" * 80)

if __name__ == "__main__":
    file_path = '/Users/abelsalmona/Documents/Retinax/Training Repo/data/data_patho_matrix_full.csv'
    save_path = '/Users/abelsalmona/Documents/Retinax/Training Repo/data/class_distributions_full.png'
    stats_global(file_path, save_path)
