import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Load the raw data
data_path = '/projects/bgde/grace/data/atlas-higgs-challenge-2014-v2.csv'
df = pd.read_csv(data_path)

print(f'Loaded dataset with shape: {df.shape}')
print(f'Columns: {list(df.columns)}')

# Handle missing values (-999.0)
missing_indicator = -999.0
missing_mask = (df == missing_indicator)
missing_counts = missing_mask.sum()
print('\nMissing value counts by column:')
for col, count in missing_counts.items():
    if count > 0:
        pct = count / len(df) * 100
        print(f'  {col}: {count} ({pct:.1f}%)')

# Separate features by physics type
pri_features = [col for col in df.columns if col.startswith('PRI_')]
der_features = [col for col in df.columns if col.startswith('DER_')]
other_features = [col for col in df.columns if not col.startswith(('PRI_', 'DER_')) and col not in ['EventId', 'Label', 'Weight']]

print(f'\nFeature categorization:')
print(f'  Primary (PRI_*): {len(pri_features)} features')
print(f'  Derived (DER_*): {len(der_features)} features')
print(f'  Other: {len(other_features)} features')

# Analyze event weights
if 'Weight' in df.columns:
    weights = df['Weight']
    print(f'\nEvent weight analysis:')
    print(f'  Min weight: {weights.min():.6f}')
    print(f'  Max weight: {weights.max():.6f}')
    print(f'  Mean weight: {weights.mean():.6f}')
    print(f'  Std weight: {weights.std():.6f}')
    print(f'  Median weight: {weights.median():.6f}')
    
    # Plot weight distribution
    plt.figure(figsize=(10, 6))
    plt.hist(weights, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Event Weight')
    plt.ylabel('Frequency')
    plt.title('Distribution of Event Weights')
    plt.yscale('log')
    plt.savefig('event_weights_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

# Create cleaned dataset
df_clean = df.copy()

# Strategy for missing values: Create indicator variables for key features with high missing rates
high_missing_features = [col for col, count in missing_counts.items() if count > len(df) * 0.1]
print(f'\nFeatures with >10% missing values: {high_missing_features}')

# For DER_mass_MMC (most important feature), create indicator and fill with median of available values
if 'DER_mass_MMC' in df_clean.columns:
    df_clean['DER_mass_MMC_missing'] = (df_clean['DER_mass_MMC'] == missing_indicator).astype(int)
    available_mmc = df_clean[df_clean['DER_mass_MMC'] != missing_indicator]['DER_mass_MMC']
    median_mmc = available_mmc.median()
    df_clean.loc[df_clean['DER_mass_MMC'] == missing_indicator, 'DER_mass_MMC'] = median_mmc
    print(f'  DER_mass_MMC: filled {missing_counts["DER_mass_MMC"]} missing values with median {median_mmc:.2f}')

# For other features with missing values, use physics-informed imputation
for col in missing_counts.index:
    if missing_counts[col] > 0 and col != 'DER_mass_MMC':
        if col.startswith('DER_'):
            # For derived features, use median imputation
            available_values = df_clean[df_clean[col] != missing_indicator][col]
            if len(available_values) > 0:
                median_val = available_values.median()
                df_clean.loc[df_clean[col] == missing_indicator, col] = median_val
                print(f'  {col}: filled {missing_counts[col]} missing values with median {median_val:.3f}')
        elif col.startswith('PRI_'):
            # For primary features, use zero or physics-motivated defaults
            df_clean.loc[df_clean[col] == missing_indicator, col] = 0.0
            print(f'  {col}: filled {missing_counts[col]} missing values with 0.0')

# Verify no missing values remain
remaining_missing = (df_clean == missing_indicator).sum().sum()
print(f'\nRemaining missing values after cleaning: {remaining_missing}')

# Create feature groups for ML
feature_groups = {
    'primary_features': pri_features,
    'derived_features': der_features,
    'kinematic_features': [col for col in df_clean.columns if any(x in col.lower() for x in ['pt', 'eta', 'phi', 'mass'])],
    'angular_features': [col for col in df_clean.columns if any(x in col.lower() for x in ['delta', 'deltar', 'prodeta'])],
    'all_physics_features': pri_features + der_features
}

# Prepare final dataset for ML
features_for_ml = [col for col in df_clean.columns if col not in ['EventId']]
if 'Label' in df_clean.columns:
    target_col = 'Label'
else:
    target_col = None

print(f'\nML-ready dataset:')
print(f'  Total features: {len(features_for_ml)}')
print(f'  Target column: {target_col}')
print(f'  Dataset shape: {df_clean.shape}')

# Save processed data and metadata
df_clean.to_csv('preprocessed_physics_data.csv', index=False)

# Save feature metadata
metadata = {
    'feature_groups': feature_groups,
    'missing_value_strategy': {
        'DER_mass_MMC': 'median_imputation_with_indicator',
        'other_derived': 'median_imputation',
        'primary': 'zero_fill'
    },
    'original_missing_counts': missing_counts.to_dict(),
    'ml_features': features_for_ml,
    'target_column': target_col,
    'dataset_shape': list(df_clean.shape),
    'weight_statistics': {
        'min': float(weights.min()) if 'Weight' in df.columns else None,
        'max': float(weights.max()) if 'Weight' in df.columns else None,
        'mean': float(weights.mean()) if 'Weight' in df.columns else None,
        'std': float(weights.std()) if 'Weight' in df.columns else None
    }
}

with open('preprocessing_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# Summary statistics
print('\nPreprocessing completed successfully!')
print(f'RESULT:dataset_shape={df_clean.shape[0]}x{df_clean.shape[1]}')
print(f'RESULT:primary_features_count={len(pri_features)}')
print(f'RESULT:derived_features_count={len(der_features)}')
print(f'RESULT:missing_values_handled={missing_counts.sum()}')
print('RESULT:output_file=preprocessed_physics_data.csv')
print('RESULT:metadata_file=preprocessing_metadata.json')
if 'Weight' in df.columns:
    print('RESULT:weight_plot=event_weights_distribution.png')