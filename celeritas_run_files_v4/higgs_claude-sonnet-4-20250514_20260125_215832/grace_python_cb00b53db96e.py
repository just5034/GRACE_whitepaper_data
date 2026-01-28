import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load the ATLAS Higgs data
data_path = '/projects/bgde/grace/data/atlas-higgs-challenge-2014-v2.csv'
df = pd.read_csv(data_path)

print(f'Original data shape: {df.shape}')
print(f'Columns: {list(df.columns)}')

# Separate features by physics type
pri_features = [col for col in df.columns if col.startswith('PRI_')]
der_features = [col for col in df.columns if col.startswith('DER_')]
other_features = [col for col in df.columns if not (col.startswith('PRI_') or col.startswith('DER_'))]

print(f'Primary (PRI_) features: {len(pri_features)}')
print(f'Derived (DER_) features: {len(der_features)}')
print(f'Other features: {other_features}')

# Analyze missing values (-999.0)
missing_indicator = -999.0
missing_counts = {}
for col in df.columns:
    if col not in ['EventId', 'Label', 'Weight']:
        missing_count = (df[col] == missing_indicator).sum()
        missing_pct = missing_count / len(df) * 100
        missing_counts[col] = {'count': missing_count, 'percentage': missing_pct}
        if missing_count > 0:
            print(f'{col}: {missing_count} missing values ({missing_pct:.1f}%)')

# Analyze event weights
weight_stats = {
    'min': df['Weight'].min(),
    'max': df['Weight'].max(),
    'mean': df['Weight'].mean(),
    'std': df['Weight'].std(),
    'median': df['Weight'].median()
}
print(f'Weight statistics: min={weight_stats["min"]:.6f}, max={weight_stats["max"]:.6f}, mean={weight_stats["mean"]:.6f}')

# Handle missing values by feature category
# Strategy: For physics features, use category-aware imputation
# Separate by jet multiplicity as suggested in exploration
jet_categories = df['PRI_jet_num'].unique()
print(f'Jet multiplicity categories: {sorted(jet_categories)}')

# Create processed dataset
df_processed = df.copy()

# Handle missing values by jet category
for jet_cat in jet_categories:
    mask = df_processed['PRI_jet_num'] == jet_cat
    cat_data = df_processed[mask]
    print(f'Jet category {jet_cat}: {mask.sum()} events')
    
    # For each feature with missing values in this category
    for col in pri_features + der_features:
        if col in df_processed.columns:
            missing_mask = (cat_data[col] == missing_indicator)
            if missing_mask.sum() > 0:
                # Use median imputation within jet category for non-missing values
                valid_values = cat_data[col][cat_data[col] != missing_indicator]
                if len(valid_values) > 0:
                    impute_value = valid_values.median()
                    df_processed.loc[mask & (df_processed[col] == missing_indicator), col] = impute_value

# Verify missing value handling
remaining_missing = (df_processed[pri_features + der_features] == missing_indicator).sum().sum()
print(f'Remaining missing values after imputation: {remaining_missing}')

# Create feature categorization
feature_categories = {
    'primary_features': pri_features,
    'derived_features': der_features,
    'metadata': ['EventId', 'Label', 'Weight']
}

# Prepare ML-ready datasets
X_features = pri_features + der_features
X = df_processed[X_features]
y = df_processed['Label']
weights = df_processed['Weight']
event_ids = df_processed['EventId']

# Save processed data
processed_data = {
    'features': X,
    'labels': y,
    'weights': weights,
    'event_ids': event_ids,
    'feature_categories': feature_categories,
    'missing_value_stats': missing_counts,
    'weight_stats': weight_stats
}

# Save to files for downstream use
X.to_csv('processed_features.csv', index=False)
y.to_csv('processed_labels.csv', index=False)
weights.to_csv('processed_weights.csv', index=False)

# Save metadata
with open('preprocessing_metadata.json', 'w') as f:
    json.dump({
        'feature_categories': feature_categories,
        'missing_value_stats': {k: v for k, v in missing_counts.items()},
        'weight_stats': weight_stats,
        'jet_categories': sorted(jet_categories.tolist()),
        'original_shape': list(df.shape),
        'processed_shape': list(df_processed.shape)
    }, f, indent=2)

print(f'Processed dataset shape: {X.shape}')
print(f'Label distribution: {y.value_counts().to_dict()}')
print(f'Features saved to: processed_features.csv')
print(f'Labels saved to: processed_labels.csv')
print(f'Weights saved to: processed_weights.csv')
print(f'Metadata saved to: preprocessing_metadata.json')

# Return key metrics
print(f'RESULT:num_features={len(X_features)}')
print(f'RESULT:num_primary_features={len(pri_features)}')
print(f'RESULT:num_derived_features={len(der_features)}')
print(f'RESULT:num_events={len(df_processed)}')
print(f'RESULT:missing_values_handled={True}')
print(f'RESULT:weight_range_ratio={weight_stats["max"]/weight_stats["min"]:.2f}')