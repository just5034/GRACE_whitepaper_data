import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Load the ATLAS Higgs data
data_path = '/projects/bgde/grace/data/atlas-higgs-challenge-2014-v2.csv'
df = pd.read_csv(data_path)
print(f'Loaded data shape: {df.shape}')

# Separate features by physics type
pri_features = [col for col in df.columns if col.startswith('PRI_')]
der_features = [col for col in df.columns if col.startswith('DER_')]
other_features = ['EventId', 'Label', 'Weight']

print(f'Primary features (PRI_*): {len(pri_features)}')
print(f'Derived features (DER_*): {len(der_features)}')
print(f'Primary: {pri_features}')
print(f'Derived: {der_features}')

# Analyze missing values (-999.0)
missing_indicator = -999.0
missing_analysis = {}
for col in pri_features + der_features:
    missing_count = (df[col] == missing_indicator).sum()
    missing_pct = missing_count / len(df) * 100
    missing_analysis[col] = {'count': missing_count, 'percentage': missing_pct}
    print(f'{col}: {missing_count} missing ({missing_pct:.1f}%)')

# Analyze event weights
weight_stats = {
    'min': df['Weight'].min(),
    'max': df['Weight'].max(),
    'mean': df['Weight'].mean(),
    'std': df['Weight'].std(),
    'median': df['Weight'].median()
}
print(f'Weight statistics: {weight_stats}')

# Stratify by jet multiplicity (key physics insight)
jet_categories = df['PRI_jet_num'].value_counts().sort_index()
print(f'Jet multiplicity distribution: {dict(jet_categories)}')

# Create stratified preprocessing approach
processed_data = {}
for jet_num in sorted(df['PRI_jet_num'].unique()):
    subset = df[df['PRI_jet_num'] == jet_num].copy()
    print(f'\nProcessing jet category {jet_num}: {len(subset)} events')
    
    # Handle missing values by category-specific strategy
    for col in pri_features + der_features:
        missing_mask = subset[col] == missing_indicator
        if missing_mask.any():
            if col == 'DER_mass_MMC':  # Most important feature
                # Keep as missing indicator for model to handle
                subset.loc[missing_mask, col] = np.nan
            else:
                # Category-specific median imputation
                valid_values = subset[subset[col] != missing_indicator][col]
                if len(valid_values) > 0:
                    median_val = valid_values.median()
                    subset.loc[missing_mask, col] = median_val
                else:
                    subset.loc[missing_mask, col] = 0
    
    processed_data[f'jet_{jet_num}'] = subset

# Combine processed categories
processed_df = pd.concat(processed_data.values(), ignore_index=True)
print(f'\nProcessed data shape: {processed_df.shape}')

# Create feature importance ranking based on physics knowledge
feature_importance = {
    'high_importance': ['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h'],
    'medium_importance': ['DER_deltar_tau_lep', 'DER_pt_tot', 'PRI_met', 'PRI_met_phi'],
    'jet_dependent': ['DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet']
}

# Prepare ML-ready dataset
feature_cols = pri_features + der_features
X = processed_df[feature_cols]
y = processed_df['Label']
weights = processed_df['Weight']
event_ids = processed_df['EventId']

# Create train/validation split preserving jet categories
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
    X, y, weights, test_size=0.2, random_state=42, stratify=processed_df['PRI_jet_num']
)

print(f'Training set: {X_train.shape[0]} events')
print(f'Validation set: {X_val.shape[0]} events')
print(f'Signal/background ratio - Train: {y_train.value_counts()["s"]}/{y_train.value_counts()["b"]}')
print(f'Signal/background ratio - Val: {y_val.value_counts()["s"]}/{y_val.value_counts()["b"]}')

# Save preprocessed data
preprocessed_data = {
    'X_train': X_train,
    'X_val': X_val,
    'y_train': y_train,
    'y_val': y_val,
    'w_train': w_train,
    'w_val': w_val,
    'feature_names': feature_cols,
    'pri_features': pri_features,
    'der_features': der_features,
    'jet_categories': dict(jet_categories),
    'missing_analysis': missing_analysis,
    'weight_stats': weight_stats,
    'feature_importance': feature_importance
}

# Save to files for downstream use
X_train.to_parquet('X_train.parquet')
X_val.to_parquet('X_val.parquet')
y_train.to_frame('Label').to_parquet('y_train.parquet')
y_val.to_frame('Label').to_parquet('y_val.parquet')
w_train.to_frame('Weight').to_parquet('w_train.parquet')
w_val.to_frame('Weight').to_parquet('w_val.parquet')

with open('preprocessing_metadata.json', 'w') as f:
    json.dump({
        'feature_names': feature_cols,
        'pri_features': pri_features,
        'der_features': der_features,
        'jet_categories': dict(jet_categories),
        'weight_stats': weight_stats,
        'feature_importance': feature_importance,
        'train_size': len(X_train),
        'val_size': len(X_val)
    }, f, indent=2)

# Create visualization of preprocessing results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Missing value pattern
missing_pcts = [missing_analysis[col]['percentage'] for col in feature_cols]
axes[0,0].bar(range(len(feature_cols)), missing_pcts)
axes[0,0].set_title('Missing Value Percentages by Feature')
axes[0,0].set_xlabel('Feature Index')
axes[0,0].set_ylabel('Missing %')

# Weight distribution (log scale)
axes[0,1].hist(weights, bins=50, alpha=0.7, log=True)
axes[0,1].set_title('Event Weight Distribution (Log Scale)')
axes[0,1].set_xlabel('Weight')
axes[0,1].set_ylabel('Count')

# Jet multiplicity distribution
axes[1,0].bar(jet_categories.index, jet_categories.values)
axes[1,0].set_title('Jet Multiplicity Distribution')
axes[1,0].set_xlabel('Number of Jets')
axes[1,0].set_ylabel('Event Count')

# Signal vs background by jet category
for i, jet_num in enumerate(sorted(df['PRI_jet_num'].unique())):
    subset = processed_df[processed_df['PRI_jet_num'] == jet_num]
    signal_count = (subset['Label'] == 's').sum()
    bg_count = (subset['Label'] == 'b').sum()
    axes[1,1].bar([f'{jet_num}_s', f'{jet_num}_b'], [signal_count, bg_count], alpha=0.7)
axes[1,1].set_title('Signal/Background by Jet Category')
axes[1,1].set_xlabel('Jet Category_Label')
axes[1,1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('preprocessing_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('preprocessing_analysis.pdf', bbox_inches='tight')

print('RESULT:train_events=' + str(len(X_train)))
print('RESULT:val_events=' + str(len(X_val)))
print('RESULT:num_features=' + str(len(feature_cols)))
print('RESULT:missing_features=' + str(sum(1 for col in missing_analysis if missing_analysis[col]['percentage'] > 0)))
print('RESULT:max_missing_pct=' + str(max(missing_analysis[col]['percentage'] for col in missing_analysis)))
print('RESULT:weight_range=' + str(weight_stats['max'] - weight_stats['min']))
print('RESULT:preprocessing_plot=preprocessing_analysis.png')
print('RESULT:data_ready=true')