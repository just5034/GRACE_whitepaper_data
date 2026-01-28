import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load the ATLAS data
data_path = '/projects/bgde/grace/data/atlas-higgs-challenge-2014-v2.csv'
print(f'Loading data from {data_path}')
df = pd.read_csv(data_path)
print(f'Loaded dataset shape: {df.shape}')
print(f'Columns: {list(df.columns)}')

# Handle missing values (marked as -999.0)
missing_indicator = -999.0
print(f'\nMissing value analysis:')
missing_counts = (df == missing_indicator).sum()
for col, count in missing_counts.items():
    if count > 0:
        pct = count / len(df) * 100
        print(f'{col}: {count} missing ({pct:.1f}%)')

# Categorize features by physics type
kinematic_features = ['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h', 'DER_pt_tot']
angular_features = ['DER_deltaeta_jet_jet', 'DER_deltar_tau_lep']
jet_features = ['DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'PRI_jet_num', 'PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt', 'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt']
lepton_features = ['PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi']
met_features = ['PRI_met', 'PRI_met_phi', 'PRI_met_sumet']

feature_categories = {
    'kinematic': kinematic_features,
    'angular': angular_features, 
    'jet': jet_features,
    'lepton': lepton_features,
    'met': met_features
}

print(f'\nFeature categorization:')
for category, features in feature_categories.items():
    available_features = [f for f in features if f in df.columns]
    print(f'{category}: {len(available_features)} features')

# Handle missing values by jet multiplicity category
print(f'\nProcessing by jet multiplicity:')
jet_categories = df['PRI_jet_num'].unique()
print(f'Jet categories: {sorted(jet_categories)}')

# Create processed dataset
processed_df = df.copy()

# Strategy: For each jet category, handle missing values appropriately
for jet_cat in jet_categories:
    mask = processed_df['PRI_jet_num'] == jet_cat
    subset = processed_df[mask]
    print(f'\nJet category {int(jet_cat)}: {len(subset)} events')
    
    # For kinematic features, use median imputation within category
    for feature in kinematic_features:
        if feature in processed_df.columns:
            missing_mask = (processed_df[feature] == missing_indicator) & mask
            if missing_mask.sum() > 0:
                valid_values = processed_df.loc[mask & (processed_df[feature] != missing_indicator), feature]
                if len(valid_values) > 0:
                    median_val = valid_values.median()
                    processed_df.loc[missing_mask, feature] = median_val
                    print(f'  Imputed {missing_mask.sum()} values for {feature} with median {median_val:.3f}')

# Convert numpy types to Python types for JSON serialization
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# Apply type conversion to entire DataFrame
for col in processed_df.columns:
    if processed_df[col].dtype == 'int64':
        processed_df[col] = processed_df[col].apply(lambda x: int(x) if pd.notna(x) else x)
    elif processed_df[col].dtype == 'float64':
        processed_df[col] = processed_df[col].apply(lambda x: float(x) if pd.notna(x) else x)

print(f'\nType conversion completed')
print(f'Data types after conversion:')
for col in processed_df.columns[:10]:  # Show first 10
    print(f'{col}: {processed_df[col].dtype}')

# Preserve event weights
if 'Weight' in processed_df.columns:
    weight_stats = processed_df['Weight'].describe()
    print(f'\nEvent weight statistics:')
    print(f'Mean: {weight_stats["mean"]:.6f}, Std: {weight_stats["std"]:.6f}')
    print(f'Min: {weight_stats["min"]:.6f}, Max: {weight_stats["max"]:.6f}')
    
    # Convert weights to Python float
    processed_df['Weight'] = processed_df['Weight'].apply(lambda x: float(x) if pd.notna(x) else x)

# Create ML-ready features (exclude EventId and Label if present)
feature_cols = [col for col in processed_df.columns if col not in ['EventId', 'Label', 'KaggleSet', 'KaggleWeight']]
ml_features = processed_df[feature_cols].copy()

print(f'\nML-ready dataset:')
print(f'Shape: {ml_features.shape}')
print(f'Features: {len(feature_cols)}')

# Check for remaining missing values
remaining_missing = (ml_features == missing_indicator).sum().sum()
print(f'Remaining missing values: {remaining_missing}')

# Save processed data
processed_df.to_csv('processed_atlas_data.csv', index=False)
ml_features.to_csv('ml_ready_features.csv', index=False)

# Create metadata with type checking
metadata = {
    'original_shape': [int(df.shape[0]), int(df.shape[1])],
    'processed_shape': [int(processed_df.shape[0]), int(processed_df.shape[1])],
    'feature_categories': {k: v for k, v in feature_categories.items()},
    'jet_categories': [int(x) for x in sorted(jet_categories)],
    'missing_indicator': float(missing_indicator),
    'remaining_missing_values': int(remaining_missing),
    'ml_ready_features': int(len(feature_cols))
}

# Apply type checking before JSON serialization
def check_json_serializable(obj, path=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            check_json_serializable(v, f'{path}.{k}')
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            check_json_serializable(v, f'{path}[{i}]')
    elif isinstance(obj, (np.integer, np.int64)):
        raise TypeError(f'Found numpy int64 at {path}: {obj} (type: {type(obj)})')
    elif isinstance(obj, (np.floating, np.float64)):
        raise TypeError(f'Found numpy float64 at {path}: {obj} (type: {type(obj)})')

# Check metadata before saving
try:
    check_json_serializable(metadata)
    print('Type checking passed - all values are JSON serializable')
except TypeError as e:
    print(f'Type checking failed: {e}')
    # Fix any remaining numpy types
    metadata = json.loads(json.dumps(metadata, default=convert_numpy_types))

# Save metadata with JSON serialization fix
with open('preprocessing_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2, default=convert_numpy_types)

print(f'\nPreprocessing completed successfully!')
print(f'Files saved:')
print(f'- processed_atlas_data.csv: Full processed dataset')
print(f'- ml_ready_features.csv: ML-ready features only')
print(f'- preprocessing_metadata.json: Processing metadata')

# Return results
print(f'RESULT:processed_events={len(processed_df)}')
print(f'RESULT:ml_features={len(feature_cols)}')
print(f'RESULT:remaining_missing={remaining_missing}')
print(f'RESULT:jet_categories={len(jet_categories)}')
print('RESULT:preprocessing_status=success')