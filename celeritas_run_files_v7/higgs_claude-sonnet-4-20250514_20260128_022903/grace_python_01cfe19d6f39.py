import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load the ATLAS dataset
print('Loading ATLAS dataset...')
df = pd.read_csv('atlas_data.csv')

print(f'Dataset shape: {df.shape}')
print(f'Columns: {list(df.columns)}')

# Check for missing values (-999.0)
missing_mask = (df == -999.0)
missing_counts = missing_mask.sum()
print(f'\nMissing values (-999.0) per column:')
for col, count in missing_counts.items():
    if count > 0:
        print(f'  {col}: {count} ({count/len(df)*100:.1f}%)')

# Separate features, labels, and weights
if 'Label' in df.columns:
    labels = df['Label'].copy()
    df_features = df.drop('Label', axis=1)
else:
    labels = None
    df_features = df.copy()

# Preserve event weights if present
if 'Weight' in df_features.columns:
    weights = df_features['Weight'].copy()
    df_features = df_features.drop('Weight', axis=1)
elif 'EventWeight' in df_features.columns:
    weights = df_features['EventWeight'].copy()
    df_features = df_features.drop('EventWeight', axis=1)
else:
    weights = pd.Series(np.ones(len(df)), index=df.index)
    print('No weight column found, using uniform weights')

# Handle KaggleSet/KaggleWeight columns if present
kaggle_cols = [col for col in df_features.columns if 'Kaggle' in col]
for col in kaggle_cols:
    if col in df_features.columns:
        df_features = df_features.drop(col, axis=1)
        print(f'Dropped {col} column')

# Physics-aware missing value handling
df_clean = df_features.copy()

# Replace -999.0 with NaN for proper handling
df_clean[df_clean == -999.0] = np.nan

# Physics-motivated feature engineering
print('\nEngineering physics features...')

# Invariant masses and transverse quantities
if all(col in df_clean.columns for col in ['PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi']):
    df_clean['PRI_lep_px'] = df_clean['PRI_lep_pt'] * np.cos(df_clean['PRI_lep_phi'])
    df_clean['PRI_lep_py'] = df_clean['PRI_lep_pt'] * np.sin(df_clean['PRI_lep_phi'])
    print('Added lepton momentum components')

# Missing energy significance
if 'PRI_met' in df_clean.columns and 'PRI_met_sumet' in df_clean.columns:
    df_clean['MET_significance'] = df_clean['PRI_met'] / np.sqrt(df_clean['PRI_met_sumet'] + 1e-6)
    print('Added MET significance')

# Jet multiplicity and energy ratios
jet_cols = [col for col in df_clean.columns if 'jet' in col.lower()]
if len(jet_cols) > 0:
    print(f'Found {len(jet_cols)} jet-related features')

# Transverse mass calculations
if all(col in df_clean.columns for col in ['PRI_lep_pt', 'PRI_met', 'DER_deltar_tau_lep']):
    # Approximate transverse mass
    df_clean['MT_lep_met'] = np.sqrt(2 * df_clean['PRI_lep_pt'] * df_clean['PRI_met'] * 
                                    (1 - np.cos(df_clean['DER_deltar_tau_lep'])))
    print('Added transverse mass feature')

# Handle remaining missing values with physics-aware strategy
for col in df_clean.columns:
    if df_clean[col].isna().any():
        if 'tau' in col.lower():
            # Tau-related features: missing often means no tau candidate
            df_clean[col] = df_clean[col].fillna(0)
        elif 'jet' in col.lower():
            # Jet features: missing means no jet
            df_clean[col] = df_clean[col].fillna(0)
        elif 'met' in col.lower():
            # MET features: use median (conservative)
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        else:
            # Other features: use median
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

print(f'\nFinal dataset shape: {df_clean.shape}')
print(f'Remaining missing values: {df_clean.isna().sum().sum()}')

# Create final dataset with preserved weights
final_data = df_clean.copy()
final_data['EventWeight'] = weights
if labels is not None:
    final_data['Label'] = labels

# Save preprocessed data
final_data.to_csv('preprocessed_atlas_data.csv', index=False)
print('Saved preprocessed data to preprocessed_atlas_data.csv')

# Summary statistics
stats = {
    'original_shape': list(df.shape),
    'final_shape': list(final_data.shape),
    'features_engineered': len([col for col in final_data.columns if col not in df.columns]),
    'missing_values_handled': int(missing_counts.sum()),
    'weight_column_preserved': 'EventWeight' in final_data.columns,
    'label_column_preserved': 'Label' in final_data.columns if labels is not None else False
}

# Save preprocessing metadata
with open('preprocessing_metadata.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f'\nPreprocessing Summary:')
print(f'Original features: {df.shape[1]}')
print(f'Final features: {final_data.shape[1]}')
print(f'Features engineered: {stats["features_engineered"]}')
print(f'Missing values handled: {stats["missing_values_handled"]}')
print(f'Event weights preserved: {stats["weight_column_preserved"]}')

print('RESULT:preprocessing_complete=True')
print(f'RESULT:final_features={final_data.shape[1]}')
print(f'RESULT:samples_processed={final_data.shape[0]}')
print(f'RESULT:missing_values_handled={stats["missing_values_handled"]}')
print(f'RESULT:features_engineered={stats["features_engineered"]}')
print('RESULT:output_file=preprocessed_atlas_data.csv')