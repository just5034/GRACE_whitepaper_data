import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load preprocessed data
df = pd.read_parquet('atlas_higgs_processed.parquet')
print(f'Loaded {len(df)} events with {df.shape[1]} features')

# Create copy for feature engineering
df_features = df.copy()

# Physics-motivated feature engineering for H->tau tau
print('Engineering physics-motivated features...')

# 1. Kinematic ratios and normalized variables
df_features['pt_ratio_lep_met'] = np.where(df['PRI_met'] > 0, df['PRI_lep_pt'] / df['PRI_met'], 0)
df_features['pt_ratio_tau_met'] = np.where(df['PRI_met'] > 0, df['PRI_tau_pt'] / df['PRI_met'], 0)
df_features['pt_balance'] = (df['PRI_lep_pt'] + df['PRI_tau_pt']) / (df['PRI_met'] + 1e-6)

# 2. Enhanced angular features
df_features['cos_deltar_tau_lep'] = np.cos(df['DER_deltar_tau_lep'])
df_features['sin_deltar_tau_lep'] = np.sin(df['DER_deltar_tau_lep'])
df_features['deltar_tau_lep_squared'] = df['DER_deltar_tau_lep'] ** 2

# 3. Mass-based discriminants (handle -999 missing values)
mask_mmc = df['DER_mass_MMC'] != -999
df_features['mass_mmc_valid'] = mask_mmc.astype(int)
df_features['mass_mmc_higgs_diff'] = np.where(mask_mmc, np.abs(df['DER_mass_MMC'] - 125.0), -999)
df_features['mass_vis_over_mmc'] = np.where(mask_mmc & (df['DER_mass_MMC'] > 0), 
                                          df['DER_mass_vis'] / df['DER_mass_MMC'], -999)

# 4. Jet-specific features (stratified by jet multiplicity)
for n_jets in [0, 1, 2, 3]:
    mask = df['PRI_jet_num'] == n_jets
    df_features[f'is_{n_jets}_jet'] = mask.astype(int)

# Handle jet features for events with jets
mask_jets = df['PRI_jet_num'] > 0
df_features['jet_pt_sum'] = np.where(mask_jets, 
                                   df['PRI_jet_leading_pt'].fillna(0) + df['PRI_jet_subleading_pt'].fillna(0), 0)
df_features['jet_pt_ratio'] = np.where((df['PRI_jet_num'] >= 2) & (df['PRI_jet_subleading_pt'] > 0),
                                     df['PRI_jet_leading_pt'] / df['PRI_jet_subleading_pt'], -999)

# 5. Tau decay signature features
df_features['tau_lep_pt_sum'] = df['PRI_tau_pt'] + df['PRI_lep_pt']
df_features['tau_lep_pt_diff'] = np.abs(df['PRI_tau_pt'] - df['PRI_lep_pt'])
df_features['tau_lep_asymmetry'] = (df['PRI_tau_pt'] - df['PRI_lep_pt']) / (df['PRI_tau_pt'] + df['PRI_lep_pt'] + 1e-6)

# 6. Missing energy significance
df_features['met_significance'] = df['PRI_met'] / np.sqrt(df['PRI_met'] + 50.0)  # Approximate MET significance
df_features['met_over_sqrt_ht'] = df['PRI_met'] / np.sqrt(df['tau_lep_pt_sum'] + df['jet_pt_sum'] + 1e-6)

# 7. Centrality and sphericity-like variables
df_features['total_pt'] = df['tau_lep_pt_sum'] + df['jet_pt_sum'] + df['PRI_met']
df_features['visible_pt_fraction'] = (df['tau_lep_pt_sum'] + df['jet_pt_sum']) / (df['total_pt'] + 1e-6)

# 8. Higgs transverse momentum features
mask_pt_h = df['DER_pt_h'] != -999
df_features['pt_h_valid'] = mask_pt_h.astype(int)
df_features['pt_h_over_mass_vis'] = np.where(mask_pt_h & (df['DER_mass_vis'] > 0),
                                           df['DER_pt_h'] / df['DER_mass_vis'], -999)

# 9. Log-transformed features for better ML performance
for col in ['PRI_tau_pt', 'PRI_lep_pt', 'PRI_met', 'DER_mass_vis']:
    df_features[f'log_{col}'] = np.log(df[col] + 1.0)

# 10. Interaction terms for important physics correlations
df_features['mass_vis_times_pt_ratio'] = df['DER_mass_vis'] * df['pt_ratio_lep_met']
df_features['deltar_times_pt_balance'] = df['DER_deltar_tau_lep'] * df['pt_balance']

# Count new features created
original_features = df.shape[1]
new_features = df_features.shape[1] - original_features
print(f'Created {new_features} new physics-motivated features')
print(f'Total features: {df_features.shape[1]}')

# Feature importance analysis - correlation with signal
signal_mask = df_features['Label'] == 's'
background_mask = df_features['Label'] == 'b'

# Calculate feature discriminative power
feature_discrimination = {}
for col in df_features.select_dtypes(include=[np.number]).columns:
    if col not in ['EventId', 'Weight']:
        # Handle missing values
        valid_mask = df_features[col] != -999
        if valid_mask.sum() > 100:  # Enough valid values
            signal_vals = df_features.loc[signal_mask & valid_mask, col]
            background_vals = df_features.loc[background_mask & valid_mask, col]
            
            if len(signal_vals) > 0 and len(background_vals) > 0:
                # Simple separation metric
                signal_mean = signal_vals.mean()
                background_mean = background_vals.mean()
                signal_std = signal_vals.std()
                background_std = background_vals.std()
                
                if signal_std > 0 and background_std > 0:
                    separation = abs(signal_mean - background_mean) / np.sqrt(0.5 * (signal_std**2 + background_std**2))
                    feature_discrimination[col] = separation

# Top discriminative features
top_features = sorted(feature_discrimination.items(), key=lambda x: x[1], reverse=True)[:15]
print('\nTop 15 most discriminative features:')
for i, (feature, score) in enumerate(top_features, 1):
    print(f'{i:2d}. {feature:<30} {score:.4f}')

# Save enhanced dataset
output_file = 'atlas_higgs_physics_features.parquet'
df_features.to_parquet(output_file, index=False)
print(f'\nSaved enhanced dataset to {output_file}')

# Create feature metadata
feature_metadata = {
    'original_features': original_features,
    'new_features': new_features,
    'total_features': df_features.shape[1],
    'feature_categories': {
        'kinematic_ratios': ['pt_ratio_lep_met', 'pt_ratio_tau_met', 'pt_balance'],
        'angular_features': ['cos_deltar_tau_lep', 'sin_deltar_tau_lep', 'deltar_tau_lep_squared'],
        'mass_discriminants': ['mass_mmc_higgs_diff', 'mass_vis_over_mmc', 'pt_h_over_mass_vis'],
        'jet_features': ['jet_pt_sum', 'jet_pt_ratio'] + [f'is_{n}_jet' for n in range(4)],
        'tau_signatures': ['tau_lep_pt_sum', 'tau_lep_pt_diff', 'tau_lep_asymmetry'],
        'met_features': ['met_significance', 'met_over_sqrt_ht'],
        'global_features': ['total_pt', 'visible_pt_fraction'],
        'log_features': [f'log_{col}' for col in ['PRI_tau_pt', 'PRI_lep_pt', 'PRI_met', 'DER_mass_vis']],
        'interactions': ['mass_vis_times_pt_ratio', 'deltar_times_pt_balance']
    },
    'top_discriminative_features': dict(top_features),
    'missing_value_handling': 'Physics-motivated imputation and validity flags for -999 values'
}

with open('physics_features_metadata.json', 'w') as f:
    json.dump(feature_metadata, f, indent=2)

# Summary statistics
print(f'\nFeature engineering summary:')
print(f'- Events processed: {len(df_features):,}')
print(f'- Original features: {original_features}')
print(f'- New physics features: {new_features}')
print(f'- Total features: {df_features.shape[1]}')
print(f'- Signal events: {signal_mask.sum():,}')
print(f'- Background events: {background_mask.sum():,}')

# Return values for downstream steps
print(f'RESULT:enhanced_dataset={output_file}')
print(f'RESULT:metadata_file=physics_features_metadata.json')
print(f'RESULT:total_features={df_features.shape[1]}')
print(f'RESULT:new_features_count={new_features}')
print(f'RESULT:top_feature={top_features[0][0]}')
print(f'RESULT:top_feature_score={top_features[0][1]:.4f}')
print('RESULT:feature_engineering_complete=true')