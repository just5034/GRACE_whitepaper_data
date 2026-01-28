import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Step 1: Inspect metadata first (as requested in modifications)
print('=== DEBUGGING METADATA STRUCTURE ===')
metadata_file = 'preprocessing_metadata.json'
if os.path.exists(metadata_file):
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    print('Metadata keys:', list(metadata.keys()))
    print('Metadata structure:', json.dumps(metadata, indent=2)[:500] + '...')
else:
    print('Metadata file not found, using fallback approach')
    metadata = {}

# Step 2: Load preprocessed data with flexible access
print('\n=== LOADING PREPROCESSED DATA ===')
data_file = 'preprocessed_physics_data.csv'
if os.path.exists(data_file):
    df = pd.read_csv(data_file)
    print(f'Loaded data shape: {df.shape}')
else:
    print('ERROR: Preprocessed data file not found')
    exit(1)

# Step 3: Inspect data structure
print('\nData columns:', list(df.columns))
print('Data types:', df.dtypes.to_dict())
print('Missing values per column:')
print(df.isnull().sum())

# Step 4: Identify physics features for H->tau tau channel
print('\n=== ENGINEERING PHYSICS FEATURES ===')

# Original features (from exploration context)
original_features = ['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 
                    'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 
                    'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot']

# Check which features are available
available_features = [f for f in original_features if f in df.columns]
print(f'Available original features: {len(available_features)}')

# Engineer new physics-motivated features for H->tau tau
feature_count = 0

# 1. Mass-based features (critical for Higgs at 125 GeV)
if 'DER_mass_vis' in df.columns:
    df['mass_vis_ratio_higgs'] = df['DER_mass_vis'] / 125.0  # Ratio to Higgs mass
    feature_count += 1
    
if 'DER_mass_MMC' in df.columns and df['DER_mass_MMC'].notna().sum() > 0:
    df['mass_mmc_ratio_higgs'] = df['DER_mass_MMC'] / 125.0
    df['mass_mmc_available'] = (df['DER_mass_MMC'] != -999).astype(int)
    feature_count += 2

# 2. Transverse momentum features
if 'DER_pt_h' in df.columns:
    df['pt_h_log'] = np.log1p(df['DER_pt_h'])  # Log transform for better distribution
    feature_count += 1
    
if 'DER_pt_tot' in df.columns:
    df['pt_tot_log'] = np.log1p(df['DER_pt_tot'])
    feature_count += 1
    
if 'DER_pt_h' in df.columns and 'DER_pt_tot' in df.columns:
    df['pt_ratio'] = df['DER_pt_h'] / (df['DER_pt_tot'] + 1e-6)  # Avoid division by zero
    feature_count += 1

# 3. Angular separation features (important for tau identification)
if 'DER_deltar_tau_lep' in df.columns:
    df['deltar_tau_lep_squared'] = df['DER_deltar_tau_lep'] ** 2
    df['deltar_tau_lep_inv'] = 1.0 / (df['DER_deltar_tau_lep'] + 0.1)
    feature_count += 2

# 4. Jet-based features (for different jet multiplicities)
if 'PRI_jet_num' in df.columns:
    df['has_jets'] = (df['PRI_jet_num'] > 0).astype(int)
    df['jet_multiplicity_high'] = (df['PRI_jet_num'] >= 2).astype(int)
    feature_count += 2
    
if 'DER_deltaeta_jet_jet' in df.columns:
    # Only valid for events with 2+ jets
    df['deltaeta_jet_jet_abs'] = np.abs(df['DER_deltaeta_jet_jet'])
    df['deltaeta_jet_jet_available'] = (df['DER_deltaeta_jet_jet'] != -999).astype(int)
    feature_count += 2

# 5. Missing energy features (crucial for neutrinos from tau decays)
if 'PRI_met' in df.columns:
    df['met_log'] = np.log1p(df['PRI_met'])
    df['met_significance'] = df['PRI_met'] / (df['PRI_met'].std() + 1e-6)
    feature_count += 2

# 6. Composite kinematic features
if 'DER_mass_vis' in df.columns and 'DER_pt_h' in df.columns:
    df['mass_pt_ratio'] = df['DER_mass_vis'] / (df['DER_pt_h'] + 1e-6)
    feature_count += 1
    
if 'PRI_met' in df.columns and 'DER_pt_tot' in df.columns:
    df['met_pt_ratio'] = df['PRI_met'] / (df['DER_pt_tot'] + 1e-6)
    feature_count += 1

# 7. Tau-specific features
if 'PRI_tau_pt' in df.columns:
    df['tau_pt_log'] = np.log1p(df['PRI_tau_pt'])
    feature_count += 1
    
if 'PRI_lep_pt' in df.columns:
    df['lep_pt_log'] = np.log1p(df['PRI_lep_pt'])
    feature_count += 1
    
if 'PRI_tau_pt' in df.columns and 'PRI_lep_pt' in df.columns:
    df['tau_lep_pt_ratio'] = df['PRI_tau_pt'] / (df['PRI_lep_pt'] + 1e-6)
    feature_count += 1

# Step 5: Fallback feature counting (as requested)
print(f'\n=== FALLBACK FEATURE COUNTING ===')
original_feature_count = len([c for c in df.columns if not c.startswith('engineered_')])
engineered_feature_count = feature_count
total_features = df.shape[1]

print(f'Original features: {original_feature_count}')
print(f'Newly engineered features: {engineered_feature_count}')
print(f'Total features: {total_features}')

# Step 6: Handle remaining missing values
print('\n=== HANDLING MISSING VALUES ===')
missing_before = df.isnull().sum().sum()

# Fill remaining -999 values with appropriate strategies
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        # Replace -999 with NaN first
        df[col] = df[col].replace(-999, np.nan)
        
        # Fill NaN with median for continuous variables
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

missing_after = df.isnull().sum().sum()
print(f'Missing values before: {missing_before}')
print(f'Missing values after: {missing_after}')

# Step 7: Save enhanced dataset
output_file = 'enhanced_physics_features.csv'
df.to_csv(output_file, index=False)
print(f'\nSaved enhanced dataset to: {output_file}')

# Step 8: Create feature importance visualization
plt.figure(figsize=(12, 8))
if 'Label' in df.columns or 'label' in df.columns:
    label_col = 'Label' if 'Label' in df.columns else 'label'
    signal_data = df[df[label_col] == 's']
    background_data = df[df[label_col] == 'b']
    
    # Plot distribution of key engineered features
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    if 'mass_vis_ratio_higgs' in df.columns:
        axes[0,0].hist(signal_data['mass_vis_ratio_higgs'], bins=50, alpha=0.7, label='Signal', density=True)
        axes[0,0].hist(background_data['mass_vis_ratio_higgs'], bins=50, alpha=0.7, label='Background', density=True)
        axes[0,0].set_xlabel('Visible Mass / Higgs Mass')
        axes[0,0].set_ylabel('Density')
        axes[0,0].legend()
        axes[0,0].axvline(1.0, color='red', linestyle='--', label='Higgs Mass')
    
    if 'pt_ratio' in df.columns:
        axes[0,1].hist(signal_data['pt_ratio'], bins=50, alpha=0.7, label='Signal', density=True)
        axes[0,1].hist(background_data['pt_ratio'], bins=50, alpha=0.7, label='Background', density=True)
        axes[0,1].set_xlabel('pT_h / pT_tot Ratio')
        axes[0,1].set_ylabel('Density')
        axes[0,1].legend()
    
    if 'deltar_tau_lep_squared' in df.columns:
        axes[1,0].hist(signal_data['deltar_tau_lep_squared'], bins=50, alpha=0.7, label='Signal', density=True)
        axes[1,0].hist(background_data['deltar_tau_lep_squared'], bins=50, alpha=0.7, label='Background', density=True)
        axes[1,0].set_xlabel('(ΔR τ-lepton)²')
        axes[1,0].set_ylabel('Density')
        axes[1,0].legend()
    
    if 'met_pt_ratio' in df.columns:
        axes[1,1].hist(signal_data['met_pt_ratio'], bins=50, alpha=0.7, label='Signal', density=True)
        axes[1,1].hist(background_data['met_pt_ratio'], bins=50, alpha=0.7, label='Background', density=True)
        axes[1,1].set_xlabel('MET / pT_tot Ratio')
        axes[1,1].set_ylabel('Density')
        axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('engineered_features_distributions.png', dpi=150, bbox_inches='tight')
    plt.savefig('engineered_features_distributions.pdf', bbox_inches='tight')
    print('Saved feature distributions plot')

# Step 9: Create summary metadata
summary_metadata = {
    'original_features': original_feature_count,
    'engineered_features': engineered_feature_count,
    'total_features': total_features,
    'dataset_shape': list(df.shape),
    'missing_values_handled': int(missing_before - missing_after),
    'decay_channel': 'H->tau tau',
    'physics_features_added': [
        'mass_vis_ratio_higgs', 'mass_mmc_ratio_higgs', 'pt_h_log', 'pt_tot_log',
        'pt_ratio', 'deltar_tau_lep_squared', 'deltar_tau_lep_inv', 'has_jets',
        'jet_multiplicity_high', 'deltaeta_jet_jet_abs', 'met_log', 'met_significance',
        'mass_pt_ratio', 'met_pt_ratio', 'tau_pt_log', 'lep_pt_log', 'tau_lep_pt_ratio'
    ]
}

with open('enhanced_features_metadata.json', 'w') as f:
    json.dump(summary_metadata, f, indent=2)

print('\n=== FEATURE ENGINEERING COMPLETE ===')
print(f'RESULT:enhanced_features_count={engineered_feature_count}')
print(f'RESULT:total_features_count={total_features}')
print(f'RESULT:dataset_shape={df.shape[0]}x{df.shape[1]}')
print(f'RESULT:output_file={output_file}')
print(f'RESULT:metadata_file=enhanced_features_metadata.json')
print(f'RESULT:feature_plot=engineered_features_distributions.png')
print(f'RESULT:missing_values_handled={int(missing_before - missing_after)}')