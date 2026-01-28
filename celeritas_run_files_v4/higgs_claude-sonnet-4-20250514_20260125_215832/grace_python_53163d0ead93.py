import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load the preprocessed data
print('Loading preprocessed ATLAS Higgs data...')
df = pd.read_parquet('atlas_higgs_processed.parquet')
print(f'Loaded data shape: {df.shape}')

# Check available columns
print('\nAvailable columns:')
for i, col in enumerate(df.columns):
    print(f'{i+1:2d}. {col}')

# Validate required columns exist
required_base_cols = ['PRI_tau_pt', 'PRI_lep_pt', 'PRI_jet_leading_pt', 'PRI_jet_subleading_pt',
                      'DER_mass_vis', 'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
                      'DER_deltar_tau_lep', 'PRI_met', 'PRI_met_phi', 'PRI_jet_num']

missing_cols = [col for col in required_base_cols if col not in df.columns]
if missing_cols:
    print(f'Warning: Missing columns: {missing_cols}')
else:
    print('All required base columns found!')

# Create atlas column mapping features first
print('\nCreating mapped features...')
if 'PRI_tau_pt' in df.columns and 'PRI_lep_pt' in df.columns:
    df['tau_lep_pt_sum'] = df['PRI_tau_pt'] + df['PRI_lep_pt']
    print('Created tau_lep_pt_sum')

if 'PRI_jet_leading_pt' in df.columns and 'PRI_jet_subleading_pt' in df.columns:
    # Handle missing values (-999) in jet features
    jet_lead = df['PRI_jet_leading_pt'].replace(-999, 0)
    jet_sub = df['PRI_jet_subleading_pt'].replace(-999, 0)
    df['jet_pt_sum'] = jet_lead + jet_sub
    print('Created jet_pt_sum')

# Engineer physics-motivated features for H->tau tau discrimination
print('\nEngineering physics features...')

# 1. Transverse mass features (key for tau reconstruction)
if 'PRI_met' in df.columns and 'PRI_lep_pt' in df.columns and 'PRI_met_phi' in df.columns:
    # Transverse mass of lepton and MET system
    df['mt_lep_met'] = np.sqrt(2 * df['PRI_lep_pt'] * df['PRI_met'] * 
                               (1 - np.cos(df['PRI_met_phi'])))
    print('Created mt_lep_met (transverse mass)')

# 2. Momentum ratios (discriminate tau decay signatures)
if 'tau_lep_pt_sum' in df.columns and 'DER_pt_h' in df.columns:
    df['pt_ratio_tau_lep_h'] = df['tau_lep_pt_sum'] / (df['DER_pt_h'] + 1e-6)
    print('Created pt_ratio_tau_lep_h')

if 'PRI_met' in df.columns and 'DER_pt_h' in df.columns:
    df['met_pt_h_ratio'] = df['PRI_met'] / (df['DER_pt_h'] + 1e-6)
    print('Created met_pt_h_ratio')

# 3. Angular features (tau decay topology)
if 'DER_deltar_tau_lep' in df.columns:
    df['deltar_tau_lep_sq'] = df['DER_deltar_tau_lep'] ** 2
    print('Created deltar_tau_lep_sq')

# 4. Jet-based features (background discrimination)
if 'PRI_jet_num' in df.columns:
    # One-hot encode jet multiplicity (important physics categories)
    df['is_0_jet'] = (df['PRI_jet_num'] == 0).astype(int)
    df['is_1_jet'] = (df['PRI_jet_num'] == 1).astype(int)
    df['is_2plus_jet'] = (df['PRI_jet_num'] >= 2).astype(int)
    print('Created jet multiplicity features')

if 'DER_mass_jet_jet' in df.columns and 'DER_deltaeta_jet_jet' in df.columns:
    # VBF (Vector Boson Fusion) discriminant features
    # Handle missing values for events with <2 jets
    mask_2jets = df['PRI_jet_num'] >= 2
    df['vbf_mass_eta_product'] = 0.0
    df.loc[mask_2jets, 'vbf_mass_eta_product'] = (df.loc[mask_2jets, 'DER_mass_jet_jet'] * 
                                                   df.loc[mask_2jets, 'DER_deltaeta_jet_jet'])
    print('Created vbf_mass_eta_product')

# 5. Mass-based features (Higgs mass peak discrimination)
if 'DER_mass_vis' in df.columns:
    # Distance from Higgs mass (125 GeV)
    df['mass_vis_higgs_diff'] = np.abs(df['DER_mass_vis'] - 125.0)
    print('Created mass_vis_higgs_diff')

# 6. Energy scale features
if 'DER_pt_h' in df.columns and 'DER_mass_vis' in df.columns:
    df['pt_mass_ratio'] = df['DER_pt_h'] / (df['DER_mass_vis'] + 1e-6)
    print('Created pt_mass_ratio')

# 7. Missing energy significance
if 'PRI_met' in df.columns and 'tau_lep_pt_sum' in df.columns:
    df['met_significance'] = df['PRI_met'] / (df['tau_lep_pt_sum'] + 1e-6)
    print('Created met_significance')

# Count new features created
original_features = 35  # From preprocessing step
new_features = df.shape[1] - original_features
print(f'\nFeature engineering summary:')
print(f'Original features: {original_features}')
print(f'New features created: {new_features}')
print(f'Total features: {df.shape[1]}')

# Validate no infinite or NaN values in new features
print('\nValidating new features...')
new_feature_cols = df.columns[-new_features:] if new_features > 0 else []
for col in new_feature_cols:
    n_inf = np.isinf(df[col]).sum()
    n_nan = df[col].isna().sum()
    if n_inf > 0 or n_nan > 0:
        print(f'Warning: {col} has {n_inf} inf and {n_nan} NaN values')
        # Replace inf with large finite value, NaN with 0
        df[col] = df[col].replace([np.inf, -np.inf], [1e6, -1e6]).fillna(0)

# Save enhanced dataset
output_file = 'atlas_higgs_features_enhanced.parquet'
df.to_parquet(output_file, index=False)
print(f'\nSaved enhanced dataset to: {output_file}')

# Create feature metadata
feature_metadata = {
    'total_features': int(df.shape[1]),
    'new_features_count': int(new_features),
    'physics_features': {
        'transverse_mass': ['mt_lep_met'],
        'momentum_ratios': ['pt_ratio_tau_lep_h', 'met_pt_h_ratio', 'pt_mass_ratio'],
        'angular_features': ['deltar_tau_lep_sq'],
        'jet_features': ['is_0_jet', 'is_1_jet', 'is_2plus_jet', 'vbf_mass_eta_product'],
        'mass_features': ['mass_vis_higgs_diff'],
        'energy_features': ['met_significance'],
        'composite_features': ['tau_lep_pt_sum', 'jet_pt_sum']
    },
    'feature_descriptions': {
        'mt_lep_met': 'Transverse mass of lepton-MET system (tau decay signature)',
        'pt_ratio_tau_lep_h': 'Ratio of tau+lepton pT to Higgs pT (momentum balance)',
        'met_pt_h_ratio': 'Missing ET to Higgs pT ratio (neutrino signature)',
        'deltar_tau_lep_sq': 'Squared angular separation of tau-lepton (decay topology)',
        'vbf_mass_eta_product': 'VBF discriminant (dijet mass × eta separation)',
        'mass_vis_higgs_diff': 'Distance from Higgs mass peak (125 GeV)',
        'met_significance': 'MET significance relative to visible momentum'
    }
}

with open('atlas_higgs_features_metadata.json', 'w') as f:
    json.dump(feature_metadata, f, indent=2)

# Print feature importance insights
print('\nPhysics-motivated feature insights:')
print('1. Transverse mass (mt_lep_met): Key for tau->lepton+neutrino reconstruction')
print('2. Momentum ratios: Discriminate tau decay kinematics from background')
print('3. Jet multiplicity: Separate VBF, VH, and ggH production modes')
print('4. VBF features: Identify vector boson fusion topology')
print('5. Mass features: Exploit Higgs mass peak at 125 GeV')
print('6. Angular features: Tau decay topology discrimination')

print(f'RESULT:enhanced_features_file={output_file}')
print(f'RESULT:total_features={df.shape[1]}')
print(f'RESULT:new_features_created={new_features}')
print(f'RESULT:metadata_file=atlas_higgs_features_metadata.json')
print(f'RESULT:feature_engineering_success=True')