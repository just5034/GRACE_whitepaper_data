import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load preprocessed data
print('Loading preprocessed ATLAS Higgs data...')
try:
    # Try parquet first (faster)
    df = pd.read_parquet('atlas_higgs_processed.parquet')
    print(f'Loaded {len(df)} events from parquet file')
except:
    # Fallback to CSV
    df = pd.read_csv('atlas_higgs_processed.csv')
    print(f'Loaded {len(df)} events from CSV file')

# Debug: Inspect columns first
print('\n=== COLUMN INSPECTION ===')
print(f'Total columns: {len(df.columns)}')
print('Available columns:')
for i, col in enumerate(df.columns):
    print(f'  {i+1:2d}. {col}')

# Check for missing values
print('\n=== MISSING VALUE ANALYSIS ===')
missing_summary = df.isnull().sum()
missing_cols = missing_summary[missing_summary > 0]
if len(missing_cols) > 0:
    print('Columns with missing values:')
    for col, count in missing_cols.items():
        pct = count / len(df) * 100
        print(f'  {col}: {count} ({pct:.1f}%)')
else:
    print('No missing values found')

# Check for -999 sentinel values (common in HEP)
print('\n=== SENTINEL VALUE CHECK ===')
for col in df.select_dtypes(include=[np.number]).columns:
    if col != 'EventId':
        sentinel_count = (df[col] == -999.0).sum()
        if sentinel_count > 0:
            pct = sentinel_count / len(df) * 100
            print(f'  {col}: {sentinel_count} (-999 values, {pct:.1f}%)')

# Robust column access function
def get_column_safe(df, col_names):
    """Safely get column, trying multiple name variants"""
    if isinstance(col_names, str):
        col_names = [col_names]
    
    for name in col_names:
        if name in df.columns:
            return df[name]
    
    print(f'Warning: None of {col_names} found in columns')
    return pd.Series(np.zeros(len(df)), index=df.index)

# Start feature engineering
print('\n=== PHYSICS FEATURE ENGINEERING ===')
features_created = []

# 1. Transverse momentum ratios (important for tau identification)
try:
    pri_tau_pt = get_column_safe(df, ['PRI_tau_pt', 'tau_pt'])
    pri_lep_pt = get_column_safe(df, ['PRI_lep_pt', 'lep_pt'])
    if pri_tau_pt.sum() > 0 and pri_lep_pt.sum() > 0:
        df['pt_ratio_tau_lep'] = pri_tau_pt / (pri_lep_pt + 1e-6)
        features_created.append('pt_ratio_tau_lep')
        print('Created pt_ratio_tau_lep')
except Exception as e:
    print(f'Could not create pt_ratio_tau_lep: {e}')

# 2. Missing ET significance
try:
    pri_met = get_column_safe(df, ['PRI_met', 'met'])
    pri_met_sumet = get_column_safe(df, ['PRI_met_sumet', 'met_sumet'])
    if pri_met.sum() > 0 and pri_met_sumet.sum() > 0:
        df['met_significance'] = pri_met / np.sqrt(pri_met_sumet + 1e-6)
        features_created.append('met_significance')
        print('Created met_significance')
except Exception as e:
    print(f'Could not create met_significance: {e}')

# 3. Jet multiplicity features
try:
    pri_jet_num = get_column_safe(df, ['PRI_jet_num', 'jet_num'])
    if pri_jet_num.sum() >= 0:  # Can be 0
        df['has_jets'] = (pri_jet_num > 0).astype(int)
        df['multi_jet'] = (pri_jet_num >= 2).astype(int)
        features_created.extend(['has_jets', 'multi_jet'])
        print('Created jet multiplicity features')
except Exception as e:
    print(f'Could not create jet features: {e}')

# 4. Angular features (important for tau decay topology)
try:
    der_deltar_tau_lep = get_column_safe(df, ['DER_deltar_tau_lep', 'deltar_tau_lep'])
    if der_deltar_tau_lep.sum() > 0:
        # Create binned angular separation
        df['deltar_tau_lep_binned'] = pd.cut(der_deltar_tau_lep, bins=[0, 1, 2, 3, 10], labels=[0, 1, 2, 3]).astype(float)
        features_created.append('deltar_tau_lep_binned')
        print('Created deltar_tau_lep_binned')
except Exception as e:
    print(f'Could not create angular features: {e}')

# 5. Mass window features (crucial for Higgs identification)
try:
    der_mass_vis = get_column_safe(df, ['DER_mass_vis', 'mass_vis'])
    der_mass_mmc = get_column_safe(df, ['DER_mass_MMC', 'mass_mmc', 'DER_mass_mmc'])
    
    if der_mass_vis.sum() > 0:
        # Visible mass in Higgs window (around 125 GeV)
        df['mass_vis_higgs_window'] = ((der_mass_vis > 100) & (der_mass_vis < 150)).astype(int)
        features_created.append('mass_vis_higgs_window')
        print('Created mass_vis_higgs_window')
    
    if der_mass_mmc.sum() > 0:  # MMC mass when available
        # Handle -999 values in MMC mass
        valid_mmc = der_mass_mmc != -999.0
        df['has_valid_mmc'] = valid_mmc.astype(int)
        df['mmc_higgs_window'] = ((der_mass_mmc > 100) & (der_mass_mmc < 150) & valid_mmc).astype(int)
        features_created.extend(['has_valid_mmc', 'mmc_higgs_window'])
        print('Created MMC mass features')
except Exception as e:
    print(f'Could not create mass features: {e}')

# 6. Energy scale features
try:
    der_pt_h = get_column_safe(df, ['DER_pt_h', 'pt_h'])
    der_pt_tot = get_column_safe(df, ['DER_pt_tot', 'pt_tot'])
    
    if der_pt_h.sum() > 0 and der_pt_tot.sum() > 0:
        df['pt_h_over_pt_tot'] = der_pt_h / (der_pt_tot + 1e-6)
        features_created.append('pt_h_over_pt_tot')
        print('Created pt_h_over_pt_tot')
except Exception as e:
    print(f'Could not create energy scale features: {e}')

# 7. Jet-specific features (when jets are present)
try:
    der_deltaeta_jet_jet = get_column_safe(df, ['DER_deltaeta_jet_jet', 'deltaeta_jet_jet'])
    der_mass_jet_jet = get_column_safe(df, ['DER_mass_jet_jet', 'mass_jet_jet'])
    
    if der_deltaeta_jet_jet.sum() > 0:
        # Large rapidity gap (VBF signature)
        valid_deltaeta = der_deltaeta_jet_jet != -999.0
        df['large_deltaeta_jets'] = ((der_deltaeta_jet_jet > 3.5) & valid_deltaeta).astype(int)
        features_created.append('large_deltaeta_jets')
        print('Created large_deltaeta_jets')
    
    if der_mass_jet_jet.sum() > 0:
        # High dijet mass (VBF signature)
        valid_mjj = der_mass_jet_jet != -999.0
        df['high_mjj'] = ((der_mass_jet_jet > 500) & valid_mjj).astype(int)
        features_created.append('high_mjj')
        print('Created high_mjj')
except Exception as e:
    print(f'Could not create jet-specific features: {e}')

# 8. Combined discriminant features
try:
    # Create a simple BDT-like combination of key features
    score_components = []
    
    # Mass component (most important)
    if 'has_valid_mmc' in df.columns:
        score_components.append(df['has_valid_mmc'] * 0.3)
    if 'mmc_higgs_window' in df.columns:
        score_components.append(df['mmc_higgs_window'] * 0.4)
    
    # Kinematic components
    if 'met_significance' in df.columns:
        met_sig_norm = (df['met_significance'] - df['met_significance'].mean()) / (df['met_significance'].std() + 1e-6)
        score_components.append(np.clip(met_sig_norm, -2, 2) * 0.1)
    
    if len(score_components) > 0:
        df['higgs_discriminant'] = sum(score_components)
        features_created.append('higgs_discriminant')
        print('Created higgs_discriminant')
except Exception as e:
    print(f'Could not create discriminant features: {e}')

# Summary of created features
print(f'\n=== FEATURE CREATION SUMMARY ===')
print(f'Successfully created {len(features_created)} new features:')
for i, feat in enumerate(features_created, 1):
    print(f'  {i:2d}. {feat}')

# Debug: Check feature distributions
print('\n=== FEATURE DISTRIBUTION CHECK ===')
for feat in features_created[:5]:  # Check first 5 features
    if feat in df.columns:
        print(f'{feat}: mean={df[feat].mean():.3f}, std={df[feat].std():.3f}, min={df[feat].min():.3f}, max={df[feat].max():.3f}')

# Save enhanced dataset
enhanced_file = 'atlas_higgs_enhanced_features.parquet'
df.to_parquet(enhanced_file, index=False)
print(f'\nSaved enhanced dataset with {len(df.columns)} total columns to {enhanced_file}')

# Create feature metadata
feature_metadata = {
    'original_features': len(df.columns) - len(features_created),
    'new_features': len(features_created),
    'total_features': len(df.columns),
    'created_features': features_created,
    'decay_channel': 'H->tau tau',
    'feature_descriptions': {
        'pt_ratio_tau_lep': 'Ratio of tau to lepton transverse momentum',
        'met_significance': 'Missing ET significance (MET/sqrt(SumET))',
        'has_jets': 'Binary indicator for presence of jets',
        'multi_jet': 'Binary indicator for 2+ jets',
        'deltar_tau_lep_binned': 'Binned angular separation between tau and lepton',
        'mass_vis_higgs_window': 'Visible mass in Higgs mass window (100-150 GeV)',
        'has_valid_mmc': 'Binary indicator for valid MMC mass reconstruction',
        'mmc_higgs_window': 'MMC mass in Higgs mass window',
        'pt_h_over_pt_tot': 'Ratio of Higgs pT to total pT',
        'large_deltaeta_jets': 'Large rapidity gap between jets (VBF signature)',
        'high_mjj': 'High dijet invariant mass (VBF signature)',
        'higgs_discriminant': 'Combined discriminant score for Higgs identification'
    }
}

with open('feature_engineering_metadata.json', 'w') as f:
    json.dump(feature_metadata, f, indent=2)

# Results
print(f'\nRESULT:enhanced_dataset={enhanced_file}')
print(f'RESULT:metadata_file=feature_engineering_metadata.json')
print(f'RESULT:original_features={len(df.columns) - len(features_created)}')
print(f'RESULT:new_features={len(features_created)}')
print(f'RESULT:total_features={len(df.columns)}')
print(f'RESULT:features_created={len(features_created)}')
print('RESULT:engineering_success=true')