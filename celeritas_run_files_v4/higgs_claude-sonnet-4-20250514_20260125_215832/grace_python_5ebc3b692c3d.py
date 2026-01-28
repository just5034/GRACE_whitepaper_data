import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load preprocessed data
print('Loading preprocessed physics data...')
data_path = '/projects/bgde/grace/data/atlas-higgs-challenge-2014-v2.csv'
df = pd.read_csv(data_path)

print('Original features:', len(df.columns))
print('Events loaded:', len(df))

# Physics feature engineering for H->tau tau channel
print('Engineering physics-motivated features...')

# 1. Transverse momentum ratios and asymmetries
df['pt_ratio_lep_met'] = df['PRI_lep_pt'] / (df['PRI_met'] + 1e-6)
df['pt_asymmetry_tau'] = (df['PRI_tau_pt'] - df['PRI_lep_pt']) / (df['PRI_tau_pt'] + df['PRI_lep_pt'] + 1e-6)

# 2. Angular correlations (important for tau decay topology)
df['cos_deltar_tau_lep'] = np.cos(df['DER_deltar_tau_lep'].fillna(0))
df['sin_deltar_tau_lep'] = np.sin(df['DER_deltar_tau_lep'].fillna(0))

# 3. Missing energy significance
df['met_significance'] = df['PRI_met'] / np.sqrt(df['PRI_met_sumet'] + 1e-6)

# 4. Higgs candidate momentum features
df['pt_h_over_met'] = df['DER_pt_h'] / (df['PRI_met'] + 1e-6)
df['pt_h_significance'] = df['DER_pt_h'] / np.sqrt(df['PRI_met_sumet'] + 1e-6)

# 5. Jet-related features (stratified by jet multiplicity)
for njet in range(4):
    mask = (df['PRI_jet_num'] == njet)
    if mask.sum() > 0:
        jet_suffix = str(njet) + 'j'
        # Jet momentum fractions
        if njet >= 1:
            df.loc[mask, 'jet1_pt_frac'] = df.loc[mask, 'PRI_jet_leading_pt'] / (df.loc[mask, 'DER_pt_h'] + 1e-6)
        if njet >= 2:
            df.loc[mask, 'jet2_pt_frac'] = df.loc[mask, 'PRI_jet_subleading_pt'] / (df.loc[mask, 'DER_pt_h'] + 1e-6)
            df.loc[mask, 'dijet_pt_balance'] = abs(df.loc[mask, 'PRI_jet_leading_pt'] - df.loc[mask, 'PRI_jet_subleading_pt']) / (df.loc[mask, 'PRI_jet_leading_pt'] + df.loc[mask, 'PRI_jet_subleading_pt'] + 1e-6)

# 6. Mass-related discriminants
# Transverse mass combinations
df['mt_lep_met_squared'] = df['DER_mass_transverse_met_lep'] ** 2
df['mass_vis_over_mmc'] = df['DER_mass_vis'] / (df['DER_mass_MMC'].fillna(125.0) + 1e-6)

# 7. Centrality measures
df['centrality'] = (df['PRI_lep_pt'] + df['PRI_tau_pt']) / (df['DER_pt_h'] + 1e-6)

# 8. Acoplanarity (important for tau pair topology)
df['acoplanarity_approx'] = 1.0 - abs(np.cos(df['DER_deltar_tau_lep'].fillna(np.pi)))

# 9. Energy scale features
df['total_visible_pt'] = df['PRI_lep_pt'] + df['PRI_tau_pt'] + df['PRI_jet_all_pt']
df['visible_energy_frac'] = df['total_visible_pt'] / (df['total_visible_pt'] + df['PRI_met'] + 1e-6)

# 10. Log-transformed features (for highly skewed distributions)
for col in ['DER_mass_MMC', 'DER_mass_vis', 'PRI_met', 'DER_pt_h']:
    if col in df.columns:
        df['log_' + col.lower()] = np.log1p(df[col].fillna(0))

# Handle remaining missing values with physics-motivated imputation
print('Handling missing values with physics-motivated defaults...')

# MMC mass: use Higgs mass hypothesis when missing
df['DER_mass_MMC'] = df['DER_mass_MMC'].fillna(125.0)

# Jet features: fill with zeros when no jets
jet_cols = ['DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'PRI_jet_leading_pt', 'PRI_jet_subleading_pt']
for col in jet_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0.0)

# Fill any remaining NaN values
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# Count engineered features
original_cols = ['EventId', 'DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi', 'PRI_met_sumet', 'PRI_jet_num', 'PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt', 'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt', 'Label', 'KaggleSet', 'KaggleWeight']
engineered_cols = [col for col in df.columns if col not in original_cols]
num_engineered = len(engineered_cols)

print('Physics features engineered:')
for i, col in enumerate(engineered_cols, 1):
    print('  ' + str(i) + '. ' + col)

# Validate feature quality
print('\nFeature validation:')
feature_stats = {}
for col in engineered_cols:
    stats = {
        'mean': float(df[col].mean()),
        'std': float(df[col].std()),
        'min': float(df[col].min()),
        'max': float(df[col].max()),
        'missing': int(df[col].isnull().sum())
    }
    feature_stats[col] = stats
    print('  ' + col + ': mean=' + str(round(stats['mean'], 4)) + ', std=' + str(round(stats['std'], 4)))

# Save enhanced dataset
output_path = 'enhanced_physics_features.csv'
df.to_csv(output_path, index=False)

# Save feature metadata
feature_metadata = {
    'original_features': len(original_cols),
    'engineered_features': num_engineered,
    'total_features': len(df.columns),
    'events': len(df),
    'decay_channel': 'H->tau tau',
    'feature_categories': {
        'momentum_ratios': ['pt_ratio_lep_met', 'pt_asymmetry_tau', 'pt_h_over_met'],
        'angular_features': ['cos_deltar_tau_lep', 'sin_deltar_tau_lep', 'acoplanarity_approx'],
        'missing_energy': ['met_significance', 'pt_h_significance'],
        'jet_features': ['jet1_pt_frac', 'jet2_pt_frac', 'dijet_pt_balance'],
        'mass_features': ['mt_lep_met_squared', 'mass_vis_over_mmc'],
        'topology': ['centrality', 'visible_energy_frac'],
        'log_transforms': ['log_der_mass_mmc', 'log_der_mass_vis', 'log_pri_met', 'log_der_pt_h']
    },
    'feature_stats': feature_stats
}

with open('feature_metadata.json', 'w') as f:
    json.dump(feature_metadata, f, indent=2)

print('\nPhysics feature engineering completed successfully!')
print('Enhanced dataset saved to:', output_path)
print('Feature metadata saved to: feature_metadata.json')

# Return values for downstream steps
print('RESULT:enhanced_features=' + str(len(df.columns)))
print('RESULT:engineered_features=' + str(num_engineered))
print('RESULT:total_events=' + str(len(df)))
print('RESULT:feature_file=' + output_path)
print('RESULT:physics_features_status=success')