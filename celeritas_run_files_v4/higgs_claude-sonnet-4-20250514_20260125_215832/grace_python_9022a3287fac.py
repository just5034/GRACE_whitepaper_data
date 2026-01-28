import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

# Load preprocessed data
df = pd.read_csv('preprocessed_physics_data.csv')
with open('preprocessing_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f'Loaded preprocessed data: {df.shape}')
print(f'Original features: {metadata["primary_features_count"]}')
print(f'Derived features from preprocessing: {metadata["derived_features_count"]}')

# Create physics-motivated features for Higgs->tau tau detection
original_cols = df.columns.tolist()

# 1. Kinematic ratios and normalized variables
df['pt_ratio_lep_met'] = df['PRI_lep_pt'] / (df['PRI_met'] + 1e-6)
df['pt_ratio_tau_lep'] = df['PRI_tau_pt'] / (df['PRI_lep_pt'] + 1e-6)
df['met_significance'] = df['PRI_met'] / np.sqrt(df['PRI_met_sumet'] + 1e-6)
df['total_pt'] = df['PRI_lep_pt'] + df['PRI_tau_pt'] + df['PRI_met']
df['visible_pt_fraction'] = (df['PRI_lep_pt'] + df['PRI_tau_pt']) / (df['total_pt'] + 1e-6)

# 2. Enhanced angular and geometric features
df['cos_deltar_tau_lep'] = np.cos(df['DER_deltar_tau_lep'])
df['sin_deltar_tau_lep'] = np.sin(df['DER_deltar_tau_lep'])
df['deltar_tau_lep_squared'] = df['DER_deltar_tau_lep'] ** 2

# 3. Mass-related discriminative features
# Higgs mass window indicators (signal peaks around 125 GeV)
df['mass_mmc_higgs_window'] = ((df['DER_mass_MMC'] > 100) & (df['DER_mass_MMC'] < 150)).astype(int)
df['mass_vis_normalized'] = df['DER_mass_vis'] / 125.0  # Normalize by Higgs mass
df['mass_ratio_vis_transverse'] = df['DER_mass_vis'] / (df['DER_mass_transverse_met_lep'] + 1e-6)

# 4. Jet-specific features for different multiplicities
df['has_jets'] = (df['PRI_jet_num'] > 0).astype(int)
df['jet_pt_density'] = np.where(df['PRI_jet_num'] > 0, 
                               (df['PRI_jet_leading_pt'] + df['PRI_jet_subleading_pt']) / df['PRI_jet_num'], 0)

# Jet mass features (handle missing values appropriately)
df['jet_mass_total'] = np.where(df['DER_mass_jet_jet'] > -900, df['DER_mass_jet_jet'], 0)
df['jet_eta_span'] = np.where(df['DER_deltaeta_jet_jet'] > -900, np.abs(df['DER_deltaeta_jet_jet']), 0)
df['jet_prod_eta'] = np.where(df['DER_prodeta_jet_jet'] > -900, df['DER_prodeta_jet_jet'], 0)

# 5. Tau decay topology features
df['tau_isolation'] = df['PRI_tau_pt'] / (df['total_pt'] + 1e-6)
df['lep_isolation'] = df['PRI_lep_pt'] / (df['total_pt'] + 1e-6)
df['met_balance'] = df['PRI_met'] / (df['PRI_lep_pt'] + df['PRI_tau_pt'] + 1e-6)

# 6. Composite discriminants
df['higgs_pt_estimate'] = np.sqrt(df['DER_pt_h'] ** 2 + df['PRI_met'] ** 2)
df['total_transverse_energy'] = df['PRI_lep_pt'] + df['PRI_tau_pt'] + df['PRI_met'] + df['PRI_jet_leading_pt'] + df['PRI_jet_subleading_pt']
df['sphericity_approx'] = df['DER_pt_tot'] / (df['total_transverse_energy'] + 1e-6)

# 7. Missing value indicators (important for tree-based models)
df['mmc_available'] = (df['DER_mass_MMC'] > -900).astype(int)
df['jet_features_available'] = (df['DER_mass_jet_jet'] > -900).astype(int)

# 8. Binned categorical features for non-linear relationships
df['lep_eta_bin'] = pd.cut(df['PRI_lep_eta'], bins=[-3, -1, 0, 1, 3], labels=[0,1,2,3]).astype(int)
df['tau_eta_bin'] = pd.cut(df['PRI_tau_eta'], bins=[-3, -1, 0, 1, 3], labels=[0,1,2,3]).astype(int)
df['met_bin'] = pd.cut(df['PRI_met'], bins=[0, 20, 40, 80, 200], labels=[0,1,2,3]).astype(int)

# Count new features created
new_cols = [col for col in df.columns if col not in original_cols]
num_new_features = len(new_cols)
total_features = len(df.columns) - 2  # Exclude EventId and Label

print(f'Created {num_new_features} new physics-motivated features:')
for i, col in enumerate(new_cols, 1):
    print(f'  {i:2d}. {col}')

# Analyze feature importance indicators
print('\nFeature engineering summary:')
print(f'- Kinematic ratios and normalized variables: 5')
print(f'- Enhanced angular features: 3')
print(f'- Mass-related discriminants: 3')
print(f'- Jet-specific features: 4')
print(f'- Tau decay topology: 3')
print(f'- Composite discriminants: 3')
print(f'- Missing value indicators: 2')
print(f'- Binned categorical features: 3')

# Create feature importance plot based on physics motivation
feature_categories = ['Kinematic Ratios', 'Angular Features', 'Mass Features', 
                     'Jet Features', 'Tau Topology', 'Composite', 'Missing Indicators', 'Binned Features']
feature_counts = [5, 3, 3, 4, 3, 3, 2, 3]

plt.figure(figsize=(12, 6))
plt.bar(feature_categories, feature_counts, color='skyblue', alpha=0.7)
plt.title('Physics-Motivated Feature Categories for Higgs Detection')
plt.xlabel('Feature Category')
plt.ylabel('Number of Features')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('feature_engineering_summary.png', dpi=150, bbox_inches='tight')
plt.savefig('feature_engineering_summary.pdf', bbox_inches='tight')

# Save enhanced dataset
output_file = 'physics_features_enhanced.csv'
df.to_csv(output_file, index=False)

# Create feature metadata
feature_metadata = {
    'total_features': total_features,
    'original_features': len(original_cols) - 2,
    'new_features': num_new_features,
    'feature_categories': {
        'kinematic_ratios': ['pt_ratio_lep_met', 'pt_ratio_tau_lep', 'met_significance', 'total_pt', 'visible_pt_fraction'],
        'angular_features': ['cos_deltar_tau_lep', 'sin_deltar_tau_lep', 'deltar_tau_lep_squared'],
        'mass_features': ['mass_mmc_higgs_window', 'mass_vis_normalized', 'mass_ratio_vis_transverse'],
        'jet_features': ['has_jets', 'jet_pt_density', 'jet_mass_total', 'jet_eta_span', 'jet_prod_eta'],
        'tau_topology': ['tau_isolation', 'lep_isolation', 'met_balance'],
        'composite': ['higgs_pt_estimate', 'total_transverse_energy', 'sphericity_approx'],
        'missing_indicators': ['mmc_available', 'jet_features_available'],
        'binned_features': ['lep_eta_bin', 'tau_eta_bin', 'met_bin']
    },
    'physics_motivation': {
        'higgs_mass_window': 'Signal events cluster around 125 GeV Higgs mass',
        'tau_isolation': 'Tau leptons from Higgs decay have characteristic isolation patterns',
        'angular_correlations': 'Spin correlations in H->tau tau create angular preferences',
        'missing_energy': 'Neutrinos from tau decays create specific MET signatures',
        'jet_veto': 'Higgs production mechanisms affect jet multiplicity distributions'
    }
}

with open('physics_features_metadata.json', 'w') as f:
    json.dump(feature_metadata, f, indent=2)

print(f'\nEnhanced dataset saved: {output_file}')
print(f'Dataset shape: {df.shape}')
print(f'Total features for ML: {total_features}')

# Return values for downstream steps
print(f'RESULT:total_features={total_features}')
print(f'RESULT:new_features_count={num_new_features}')
print(f'RESULT:enhanced_dataset_file={output_file}')
print(f'RESULT:feature_metadata_file=physics_features_metadata.json')
print(f'RESULT:feature_plot=feature_engineering_summary.png')
print(f'RESULT:dataset_shape={df.shape[0]}x{df.shape[1]}')