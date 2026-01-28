import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load the preprocessed ATLAS Higgs data
data_path = '/projects/bgde/grace/data/atlas-higgs-challenge-2014-v2.csv'
df = pd.read_csv(data_path)

print(f'Loaded {len(df)} events with {df.shape[1]} features')
print(f'Signal events: {(df["Label"] == "s").sum()}')
print(f'Background events: {(df["Label"] == "b").sum()}')

# Create physics-motivated features for H->tau tau detection
# 1. Kinematic variables
df['DER_pt_ratio_lep_tau'] = df['PRI_lep_pt'] / (df['PRI_tau_pt'] + 1e-6)
df['DER_eta_centrality'] = np.abs(df['PRI_lep_eta'] + df['PRI_tau_eta']) / 2.0
df['DER_phi_separation'] = np.abs(df['PRI_lep_phi'] - df['PRI_tau_phi'])
df['DER_phi_separation'] = np.where(df['DER_phi_separation'] > np.pi, 2*np.pi - df['DER_phi_separation'], df['DER_phi_separation'])

# 2. Enhanced angular separations
df['DER_deltaR_lep_tau'] = np.sqrt((df['PRI_lep_eta'] - df['PRI_tau_eta'])**2 + df['DER_phi_separation']**2)
df['DER_cos_theta_lep_tau'] = np.cos(df['DER_phi_separation'])

# 3. Transverse mass combinations
df['DER_mt_lep_met'] = np.sqrt(2 * df['PRI_lep_pt'] * df['PRI_met'] * (1 - np.cos(np.abs(df['PRI_lep_phi'] - df['PRI_met_phi']))))
df['DER_mt_tau_met'] = np.sqrt(2 * df['PRI_tau_pt'] * df['PRI_met'] * (1 - np.cos(np.abs(df['PRI_tau_phi'] - df['PRI_met_phi']))))

# 4. Tau-specific observables
df['DER_tau_isolation'] = df['PRI_tau_pt'] / (df['DER_pt_tot'] + 1e-6)
df['DER_met_significance'] = df['PRI_met'] / np.sqrt(df['DER_sum_pt'] + 1e-6)
df['DER_sphericity'] = (df['PRI_lep_pt'] + df['PRI_tau_pt']) / (df['DER_pt_tot'] + 1e-6)

# 5. Jet-related features (when jets are present)
jet_mask = df['PRI_jet_num'] > 0
df['DER_mjj_over_mvis'] = np.where(jet_mask, df['DER_mass_jet_jet'] / (df['DER_mass_vis'] + 1e-6), 0)
df['DER_jet_centrality'] = np.where(jet_mask, np.abs(df['DER_deltaeta_jet_jet']) / 5.0, 0)

# 6. Higgs-specific kinematic features
df['DER_pt_balance'] = np.abs(df['PRI_lep_pt'] - df['PRI_tau_pt']) / (df['PRI_lep_pt'] + df['PRI_tau_pt'] + 1e-6)
df['DER_visible_mass_ratio'] = df['DER_mass_vis'] / 125.0  # Ratio to Higgs mass
df['DER_collinear_mass'] = np.where(df['DER_mass_MMC'] != -999, df['DER_mass_MMC'], df['DER_mass_vis'])

# 7. Missing energy features for neutrinos from tau decays
df['DER_met_proj_lep'] = df['PRI_met'] * np.cos(df['PRI_met_phi'] - df['PRI_lep_phi'])
df['DER_met_proj_tau'] = df['PRI_met'] * np.cos(df['PRI_met_phi'] - df['PRI_tau_phi'])
df['DER_met_perp'] = np.sqrt(df['PRI_met']**2 - df['DER_met_proj_lep']**2 - df['DER_met_proj_tau']**2 + 1e-6)

# Handle missing values in new features
new_features = ['DER_pt_ratio_lep_tau', 'DER_eta_centrality', 'DER_phi_separation', 'DER_deltaR_lep_tau',
               'DER_cos_theta_lep_tau', 'DER_mt_lep_met', 'DER_mt_tau_met', 'DER_tau_isolation',
               'DER_met_significance', 'DER_sphericity', 'DER_mjj_over_mvis', 'DER_jet_centrality',
               'DER_pt_balance', 'DER_visible_mass_ratio', 'DER_collinear_mass', 'DER_met_proj_lep',
               'DER_met_proj_tau', 'DER_met_perp']

# Replace infinities and extreme values
for feature in new_features:
    df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
    df[feature] = df[feature].fillna(df[feature].median())
    df[feature] = np.clip(df[feature], df[feature].quantile(0.001), df[feature].quantile(0.999))

# Calculate feature importance metrics
signal_mask = df['Label'] == 's'
background_mask = df['Label'] == 'b'

feature_discrimination = {}
for feature in new_features:
    if feature in df.columns:
        signal_mean = df.loc[signal_mask, feature].mean()
        background_mean = df.loc[background_mask, feature].mean()
        signal_std = df.loc[signal_mask, feature].std()
        background_std = df.loc[background_mask, feature].std()
        
        # Calculate separation power (simplified)
        pooled_std = np.sqrt((signal_std**2 + background_std**2) / 2)
        separation = abs(signal_mean - background_mean) / (pooled_std + 1e-6)
        feature_discrimination[feature] = separation

# Sort features by discrimination power
sorted_features = sorted(feature_discrimination.items(), key=lambda x: x[1], reverse=True)

print('\nTop 10 most discriminative new features:')
for i, (feature, score) in enumerate(sorted_features[:10]):
    print(f'{i+1:2d}. {feature}: {score:.4f}')

# Save enhanced dataset
output_path = 'enhanced_higgs_features.csv'
df.to_csv(output_path, index=False)

# Create feature summary
feature_summary = {
    'original_features': df.shape[1] - len(new_features),
    'new_features': len(new_features),
    'total_features': df.shape[1],
    'events': len(df),
    'top_discriminative_features': dict(sorted_features[:5]),
    'feature_categories': {
        'kinematic_variables': 3,
        'angular_separations': 2,
        'transverse_masses': 2,
        'tau_specific': 3,
        'jet_related': 2,
        'higgs_specific': 3,
        'missing_energy': 3
    }
}

with open('feature_engineering_summary.json', 'w') as f:
    json.dump(feature_summary, f, indent=2)

print(f'\nFeature engineering completed:')
print(f'- Original features: {feature_summary["original_features"]}')
print(f'- New physics features: {feature_summary["new_features"]}')
print(f'- Total features: {feature_summary["total_features"]}')
print(f'- Enhanced dataset saved to: {output_path}')

# Return values for downstream steps
print(f'RESULT:enhanced_features={feature_summary["new_features"]}')
print(f'RESULT:total_features={feature_summary["total_features"]}')
print(f'RESULT:events_processed={len(df)}')
print(f'RESULT:top_discriminator={sorted_features[0][0]}')
print(f'RESULT:discrimination_score={sorted_features[0][1]:.4f}')
print(f'RESULT:enhanced_dataset=enhanced_higgs_features.csv')
print(f'RESULT:feature_summary=feature_engineering_summary.json')
print('RESULT:engineering_status=success')