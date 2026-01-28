import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Load the preprocessed data
data_path = '/projects/bgde/grace/data/atlas-higgs-challenge-2014-v2.csv'
df = pd.read_csv(data_path)

print(f'Loaded {len(df)} events with {df.shape[1]} features')
print(f'Signal events: {(df["Label"] == "s").sum()}')
print(f'Background events: {(df["Label"] == "b").sum()}')

# Separate by jet multiplicity categories for targeted feature engineering
jet_categories = df['PRI_jet_num'].value_counts().sort_index()
print(f'Jet multiplicity distribution: {dict(jet_categories)}')

# Create physics-motivated features for Higgs->tau tau detection
df_enhanced = df.copy()

# 1. Tau decay kinematics - visible mass ratios
df_enhanced['vis_mass_ratio'] = df['DER_mass_vis'] / 125.0  # Ratio to Higgs mass
df_enhanced['mmc_vis_ratio'] = np.where(df['DER_mass_MMC'] != -999, 
                                       df['DER_mass_MMC'] / df['DER_mass_vis'], -999)

# 2. Transverse momentum features
df_enhanced['pt_ratio_lep_tau'] = df['PRI_lep_pt'] / (df['PRI_tau_pt'] + 1e-6)
df_enhanced['pt_balance'] = (df['PRI_lep_pt'] + df['PRI_tau_pt']) / (df['DER_pt_h'] + 1e-6)
df_enhanced['met_significance'] = df['PRI_met'] / np.sqrt(df['DER_sum_pt'] + 1e-6)

# 3. Angular separation features (tau decay signatures)
df_enhanced['delta_phi_lep_tau'] = np.abs(df['PRI_lep_phi'] - df['PRI_tau_phi'])
df_enhanced['delta_phi_lep_tau'] = np.where(df_enhanced['delta_phi_lep_tau'] > np.pi,
                                           2*np.pi - df_enhanced['delta_phi_lep_tau'],
                                           df_enhanced['delta_phi_lep_tau'])

# 4. Missing energy features (neutrinos from tau decays)
df_enhanced['met_centrality'] = np.abs(df['PRI_met_phi'] - (df['PRI_lep_phi'] + df['PRI_tau_phi'])/2)
df_enhanced['met_centrality'] = np.where(df_enhanced['met_centrality'] > np.pi,
                                        2*np.pi - df_enhanced['met_centrality'],
                                        df_enhanced['met_centrality'])

# 5. Jet-specific features (different for 0-jet, 1-jet, 2+ jet categories)
for jet_cat in [0, 1, 2, 3]:
    mask = df['PRI_jet_num'] == jet_cat
    if jet_cat >= 2:
        # VBF-like features for 2+ jets
        df_enhanced.loc[mask, 'vbf_score'] = np.where(
            (df.loc[mask, 'DER_deltaeta_jet_jet'] > 3.5) & 
            (df.loc[mask, 'DER_mass_jet_jet'] > 400),
            df.loc[mask, 'DER_deltaeta_jet_jet'] * np.log(df.loc[mask, 'DER_mass_jet_jet']),
            0
        )
    else:
        df_enhanced.loc[mask, 'vbf_score'] = 0

# 6. Invariant mass combinations
df_enhanced['mass_lep_met'] = np.sqrt(2 * df['PRI_lep_pt'] * df['PRI_met'] * 
                                     (1 - np.cos(df['PRI_lep_phi'] - df['PRI_met_phi'])))
df_enhanced['mass_tau_met'] = np.sqrt(2 * df['PRI_tau_pt'] * df['PRI_met'] * 
                                     (1 - np.cos(df['PRI_tau_phi'] - df['PRI_met_phi'])))

# 7. Collinear approximation for tau momentum reconstruction
df_enhanced['x1_col'] = df['PRI_lep_pt'] / (df['PRI_lep_pt'] + df['PRI_met'] * np.cos(df['PRI_lep_phi'] - df['PRI_met_phi']))
df_enhanced['x2_col'] = df['PRI_tau_pt'] / (df['PRI_tau_pt'] + df['PRI_met'] * np.cos(df['PRI_tau_phi'] - df['PRI_met_phi']))
df_enhanced['collinear_mass'] = np.where(
    (df_enhanced['x1_col'] > 0) & (df_enhanced['x1_col'] < 1) & 
    (df_enhanced['x2_col'] > 0) & (df_enhanced['x2_col'] < 1),
    df['DER_mass_vis'] / np.sqrt(df_enhanced['x1_col'] * df_enhanced['x2_col']),
    -999
)

# 8. Stransverse mass (for missing energy events)
df_enhanced['mt_lep'] = np.sqrt(2 * df['PRI_lep_pt'] * df['PRI_met'] * 
                               (1 - np.cos(df['PRI_lep_phi'] - df['PRI_met_phi'])))
df_enhanced['mt_tau'] = np.sqrt(2 * df['PRI_tau_pt'] * df['PRI_met'] * 
                               (1 - np.cos(df['PRI_tau_phi'] - df['PRI_met_phi'])))

# Handle missing values in new features
for col in df_enhanced.columns:
    if col not in df.columns:
        df_enhanced[col] = df_enhanced[col].replace([np.inf, -np.inf], -999)

# Count new features created
original_features = df.shape[1]
new_features = df_enhanced.shape[1] - original_features
print(f'Created {new_features} new physics-motivated features')

# Analyze feature importance by signal/background separation
signal_mask = df_enhanced['Label'] == 's'
background_mask = df_enhanced['Label'] == 'b'

feature_importance = {}
for col in df_enhanced.columns:
    if col not in ['EventId', 'Label', 'Weight'] and df_enhanced[col].dtype in ['float64', 'int64']:
        # Skip features with too many missing values
        valid_mask = df_enhanced[col] != -999
        if valid_mask.sum() < 1000:
            continue
            
        signal_vals = df_enhanced.loc[signal_mask & valid_mask, col]
        background_vals = df_enhanced.loc[background_mask & valid_mask, col]
        
        if len(signal_vals) > 0 and len(background_vals) > 0:
            # Calculate separation power using KL divergence approximation
            s_mean, s_std = signal_vals.mean(), signal_vals.std()
            b_mean, b_std = background_vals.mean(), background_vals.std()
            
            if s_std > 0 and b_std > 0:
                separation = abs(s_mean - b_mean) / np.sqrt(0.5 * (s_std**2 + b_std**2))
                feature_importance[col] = separation

# Sort features by importance
top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=true)[:15]
print('\nTop 15 discriminative features:')
for feat, importance in top_features:
    print(f'{feat}: {importance:.3f}')

# Create visualization of key new features
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Physics-Motivated Features for Higgs Detection', fontsize=14)

# Plot 1: Visible mass ratio
axes[0,0].hist(df_enhanced.loc[signal_mask, 'vis_mass_ratio'], bins=50, alpha=0.7, 
               label='Signal', density=true, histtype='step', linewidth=2)
axes[0,0].hist(df_enhanced.loc[background_mask, 'vis_mass_ratio'], bins=50, alpha=0.7, 
               label='Background', density=true, histtype='step', linewidth=2)
axes[0,0].axvline(1.0, color='red', linestyle='--', alpha=0.7, label='Higgs mass')
axes[0,0].set_xlabel('Visible Mass / 125 GeV')
axes[0,0].set_ylabel('Density')
axes[0,0].legend()
axes[0,0].set_title('Visible Mass Ratio')

# Plot 2: MET significance
valid_met = df_enhanced['met_significance'] != -999
axes[0,1].hist(df_enhanced.loc[signal_mask & valid_met, 'met_significance'], bins=50, 
               alpha=0.7, label='Signal', density=true, histtype='step', linewidth=2, range=(0, 5))
axes[0,1].hist(df_enhanced.loc[background_mask & valid_met, 'met_significance'], bins=50, 
               alpha=0.7, label='Background', density=true, histtype='step', linewidth=2, range=(0, 5))
axes[0,1].set_xlabel('MET Significance')
axes[0,1].set_ylabel('Density')
axes[0,1].legend()
axes[0,1].set_title('Missing Energy Significance')

# Plot 3: Angular separation
axes[1,0].hist(df_enhanced.loc[signal_mask, 'delta_phi_lep_tau'], bins=50, alpha=0.7, 
               label='Signal', density=true, histtype='step', linewidth=2)
axes[1,0].hist(df_enhanced.loc[background_mask, 'delta_phi_lep_tau'], bins=50, alpha=0.7, 
               label='Background', density=true, histtype='step', linewidth=2)
axes[1,0].set_xlabel('Δφ(lepton, tau)')
axes[1,0].set_ylabel('Density')
axes[1,0].legend()
axes[1,0].set_title('Lepton-Tau Angular Separation')

# Plot 4: Collinear mass (when available)
valid_col = df_enhanced['collinear_mass'] != -999
if valid_col.sum() > 100:
    col_range = (0, 300)
    axes[1,1].hist(df_enhanced.loc[signal_mask & valid_col, 'collinear_mass'], bins=50, 
                   alpha=0.7, label='Signal', density=true, histtype='step', linewidth=2, range=col_range)
    axes[1,1].hist(df_enhanced.loc[background_mask & valid_col, 'collinear_mass'], bins=50, 
                   alpha=0.7, label='Background', density=true, histtype='step', linewidth=2, range=col_range)
    axes[1,1].axvline(125, color='red', linestyle='--', alpha=0.7, label='Higgs mass')
    axes[1,1].set_xlabel('Collinear Mass (GeV)')
    axes[1,1].set_ylabel('Density')
    axes[1,1].legend()
    axes[1,1].set_title('Collinear Mass Approximation')
else:
    axes[1,1].text(0.5, 0.5, 'Insufficient valid\ncollinear mass data', 
                   ha='center', va='center', transform=axes[1,1].transAxes)
    axes[1,1].set_title('Collinear Mass (Limited Data)')

plt.tight_layout()
plt.savefig('physics_features_distributions.png', dpi=150, bbox_inches='tight')
plt.savefig('physics_features_distributions.pdf', bbox_inches='tight')
plt.show()

# Save enhanced dataset
output_path = 'enhanced_higgs_features.csv'
df_enhanced.to_csv(output_path, index=false)

# Save feature metadata
feature_metadata = {
    'original_features': original_features,
    'new_features': new_features,
    'total_features': df_enhanced.shape[1],
    'top_discriminative_features': dict(top_features),
    'feature_descriptions': {
        'vis_mass_ratio': 'Visible mass normalized to Higgs mass (125 GeV)',
        'mmc_vis_ratio': 'MMC mass to visible mass ratio',
        'pt_ratio_lep_tau': 'Lepton to tau transverse momentum ratio',
        'pt_balance': 'Momentum balance between leptons and total system',
        'met_significance': 'Missing energy significance',
        'delta_phi_lep_tau': 'Angular separation between lepton and tau',
        'met_centrality': 'Missing energy direction relative to lepton-tau system',
        'vbf_score': 'Vector boson fusion topology score',
        'collinear_mass': 'Tau pair mass from collinear approximation',
        'mt_lep': 'Transverse mass of lepton-MET system',
        'mt_tau': 'Transverse mass of tau-MET system'
    }
}

with open('feature_metadata.json', 'w') as f:
    json.dump(feature_metadata, f, indent=2)

print(f'\n