import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Load preprocessed data
df = pd.read_parquet('preprocessed_physics_data.csv')
with open('preprocessing_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Loaded dataset with shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Physics-motivated feature engineering for H->tau tau
print("\nEngineering physics-motivated features...")

# 1. Kinematic ratios and normalized variables
df['pt_ratio_tau_lep'] = df['PRI_tau_pt'] / (df['PRI_lep_pt'] + 1e-6)
df['eta_diff_tau_lep'] = np.abs(df['PRI_tau_eta'] - df['PRI_lep_eta'])
df['phi_diff_tau_lep'] = np.abs(df['PRI_tau_phi'] - df['PRI_lep_phi'])
df['phi_diff_tau_lep'] = np.minimum(df['phi_diff_tau_lep'], 2*np.pi - df['phi_diff_tau_lep'])

# 2. Enhanced angular separations
df['delta_R_tau_lep_enhanced'] = np.sqrt(df['eta_diff_tau_lep']**2 + df['phi_diff_tau_lep']**2)
df['cos_theta_tau_lep'] = np.cos(df['phi_diff_tau_lep'])

# 3. Transverse mass combinations
valid_met = df['PRI_met'] > 0
df['mt_tau_met'] = np.where(valid_met, 
    np.sqrt(2 * df['PRI_tau_pt'] * df['PRI_met'] * (1 - np.cos(df['PRI_tau_phi'] - df['PRI_met_phi']))), 0)
df['mt_lep_met'] = np.where(valid_met,
    np.sqrt(2 * df['PRI_lep_pt'] * df['PRI_met'] * (1 - np.cos(df['PRI_lep_phi'] - df['PRI_met_phi']))), 0)

# 4. Tau-specific observables
df['tau_isolation'] = df['PRI_tau_pt'] / (df['DER_pt_tot'] + 1e-6)
df['visible_mass_ratio'] = df['DER_mass_vis'] / 125.0  # Ratio to Higgs mass
df['transverse_mass_ratio'] = df['DER_mass_transverse_met_lep'] / 125.0

# 5. Jet-related features (when jets are present)
has_jets = df['PRI_jet_num'] > 0
df['leading_jet_pt_ratio'] = np.where(has_jets, df['PRI_jet_leading_pt'] / df['DER_pt_tot'], 0)
df['subleading_jet_pt_ratio'] = np.where(df['PRI_jet_num'] > 1, 
    df['PRI_jet_subleading_pt'] / df['DER_pt_tot'], 0)

# 6. Centrality and sphericity measures
df['centrality'] = (df['PRI_tau_pt'] + df['PRI_lep_pt']) / df['DER_pt_tot']
df['pt_asymmetry'] = np.abs(df['PRI_tau_pt'] - df['PRI_lep_pt']) / (df['PRI_tau_pt'] + df['PRI_lep_pt'] + 1e-6)

# 7. Missing energy significance
df['met_significance'] = df['PRI_met'] / np.sqrt(df['DER_sum_pt'] + 1e-6)
df['met_centrality'] = df['PRI_met'] / (df['DER_sum_pt'] + 1e-6)

# 8. Collinear mass approximation (when MMC is missing)
mmc_missing = df['DER_mass_MMC'] == -999.0
valid_collinear = mmc_missing & (df['PRI_met'] > 0)
df['collinear_mass'] = np.where(valid_collinear,
    df['DER_mass_vis'] / np.sqrt(df['PRI_tau_pt'] * df['PRI_lep_pt'] / ((df['PRI_tau_pt'] + df['PRI_met']) * (df['PRI_lep_pt'] + df['PRI_met']) + 1e-6)), 0)

# 9. Higgs candidate momentum features
df['higgs_pt_estimate'] = np.sqrt((df['PRI_tau_pt'] * np.cos(df['PRI_tau_phi']) + df['PRI_lep_pt'] * np.cos(df['PRI_lep_phi']) + df['PRI_met'] * np.cos(df['PRI_met_phi']))**2 + 
                                 (df['PRI_tau_pt'] * np.sin(df['PRI_tau_phi']) + df['PRI_lep_pt'] * np.sin(df['PRI_lep_phi']) + df['PRI_met'] * np.sin(df['PRI_met_phi']))**2)

# 10. Tau decay topology features
df['tau_lep_opening_angle'] = np.arccos(np.clip(
    np.cos(df['eta_diff_tau_lep']) * np.cos(df['phi_diff_tau_lep']) + 
    np.sin(df['eta_diff_tau_lep']) * np.sin(df['phi_diff_tau_lep']), -1, 1))

# Count new features
original_features = set(metadata.get('feature_columns', []))
new_features = [col for col in df.columns if col not in original_features]
print(f"\nCreated {len(new_features)} new physics-motivated features:")
for feat in new_features[:10]:  # Show first 10
    print(f"  - {feat}")
if len(new_features) > 10:
    print(f"  ... and {len(new_features) - 10} more")

# Handle any infinite or NaN values
df = df.replace([np.inf, -np.inf], np.nan)
nan_counts = df.isnull().sum()
if nan_counts.sum() > 0:
    print(f"\nHandling {nan_counts.sum()} NaN values created during feature engineering")
    df = df.fillna(0)  # Fill with 0 for physics features

# Create feature importance plot based on correlation with label
if 'Label' in df.columns:
    signal_data = df[df['Label'] == 's']
    background_data = df[df['Label'] == 'b']
    
    # Calculate discriminative power for new features
    discrimination_scores = {}
    for feat in new_features:
        if df[feat].std() > 0:  # Avoid constant features
            signal_mean = signal_data[feat].mean()
            background_mean = background_data[feat].mean()
            pooled_std = np.sqrt((signal_data[feat].var() + background_data[feat].var()) / 2)
            if pooled_std > 0:
                discrimination_scores[feat] = abs(signal_mean - background_mean) / pooled_std
    
    # Plot top discriminative features
    top_features = sorted(discrimination_scores.items(), key=lambda x: x[1], reverse=True)[:15]
    
    plt.figure(figsize=(12, 8))
    features, scores = zip(*top_features)
    plt.barh(range(len(features)), scores)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Discrimination Score (|μ_s - μ_b| / σ_pooled)')
    plt.title('Top Physics-Motivated Features for H→ττ Discrimination')
    plt.tight_layout()
    plt.savefig('physics_features_discrimination.png', dpi=150, bbox_inches='tight')
    plt.savefig('physics_features_discrimination.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"\nTop 5 discriminative features:")
    for feat, score in top_features[:5]:
        print(f"  {feat}: {score:.3f}")

# Save enhanced dataset
output_file = 'physics_enhanced_features.parquet'
df.to_parquet(output_file, index=False)

# Create metadata for enhanced features
enhanced_metadata = {
    'original_shape': metadata.get('dataset_shape', 'unknown'),
    'enhanced_shape': f"{df.shape[0]}x{df.shape[1]}",
    'original_features': len(original_features),
    'new_physics_features': len(new_features),
    'total_features': df.shape[1],
    'new_feature_names': new_features,
    'feature_categories': {
        'kinematic_ratios': ['pt_ratio_tau_lep', 'eta_diff_tau_lep', 'phi_diff_tau_lep'],
        'angular_separations': ['delta_R_tau_lep_enhanced', 'cos_theta_tau_lep', 'tau_lep_opening_angle'],
        'transverse_masses': ['mt_tau_met', 'mt_lep_met'],
        'tau_observables': ['tau_isolation', 'visible_mass_ratio', 'transverse_mass_ratio'],
        'jet_features': ['leading_jet_pt_ratio', 'subleading_jet_pt_ratio'],
        'topology_features': ['centrality', 'pt_asymmetry', 'met_significance'],
        'higgs_features': ['collinear_mass', 'higgs_pt_estimate']
    },
    'discrimination_scores': dict(top_features) if 'Label' in df.columns else {}
}

with open('physics_features_metadata.json', 'w') as f:
    json.dump(enhanced_metadata, f, indent=2)

print(f"\nPhysics feature engineering completed:")
print(f"RESULT:enhanced_dataset_shape={df.shape[0]}x{df.shape[1]}")
print(f"RESULT:new_physics_features_count={len(new_features)}")
print(f"RESULT:total_features_count={df.shape[1]}")
print(f"RESULT:output_file={output_file}")
print(f"RESULT:metadata_file=physics_features_metadata.json")
if 'Label' in df.columns:
    print(f"RESULT:discrimination_plot=physics_features_discrimination.png")
    print(f"RESULT:top_discriminative_feature={top_features[0][0]}")
    print(f"RESULT:top_discrimination_score={top_features[0][1]:.4f}")
print("RESULT:feature_categories=kinematic_ratios,angular_separations,transverse_masses,tau_observables,jet_features,topology_features,higgs_features")