import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load preprocessed data
print('Loading preprocessed ATLAS Higgs data...')
train_df = pd.read_parquet('atlas_higgs_train.parquet')
val_df = pd.read_parquet('atlas_higgs_val.parquet')

# Load metadata for feature information
with open('atlas_higgs_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f'Training data shape: {train_df.shape}')
print(f'Validation data shape: {val_df.shape}')

def engineer_physics_features(df):
    """Engineer physics-motivated features for Higgs detection"""
    df_enhanced = df.copy()
    
    # 1. Tau decay kinematics - ratios and combinations
    # Visible mass to transverse mass ratio (tau decay signature)
    df_enhanced['mass_vis_to_transmet_ratio'] = np.where(
        df['DER_mass_transverse_met_lep'] > 0,
        df['DER_mass_vis'] / df['DER_mass_transverse_met_lep'],
        -999
    )
    
    # Missing energy significance
    df_enhanced['met_significance'] = np.where(
        df['DER_pt_tot'] > 0,
        df['PRI_met'] / df['DER_pt_tot'],
        -999
    )
    
    # 2. Angular correlations and separations
    # Tau-lepton angular correlation with missing energy direction
    df_enhanced['deltar_tau_met'] = np.sqrt(
        (df['PRI_tau_phi'] - df['PRI_met_phi'])**2 + 
        (df['PRI_tau_eta'] - 0)**2  # MET has eta=0 by definition
    )
    
    # Lepton-MET angular separation
    df_enhanced['deltar_lep_met'] = np.sqrt(
        (df['PRI_lep_phi'] - df['PRI_met_phi'])**2 + 
        (df['PRI_lep_eta'] - 0)**2
    )
    
    # 3. Higgs candidate momentum features
    # Higgs transverse momentum to mass ratio
    df_enhanced['pt_h_to_mass_ratio'] = np.where(
        (df['DER_mass_vis'] > 0) & (df['DER_mass_vis'] != -999),
        df['DER_pt_h'] / df['DER_mass_vis'],
        -999
    )
    
    # Total transverse energy
    df_enhanced['total_et'] = df['PRI_tau_pt'] + df['PRI_lep_pt'] + df['PRI_met']
    
    # 4. Jet activity features (important for different production modes)
    # Jet momentum balance
    df_enhanced['jet_pt_balance'] = np.where(
        df['PRI_jet_num'] >= 2,
        np.abs(df['PRI_jet_leading_pt'] - df['PRI_jet_subleading_pt']) / 
        (df['PRI_jet_leading_pt'] + df['PRI_jet_subleading_pt']),
        -999
    )
    
    # Jet-tau angular correlations
    df_enhanced['min_deltar_jet_tau'] = np.where(
        df['PRI_jet_num'] >= 1,
        np.minimum(
            np.sqrt((df['PRI_jet_leading_eta'] - df['PRI_tau_eta'])**2 + 
                   (df['PRI_jet_leading_phi'] - df['PRI_tau_phi'])**2),
            np.where(df['PRI_jet_num'] >= 2,
                    np.sqrt((df['PRI_jet_subleading_eta'] - df['PRI_tau_eta'])**2 + 
                           (df['PRI_jet_subleading_phi'] - df['PRI_tau_phi'])**2),
                    999)
        ),
        -999
    )
    
    # 5. Mass reconstruction quality indicators
    # MMC availability flag (important discriminator)
    df_enhanced['has_mmc'] = (df['DER_mass_MMC'] != -999).astype(int)
    
    # Mass consistency check between different estimators
    df_enhanced['mass_consistency'] = np.where(
        (df['DER_mass_MMC'] != -999) & (df['DER_mass_vis'] > 0),
        np.abs(df['DER_mass_MMC'] - df['DER_mass_vis']) / df['DER_mass_vis'],
        -999
    )
    
    # 6. Centrality and event shape
    # Centrality measure
    df_enhanced['centrality'] = np.where(
        df['total_et'] > 0,
        df['DER_pt_h'] / df['total_et'],
        -999
    )
    
    # Sphericity-like measure using available momenta
    df_enhanced['momentum_imbalance'] = np.where(
        df['DER_pt_tot'] > 0,
        df['DER_pt_h'] / df['DER_pt_tot'],
        -999
    )
    
    # 7. Tau-specific observables
    # Tau isolation proxy
    df_enhanced['tau_isolation'] = np.where(
        df['PRI_jet_num'] > 0,
        df['min_deltar_jet_tau'],  # Larger = more isolated
        10.0  # No jets = well isolated
    )
    
    # Tau energy fraction
    df_enhanced['tau_energy_fraction'] = np.where(
        df['total_et'] > 0,
        df['PRI_tau_pt'] / df['total_et'],
        -999
    )
    
    # 8. VBF (Vector Boson Fusion) discriminants
    # VBF topology indicators
    df_enhanced['vbf_like'] = np.where(
        (df['PRI_jet_num'] >= 2) & (df['DER_deltaeta_jet_jet'] != -999),
        (df['DER_deltaeta_jet_jet'] > 3.5) & (df['DER_mass_jet_jet'] > 400),
        0
    ).astype(int)
    
    return df_enhanced

# Apply feature engineering
print('Engineering physics features...')
train_enhanced = engineer_physics_features(train_df)
val_enhanced = engineer_physics_features(val_df)

# Count new features
original_features = len(train_df.columns)
new_features = len(train_enhanced.columns) - original_features
print(f'Added {new_features} new physics-motivated features')
print(f'Total features: {len(train_enhanced.columns)}')

# Analyze feature importance for signal/background discrimination
print('\nAnalyzing feature discriminative power...')
signal_mask = train_enhanced['Label'] == 's'
background_mask = train_enhanced['Label'] == 'b'

# Calculate separation power for new features
new_feature_names = [col for col in train_enhanced.columns if col not in train_df.columns]
separation_scores = {}

for feature in new_feature_names:
    if feature in ['Label', 'Weight']:  # Skip non-physics columns
        continue
    
    # Only use events where feature is not missing
    valid_mask = train_enhanced[feature] != -999
    if valid_mask.sum() < 100:  # Skip if too few valid events
        continue
    
    signal_vals = train_enhanced[signal_mask & valid_mask][feature]
    background_vals = train_enhanced[background_mask & valid_mask][feature]
    
    if len(signal_vals) > 0 and len(background_vals) > 0:
        # Simple separation metric: |mean_s - mean_b| / sqrt(var_s + var_b)
        mean_diff = abs(signal_vals.mean() - background_vals.mean())
        var_sum = signal_vals.var() + background_vals.var()
        if var_sum > 0:
            separation = mean_diff / np.sqrt(var_sum)
            separation_scores[feature] = separation

# Sort features by discriminative power
top_features = sorted(separation_scores.items(), key=lambda x: x[1], reverse=True)[:10]

print('\nTop 10 most discriminative new features:')
for i, (feature, score) in enumerate(top_features, 1):
    print(f'{i:2d}. {feature:<25} separation: {score:.4f}')

# Save enhanced datasets
print('\nSaving enhanced datasets...')
train_enhanced.to_parquet('atlas_higgs_train_enhanced.parquet', index=False)
val_enhanced.to_parquet('atlas_higgs_val_enhanced.parquet', index=False)

# Create feature metadata
feature_metadata = {
    'original_features': original_features,
    'new_features': new_features,
    'total_features': len(train_enhanced.columns),
    'new_feature_names': new_feature_names,
    'top_discriminative_features': dict(top_features),
    'feature_descriptions': {
        'mass_vis_to_transmet_ratio': 'Visible to transverse mass ratio (tau signature)',
        'met_significance': 'Missing energy significance relative to total pt',
        'deltar_tau_met': 'Angular separation between tau and missing energy',
        'pt_h_to_mass_ratio': 'Higgs pt to mass ratio (boost indicator)',
        'jet_pt_balance': 'Momentum balance between leading jets',
        'has_mmc': 'MMC mass reconstruction availability flag',
        'centrality': 'Event centrality measure',
        'tau_isolation': 'Tau isolation from jets',
        'vbf_like': 'VBF topology indicator'
    }
}

with open('physics_features_metadata.json', 'w') as f:
    json.dump(feature_metadata, f, indent=2)

# Summary statistics
print(f'\nFeature engineering completed:')
print(f'RESULT:original_features={original_features}')
print(f'RESULT:new_features={new_features}')
print(f'RESULT:total_features={len(train_enhanced.columns)}')
print(f'RESULT:train_enhanced_file=atlas_higgs_train_enhanced.parquet')
print(f'RESULT:val_enhanced_file=atlas_higgs_val_enhanced.parquet')
print(f'RESULT:metadata_file=physics_features_metadata.json')
print(f'RESULT:top_feature={top_features[0][0] if top_features else "none"}')
print(f'RESULT:top_separation_score={top_features[0][1]:.4f if top_features else 0.0}')

print('\nPhysics-motivated feature engineering completed successfully!')