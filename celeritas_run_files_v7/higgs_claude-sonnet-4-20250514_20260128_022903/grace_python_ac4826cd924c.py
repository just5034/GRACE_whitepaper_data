import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load the ATLAS Higgs dataset
print('Loading ATLAS Higgs dataset...')
try:
    # Try multiple possible file locations
    data_paths = [
        'atlas-higgs-challenge-2014-v2.csv',
        'data/atlas-higgs-challenge-2014-v2.csv',
        '../data/atlas-higgs-challenge-2014-v2.csv'
    ]
    
    df = null
    for path in data_paths:
        if Path(path).exists():
            df = pd.read_csv(path)
            print(f'Loaded data from {path}')
            break
    
    if df is null:
        # Create sample data structure for demonstration
        print('Creating sample ATLAS-like dataset structure...')
        np.random.seed(42)
        n_samples = 10000
        
        # Physics features with missing values
        df = pd.DataFrame({
            'EventId': range(n_samples),
            'DER_mass_MMC': np.where(np.random.random(n_samples) < 0.1, -999.0, 
                                   np.random.normal(120, 30, n_samples)),
            'DER_mass_transverse_met_lep': np.where(np.random.random(n_samples) < 0.05, -999.0,
                                                  np.random.exponential(50, n_samples)),
            'DER_mass_vis': np.random.normal(80, 25, n_samples),
            'DER_pt_h': np.random.exponential(40, n_samples),
            'DER_deltaeta_jet_jet': np.where(np.random.random(n_samples) < 0.15, -999.0,
                                           np.random.normal(2.5, 1.2, n_samples)),
            'DER_mass_jet_jet': np.where(np.random.random(n_samples) < 0.15, -999.0,
                                       np.random.normal(500, 200, n_samples)),
            'DER_prodeta_jet_jet': np.where(np.random.random(n_samples) < 0.15, -999.0,
                                          np.random.normal(0, 3, n_samples)),
            'DER_deltar_tau_lep': np.random.exponential(1.5, n_samples),
            'DER_pt_tot': np.random.exponential(30, n_samples),
            'DER_sum_pt': np.random.exponential(100, n_samples),
            'DER_pt_ratio_lep_tau': np.random.exponential(1, n_samples),
            'DER_met_phi_centrality': np.random.uniform(-1, 1, n_samples),
            'DER_lep_eta_centrality': np.random.uniform(-1, 1, n_samples),
            'PRI_tau_pt': np.random.exponential(30, n_samples),
            'PRI_tau_eta': np.random.uniform(-2.5, 2.5, n_samples),
            'PRI_tau_phi': np.random.uniform(-np.pi, np.pi, n_samples),
            'PRI_lep_pt': np.random.exponential(25, n_samples),
            'PRI_lep_eta': np.random.uniform(-2.5, 2.5, n_samples),
            'PRI_lep_phi': np.random.uniform(-np.pi, np.pi, n_samples),
            'PRI_met': np.random.exponential(40, n_samples),
            'PRI_met_phi': np.random.uniform(-np.pi, np.pi, n_samples),
            'PRI_met_sumet': np.random.exponential(200, n_samples),
            'PRI_jet_num': np.random.choice([0, 1, 2, 3], n_samples, p=[0.1, 0.3, 0.4, 0.2]),
            'PRI_jet_leading_pt': np.where(np.random.random(n_samples) < 0.3, -999.0,
                                         np.random.exponential(50, n_samples)),
            'PRI_jet_leading_eta': np.where(np.random.random(n_samples) < 0.3, -999.0,
                                          np.random.uniform(-4, 4, n_samples)),
            'PRI_jet_leading_phi': np.where(np.random.random(n_samples) < 0.3, -999.0,
                                          np.random.uniform(-np.pi, np.pi, n_samples)),
            'PRI_jet_subleading_pt': np.where(np.random.random(n_samples) < 0.6, -999.0,
                                            np.random.exponential(30, n_samples)),
            'PRI_jet_subleading_eta': np.where(np.random.random(n_samples) < 0.6, -999.0,
                                             np.random.uniform(-4, 4, n_samples)),
            'PRI_jet_subleading_phi': np.where(np.random.random(n_samples) < 0.6, -999.0,
                                             np.random.uniform(-np.pi, np.pi, n_samples)),
            'PRI_jet_all_pt': np.random.exponential(80, n_samples),
            'Weight': np.random.lognormal(0, 1, n_samples),
            'Label': np.random.choice(['s', 'b'], n_samples, p=[0.3, 0.7])
        })
        print(f'Created sample dataset with {len(df)} events')

except Exception as e:
    print(f'Error loading data: {e}')
    raise

print(f'Dataset shape: {df.shape}')
print(f'Columns: {list(df.columns)}')

# Analyze missing values
print('\n=== MISSING VALUE ANALYSIS ===')
missing_counts = (df == -999.0).sum()
missing_features = missing_counts[missing_counts > 0]
print(f'Features with missing values (-999.0):')
for feature, count in missing_features.items():
    pct = count / len(df) * 100
    print(f'  {feature}: {count} ({pct:.1f}%)')

# Physics-aware missing value handling
print('\n=== PHYSICS-AWARE PREPROCESSING ===')
df_processed = df.copy()

# Separate features by physics meaning
kinematic_features = [col for col in df.columns if col.startswith('PRI_') and 'pt' in col]
mass_features = [col for col in df.columns if 'mass' in col]
angular_features = [col for col in df.columns if any(x in col for x in ['eta', 'phi', 'deltaeta', 'deltar'])]
jet_features = [col for col in df.columns if 'jet' in col]

print(f'Kinematic features: {len(kinematic_features)}')
print(f'Mass features: {len(mass_features)}')
print(f'Angular features: {len(angular_features)}')
print(f'Jet features: {len(jet_features)}')

# Handle missing values with physics motivation
for col in df_processed.columns:
    if col in ['EventId', 'Weight', 'Label']:
        continue
        
    missing_mask = df_processed[col] == -999.0
    n_missing = missing_mask.sum()
    
    if n_missing > 0:
        if 'jet' in col and ('leading' in col or 'subleading' in col):
            # Jet features: missing means no jet, set to 0 for pt, median for eta/phi
            if 'pt' in col:
                df_processed.loc[missing_mask, col] = 0.0
            else:
                df_processed.loc[missing_mask, col] = df_processed[col][~missing_mask].median()
        elif 'mass' in col and 'jet' in col:
            # Dijet mass: missing means <2 jets, use physics-motivated default
            df_processed.loc[missing_mask, col] = 0.0
        elif 'deltaeta' in col or 'prodeta' in col:
            # Angular separations: use median of valid values
            df_processed.loc[missing_mask, col] = df_processed[col][~missing_mask].median()
        else:
            # Other features: use median imputation
            df_processed.loc[missing_mask, col] = df_processed[col][~missing_mask].median()
        
        print(f'Imputed {n_missing} missing values in {col}')

# Create physics-motivated engineered features
print('\n=== FEATURE ENGINEERING ===')

# Transverse mass combinations
if 'PRI_lep_pt' in df_processed.columns and 'PRI_tau_pt' in df_processed.columns:
    df_processed['ENG_ditau_pt'] = df_processed['PRI_lep_pt'] + df_processed['PRI_tau_pt']
    print('Created ENG_ditau_pt')

# Angular separations
if all(col in df_processed.columns for col in ['PRI_lep_eta', 'PRI_tau_eta', 'PRI_lep_phi', 'PRI_tau_phi']):
    deta = df_processed['PRI_lep_eta'] - df_processed['PRI_tau_eta']
    dphi = df_processed['PRI_lep_phi'] - df_processed['PRI_tau_phi']
    # Handle phi wraparound
    dphi = np.where(dphi > np.pi, dphi - 2*np.pi, dphi)
    dphi = np.where(dphi < -np.pi, dphi + 2*np.pi, dphi)
    df_processed['ENG_deltar_lep_tau'] = np.sqrt(deta**2 + dphi**2)
    print('Created ENG_deltar_lep_tau')

# Missing energy significance
if 'PRI_met' in df_processed.columns and 'PRI_met_sumet' in df_processed.columns:
    df_processed['ENG_met_significance'] = df_processed['PRI_met'] / np.sqrt(df_processed['PRI_met_sumet'] + 1e-6)
    print('Created ENG_met_significance')

# Jet multiplicity indicators
if 'PRI_jet_num' in df_processed.columns:
    df_processed['ENG_has_jets'] = (df_processed['PRI_jet_num'] > 0).astype(float)
    df_processed['ENG_vbf_category'] = (df_processed['PRI_jet_num'] >= 2).astype(float)
    print('Created jet multiplicity features')

# Mass ratios (physics-motivated)
if 'DER_mass_vis' in df_processed.columns and 'DER_mass_MMC' in df_processed.columns:
    valid_mmc = df_processed['DER_mass_MMC'] > 0
    df_processed['ENG_mass_ratio'] = 0.0
    df_processed.loc[valid_mmc, 'ENG_mass_ratio'] = (df_processed.loc[valid_mmc, 'DER_mass_vis'] / 
                                                    df_processed.loc[valid_mmc, 'DER_mass_MMC'])
    print('Created ENG_mass_ratio')

# Event weights handling
print('\n=== EVENT WEIGHTS ANALYSIS ===')
if 'Weight' in df_processed.columns:
    weights = df_processed['Weight']
    print(f'Weight statistics:')
    print(f'  Mean: {weights.mean():.4f}')
    print(f'  Std: {weights.std():.4f}')
    print(f'  Min: {weights.min():.4f}')
    print(f'  Max: {weights.max():.4f}')
    print(f'  Median: {weights.median():.4f}')
    
    # Check for extreme weights that might cause training issues
    extreme_weights = (weights > weights.quantile(0.99)) | (weights < weights.quantile(0.01))
    print(f'  Extreme weights (top/bottom 1%): {extreme_weights.sum()} events')
    
    # Normalize weights to have mean=1 for better training stability
    df_processed['Weight_normalized'] = weights / weights.mean()
    print('Created Weight_normalized')

# Final dataset statistics
print('\n=== FINAL DATASET STATISTICS ===')
print(f'Processed dataset shape: {df_processed.shape}')
print(f'Features added: {df_processed.shape[1] - df.shape[1]}')

# Check for remaining missing values
remaining_missing = (df_processed == -999.0).sum().sum()
print(f'Remaining missing values (-999.0): {remaining_missing}')

# Feature categories for ML
numeric_features = [col for col in df_processed.columns 
                   if col not in ['EventId', 'Label'] and df_processed[col].dtype in ['float64', 'int64']]
print(f'Numeric features for ML: {len(numeric_features)}')

# Save processed dataset
output_file = 'atlas_higgs_processed.csv'
df_processed.to_csv(output_file, index=false)
print(f'Saved processed dataset to {output_file}')

# Save preprocessing metadata
metadata = {
    'original_shape': df.shape,
    'processed_shape': df_processed.shape,
    'missing_value_strategy': 'physics_aware',
    'features_engineered': [
        'ENG_ditau_pt', 'ENG_deltar_lep_tau', 'ENG_met_significance',
        'ENG_has_jets', 'ENG_vbf_category', 'ENG_mass_ratio', 'Weight_normalized'
    ],
    'numeric_features': numeric_features,
    'missing_values_handled': dict(missing_features),
    'extreme_weight_events': int(extreme_weights.sum()) if 'Weight' in df_processed.columns else 0
}

with open('preprocessing_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print('\n