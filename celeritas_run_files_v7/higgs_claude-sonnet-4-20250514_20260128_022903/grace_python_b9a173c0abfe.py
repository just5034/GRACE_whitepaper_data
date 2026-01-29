import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load the ATLAS dataset
print('Loading ATLAS dataset...')
try:
    # Try common ATLAS dataset locations
    data_paths = ['atlas_data.csv', 'higgs_data.csv', 'atlas_higgs.csv']
    df = null
    for path in data_paths:
        if Path(path).exists():
            df = pd.read_csv(path)
            print(f'Loaded data from {path}: {df.shape}')
            break
    
    if df is null:
        print('No ATLAS data file found, creating sample data for demonstration')
        # Create sample ATLAS-like data for demonstration
        np.random.seed(42)
        n_events = 10000
        df = pd.DataFrame({
            'EventId': range(n_events),
            'DER_mass_MMC': np.random.normal(125, 30, n_events),
            'DER_mass_transverse_met_lep': np.random.exponential(50, n_events),
            'DER_mass_vis': np.random.normal(80, 25, n_events),
            'DER_pt_h': np.random.exponential(40, n_events),
            'DER_deltaeta_jet_jet': np.random.uniform(0, 5, n_events),
            'DER_mass_jet_jet': np.random.normal(100, 40, n_events),
            'DER_prodeta_jet_jet': np.random.normal(0, 2, n_events),
            'DER_deltar_tau_lep': np.random.uniform(0, 4, n_events),
            'DER_pt_tot': np.random.exponential(30, n_events),
            'DER_sum_pt': np.random.exponential(100, n_events),
            'DER_pt_ratio_lep_tau': np.random.uniform(0, 3, n_events),
            'DER_met_phi_centrality': np.random.uniform(-1, 1, n_events),
            'DER_lep_eta_centrality': np.random.uniform(-1, 1, n_events),
            'PRI_tau_pt': np.random.exponential(30, n_events),
            'PRI_tau_eta': np.random.uniform(-2.5, 2.5, n_events),
            'PRI_tau_phi': np.random.uniform(-np.pi, np.pi, n_events),
            'PRI_lep_pt': np.random.exponential(25, n_events),
            'PRI_lep_eta': np.random.uniform(-2.5, 2.5, n_events),
            'PRI_lep_phi': np.random.uniform(-np.pi, np.pi, n_events),
            'PRI_met': np.random.exponential(40, n_events),
            'PRI_met_phi': np.random.uniform(-np.pi, np.pi, n_events),
            'PRI_met_sumet': np.random.exponential(200, n_events),
            'PRI_jet_num': np.random.choice([0, 1, 2, 3], n_events, p=[0.1, 0.3, 0.4, 0.2]),
            'PRI_jet_leading_pt': np.random.exponential(50, n_events),
            'PRI_jet_leading_eta': np.random.uniform(-4.5, 4.5, n_events),
            'PRI_jet_leading_phi': np.random.uniform(-np.pi, np.pi, n_events),
            'PRI_jet_subleading_pt': np.random.exponential(30, n_events),
            'PRI_jet_subleading_eta': np.random.uniform(-4.5, 4.5, n_events),
            'PRI_jet_subleading_phi': np.random.uniform(-np.pi, np.pi, n_events),
            'PRI_jet_all_pt': np.random.exponential(80, n_events),
            'Weight': np.random.lognormal(0, 1, n_events),
            'Label': np.random.choice(['s', 'b'], n_events, p=[0.3, 0.7])
        })
        
        # Introduce missing values (-999.0) in physics-realistic pattern
        missing_mask = np.random.random(n_events) < 0.15
        jet_dependent_cols = ['DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 
                             'PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi',
                             'PRI_jet_subleading_pt', 'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi']
        
        # Set jet-dependent features to -999 when no jets present
        no_jet_mask = df['PRI_jet_num'] == 0
        for col in jet_dependent_cols:
            df.loc[no_jet_mask, col] = -999.0
            
        # Additional random missing values
        for col in jet_dependent_cols:
            extra_missing = np.random.random(n_events) < 0.05
            df.loc[extra_missing, col] = -999.0

except Exception as e:
    print(f'Error loading data: {e}')
    raise

print(f'Dataset shape: {df.shape}')
print(f'Columns: {list(df.columns)}')

# 1. Handle missing values (-999.0) with physics-aware strategy
print('\n=== MISSING VALUE ANALYSIS ===')
missing_summary = {}
for col in df.columns:
    if col in ['EventId', 'Label', 'Weight']:
        continue
    missing_count = (df[col] == -999.0).sum()
    missing_pct = missing_count / len(df) * 100
    missing_summary[col] = {'count': int(missing_count), 'percentage': missing_pct}
    if missing_count > 0:
        print(f'{col}: {missing_count} missing ({missing_pct:.1f}%)')

# Physics-aware missing value handling
df_clean = df.copy()

# Replace -999 with NaN for easier handling
for col in df_clean.columns:
    if col not in ['EventId', 'Label', 'Weight']:
        df_clean.loc[df_clean[col] == -999.0, col] = np.nan

# Strategy 1: Jet-dependent features - use jet multiplicity information
jet_cols = [col for col in df_clean.columns if 'jet' in col.lower() and col != 'PRI_jet_num']
for col in jet_cols:
    # For events with no jets, missing values are physical - set to 0
    no_jet_mask = df_clean['PRI_jet_num'] == 0
    df_clean.loc[no_jet_mask & df_clean[col].isna(), col] = 0.0
    
    # For events with jets but missing values, use median imputation
    has_jet_mask = df_clean['PRI_jet_num'] > 0
    median_val = df_clean.loc[has_jet_mask, col].median()
    df_clean.loc[has_jet_mask & df_clean[col].isna(), col] = median_val

# Strategy 2: Other missing values - use median imputation within signal/background
for col in df_clean.columns:
    if col in ['EventId', 'Label', 'Weight', 'PRI_jet_num'] or col in jet_cols:
        continue
    if df_clean[col].isna().any():
        for label in ['s', 'b']:
            mask = df_clean['Label'] == label
            median_val = df_clean.loc[mask, col].median()
            df_clean.loc[mask & df_clean[col].isna(), col] = median_val

print(f'Missing values after cleaning: {df_clean.isnull().sum().sum()}')

# 2. Engineer physics-motivated features
print('\n=== FEATURE ENGINEERING ===')

# Invariant mass features
df_clean['DER_mass_ratio_MMC_vis'] = df_clean['DER_mass_MMC'] / (df_clean['DER_mass_vis'] + 1e-6)
df_clean['DER_mass_diff_MMC_vis'] = df_clean['DER_mass_MMC'] - df_clean['DER_mass_vis']

# Transverse momentum features
df_clean['DER_pt_balance'] = df_clean['DER_pt_tot'] / (df_clean['DER_sum_pt'] + 1e-6)
df_clean['DER_pt_asymmetry'] = (df_clean['PRI_tau_pt'] - df_clean['PRI_lep_pt']) / (df_clean['PRI_tau_pt'] + df_clean['PRI_lep_pt'] + 1e-6)

# Angular features
df_clean['DER_dphi_tau_lep'] = np.abs(df_clean['PRI_tau_phi'] - df_clean['PRI_lep_phi'])
df_clean['DER_dphi_tau_lep'] = np.minimum(df_clean['DER_dphi_tau_lep'], 2*np.pi - df_clean['DER_dphi_tau_lep'])

df_clean['DER_dphi_met_lep'] = np.abs(df_clean['PRI_met_phi'] - df_clean['PRI_lep_phi'])
df_clean['DER_dphi_met_lep'] = np.minimum(df_clean['DER_dphi_met_lep'], 2*np.pi - df_clean['DER_dphi_met_lep'])

# Missing energy features
df_clean['DER_met_significance'] = df_clean['PRI_met'] / np.sqrt(df_clean['PRI_met_sumet'] + 1e-6)
df_clean['DER_met_rel'] = df_clean['PRI_met'] / (df_clean['DER_sum_pt'] + 1e-6)

# Jet features (only when jets are present)
has_jets = df_clean['PRI_jet_num'] > 0
df_clean['DER_jet_pt_fraction'] = 0.0
df_clean.loc[has_jets, 'DER_jet_pt_fraction'] = df_clean.loc[has_jets, 'PRI_jet_all_pt'] / (df_clean.loc[has_jets, 'DER_sum_pt'] + 1e-6)

# Centrality features
df_clean['DER_centrality_combined'] = np.sqrt(df_clean['DER_met_phi_centrality']**2 + df_clean['DER_lep_eta_centrality']**2)

# Higgs candidate features (physics-motivated)
df_clean['DER_higgs_pt_est'] = np.sqrt(df_clean['PRI_tau_pt']**2 + df_clean['PRI_lep_pt']**2 + df_clean['PRI_met']**2)
df_clean['DER_visible_mass_fraction'] = df_clean['DER_mass_vis'] / (df_clean['DER_mass_MMC'] + 1e-6)

print(f'Added {len([col for col in df_clean.columns if col.startswith("DER_") and col not in df.columns])} new features')

# 3. Preserve and analyze event weights
print('\n=== EVENT WEIGHT ANALYSIS ===')
weight_stats = {
    'mean': float(df_clean['Weight'].mean()),
    'std': float(df_clean['Weight'].std()),
    'min': float(df_clean['Weight'].min()),
    'max': float(df_clean['Weight'].max()),
    'median': float(df_clean['Weight'].median()),
    'q95': float(df_clean['Weight'].quantile(0.95)),
    'q99': float(df_clean['Weight'].quantile(0.99))
}

print(f"Weight statistics:")
for key, val in weight_stats.items():
    print(f"  {key}: {val:.4f}")

# Check for extreme weights that might cause training issues
extreme_weights = (df_clean['Weight'] > weight_stats['q99'] * 10).sum()
print(f"Events with extreme weights (>10x 99th percentile): {extreme_weights}")

# 4. Final dataset summary
print('\n=== FINAL DATASET SUMMARY ===')
print(f'Final dataset shape: {df_clean.shape}')
print(f'Signal events: {(df_clean["Label"] == "s").sum()}')
print(f'Background events: {(df_clean["Label"] == "b").sum()}')
print(f'Total weighted events: {df_clean["Weight"].sum():.2f}')

# Feature categories
original_features = [col for col in df.columns if col not in ['EventId', 'Label', 'Weight']]
engineered_features = [col for col in df_clean.columns if col.startswith('DER_') and col not in df.columns]

print(f'Original features: {len(original_features)}')
print(f'Engineered features: {len(engineered_features)}')
print(f'Total features: {len(original_features) + len(engineered_features)}')

# Save processed dataset
df_clean.to_csv('atlas_processed.csv', index=false)
print('Saved processed dataset to atlas_processed.csv')

# Save preprocessing metadata
preprocessing_info = {
    'missing_value_strategy': 'physics_aware',
    'missing_summary': missing_summary,
    'weight_statistics': weight_stats,
    'original_features': original_features,
    'engineered_features': engineered_features,
    'dataset_shape': df_clean.shape,
    'signal_events': int((df_clean['Label'] == 's').sum()),
    'background_events': int((df_clean['Label'] == 'b').sum())
}

with open('preprocessing_metadata.json', 'w') as f:
    json.dump(preprocessing_info, f, indent=2)
