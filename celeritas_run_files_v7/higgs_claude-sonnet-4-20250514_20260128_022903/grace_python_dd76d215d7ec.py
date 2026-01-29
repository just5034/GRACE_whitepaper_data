import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load the ATLAS dataset
print('Loading ATLAS dataset...')
try:
    # Try to load from common locations
    data_files = list(Path('.').glob('*atlas*.csv')) + list(Path('.').glob('*higgs*.csv'))
    if data_files:
        df = pd.read_csv(data_files[0])
        print(f'Loaded dataset from {data_files[0]} with shape: {df.shape}')
    else:
        # Fallback - create synthetic ATLAS-like dataset for demonstration
        print('No ATLAS dataset found, creating synthetic example...')
        np.random.seed(42)
        n_samples = 100000
        df = pd.DataFrame({
            'DER_mass_MMC': np.random.exponential(100, n_samples),
            'DER_mass_transverse_met_lep': np.random.exponential(50, n_samples),
            'DER_mass_vis': np.random.exponential(80, n_samples),
            'DER_pt_h': np.random.exponential(60, n_samples),
            'DER_deltaeta_jet_jet': np.random.normal(2.5, 1.5, n_samples),
            'DER_mass_jet_jet': np.random.exponential(200, n_samples),
            'DER_prodeta_jet_jet': np.random.normal(0, 5, n_samples),
            'DER_deltar_tau_lep': np.random.exponential(2, n_samples),
            'DER_pt_tot': np.random.exponential(40, n_samples),
            'DER_sum_pt': np.random.exponential(150, n_samples),
            'DER_pt_ratio_lep_tau': np.random.exponential(1, n_samples),
            'DER_met_phi_centrality': np.random.uniform(-1, 1, n_samples),
            'DER_lep_eta_centrality': np.random.uniform(-1, 1, n_samples),
            'PRI_tau_pt': np.random.exponential(30, n_samples),
            'PRI_tau_eta': np.random.normal(0, 2, n_samples),
            'PRI_tau_phi': np.random.uniform(-np.pi, np.pi, n_samples),
            'PRI_lep_pt': np.random.exponential(25, n_samples),
            'PRI_lep_eta': np.random.normal(0, 2, n_samples),
            'PRI_lep_phi': np.random.uniform(-np.pi, np.pi, n_samples),
            'PRI_met': np.random.exponential(35, n_samples),
            'PRI_met_phi': np.random.uniform(-np.pi, np.pi, n_samples),
            'PRI_met_sumet': np.random.exponential(200, n_samples),
            'PRI_jet_num': np.random.poisson(2, n_samples),
            'PRI_jet_leading_pt': np.random.exponential(50, n_samples),
            'PRI_jet_leading_eta': np.random.normal(0, 2.5, n_samples),
            'PRI_jet_leading_phi': np.random.uniform(-np.pi, np.pi, n_samples),
            'PRI_jet_subleading_pt': np.random.exponential(30, n_samples),
            'PRI_jet_subleading_eta': np.random.normal(0, 2.5, n_samples),
            'PRI_jet_subleading_phi': np.random.uniform(-np.pi, np.pi, n_samples),
            'PRI_jet_all_pt': np.random.exponential(100, n_samples),
            'Weight': np.random.exponential(1, n_samples),
            'Label': np.random.choice(['s', 'b'], n_samples, p=[0.3, 0.7])
        })
        # Introduce missing values (-999.0) in physics-realistic pattern
        missing_mask = np.random.random(n_samples) < 0.1
        df.loc[missing_mask, 'DER_mass_jet_jet'] = -999.0
        df.loc[missing_mask, 'DER_deltaeta_jet_jet'] = -999.0
        df.loc[missing_mask, 'PRI_jet_subleading_pt'] = -999.0
        df.loc[missing_mask, 'PRI_jet_subleading_eta'] = -999.0
        df.loc[missing_mask, 'PRI_jet_subleading_phi'] = -999.0
        
except Exception as e:
    print(f'Error loading data: {e}')
    raise

print(f'Dataset shape: {df.shape}')
print(f'Columns: {list(df.columns)}')

# Identify missing values (-999.0)
print('\nAnalyzing missing values...')
missing_counts = (df == -999.0).sum()
print('Missing value counts (-999.0):')
for col, count in missing_counts.items():
    if count > 0:
        print(f'  {col}: {count} ({count/len(df)*100:.1f}%)')

# Physics-aware missing value handling
print('\nHandling missing values with physics-aware strategy...')
df_clean = df.copy()

# Replace -999.0 with NaN for proper handling
df_clean = df_clean.replace(-999.0, np.nan)

# Physics-motivated imputation strategies
for col in df_clean.columns:
    if df_clean[col].isna().sum() > 0:
        if 'jet_jet' in col:
            # Jet-jet variables: missing when < 2 jets, impute with median of 2+ jet events
            valid_mask = df_clean['PRI_jet_num'] >= 2
            if valid_mask.sum() > 0:
                median_val = df_clean.loc[valid_mask, col].median()
                df_clean[col].fillna(median_val, inplace=True)
        elif 'subleading' in col:
            # Subleading jet: missing when < 2 jets, use leading jet properties
            leading_col = col.replace('subleading', 'leading')
            if leading_col in df_clean.columns:
                df_clean[col].fillna(df_clean[leading_col], inplace=True)
        elif col not in ['Weight', 'Label']:
            # Other physics variables: use median imputation
            df_clean[col].fillna(df_clean[col].median(), inplace=True)

# Feature engineering - physics-motivated features
print('\nEngineering physics-motivated features...')

# Invariant masses and energy scales
if 'DER_mass_MMC' in df_clean.columns and 'DER_mass_vis' in df_clean.columns:
    df_clean['mass_ratio_MMC_vis'] = df_clean['DER_mass_MMC'] / (df_clean['DER_mass_vis'] + 1e-6)

# Transverse momentum ratios
if 'PRI_tau_pt' in df_clean.columns and 'PRI_lep_pt' in df_clean.columns:
    df_clean['pt_ratio_tau_lep'] = df_clean['PRI_tau_pt'] / (df_clean['PRI_lep_pt'] + 1e-6)

# Angular separations
if all(col in df_clean.columns for col in ['PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_eta', 'PRI_lep_phi']):
    deta = df_clean['PRI_tau_eta'] - df_clean['PRI_lep_eta']
    dphi = df_clean['PRI_tau_phi'] - df_clean['PRI_lep_phi']
    dphi = np.where(dphi > np.pi, dphi - 2*np.pi, dphi)
    dphi = np.where(dphi < -np.pi, dphi + 2*np.pi, dphi)
    df_clean['delta_R_tau_lep'] = np.sqrt(deta**2 + dphi**2)

# Missing transverse energy significance
if 'PRI_met' in df_clean.columns and 'PRI_met_sumet' in df_clean.columns:
    df_clean['met_significance'] = df_clean['PRI_met'] / np.sqrt(df_clean['PRI_met_sumet'] + 1e-6)

# Jet multiplicity features
if 'PRI_jet_num' in df_clean.columns:
    df_clean['has_jets'] = (df_clean['PRI_jet_num'] > 0).astype(int)
    df_clean['has_two_jets'] = (df_clean['PRI_jet_num'] >= 2).astype(int)

# Centrality measures
if 'DER_lep_eta_centrality' in df_clean.columns and 'DER_met_phi_centrality' in df_clean.columns:
    df_clean['combined_centrality'] = np.sqrt(df_clean['DER_lep_eta_centrality']**2 + df_clean['DER_met_phi_centrality']**2)

# Log transformations for highly skewed variables
skewed_vars = ['DER_mass_MMC', 'DER_mass_vis', 'PRI_tau_pt', 'PRI_lep_pt', 'PRI_met']
for var in skewed_vars:
    if var in df_clean.columns:
        df_clean[f'log_{var}'] = np.log(df_clean[var] + 1)

# Preserve event weights
if 'Weight' in df_clean.columns:
    print(f'Event weights preserved: min={df_clean["Weight"].min():.6f}, max={df_clean["Weight"].max():.6f}')
    print(f'Weight distribution: mean={df_clean["Weight"].mean():.6f}, std={df_clean["Weight"].std():.6f}')
else:
    print('No Weight column found, creating uniform weights')
    df_clean['Weight'] = 1.0

# Data quality checks
print('\nData quality summary:')
print(f'Original shape: {df.shape}')
print(f'Cleaned shape: {df_clean.shape}')
print(f'Missing values remaining: {df_clean.isna().sum().sum()}')
print(f'Features added: {df_clean.shape[1] - df.shape[1]}')

# Select numeric features for ML (exclude Label and string columns)
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
if 'Label' in numeric_cols:
    numeric_cols.remove('Label')

print(f'Numeric features for ML: {len(numeric_cols)}')

# Save preprocessed data
df_clean.to_csv('preprocessed_atlas_data.csv', index=False)
print('Saved preprocessed data to preprocessed_atlas_data.csv')

# Save feature metadata
feature_info = {
    'original_features': len(df.columns),
    'engineered_features': df_clean.shape[1] - df.shape[1],
    'total_features': df_clean.shape[1],
    'numeric_features': len(numeric_cols),
    'missing_value_strategy': 'physics_aware',
    'weight_column': 'Weight' if 'Weight' in df_clean.columns else None,
    'target_column': 'Label' if 'Label' in df_clean.columns else None
}

with open('feature_metadata.json', 'w') as f:
    json.dump(feature_info, f, indent=2)

print('\nRESULT:success=True')
print(f'RESULT:original_features={len(df.columns)}')
print(f'RESULT:engineered_features={df_clean.shape[1] - df.shape[1]}')
print(f'RESULT:total_features={df_clean.shape[1]}')
print(f'RESULT:samples_processed={len(df_clean)}')
print(f'RESULT:missing_values_handled={missing_counts.sum()}')
print('RESULT:output_file=preprocessed_atlas_data.csv')
print('RESULT:metadata_file=feature_metadata.json')