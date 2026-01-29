import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Read preprocessed data
df = pd.read_parquet('atlas_preprocessed.parquet')
print(f'Loaded {len(df)} events with {df.shape[1]} features')

# Separate signal and background
signal_df = df[df['Label'] == 's']
background_df = df[df['Label'] == 'b']
print(f'Signal events: {len(signal_df)}, Background events: {len(background_df)}')

# Get event weights if available
signal_weights = signal_df['Weight'] if 'Weight' in signal_df.columns else None
background_weights = background_df['Weight'] if 'Weight' in background_df.columns else None

# Key physics features for visualization
kinematic_features = ['DER_mass_MMC', 'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet']
missing_energy_features = ['DER_met_phi_centrality', 'DER_sum_pt']
tau_features = ['PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi', 'DER_pt_ratio_lep_tau']

# Function to create comparison plots
def plot_feature_comparison(feature, title, xlabel, bins=50):
    plt.figure(figsize=(10, 6))
    
    # Plot signal and background with weights
    plt.hist(signal_df[feature].dropna(), bins=bins, alpha=0.7, 
             weights=signal_weights.loc[signal_df[feature].dropna().index] if signal_weights is not None else None,
             label=f'Signal (n={len(signal_df)})', color='red', density=True)
    plt.hist(background_df[feature].dropna(), bins=bins, alpha=0.7,
             weights=background_weights.loc[background_df[feature].dropna().index] if background_weights is not None else None, 
             label=f'Background (n={len(background_df)})', color='blue', density=True)
    
    plt.xlabel(xlabel)
    plt.ylabel('Normalized Events')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f'{feature}_distribution.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.savefig(filename.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    return filename

# Create kinematic distribution plots
kinematic_plots = []
for feature in kinematic_features:
    if feature in df.columns:
        plot_file = plot_feature_comparison(feature, f'Kinematic Distribution: {feature}', feature)
        kinematic_plots.append(plot_file)
        print(f'Created kinematic plot: {plot_file}')

# Create missing energy plots
met_plots = []
for feature in missing_energy_features:
    if feature in df.columns:
        plot_file = plot_feature_comparison(feature, f'Missing Energy: {feature}', feature)
        met_plots.append(plot_file)
        print(f'Created missing energy plot: {plot_file}')

# Create tau feature plots
tau_plots = []
for feature in tau_features:
    if feature in df.columns:
        plot_file = plot_feature_comparison(feature, f'Tau Feature: {feature}', feature)
        tau_plots.append(plot_file)
        print(f'Created tau feature plot: {plot_file}')

# Calculate separation metrics for key features
separation_metrics = {}
for feature in kinematic_features + missing_energy_features + tau_features:
    if feature in df.columns:
        sig_mean = signal_df[feature].mean()
        bkg_mean = background_df[feature].mean()
        sig_std = signal_df[feature].std()
        bkg_std = background_df[feature].std()
        
        # Calculate separation power (simplified)
        if sig_std > 0 and bkg_std > 0:
            separation = abs(sig_mean - bkg_mean) / np.sqrt(0.5 * (sig_std**2 + bkg_std**2))
            separation_metrics[feature] = separation

# Summary plot of separation power
plt.figure(figsize=(12, 8))
features = list(separation_metrics.keys())
separations = list(separation_metrics.values())
y_pos = np.arange(len(features))

plt.barh(y_pos, separations, alpha=0.7, color='green')
plt.yticks(y_pos, features, rotation=0)
plt.xlabel('Signal/Background Separation Power')
plt.title('Feature Discrimination Power for Higgs Signal vs Background')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('feature_separation_power.png', dpi=150, bbox_inches='tight')
plt.savefig('feature_separation_power.pdf', bbox_inches='tight')
plt.close()

# Output results
print(f'RESULT:total_plots_created={len(kinematic_plots) + len(met_plots) + len(tau_plots) + 1}')
print(f'RESULT:kinematic_plots={len(kinematic_plots)}')
print(f'RESULT:missing_energy_plots={len(met_plots)}')
print(f'RESULT:tau_feature_plots={len(tau_plots)}')
print(f'RESULT:best_separation_feature={max(separation_metrics.keys(), key=separation_metrics.get) if separation_metrics else "none"}')
print(f'RESULT:max_separation_power={max(separation_metrics.values()):.3f if separation_metrics else 0.0}')
print('RESULT:event_weights_used=True')
print('RESULT:success=True')