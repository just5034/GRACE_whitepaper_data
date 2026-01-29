import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load preprocessed data
df = pd.read_parquet('preprocessed_atlas_data.parquet')
print(f"Loaded {len(df)} events with {len(df.columns)} features")

# Handle missing values (-999.0) by replacing with NaN
df_clean = df.copy()
df_clean[df_clean == -999.0] = np.nan

# Separate signal (s) and background (b) events
signal_mask = df_clean['Label'] == 's'
background_mask = df_clean['Label'] == 'b'

signal_data = df_clean[signal_mask]
background_data = df_clean[background_mask]

print(f"Signal events: {len(signal_data)}")
print(f"Background events: {len(background_data)}")

# Key physics features to visualize
kinematic_features = ['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h']
missing_energy_features = ['DER_met_phi_centrality', 'DER_deltaeta_jet_jet']
tau_features = ['DER_pt_ratio_lep_tau', 'DER_deltar_tau_lep']

# Create publication-quality plots
fig, axes = plt.subplots(3, 2, figsize=(15, 18))
fig.suptitle('ATLAS Higgs → ττ Signal vs Background Distributions', fontsize=16, fontweight='bold')

# Plot kinematic distributions
for i, feature in enumerate(kinematic_features[:2]):
    ax = axes[0, i]
    
    # Get clean data (remove NaN)
    sig_vals = signal_data[feature].dropna()
    bg_vals = background_data[feature].dropna()
    
    if len(sig_vals) > 0 and len(bg_vals) > 0:
        # Use event weights if available
        sig_weights = signal_data.loc[sig_vals.index, 'Weight'] if 'Weight' in df.columns else None
        bg_weights = background_data.loc[bg_vals.index, 'Weight'] if 'Weight' in df.columns else None
        
        # Create histograms
        bins = np.linspace(min(np.min(sig_vals), np.min(bg_vals)), 
                          max(np.max(sig_vals), np.max(bg_vals)), 50)
        
        ax.hist(bg_vals, bins=bins, weights=bg_weights, alpha=0.7, 
               label=f'Background (n={len(bg_vals)})', color='red', density=True)
        ax.hist(sig_vals, bins=bins, weights=sig_weights, alpha=0.7, 
               label=f'Signal (n={len(sig_vals)})', color='blue', density=True)
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Normalized Events')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Kinematic: {feature}')

# Plot missing energy features
for i, feature in enumerate(missing_energy_features):
    ax = axes[1, i]
    
    sig_vals = signal_data[feature].dropna()
    bg_vals = background_data[feature].dropna()
    
    if len(sig_vals) > 0 and len(bg_vals) > 0:
        sig_weights = signal_data.loc[sig_vals.index, 'Weight'] if 'Weight' in df.columns else None
        bg_weights = background_data.loc[bg_vals.index, 'Weight'] if 'Weight' in df.columns else None
        
        bins = np.linspace(min(np.min(sig_vals), np.min(bg_vals)), 
                          max(np.max(sig_vals), np.max(bg_vals)), 50)
        
        ax.hist(bg_vals, bins=bins, weights=bg_weights, alpha=0.7, 
               label=f'Background (n={len(bg_vals)})', color='red', density=True)
        ax.hist(sig_vals, bins=bins, weights=sig_weights, alpha=0.7, 
               label=f'Signal (n={len(sig_vals)})', color='blue', density=True)
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Normalized Events')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Missing Energy: {feature}')

# Plot tau features
for i, feature in enumerate(tau_features):
    ax = axes[2, i]
    
    sig_vals = signal_data[feature].dropna()
    bg_vals = background_data[feature].dropna()
    
    if len(sig_vals) > 0 and len(bg_vals) > 0:
        sig_weights = signal_data.loc[sig_vals.index, 'Weight'] if 'Weight' in df.columns else None
        bg_weights = background_data.loc[bg_vals.index, 'Weight'] if 'Weight' in df.columns else None
        
        bins = np.linspace(min(np.min(sig_vals), np.min(bg_vals)), 
                          max(np.max(sig_vals), np.max(bg_vals)), 50)
        
        ax.hist(bg_vals, bins=bins, weights=bg_weights, alpha=0.7, 
               label=f'Background (n={len(bg_vals)})', color='red', density=True)
        ax.hist(sig_vals, bins=bins, weights=sig_weights, alpha=0.7, 
               label=f'Signal (n={len(sig_vals)})', color='blue', density=True)
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Normalized Events')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Tau Features: {feature}')

plt.tight_layout()
plt.savefig('atlas_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.savefig('atlas_feature_distributions.pdf', bbox_inches='tight')
plt.close()

# Calculate separation metrics for key features
separation_metrics = {}
all_features = kinematic_features + missing_energy_features + tau_features

for feature in all_features:
    if feature in df_clean.columns:
        sig_vals = signal_data[feature].dropna()
        bg_vals = background_data[feature].dropna()
        
        if len(sig_vals) > 10 and len(bg_vals) > 10:
            # Calculate separation power (simple metric)
            sig_mean = sig_vals.mean()
            bg_mean = bg_vals.mean()
            sig_std = sig_vals.std()
            bg_std = bg_vals.std()
            
            # Simple separation metric: |μ_s - μ_b| / sqrt(σ_s² + σ_b²)
            separation = abs(sig_mean - bg_mean) / np.sqrt(sig_std**2 + bg_std**2)
            separation_metrics[feature] = separation

# Create separation power plot
if separation_metrics:
    fig, ax = plt.subplots(figsize=(12, 8))
    features = list(separation_metrics.keys())
    separations = list(separation_metrics.values())
    
    bars = ax.barh(features, separations, color='steelblue', alpha=0.7)
    ax.set_xlabel('Separation Power |μ_s - μ_b| / √(σ_s² + σ_b²)')
    ax.set_title('Feature Discrimination Power: Signal vs Background')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for bar, sep in zip(bars, separations):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{sep:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('feature_separation_power.png', dpi=300, bbox_inches='tight')
    plt.savefig('feature_separation_power.pdf', bbox_inches='tight')
    plt.close()

# Output results
print(f"RESULT:plots_created=2")
print(f"RESULT:features_analyzed={len(all_features)}")
print(f"RESULT:signal_events={len(signal_data)}")
print(f"RESULT:background_events={len(background_data)}")
print(f"RESULT:main_plot=atlas_feature_distributions.png")
print(f"RESULT:separation_plot=feature_separation_power.png")
print(f"RESULT:event_weights_used={bool('Weight' in df.columns)}")
print(f"RESULT:success=True")

# Print top discriminating features
if separation_metrics:
    sorted_features = sorted(separation_metrics.items(), key=lambda x: x[1], reverse=True)
    print("\nTop discriminating features:")
    for i, (feature, sep) in enumerate(sorted_features[:5]):
        print(f"{i+1}. {feature}: {sep:.3f}")
        print(f"RESULT:top_feature_{i+1}={feature}")
        print(f"RESULT:top_separation_{i+1}={sep:.3f}")