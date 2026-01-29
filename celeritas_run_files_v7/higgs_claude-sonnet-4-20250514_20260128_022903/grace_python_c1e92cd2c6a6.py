import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Read preprocessed data from previous step
df = pd.read_csv('preprocessed_atlas_data.csv')
with open('feature_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f'Loaded {len(df)} samples with {len(df.columns)} features')

# Handle missing values (-999) and convert to proper labels
df_clean = df.copy()
df_clean[df_clean == -999.0] = np.nan

# Convert Label to binary (s=1, b=0)
signal_mask = df_clean['Label'] == 's'
background_mask = df_clean['Label'] == 'b'

print(f'Signal events: {signal_mask.sum()}')
print(f'Background events: {background_mask.sum()}')

# Key physics variables for visualization
kinematic_vars = ['DER_mass_MMC', 'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet']
missing_energy_vars = ['DER_met_phi_centrality', 'DER_lep_eta_centrality']
tau_vars = ['PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi']

# Set up publication-quality plot style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Function to create comparison plots
def create_comparison_plot(var_name, bins=50, log_scale=False):
    if var_name not in df_clean.columns:
        print(f'Warning: {var_name} not found in data')
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get data for signal and background
    sig_data = df_clean[signal_mask][var_name].dropna()
    bkg_data = df_clean[background_mask][var_name].dropna()
    
    # Get weights if available
    sig_weights = None
    bkg_weights = None
    if 'Weight' in df_clean.columns:
        sig_weights = df_clean[signal_mask]['Weight'].loc[sig_data.index]
        bkg_weights = df_clean[background_mask]['Weight'].loc[bkg_data.index]
    
    # Create histograms
    range_min = min(sig_data.min(), bkg_data.min())
    range_max = max(sig_data.max(), bkg_data.max())
    
    ax.hist(bkg_data, bins=bins, range=(range_min, range_max), 
            weights=bkg_weights, alpha=0.7, color='red', 
            label=f'Background (n={len(bkg_data)})', density=True, histtype='stepfilled')
    
    ax.hist(sig_data, bins=bins, range=(range_min, range_max),
            weights=sig_weights, alpha=0.8, color='blue',
            label=f'Signal (n={len(sig_data)})', density=True, histtype='step', linewidth=2)
    
    ax.set_xlabel(var_name.replace('_', ' ').title())
    ax.set_ylabel('Normalized Events')
    ax.set_title(f'Signal vs Background: {var_name}')
    ax.legend()
    
    if log_scale:
        ax.set_yscale('log')
    
    # Add statistics
    sig_mean = sig_data.mean()
    bkg_mean = bkg_data.mean()
    separation = abs(sig_mean - bkg_mean) / np.sqrt(0.5 * (sig_data.var() + bkg_data.var()))
    
    ax.text(0.05, 0.95, f'Separation: {separation:.3f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    filename = f'{var_name}_distribution.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.savefig(filename.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    return separation, filename

# Create kinematic distribution plots
print('Creating kinematic distribution plots...')
kinematic_separations = {}
for var in kinematic_vars:
    if var in df_clean.columns:
        sep, fname = create_comparison_plot(var)
        kinematic_separations[var] = sep
        print(f'Created {fname} (separation: {sep:.3f})')

# Create missing energy plots
print('Creating missing energy plots...')
met_separations = {}
for var in missing_energy_vars:
    if var in df_clean.columns:
        sep, fname = create_comparison_plot(var)
        met_separations[var] = sep
        print(f'Created {fname} (separation: {sep:.3f})')

# Create tau feature plots
print('Creating tau feature plots...')
tau_separations = {}
for var in tau_vars:
    if var in df_clean.columns:
        sep, fname = create_comparison_plot(var)
        tau_separations[var] = sep
        print(f'Created {fname} (separation: {sep:.3f})')

# Create summary plot of feature separations
all_separations = {**kinematic_separations, **met_separations, **tau_separations}
if all_separations:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    vars_sorted = sorted(all_separations.items(), key=lambda x: x[1], reverse=True)
    var_names = [v[0].replace('_', ' ') for v in vars_sorted]
    separations = [v[1] for v in vars_sorted]
    
    bars = ax.barh(range(len(var_names)), separations, color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(var_names)))
    ax.set_yticklabels(var_names)
    ax.set_xlabel('Signal-Background Separation')
    ax.set_title('Feature Discrimination Power')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, sep) in enumerate(zip(bars, separations)):
        ax.text(sep + 0.01, i, f'{sep:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig('feature_separations_summary.png', dpi=150, bbox_inches='tight')
    plt.savefig('feature_separations_summary.pdf', bbox_inches='tight')
    plt.close()
    print('Created feature_separations_summary.png')

# Create correlation matrix for top discriminating features
top_features = [v[0] for v in sorted(all_separations.items(), key=lambda x: x[1], reverse=True)[:10]]
if len(top_features) > 1:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Select numeric columns only
    corr_data = df_clean[top_features].select_dtypes(include=[np.number])
    corr_matrix = corr_data.corr()
    
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels([c.replace('_', ' ') for c in corr_matrix.columns], rotation=45, ha='right')
    ax.set_yticklabels([c.replace('_', ' ') for c in corr_matrix.columns])
    
    # Add correlation values
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            ax.text(j, i, f'{corr_matrix.iloc[i,j]:.2f}', ha='center', va='center')
    
    ax.set_title('Feature Correlation Matrix (Top Discriminating Variables)')
    plt.colorbar(im, ax=ax, label='Correlation Coefficient')
    plt.tight_layout()
    plt.savefig('feature_correlations.png', dpi=150, bbox_inches='tight')
    plt.savefig('feature_correlations.pdf', bbox_inches='tight')
    plt.close()
    print('Created feature_correlations.png')

# Save results
results = {
    'kinematic_separations': kinematic_separations,
    'met_separations': met_separations,
    'tau_separations': tau_separations,
    'best_discriminator': max(all_separations.items(), key=lambda x: x[1]) if all_separations else None,
    'mean_separation': np.mean(list(all_separations.values())) if all_separations else 0
}

with open('visualization_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\nVisualization Summary:')
print(f'Total plots created: {len(all_separations) + 2}')
if results['best_discriminator']:
    print(f'Best discriminating variable: {results["best_discriminator"][0]} (separation: {results["best_discriminator"][1]:.3f})')
print(f'Mean separation across all variables: {results["mean_separation"]:.3f}')

# Return values for downstream steps
print(f'RESULT:plots_created={len(all_separations) + 2}')
print(f'RESULT:best_separator={results["best_discriminator"][0] if results["best_discriminator"] else "none"}')
print(f'RESULT:best_separation={results["best_discriminator"][1]:.4f if results["best_discriminator"] else 0}')
print(f'RESULT:mean_separation={results["mean_separation"]:.4f}')
print('RESULT:success=True')