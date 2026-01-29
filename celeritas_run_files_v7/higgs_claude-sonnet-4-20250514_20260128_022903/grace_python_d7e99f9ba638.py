import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Load preprocessed data and metadata
df = pd.read_csv('preprocessed_atlas_data.csv')
with open('feature_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f'Loaded {len(df)} events with {len(df.columns)} features')
print(f'Signal events: {len(df[df["Label"] == "s"])}')
print(f'Background events: {len(df[df["Label"] == "b"])}')

# Separate signal and background
signal = df[df['Label'] == 's'].copy()
background = df[df['Label'] == 'b'].copy()

# Get event weights if available
use_weights = 'Weight' in df.columns
if use_weights:
    signal_weights = signal['Weight'].values
    background_weights = background['Weight'].values
    print(f'Using event weights: Signal mean={signal_weights.mean():.4f}, Background mean={background_weights.mean():.4f}')
else:
    signal_weights = None
    background_weights = None
    print('No event weights found, using unit weights')

# Define key physics variables for each plot type
kinematic_vars = ['PRI_tau_pt', 'PRI_lep_pt', 'PRI_met', 'PRI_met_phi']
missing_energy_vars = ['PRI_met', 'PRI_met_sumet', 'PRI_met_phi']
tau_vars = ['PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi', 'DER_mass_tau_lep']

# Function to create comparison plots
def plot_distributions(variables, plot_name, title_prefix):
    n_vars = len(variables)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, var in enumerate(variables[:4]):
        if var not in df.columns:
            print(f'Warning: Variable {var} not found in data')
            continue
            
        ax = axes[i]
        
        # Get data for plotting, handle missing values
        sig_data = signal[var].replace(-999.0, np.nan).dropna()
        bg_data = background[var].replace(-999.0, np.nan).dropna()
        
        if len(sig_data) == 0 or len(bg_data) == 0:
            print(f'Warning: No valid data for {var}')
            continue
            
        # Determine bins
        all_data = np.concatenate([sig_data, bg_data])
        bins = np.linspace(np.percentile(all_data, 1), np.percentile(all_data, 99), 50)
        
        # Get corresponding weights
        if use_weights:
            sig_w = signal.loc[sig_data.index, 'Weight'].values
            bg_w = background.loc[bg_data.index, 'Weight'].values
        else:
            sig_w = None
            bg_w = None
            
        # Plot histograms
        ax.hist(bg_data, bins=bins, alpha=0.7, label='Background', 
               color='red', density=True, weights=bg_w)
        ax.hist(sig_data, bins=bins, alpha=0.7, label='Signal', 
               color='blue', density=True, weights=sig_w)
        
        ax.set_xlabel(var)
        ax.set_ylabel('Normalized Events')
        ax.set_title(f'{title_prefix}: {var}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calculate separation metric
        sig_mean = sig_data.mean()
        bg_mean = bg_data.mean()
        sig_std = sig_data.std()
        bg_std = bg_data.std()
        
        if sig_std > 0 and bg_std > 0:
            separation = abs(sig_mean - bg_mean) / np.sqrt(0.5 * (sig_std**2 + bg_std**2))
            ax.text(0.05, 0.95, f'Sep: {separation:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{plot_name}.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{plot_name}.pdf', bbox_inches='tight')
    plt.close()
    
    return f'{plot_name}.png'

# Create kinematic distributions plot
kinematic_plot = plot_distributions(kinematic_vars, 'kinematic_distributions', 'Kinematic Variables')
print(f'RESULT:kinematic_plot={kinematic_plot}')

# Create missing energy plot
met_plot = plot_distributions(missing_energy_vars, 'missing_energy_distributions', 'Missing Energy')
print(f'RESULT:missing_energy_plot={met_plot}')

# Create tau features plot
tau_plot = plot_distributions(tau_vars, 'tau_feature_distributions', 'Tau Features')
print(f'RESULT:tau_features_plot={tau_plot}')

# Calculate overall discrimination metrics
numeric_cols = df.select_dtypes(include=[np.number]).columns
numeric_cols = [col for col in numeric_cols if col not in ['Label', 'Weight', 'KaggleSet', 'KaggleWeight']]

# Find best discriminating variables
discrimination_scores = {}
for col in numeric_cols:
    if col in df.columns:
        sig_vals = signal[col].replace(-999.0, np.nan).dropna()
        bg_vals = background[col].replace(-999.0, np.nan).dropna()
        
        if len(sig_vals) > 10 and len(bg_vals) > 10:
            sig_mean, sig_std = sig_vals.mean(), sig_vals.std()
            bg_mean, bg_std = bg_vals.mean(), bg_vals.std()
            
            if sig_std > 0 and bg_std > 0:
                separation = abs(sig_mean - bg_mean) / np.sqrt(0.5 * (sig_std**2 + bg_std**2))
                discrimination_scores[col] = separation

# Sort by discrimination power
sorted_vars = sorted(discrimination_scores.items(), key=lambda x: x[1], reverse=True)

print('\nTop 10 discriminating variables:')
for i, (var, score) in enumerate(sorted_vars[:10]):
    print(f'{i+1:2d}. {var:25s}: {score:.4f}')

# Create summary plot of top discriminating variables
if len(sorted_vars) >= 4:
    top_vars = [var for var, score in sorted_vars[:4]]
    summary_plot = plot_distributions(top_vars, 'top_discriminating_variables', 'Top Discriminating Variables')
    print(f'RESULT:summary_plot={summary_plot}')

# Save discrimination results
results = {
    'total_events': len(df),
    'signal_events': len(signal),
    'background_events': len(background),
    'top_discriminating_variables': sorted_vars[:10],
    'event_weights_used': use_weights,
    'plots_created': ['kinematic_distributions.png', 'missing_energy_distributions.png', 'tau_feature_distributions.png']
}

if len(sorted_vars) > 0:
    results['best_discriminator'] = sorted_vars[0]
    best_var, best_score = sorted_vars[0]
    print(f'RESULT:best_discriminating_variable={best_var}')
    print(f'RESULT:best_discrimination_score={best_score:.4f}')
else:
    results['best_discriminator'] = None
    print('RESULT:best_discriminating_variable=None')
    print('RESULT:best_discrimination_score=0.0000')

with open('visualization_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('RESULT:success=True')
print('RESULT:plots_created=4')
print(f'RESULT:variables_analyzed={len(discrimination_scores)}')
print('RESULT:output_file=visualization_results.json')