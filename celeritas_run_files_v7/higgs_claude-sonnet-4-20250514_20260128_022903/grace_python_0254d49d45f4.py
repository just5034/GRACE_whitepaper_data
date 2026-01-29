import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Read preprocessed data
df = pd.read_parquet('atlas_preprocessed.parquet')
print(f'Loaded {len(df)} events with {len(df.columns)} features')

# Separate signal and background
signal_mask = df['Label'] == 's'
background_mask = df['Label'] == 'b'
signal_df = df[signal_mask]
background_df = df[background_mask]

print(f'Signal events: {len(signal_df)}')
print(f'Background events: {len(background_df)}')

# Get event weights
signal_weights = signal_df['Weight'] if 'Weight' in signal_df.columns else None
background_weights = background_df['Weight'] if 'Weight' in background_df.columns else None

# Define key physics variables for each plot type
kinematic_vars = ['DER_mass_MMC', 'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet']
missing_energy_vars = ['DER_met_phi_centrality', 'DER_sum_pt', 'PRI_met', 'PRI_met_phi']
tau_vars = ['PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi', 'DER_pt_ratio_lep_tau']

# Function to create distribution plots with proper error handling
def create_distribution_plot(variables, title_prefix, filename_prefix):
    n_vars = len(variables)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, var in enumerate(variables):
        if var not in df.columns:
            print(f'Warning: Variable {var} not found in data')
            continue
            
        ax = axes[i]
        
        # Get data for signal and background
        sig_data = signal_df[var].replace(-999.0, np.nan).dropna()
        bg_data = background_df[var].replace(-999.0, np.nan).dropna()
        
        if len(sig_data) == 0 or len(bg_data) == 0:
            print(f'Warning: No valid data for variable {var}')
            continue
        
        # Determine histogram range
        combined_data = pd.concat([sig_data, bg_data])
        q1, q99 = combined_data.quantile([0.01, 0.99])
        bins = np.linspace(q1, q99, 50)
        
        # Create histograms with weights
        sig_weights_clean = None
        bg_weights_clean = None
        
        if signal_weights is not None:
            sig_weights_clean = signal_weights[signal_df[var] != -999.0]
            sig_weights_clean = sig_weights_clean[signal_df[var].replace(-999.0, np.nan).notna()]
            
        if background_weights is not None:
            bg_weights_clean = background_weights[background_df[var] != -999.0]
            bg_weights_clean = bg_weights_clean[background_df[var].replace(-999.0, np.nan).notna()]
        
        # Plot histograms
        ax.hist(sig_data, bins=bins, alpha=0.7, label='Signal', 
               weights=sig_weights_clean, density=True, histtype='step', linewidth=2, color='blue')
        ax.hist(bg_data, bins=bins, alpha=0.7, label='Background',
               weights=bg_weights_clean, density=True, histtype='step', linewidth=2, color='red')
        
        ax.set_xlabel(var)
        ax.set_ylabel('Normalized Events')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calculate separation metric
        sig_mean = sig_data.mean()
        bg_mean = bg_data.mean()
        sig_std = sig_data.std()
        bg_std = bg_data.std()
        
        if sig_std > 0 and bg_std > 0:
            separation = abs(sig_mean - bg_mean) / np.sqrt(0.5 * (sig_std**2 + bg_std**2))
            title_text = f'{var}\nSeparation: {separation:.3f}'
        else:
            title_text = var
            
        ax.set_title(title_text, fontsize=10)
    
    # Hide unused subplots
    for i in range(len(variables), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'{title_prefix} Distributions', fontsize=14)
    plt.tight_layout()
    
    # Save plots
    png_file = f'{filename_prefix}_distributions.png'
    pdf_file = f'{filename_prefix}_distributions.pdf'
    plt.savefig(png_file, dpi=150, bbox_inches='tight')
    plt.savefig(pdf_file, bbox_inches='tight')
    plt.close()
    
    return png_file

# Create kinematic distribution plots
kinematic_plot = create_distribution_plot(kinematic_vars, 'Kinematic Variables', 'kinematic')
print(f'RESULT:kinematic_plot={kinematic_plot}')

# Create missing energy distribution plots  
missing_energy_plot = create_distribution_plot(missing_energy_vars, 'Missing Energy Variables', 'missing_energy')
print(f'RESULT:missing_energy_plot={missing_energy_plot}')

# Create tau feature distribution plots
tau_plot = create_distribution_plot(tau_vars, 'Tau Variables', 'tau_features')
print(f'RESULT:tau_features_plot={tau_plot}')

# Calculate overall statistics
total_events = len(df)
signal_fraction = len(signal_df) / total_events
background_fraction = len(background_df) / total_events

# Calculate weighted statistics if weights available
if 'Weight' in df.columns:
    total_weighted = df['Weight'].sum()
    signal_weighted = signal_df['Weight'].sum()
    background_weighted = background_df['Weight'].sum()
    
    print(f'RESULT:total_weighted_events={total_weighted:.1f}')
    print(f'RESULT:signal_weighted_fraction={signal_weighted/total_weighted:.4f}')
    print(f'RESULT:background_weighted_fraction={background_weighted/total_weighted:.4f}')

print(f'RESULT:total_events={total_events}')
print(f'RESULT:signal_events={len(signal_df)}')
print(f'RESULT:background_events={len(background_df)}')
print(f'RESULT:signal_fraction={signal_fraction:.4f}')
print(f'RESULT:background_fraction={background_fraction:.4f}')
print('RESULT:success=True')

# Save summary statistics
summary = {
    'total_events': total_events,
    'signal_events': len(signal_df),
    'background_events': len(background_df),
    'signal_fraction': signal_fraction,
    'background_fraction': background_fraction,
    'plots_created': [kinematic_plot, missing_energy_plot, tau_plot],
    'event_weights_used': 'Weight' in df.columns
}

with open('visualization_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('Visualization complete - created publication-quality distribution plots')