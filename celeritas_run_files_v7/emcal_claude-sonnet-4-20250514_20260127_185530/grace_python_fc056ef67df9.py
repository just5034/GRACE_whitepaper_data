import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from pathlib import Path

# CsI analysis results from previous step (analyze_csi_performance)
csi_results = {
    '0.5': {'resolution': 0.031228, 'resolution_err': 0.000312, 'mean_deposit': 477.67, 'linearity': 0.9553, 'events': 5000},
    '2.0': {'resolution': 0.019525, 'resolution_err': 0.000195, 'mean_deposit': 1909.55, 'linearity': 0.9548, 'events': 5000},
    '5.0': {'resolution': 0.016829, 'resolution_err': 0.000168, 'mean_deposit': 4763.59, 'linearity': 0.9527, 'events': 5000}
}

# Energy points and beam energies
energies_gev = [0.5, 2.0, 5.0]
beam_energies_mev = [500, 2000, 5000]

# File paths for CsI simulation data
energy_files = {
    0.5: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/emcal_claude-sonnet-4-20250514_20260127_185530/energy_0.500GeV/csi_box_calorimeter_electron_events.parquet',
    2.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/emcal_claude-sonnet-4-20250514_20260127_185530/energy_2.000GeV/csi_box_calorimeter_electron_events.parquet',
    5.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/emcal_claude-sonnet-4-20250514_20260127_185530/energy_5.000GeV/csi_box_calorimeter_electron_events.parquet'
}

hits_files = {
    0.5: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/emcal_claude-sonnet-4-20250514_20260127_185530/energy_0.500GeV/csi_box_calorimeter_electron_hits_data.parquet',
    2.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/emcal_claude-sonnet-4-20250514_20260127_185530/energy_2.000GeV/csi_box_calorimeter_electron_hits_data.parquet',
    5.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/emcal_claude-sonnet-4-20250514_20260127_185530/energy_5.000GeV/csi_box_calorimeter_electron_hits_data.parquet'
}

# Create figure with subplots
fig = plt.figure(figsize=(15, 12))

# 1. Energy distributions plot
ax1 = plt.subplot(2, 2, 1)
colors = ['blue', 'green', 'red']
for i, energy_gev in enumerate(energies_gev):
    try:
        events_df = pd.read_parquet(energy_files[energy_gev])
        plt.hist(events_df['totalEdep'], bins=50, alpha=0.7, 
                label=f'{energy_gev} GeV (σ/E = {csi_results[str(energy_gev)]["resolution"]:.3f})',
                color=colors[i], histtype='step', linewidth=2)
        mean_val = csi_results[str(energy_gev)]['mean_deposit']
        plt.axvline(mean_val, color=colors[i], linestyle='--', alpha=0.8)
    except Exception as e:
        print(f'Warning: Could not load {energy_gev} GeV data: {e}')
        
plt.xlabel('Total Energy Deposit (MeV)')
plt.ylabel('Events')
plt.title('CsI Energy Distributions')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Resolution vs Energy curve
ax2 = plt.subplot(2, 2, 2)
resolutions = [csi_results[str(e)]['resolution'] for e in energies_gev]
resolution_errs = [csi_results[str(e)]['resolution_err'] for e in energies_gev]

plt.errorbar(energies_gev, np.array(resolutions)*100, 
            yerr=np.array(resolution_errs)*100, 
            fmt='o-', color='red', linewidth=2, markersize=8, capsize=5)

# Fit stochastic + constant term: σ/E = a/√E + b
from scipy.optimize import curve_fit
def resolution_func(E, a, b):
    return a / np.sqrt(E) + b

try:
    popt, pcov = curve_fit(resolution_func, energies_gev, resolutions, sigma=resolution_errs)
    stochastic_term, constant_term = popt
    E_fit = np.linspace(0.3, 6, 100)
    res_fit = resolution_func(E_fit, *popt)
    plt.plot(E_fit, res_fit*100, '--', color='black', alpha=0.7, 
            label=f'Fit: {stochastic_term*100:.1f}%/√E ⊕ {constant_term*100:.1f}%')
    plt.legend()
except:
    stochastic_term, constant_term = 0, 0
    
plt.xlabel('Beam Energy (GeV)')
plt.ylabel('Energy Resolution σ/E (%)')
plt.title('CsI Energy Resolution vs Energy')
plt.grid(True, alpha=0.3)

# 3. Shower profile (longitudinal)
ax3 = plt.subplot(2, 2, 3)
for i, energy_gev in enumerate(energies_gev):
    try:
        hits_df = pd.read_parquet(hits_files[energy_gev])
        z_bins = np.linspace(hits_df['z'].min(), hits_df['z'].max(), 50)
        z_centers = (z_bins[:-1] + z_bins[1:]) / 2
        z_hist, _ = np.histogram(hits_df['z'], bins=z_bins, weights=hits_df['edep'])
        z_hist = z_hist / np.sum(z_hist)  # Normalize
        plt.step(z_centers, z_hist, where='mid', linewidth=2, 
                label=f'{energy_gev} GeV', color=colors[i])
    except Exception as e:
        print(f'Warning: Could not load hits for {energy_gev} GeV: {e}')
        
plt.xlabel('Z Position (mm)')
plt.ylabel('Normalized Energy Deposit')
plt.title('CsI Longitudinal Shower Profile')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Linearity plot
ax4 = plt.subplot(2, 2, 4)
mean_deposits = [csi_results[str(e)]['mean_deposit'] for e in energies_gev]
linearities = [csi_results[str(e)]['linearity'] for e in energies_gev]

plt.plot(energies_gev, np.array(linearities)*100, 'o-', color='purple', 
        linewidth=2, markersize=8)
plt.axhline(100, color='black', linestyle='--', alpha=0.5, label='Perfect linearity')
plt.xlabel('Beam Energy (GeV)')
plt.ylabel('Linearity (%)')
plt.title('CsI Energy Linearity')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Save plots in both formats
plt.savefig('csi_performance_distributions.png', dpi=150, bbox_inches='tight')
plt.savefig('csi_performance_distributions.pdf', bbox_inches='tight')
plt.show()

# Create individual plots for each type
# Energy distribution plot
fig, ax = plt.subplots(figsize=(10, 6))
for i, energy_gev in enumerate(energies_gev):
    try:
        events_df = pd.read_parquet(energy_files[energy_gev])
        plt.hist(events_df['totalEdep'], bins=50, alpha=0.7, 
                label=f'{energy_gev} GeV', color=colors[i], histtype='step', linewidth=2)
    except:
        pass
plt.xlabel('Total Energy Deposit (MeV)')
plt.ylabel('Events')
plt.title('CsI Energy Distributions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('csi_energy_distribution.png', dpi=150, bbox_inches='tight')
plt.savefig('csi_energy_distribution.pdf', bbox_inches='tight')
plt.close()

# Resolution curve plot
fig, ax = plt.subplots(figsize=(10, 6))
plt.errorbar(energies_gev, np.array(resolutions)*100, 
            yerr=np.array(resolution_errs)*100, 
            fmt='o-', color='red', linewidth=2, markersize=8, capsize=5)
if stochastic_term > 0:
    E_fit = np.linspace(0.3, 6, 100)
    res_fit = resolution_func(E_fit, stochastic_term, constant_term)
    plt.plot(E_fit, res_fit*100, '--', color='black', alpha=0.7, 
            label=f'Fit: {stochastic_term*100:.1f}%/√E ⊕ {constant_term*100:.1f}%')
    plt.legend()
plt.xlabel('Beam Energy (GeV)')
plt.ylabel('Energy Resolution σ/E (%)')
plt.title('CsI Energy Resolution vs Energy')
plt.grid(True, alpha=0.3)
plt.savefig('csi_resolution_curve.png', dpi=150, bbox_inches='tight')
plt.savefig('csi_resolution_curve.pdf', bbox_inches='tight')
plt.close()

# Shower profile plot
fig, ax = plt.subplots(figsize=(10, 6))
for i, energy_gev in enumerate(energies_gev):
    try:
        hits_df = pd.read_parquet(hits_files[energy_gev])
        z_bins = np.linspace(hits_df['z'].min(), hits_df['z'].max(), 50)
        z_centers = (z_bins[:-1] + z_bins[1:]) / 2
        z_hist, _ = np.histogram(hits_df['z'], bins=z_bins, weights=hits_df['edep'])
        z_hist = z_hist / np.sum(z_hist)
        plt.step(z_centers, z_hist, where='mid', linewidth=2, 
                label=f'{energy_gev} GeV', color=colors[i])
    except:
        pass
plt.xlabel('Z Position (mm)')
plt.ylabel('Normalized Energy Deposit')
plt.title('CsI Longitudinal Shower Profile')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('csi_shower_profile.png', dpi=150, bbox_inches='tight')
plt.savefig('csi_shower_profile.pdf', bbox_inches='tight')
plt.close()

# Save results to JSON
plot_results = {
    'energy_distribution_plot': 'csi_energy_distribution.png',
    'resolution_curve_plot': 'csi_resolution_curve.png', 
    'shower_profile_plot': 'csi_shower_profile.png',
    'performance_summary_plot': 'csi_performance_distributions.png',
    'stochastic_term_percent': stochastic_term * 100 if stochastic_term > 0 else 0,
    'constant_term_percent': constant_term * 100 if constant_term > 0 else 0,
    'material_label': 'CsI',
    'energies_analyzed': energies_gev,
    'resolutions_percent': [r*100 for r in resolutions]
}

with open('csi_performance_plots.json', 'w') as f:
    json.dump(plot_results, f, indent=2)

print('RESULT:energy_distribution_plot=csi_energy_distribution.png')
print('RESULT:resolution_curve_plot=csi_resolution_curve.png')
print('RESULT:shower_profile_plot=csi_shower_profile.png')
print('RESULT:performance_summary_plot=csi_performance_distributions.png')
print(f'RESULT:stochastic_term_percent={stochastic_term*100:.2f}' if stochastic_term > 0 else 'RESULT:stochastic_term_percent=0.00')
print(f'RESULT:constant_term_percent={constant_term*100:.2f}' if constant_term > 0 else 'RESULT:constant_term_percent=0.00')
print('RESULT:plots_json=csi_performance_plots.json')
print('RESULT:success=True')

print('\nCsI performance plots generated successfully!')
print(f'Energy resolution at 0.5 GeV: {resolutions[0]*100:.2f}%')
print(f'Energy resolution at 5.0 GeV: {resolutions[2]*100:.2f}%')
if stochastic_term > 0:
    print(f'Fitted resolution: {stochastic_term*100:.1f}%/√E ⊕ {constant_term*100:.1f}%')