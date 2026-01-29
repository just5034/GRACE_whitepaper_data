import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# CsI energy sweep file paths from Previous Step Outputs
energy_files = {
    0.5: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/emcal_claude-sonnet-4-20250514_20260127_185530/energy_0.500GeV/csi_box_calorimeter_electron_events.parquet',
    2.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/emcal_claude-sonnet-4-20250514_20260127_185530/energy_2.000GeV/csi_box_calorimeter_electron_events.parquet',
    5.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/emcal_claude-sonnet-4-20250514_20260127_185530/energy_5.000GeV/csi_box_calorimeter_electron_events.parquet'
}

# CsI geometry parameters from Previous Step Outputs
csi_sampling_fraction = 0.997319

# Analyze each energy point
results = []
energies = []
resolutions = []
resolution_errors = []
linearities = []

for energy_gev, filepath in energy_files.items():
    print(f'Analyzing CsI at {energy_gev} GeV...')
    
    # Read events data
    events = pd.read_parquet(filepath)
    beam_energy_mev = energy_gev * 1000  # Convert GeV to MeV
    
    # Calculate energy resolution
    mean_edep = events['totalEdep'].mean()
    std_edep = events['totalEdep'].std()
    num_events = len(events)
    
    energy_resolution = std_edep / mean_edep if mean_edep > 0 else 0
    resolution_err = energy_resolution / np.sqrt(2 * num_events) if num_events > 0 else 0
    
    # Calculate linearity (response vs beam energy)
    linearity = mean_edep / beam_energy_mev if beam_energy_mev > 0 else 0
    
    # Store results
    results.append({
        'energy_gev': energy_gev,
        'beam_energy_mev': beam_energy_mev,
        'mean_deposit_mev': mean_edep,
        'std_deposit_mev': std_edep,
        'energy_resolution': energy_resolution,
        'resolution_err': resolution_err,
        'linearity': linearity,
        'num_events': num_events
    })
    
    energies.append(energy_gev)
    resolutions.append(energy_resolution)
    resolution_errors.append(resolution_err)
    linearities.append(linearity)
    
    # Output individual energy point results
    energy_str = str(energy_gev).replace('.', '_')
    print(f'RESULT:csi_{energy_str}gev_energy_resolution={energy_resolution:.6f}')
    print(f'RESULT:csi_{energy_str}gev_resolution_err={resolution_err:.6f}')
    print(f'RESULT:csi_{energy_str}gev_mean_deposit_mev={mean_edep:.2f}')
    print(f'RESULT:csi_{energy_str}gev_linearity={linearity:.4f}')
    print(f'RESULT:csi_{energy_str}gev_num_events={num_events}')

# Create energy resolution curve plot
plt.figure(figsize=(10, 6))
plt.errorbar(energies, resolutions, yerr=resolution_errors, 
            marker='o', linewidth=2, markersize=8, capsize=5,
            label='CsI Energy Resolution')
plt.xlabel('Beam Energy (GeV)')
plt.ylabel('Energy Resolution (σ/E)')
plt.title('CsI Energy Resolution vs Beam Energy')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('csi_resolution_curve.png', dpi=150, bbox_inches='tight')
plt.savefig('csi_resolution_curve.pdf', bbox_inches='tight')
plt.close()

# Create energy distributions plot for all energies
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (energy_gev, filepath) in enumerate(energy_files.items()):
    events = pd.read_parquet(filepath)
    axes[i].hist(events['totalEdep'], bins=50, histtype='step', linewidth=2,
                label=f'{energy_gev} GeV')
    axes[i].set_xlabel('Energy Deposit (MeV)')
    axes[i].set_ylabel('Events')
    axes[i].set_title(f'{energy_gev} GeV Electrons')
    axes[i].grid(True, alpha=0.3)
    axes[i].legend()
plt.tight_layout()
plt.savefig('csi_energy_distributions.png', dpi=150, bbox_inches='tight')
plt.savefig('csi_energy_distributions.pdf', bbox_inches='tight')
plt.close()

# Create linearity plot
plt.figure(figsize=(10, 6))
plt.plot(energies, linearities, 'o-', linewidth=2, markersize=8, label='CsI Linearity')
plt.xlabel('Beam Energy (GeV)')
plt.ylabel('Linearity (Response/Beam Energy)')
plt.title('CsI Detector Linearity')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('csi_linearity_curve.png', dpi=150, bbox_inches='tight')
plt.savefig('csi_linearity_curve.pdf', bbox_inches='tight')
plt.close()

# Save results to JSON
results_dict = {
    'material': 'CsI',
    'sampling_fraction': csi_sampling_fraction,
    'energy_points': results,
    'plots': {
        'resolution_curve': 'csi_resolution_curve.png',
        'energy_distributions': 'csi_energy_distributions.png',
        'linearity_curve': 'csi_linearity_curve.png'
    }
}

with open('csi_performance_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print('RESULT:csi_resolution_curve_plot=csi_resolution_curve.png')
print('RESULT:csi_energy_distributions_plot=csi_energy_distributions.png')
print('RESULT:csi_linearity_plot=csi_linearity_curve.png')
print('RESULT:csi_results_json=csi_performance_results.json')
print('RESULT:success=True')

print('\nCsI Performance Analysis Complete!')
print(f'Energy resolution at 0.5 GeV: {resolutions[0]:.4f} ± {resolution_errors[0]:.4f}')
print(f'Energy resolution at 2.0 GeV: {resolutions[1]:.4f} ± {resolution_errors[1]:.4f}')
print(f'Energy resolution at 5.0 GeV: {resolutions[2]:.4f} ± {resolution_errors[2]:.4f}')