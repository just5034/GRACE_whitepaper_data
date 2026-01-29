import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Energy sweep file paths from Previous Step Outputs
muon_files = {
    5.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/muon_claude-sonnet-4-20250514_20260127_231937/energy_5.000GeV/muon_spectrometer_retry_mum_events.parquet',
    20.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/muon_claude-sonnet-4-20250514_20260127_231937/energy_20.000GeV/muon_spectrometer_retry_mum_events.parquet',
    50.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/muon_claude-sonnet-4-20250514_20260127_231937/energy_50.000GeV/muon_spectrometer_retry_mum_events.parquet'
}

pion_files = {
    5.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/muon_claude-sonnet-4-20250514_20260127_231937/energy_5.000GeV/muon_spectrometer_retry_pim_events.parquet',
    20.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/muon_claude-sonnet-4-20250514_20260127_231937/energy_20.000GeV/muon_spectrometer_retry_pim_events.parquet',
    50.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/muon_claude-sonnet-4-20250514_20260127_231937/energy_50.000GeV/muon_spectrometer_retry_pim_events.parquet'
}

# Analyze each energy point
results = {}
energies = [5.0, 20.0, 50.0]

for energy_gev in energies:
    print(f'Analyzing {energy_gev} GeV data...')
    
    # Load muon data
    muon_df = pd.read_parquet(muon_files[energy_gev])
    pion_df = pd.read_parquet(pion_files[energy_gev])
    
    # Muon analysis
    muon_mean_edep = muon_df['totalEdep'].mean()
    muon_std_edep = muon_df['totalEdep'].std()
    muon_resolution = muon_std_edep / muon_mean_edep if muon_mean_edep > 0 else 0
    muon_resolution_err = muon_resolution / np.sqrt(2 * len(muon_df)) if len(muon_df) > 0 else 0
    
    # Muon detection efficiency (events with hits)
    muon_detected = len(muon_df[muon_df['nHits'] > 0])
    muon_efficiency = muon_detected / len(muon_df) if len(muon_df) > 0 else 0
    muon_efficiency_err = np.sqrt(muon_efficiency * (1 - muon_efficiency) / len(muon_df)) if len(muon_df) > 0 else 0
    
    # Pion analysis
    pion_mean_edep = pion_df['totalEdep'].mean()
    pion_std_edep = pion_df['totalEdep'].std()
    pion_resolution = pion_std_edep / pion_mean_edep if pion_mean_edep > 0 else 0
    pion_resolution_err = pion_resolution / np.sqrt(2 * len(pion_df)) if len(pion_df) > 0 else 0
    
    # Store results with proper naming
    results[f'cylindrical_muon_{energy_gev}gev_resolution'] = muon_resolution
    results[f'cylindrical_muon_{energy_gev}gev_resolution_err'] = muon_resolution_err
    results[f'cylindrical_muon_{energy_gev}gev_mean_edep'] = muon_mean_edep
    results[f'cylindrical_muon_{energy_gev}gev_efficiency'] = muon_efficiency
    results[f'cylindrical_muon_{energy_gev}gev_efficiency_err'] = muon_efficiency_err
    
    results[f'cylindrical_pion_{energy_gev}gev_resolution'] = pion_resolution
    results[f'cylindrical_pion_{energy_gev}gev_resolution_err'] = pion_resolution_err
    results[f'cylindrical_pion_{energy_gev}gev_mean_edep'] = pion_mean_edep
    
    # Print individual results
    print(f'RESULT:cylindrical_muon_{energy_gev}gev_resolution={muon_resolution:.4f}')
    print(f'RESULT:cylindrical_muon_{energy_gev}gev_mean_edep={muon_mean_edep:.2f}')
    print(f'RESULT:cylindrical_muon_{energy_gev}gev_efficiency={muon_efficiency:.4f}')
    print(f'RESULT:cylindrical_pion_{energy_gev}gev_resolution={pion_resolution:.4f}')
    print(f'RESULT:cylindrical_pion_{energy_gev}gev_mean_edep={pion_mean_edep:.2f}')

# Create energy deposition histograms
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Cylindrical Detector Energy Deposition Profiles', fontsize=16)

for i, energy_gev in enumerate(energies):
    muon_df = pd.read_parquet(muon_files[energy_gev])
    pion_df = pd.read_parquet(pion_files[energy_gev])
    
    # Muon histogram
    axes[0, i].hist(muon_df['totalEdep'], bins=50, alpha=0.7, color='blue', label='Muons')
    axes[0, i].axvline(muon_df['totalEdep'].mean(), color='red', linestyle='--', label=f'Mean: {muon_df["totalEdep"].mean():.1f} MeV')
    axes[0, i].set_xlabel('Energy Deposit (MeV)')
    axes[0, i].set_ylabel('Events')
    axes[0, i].set_title(f'{energy_gev} GeV Muons')
    axes[0, i].legend()
    axes[0, i].grid(True, alpha=0.3)
    
    # Pion histogram
    axes[1, i].hist(pion_df['totalEdep'], bins=50, alpha=0.7, color='orange', label='Pions')
    axes[1, i].axvline(pion_df['totalEdep'].mean(), color='red', linestyle='--', label=f'Mean: {pion_df["totalEdep"].mean():.1f} MeV')
    axes[1, i].set_xlabel('Energy Deposit (MeV)')
    axes[1, i].set_ylabel('Events')
    axes[1, i].set_title(f'{energy_gev} GeV Pions')
    axes[1, i].legend()
    axes[1, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cylindrical_energy_histograms.png', dpi=150, bbox_inches='tight')
plt.savefig('cylindrical_energy_histograms.pdf', bbox_inches='tight')
plt.close()

# Resolution vs energy plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Muon resolution
muon_resolutions = [results[f'cylindrical_muon_{e}gev_resolution'] for e in energies]
muon_res_errors = [results[f'cylindrical_muon_{e}gev_resolution_err'] for e in energies]
ax1.errorbar(energies, muon_resolutions, yerr=muon_res_errors, marker='o', capsize=5, label='Muons')
ax1.set_xlabel('Energy (GeV)')
ax1.set_ylabel('Energy Resolution (σ/E)')
ax1.set_title('Cylindrical Detector - Muon Resolution')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Pion resolution
pion_resolutions = [results[f'cylindrical_pion_{e}gev_resolution'] for e in energies]
pion_res_errors = [results[f'cylindrical_pion_{e}gev_resolution_err'] for e in energies]
ax2.errorbar(energies, pion_resolutions, yerr=pion_res_errors, marker='s', capsize=5, label='Pions', color='orange')
ax2.set_xlabel('Energy (GeV)')
ax2.set_ylabel('Energy Resolution (σ/E)')
ax2.set_title('Cylindrical Detector - Pion Resolution')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('cylindrical_resolution_vs_energy.png', dpi=150, bbox_inches='tight')
plt.savefig('cylindrical_resolution_vs_energy.pdf', bbox_inches='tight')
plt.close()

# Save results to JSON
with open('cylindrical_analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Calculate pion rejection (based on energy deposit differences)
total_pion_rejection = 0
for energy_gev in energies:
    muon_mean = results[f'cylindrical_muon_{energy_gev}gev_mean_edep']
    pion_mean = results[f'cylindrical_pion_{energy_gev}gev_mean_edep']
    # Simple discrimination based on energy deposit ratio
    rejection_factor = pion_mean / muon_mean if muon_mean > 0 else 1
    total_pion_rejection += rejection_factor

avg_pion_rejection = total_pion_rejection / len(energies)

print(f'RESULT:cylindrical_avg_pion_rejection_factor={avg_pion_rejection:.2f}')
print('RESULT:histogram_plot=cylindrical_energy_histograms.png')
print('RESULT:resolution_plot=cylindrical_resolution_vs_energy.png')
print('RESULT:results_file=cylindrical_analysis_results.json')
print('RESULT:success=True')

print('\nCylindrical detector analysis completed successfully!')
print(f'Analyzed {len(energies)} energy points: {energies} GeV')
print(f'Generated energy deposition histograms and resolution plots')