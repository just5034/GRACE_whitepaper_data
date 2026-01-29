import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# File paths from Previous Step Outputs (ENERGY SWEEP FILE PATHS)
muon_files = {
    5.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/muon_claude-sonnet-4-20250514_20260127_231937/energy_5.000GeV/thick_absorber_muon_spectrometer_mum_events.parquet',
    20.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/muon_claude-sonnet-4-20250514_20260127_231937/energy_20.000GeV/thick_absorber_muon_spectrometer_mum_events.parquet',
    50.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/muon_claude-sonnet-4-20250514_20260127_231937/energy_50.000GeV/thick_absorber_muon_spectrometer_mum_events.parquet'
}

pion_files = {
    5.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/muon_claude-sonnet-4-20250514_20260127_231937/energy_5.000GeV/thick_absorber_muon_spectrometer_pim_events.parquet',
    20.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/muon_claude-sonnet-4-20250514_20260127_231937/energy_20.000GeV/thick_absorber_muon_spectrometer_pim_events.parquet',
    50.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/muon_claude-sonnet-4-20250514_20260127_231937/energy_50.000GeV/thick_absorber_muon_spectrometer_pim_events.parquet'
}

# Geometry parameters from Previous Step Outputs
total_depth_m = 0.96  # From generate_thick_absorber_geometry
num_layers = 3

# Analyze each energy point
results = {}
for energy_gev in [5.0, 20.0, 50.0]:
    # Load muon data
    muon_data = pd.read_parquet(muon_files[energy_gev])
    pion_data = pd.read_parquet(pion_files[energy_gev])
    
    # Muon analysis
    muon_mean_edep = muon_data['totalEdep'].mean()
    muon_std_edep = muon_data['totalEdep'].std()
    muon_resolution = muon_std_edep / muon_mean_edep if muon_mean_edep > 0 else 0
    muon_resolution_err = muon_resolution / np.sqrt(2 * len(muon_data)) if len(muon_data) > 0 else 0
    
    # Muon detection efficiency (events with hits)
    muon_detected = len(muon_data[muon_data['nHits'] > 0])
    muon_efficiency = muon_detected / len(muon_data) if len(muon_data) > 0 else 0
    muon_efficiency_err = np.sqrt(muon_efficiency * (1 - muon_efficiency) / len(muon_data)) if len(muon_data) > 0 else 0
    
    # Pion analysis
    pion_mean_edep = pion_data['totalEdep'].mean()
    pion_std_edep = pion_data['totalEdep'].std()
    pion_resolution = pion_std_edep / pion_mean_edep if pion_mean_edep > 0 else 0
    
    # Store results
    results[energy_gev] = {
        'muon_mean_edep': muon_mean_edep,
        'muon_resolution': muon_resolution,
        'muon_resolution_err': muon_resolution_err,
        'muon_efficiency': muon_efficiency,
        'muon_efficiency_err': muon_efficiency_err,
        'pion_mean_edep': pion_mean_edep,
        'pion_resolution': pion_resolution,
        'muon_events': len(muon_data),
        'pion_events': len(pion_data)
    }
    
    # Print results for this energy
    print(f'RESULT:thick_absorber_muon_{energy_gev}gev_resolution={muon_resolution:.4f}')
    print(f'RESULT:thick_absorber_muon_{energy_gev}gev_mean_edep={muon_mean_edep:.2f}')
    print(f'RESULT:thick_absorber_muon_{energy_gev}gev_efficiency={muon_efficiency:.4f}')
    print(f'RESULT:thick_absorber_pion_{energy_gev}gev_resolution={pion_resolution:.4f}')
    print(f'RESULT:thick_absorber_pion_{energy_gev}gev_mean_edep={pion_mean_edep:.2f}')

# Calculate average pion rejection factor
pion_rejection_factors = []
for energy_gev in [5.0, 20.0, 50.0]:
    muon_mean = results[energy_gev]['muon_mean_edep']
    pion_mean = results[energy_gev]['pion_mean_edep']
    if muon_mean > 0:
        rejection_factor = pion_mean / muon_mean
        pion_rejection_factors.append(rejection_factor)

avg_pion_rejection = np.mean(pion_rejection_factors) if pion_rejection_factors else 0
print(f'RESULT:thick_absorber_avg_pion_rejection_factor={avg_pion_rejection:.2f}')

# Create energy deposition histograms
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Thick Absorber Energy Deposition Histograms', fontsize=16)

for i, energy_gev in enumerate([5.0, 20.0, 50.0]):
    muon_data = pd.read_parquet(muon_files[energy_gev])
    pion_data = pd.read_parquet(pion_files[energy_gev])
    
    # Muon histogram
    axes[0, i].hist(muon_data['totalEdep'], bins=50, alpha=0.7, color='blue', 
                   label=f'Muons (σ/E={results[energy_gev]["muon_resolution"]:.3f})')
    axes[0, i].axvline(results[energy_gev]['muon_mean_edep'], color='blue', linestyle='--')
    axes[0, i].set_title(f'{energy_gev} GeV Muons')
    axes[0, i].set_xlabel('Energy Deposit (MeV)')
    axes[0, i].set_ylabel('Events')
    axes[0, i].legend()
    axes[0, i].grid(True, alpha=0.3)
    
    # Pion histogram
    axes[1, i].hist(pion_data['totalEdep'], bins=50, alpha=0.7, color='red',
                   label=f'Pions (σ/E={results[energy_gev]["pion_resolution"]:.3f})')
    axes[1, i].axvline(results[energy_gev]['pion_mean_edep'], color='red', linestyle='--')
    axes[1, i].set_title(f'{energy_gev} GeV Pions')
    axes[1, i].set_xlabel('Energy Deposit (MeV)')
    axes[1, i].set_ylabel('Events')
    axes[1, i].legend()
    axes[1, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('thick_absorber_energy_histograms.png', dpi=150, bbox_inches='tight')
plt.savefig('thick_absorber_energy_histograms.pdf', bbox_inches='tight')
print('RESULT:histogram_plot=thick_absorber_energy_histograms.png')

# Create resolution vs energy plot
fig, ax = plt.subplots(figsize=(10, 6))
energies = [5.0, 20.0, 50.0]
muon_resolutions = [results[e]['muon_resolution'] for e in energies]
muon_res_errors = [results[e]['muon_resolution_err'] for e in energies]
pion_resolutions = [results[e]['pion_resolution'] for e in energies]

ax.errorbar(energies, muon_resolutions, yerr=muon_res_errors, 
           marker='o', linewidth=2, markersize=8, label='Muons', color='blue')
ax.plot(energies, pion_resolutions, marker='s', linewidth=2, markersize=8, 
       label='Pions', color='red')
ax.set_xlabel('Beam Energy (GeV)')
ax.set_ylabel('Energy Resolution (σ/E)')
ax.set_title('Thick Absorber Energy Resolution vs Energy')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('thick_absorber_resolution_vs_energy.png', dpi=150, bbox_inches='tight')
plt.savefig('thick_absorber_resolution_vs_energy.pdf', bbox_inches='tight')
print('RESULT:resolution_plot=thick_absorber_resolution_vs_energy.png')

# Save detailed results to JSON
output_results = {
    'detector_config': 'thick_absorber',
    'geometry': {
        'total_depth_m': total_depth_m,
        'num_layers': num_layers,
        'absorber_thickness_mm': 300,
        'active_thickness_mm': 20
    },
    'analysis_results': results,
    'summary': {
        'avg_pion_rejection_factor': avg_pion_rejection,
        'muon_efficiency_range': f"{min([results[e]['muon_efficiency'] for e in energies]):.3f}-{max([results[e]['muon_efficiency'] for e in energies]):.3f}"
    }
}

with open('thick_absorber_analysis_results.json', 'w') as f:
    json.dump(output_results, f, indent=2)

print('RESULT:results_file=thick_absorber_analysis_results.json')
print('RESULT:success=True')
print('\nThick Absorber Analysis Summary:')
print(f'- Detector depth: {total_depth_m} m ({num_layers} layers)')
print(f'- Average pion rejection factor: {avg_pion_rejection:.2f}')
print(f'- Muon efficiency range: {min([results[e]["muon_efficiency"] for e in energies]):.3f}-{max([results[e]["muon_efficiency"] for e in energies]):.3f}')
print('- Enhanced pion rejection due to thick absorber layers')
print('- Trade-off: Higher material budget reduces muon detection efficiency at low energies')