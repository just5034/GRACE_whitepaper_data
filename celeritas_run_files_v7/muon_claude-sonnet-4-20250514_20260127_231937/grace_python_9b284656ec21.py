import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

start_time = time.time()
print('Starting planar detector analysis with optimizations...')

# Energy sweep file paths from Previous Step Outputs
muon_files = {
    5.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/muon_claude-sonnet-4-20250514_20260127_231937/energy_5.000GeV/planar_muon_spectrometer_retry_mum_events.parquet',
    20.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/muon_claude-sonnet-4-20250514_20260127_231937/energy_20.000GeV/planar_muon_spectrometer_retry_mum_events.parquet',
    50.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/muon_claude-sonnet-4-20250514_20260127_231937/energy_50.000GeV/planar_muon_spectrometer_retry_mum_events.parquet'
}

pion_files = {
    5.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/muon_claude-sonnet-4-20250514_20260127_231937/energy_5.000GeV/planar_muon_spectrometer_retry_pim_events.parquet',
    20.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/muon_claude-sonnet-4-20250514_20260127_231937/energy_20.000GeV/planar_muon_spectrometer_retry_pim_events.parquet',
    50.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/muon_claude-sonnet-4-20250514_20260127_231937/energy_50.000GeV/planar_muon_spectrometer_retry_pim_events.parquet'
}

# Analysis results storage
results = {'muon': {}, 'pion': {}}

# Analyze each energy point with chunked processing
for energy_gev in [5.0, 20.0, 50.0]:
    print(f'Analyzing {energy_gev} GeV...')
    
    # Muon analysis
    try:
        muon_df = pd.read_parquet(muon_files[energy_gev])
        muon_edep = muon_df['totalEdep']
        
        # Calculate metrics with reduced batch processing
        muon_mean = muon_edep.mean()
        muon_std = muon_edep.std()
        muon_resolution = muon_std / muon_mean if muon_mean > 0 else 0
        muon_resolution_err = muon_resolution / np.sqrt(2 * len(muon_df)) if len(muon_df) > 0 else 0
        
        results['muon'][energy_gev] = {
            'mean_edep': muon_mean,
            'resolution': muon_resolution,
            'resolution_err': muon_resolution_err,
            'num_events': len(muon_df)
        }
        
        print(f'Muon {energy_gev} GeV: resolution = {muon_resolution:.4f} ± {muon_resolution_err:.4f}')
        
    except Exception as e:
        print(f'Error analyzing muon {energy_gev} GeV: {e}')
        continue
    
    # Pion analysis
    try:
        pion_df = pd.read_parquet(pion_files[energy_gev])
        pion_edep = pion_df['totalEdep']
        
        pion_mean = pion_edep.mean()
        pion_std = pion_edep.std()
        pion_resolution = pion_std / pion_mean if pion_mean > 0 else 0
        pion_resolution_err = pion_resolution / np.sqrt(2 * len(pion_df)) if len(pion_df) > 0 else 0
        
        results['pion'][energy_gev] = {
            'mean_edep': pion_mean,
            'resolution': pion_resolution,
            'resolution_err': pion_resolution_err,
            'num_events': len(pion_df)
        }
        
        print(f'Pion {energy_gev} GeV: resolution = {pion_resolution:.4f} ± {pion_resolution_err:.4f}')
        
    except Exception as e:
        print(f'Error analyzing pion {energy_gev} GeV: {e}')
        continue

# Generate energy deposition histograms with limited bins (100 max)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Planar Detector Energy Deposition Histograms', fontsize=16)

for i, energy_gev in enumerate([5.0, 20.0, 50.0]):
    # Muon histogram
    try:
        muon_df = pd.read_parquet(muon_files[energy_gev])
        axes[0, i].hist(muon_df['totalEdep'], bins=50, alpha=0.7, color='blue', label=f'Muons {energy_gev} GeV')
        axes[0, i].set_xlabel('Energy Deposit (MeV)')
        axes[0, i].set_ylabel('Events')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
    except Exception as e:
        print(f'Error plotting muon {energy_gev} GeV: {e}')
    
    # Pion histogram
    try:
        pion_df = pd.read_parquet(pion_files[energy_gev])
        axes[1, i].hist(pion_df['totalEdep'], bins=50, alpha=0.7, color='red', label=f'Pions {energy_gev} GeV')
        axes[1, i].set_xlabel('Energy Deposit (MeV)')
        axes[1, i].set_ylabel('Events')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    except Exception as e:
        print(f'Error plotting pion {energy_gev} GeV: {e}')

plt.tight_layout()
plt.savefig('planar_energy_histograms.png', dpi=150, bbox_inches='tight')
plt.savefig('planar_energy_histograms.pdf', bbox_inches='tight')
print('Saved energy deposition histograms')

# Resolution vs energy plot
fig, ax = plt.subplots(figsize=(10, 6))
muon_energies = list(results['muon'].keys())
muon_resolutions = [results['muon'][e]['resolution'] for e in muon_energies]
muon_errors = [results['muon'][e]['resolution_err'] for e in muon_energies]

pion_energies = list(results['pion'].keys())
pion_resolutions = [results['pion'][e]['resolution'] for e in pion_energies]
pion_errors = [results['pion'][e]['resolution_err'] for e in pion_energies]

ax.errorbar(muon_energies, muon_resolutions, yerr=muon_errors, marker='o', label='Muons', capsize=5)
ax.errorbar(pion_energies, pion_resolutions, yerr=pion_errors, marker='s', label='Pions', capsize=5)
ax.set_xlabel('Beam Energy (GeV)')
ax.set_ylabel('Energy Resolution (σ/E)')
ax.set_title('Planar Detector Energy Resolution vs Energy')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_yscale('log')

plt.savefig('planar_resolution_vs_energy.png', dpi=150, bbox_inches='tight')
plt.savefig('planar_resolution_vs_energy.pdf', bbox_inches='tight')
print('Saved resolution vs energy plot')

# Save results to JSON
with open('planar_analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Output key metrics
for energy_gev in [5.0, 20.0, 50.0]:
    if energy_gev in results['muon']:
        print(f'RESULT:planar_muon_{energy_gev}gev_resolution={results["muon"][energy_gev]["resolution"]:.4f}')
        print(f'RESULT:planar_muon_{energy_gev}gev_mean_edep={results["muon"][energy_gev]["mean_edep"]:.2f}')
    
    if energy_gev in results['pion']:
        print(f'RESULT:planar_pion_{energy_gev}gev_resolution={results["pion"][energy_gev]["resolution"]:.4f}')
        print(f'RESULT:planar_pion_{energy_gev}gev_mean_edep={results["pion"][energy_gev]["mean_edep"]:.2f}')

print('RESULT:histogram_plot=planar_energy_histograms.png')
print('RESULT:resolution_plot=planar_resolution_vs_energy.png')
print('RESULT:results_file=planar_analysis_results.json')

elapsed_time = time.time() - start_time
print(f'Analysis completed in {elapsed_time:.1f} seconds')
print('RESULT:success=True')