import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

# Geometry parameters from Previous Step Outputs
num_layers = 6
total_depth_m = 9.009
sampling_fraction = 0.000999001

# Analysis results storage
results = {}

# Analyze each energy point
for energy_gev in [5.0, 20.0, 50.0]:
    print(f'Analyzing {energy_gev} GeV...')
    
    # Load muon data
    muon_df = pd.read_parquet(muon_files[energy_gev])
    pion_df = pd.read_parquet(pion_files[energy_gev])
    
    # Muon analysis
    muon_total = len(muon_df)
    muon_detected = len(muon_df[muon_df['nHits'] > 0])
    muon_efficiency = muon_detected / muon_total if muon_total > 0 else 0
    muon_eff_err = np.sqrt(muon_efficiency * (1 - muon_efficiency) / muon_total) if muon_total > 0 else 0
    
    muon_mean_edep = muon_df['totalEdep'].mean()
    muon_std_edep = muon_df['totalEdep'].std()
    muon_resolution = muon_std_edep / muon_mean_edep if muon_mean_edep > 0 else 0
    
    # Pion analysis
    pion_total = len(pion_df)
    pion_detected = len(pion_df[pion_df['nHits'] > 0])
    pion_mean_edep = pion_df['totalEdep'].mean()
    pion_std_edep = pion_df['totalEdep'].std()
    
    # Pion rejection (fraction that deposit significantly more energy than muons)
    # Use mean muon energy + 3 sigma as threshold
    threshold = muon_mean_edep + 3 * muon_std_edep
    pion_above_threshold = len(pion_df[pion_df['totalEdep'] > threshold])
    pion_rejection = pion_above_threshold / pion_total if pion_total > 0 else 0
    
    # Store results
    results[energy_gev] = {
        'muon_efficiency': muon_efficiency,
        'muon_eff_err': muon_eff_err,
        'muon_mean_edep': muon_mean_edep,
        'muon_resolution': muon_resolution,
        'pion_mean_edep': pion_mean_edep,
        'pion_rejection': pion_rejection,
        'muon_events': muon_total,
        'pion_events': pion_total
    }

# Create energy deposition histograms
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Planar Detector Energy Deposition Profiles', fontsize=16)

for i, energy_gev in enumerate([5.0, 20.0, 50.0]):
    muon_df = pd.read_parquet(muon_files[energy_gev])
    pion_df = pd.read_parquet(pion_files[energy_gev])
    
    # Energy distribution plots
    axes[0, i].hist(muon_df['totalEdep'], bins=50, alpha=0.7, label='Muons', density=True)
    axes[0, i].hist(pion_df['totalEdep'], bins=50, alpha=0.7, label='Pions', density=True)
    axes[0, i].set_xlabel('Total Energy Deposit (MeV)')
    axes[0, i].set_ylabel('Normalized Events')
    axes[0, i].set_title(f'{energy_gev} GeV')
    axes[0, i].legend()
    axes[0, i].grid(True, alpha=0.3)
    
    # Hit multiplicity plots
    axes[1, i].hist(muon_df['nHits'], bins=range(0, max(muon_df['nHits'])+2), alpha=0.7, label='Muons', density=True)
    axes[1, i].hist(pion_df['nHits'], bins=range(0, max(pion_df['nHits'])+2), alpha=0.7, label='Pions', density=True)
    axes[1, i].set_xlabel('Number of Hits')
    axes[1, i].set_ylabel('Normalized Events')
    axes[1, i].legend()
    axes[1, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('planar_energy_deposition_profiles.png', dpi=150, bbox_inches='tight')
plt.savefig('planar_energy_deposition_profiles.pdf', bbox_inches='tight')

# Performance summary plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

energies = list(results.keys())
muon_effs = [results[e]['muon_efficiency'] for e in energies]
muon_eff_errs = [results[e]['muon_eff_err'] for e in energies]
pion_rejs = [results[e]['pion_rejection'] for e in energies]
muon_ress = [results[e]['muon_resolution'] for e in energies]

# Muon efficiency vs energy
ax1.errorbar(energies, muon_effs, yerr=muon_eff_errs, marker='o', capsize=5)
ax1.set_xlabel('Beam Energy (GeV)')
ax1.set_ylabel('Muon Detection Efficiency')
ax1.set_title('Muon Efficiency vs Energy')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.1)

# Pion rejection vs energy
ax2.plot(energies, pion_rejs, marker='s', color='red')
ax2.set_xlabel('Beam Energy (GeV)')
ax2.set_ylabel('Pion Rejection Fraction')
ax2.set_title('Pion Rejection vs Energy')
ax2.grid(True, alpha=0.3)

# Energy resolution vs energy
ax3.plot(energies, muon_ress, marker='^', color='green')
ax3.set_xlabel('Beam Energy (GeV)')
ax3.set_ylabel('Energy Resolution (σ/E)')
ax3.set_title('Muon Energy Resolution vs Energy')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('planar_performance_summary.png', dpi=150, bbox_inches='tight')
plt.savefig('planar_performance_summary.pdf', bbox_inches='tight')

# Output results
print('\n=== PLANAR DETECTOR PERFORMANCE SUMMARY ===')
for energy_gev in energies:
    r = results[energy_gev]
    print(f'\n{energy_gev} GeV:')
    print(f'  Muon efficiency: {r["muon_efficiency"]:.4f} ± {r["muon_eff_err"]:.4f}')
    print(f'  Muon mean deposit: {r["muon_mean_edep"]:.2f} MeV')
    print(f'  Muon resolution: {r["muon_resolution"]:.4f}')
    print(f'  Pion mean deposit: {r["pion_mean_edep"]:.2f} MeV')
    print(f'  Pion rejection: {r["pion_rejection"]:.4f}')
    print(f'  Events analyzed: {r["muon_events"]} muons, {r["pion_events"]} pions')
    
    # Output standardized results
    print(f'RESULT:planar_{energy_gev}gev_muon_efficiency={r["muon_efficiency"]:.4f}')
    print(f'RESULT:planar_{energy_gev}gev_muon_eff_err={r["muon_eff_err"]:.4f}')
    print(f'RESULT:planar_{energy_gev}gev_muon_mean_edep_mev={r["muon_mean_edep"]:.2f}')
    print(f'RESULT:planar_{energy_gev}gev_muon_resolution={r["muon_resolution"]:.4f}')
    print(f'RESULT:planar_{energy_gev}gev_pion_mean_edep_mev={r["pion_mean_edep"]:.2f}')
    print(f'RESULT:planar_{energy_gev}gev_pion_rejection={r["pion_rejection"]:.4f}')
    print(f'RESULT:planar_{energy_gev}gev_beam_energy_gev={energy_gev}')

# Overall detector characteristics
print(f'\nRESULT:planar_num_layers={num_layers}')
print(f'RESULT:planar_total_depth_m={total_depth_m:.3f}')
print(f'RESULT:planar_sampling_fraction={sampling_fraction:.6f}')
print('RESULT:planar_energy_profiles=planar_energy_deposition_profiles.png')
print('RESULT:planar_performance_summary=planar_performance_summary.png')
print('RESULT:success=True')

# Save detailed results to JSON
import json
with open('planar_performance_results.json', 'w') as f:
    json.dump({
        'detector_config': 'planar',
        'geometry': {
            'num_layers': num_layers,
            'total_depth_m': total_depth_m,
            'sampling_fraction': sampling_fraction
        },
        'energy_points': results
    }, f, indent=2)

print('\nDetailed results saved to planar_performance_results.json')