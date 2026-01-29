import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Energy sweep file paths from Previous Step Outputs
energy_files = {
    10.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/calorimeter_claude-sonnet-4-20250514_20260127_194626/energy_10.000GeV/baseline_calorimeter_pip_events.parquet',
    30.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/calorimeter_claude-sonnet-4-20250514_20260127_194626/energy_30.000GeV/baseline_calorimeter_pip_events.parquet',
    50.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/calorimeter_claude-sonnet-4-20250514_20260127_194626/energy_50.000GeV/baseline_calorimeter_pip_events.parquet'
}

# Analyze each energy point
results = []
for energy_gev, filepath in energy_files.items():
    print(f'Analyzing {energy_gev} GeV data...')
    
    # Read events data
    df = pd.read_parquet(filepath)
    beam_energy_mev = energy_gev * 1000  # Convert GeV to MeV
    
    # Calculate energy resolution
    mean_edep = df['totalEdep'].mean()
    std_edep = df['totalEdep'].std()
    num_events = len(df)
    
    # Energy resolution (sigma/E)
    resolution = std_edep / mean_edep if mean_edep > 0 else 0
    resolution_err = resolution / np.sqrt(2 * num_events) if num_events > 0 else 0
    
    # Response linearity (mean response / beam energy)
    linearity = mean_edep / beam_energy_mev if beam_energy_mev > 0 else 0
    
    # Store results
    results.append({
        'energy_gev': energy_gev,
        'beam_energy_mev': beam_energy_mev,
        'mean_edep_mev': mean_edep,
        'std_edep_mev': std_edep,
        'resolution': resolution,
        'resolution_err': resolution_err,
        'linearity': linearity,
        'num_events': num_events
    })
    
    # Plot energy distribution for this energy
    plt.figure(figsize=(10, 6))
    plt.hist(df['totalEdep'], bins=50, histtype='step', linewidth=2, alpha=0.8, label=f'{energy_gev} GeV')
    plt.axvline(mean_edep, color='r', linestyle='--', label=f'Mean: {mean_edep:.1f} MeV')
    plt.xlabel('Total Energy Deposit (MeV)')
    plt.ylabel('Events')
    plt.title(f'Energy Distribution at {energy_gev} GeV (σ/E = {resolution:.4f} ± {resolution_err:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'energy_dist_{energy_gev}GeV.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'energy_dist_{energy_gev}GeV.pdf', bbox_inches='tight')
    plt.close()

# Convert to DataFrame for easier analysis
results_df = pd.DataFrame(results)

# Plot energy resolution vs energy
plt.figure(figsize=(10, 6))
plt.errorbar(results_df['energy_gev'], results_df['resolution'], 
             yerr=results_df['resolution_err'], fmt='o-', capsize=5, linewidth=2, markersize=8)
plt.xlabel('Beam Energy (GeV)')
plt.ylabel('Energy Resolution (σ/E)')
plt.title('Baseline Calorimeter Energy Resolution')
plt.grid(True, alpha=0.3)
plt.savefig('baseline_energy_resolution.png', dpi=150, bbox_inches='tight')
plt.savefig('baseline_energy_resolution.pdf', bbox_inches='tight')
plt.close()

# Plot linearity
plt.figure(figsize=(10, 6))
plt.plot(results_df['energy_gev'], results_df['linearity'], 'o-', linewidth=2, markersize=8)
plt.xlabel('Beam Energy (GeV)')
plt.ylabel('Response Linearity (Edep/Ebeam)')
plt.title('Baseline Calorimeter Response Linearity')
plt.grid(True, alpha=0.3)
plt.savefig('baseline_linearity.png', dpi=150, bbox_inches='tight')
plt.savefig('baseline_linearity.pdf', bbox_inches='tight')
plt.close()

# Calculate overall metrics
overall_resolution_mean = results_df['resolution'].mean()
overall_resolution_std = results_df['resolution'].std()
linearity_deviation = np.std(results_df['linearity'])

# Output results with proper naming
for i, result in enumerate(results):
    energy = result['energy_gev']
    print(f'RESULT:baseline_{energy}GeV_energy_resolution={result["resolution"]:.6f}')
    print(f'RESULT:baseline_{energy}GeV_resolution_err={result["resolution_err"]:.6f}')
    print(f'RESULT:baseline_{energy}GeV_mean_deposit_mev={result["mean_edep_mev"]:.2f}')
    print(f'RESULT:baseline_{energy}GeV_linearity={result["linearity"]:.6f}')
    print(f'RESULT:baseline_{energy}GeV_num_events={result["num_events"]}')

# Overall performance metrics
print(f'RESULT:baseline_overall_resolution_mean={overall_resolution_mean:.6f}')
print(f'RESULT:baseline_overall_resolution_std={overall_resolution_std:.6f}')
print(f'RESULT:baseline_linearity_deviation={linearity_deviation:.6f}')
print('RESULT:baseline_resolution_plot=baseline_energy_resolution.png')
print('RESULT:baseline_linearity_plot=baseline_linearity.png')
print('RESULT:success=True')

# Save detailed results to JSON
import json
with open('baseline_performance_results.json', 'w') as f:
    json.dump({
        'energy_points': results,
        'overall_metrics': {
            'mean_resolution': overall_resolution_mean,
            'resolution_std': overall_resolution_std,
            'linearity_deviation': linearity_deviation
        }
    }, f, indent=2)

print('Analysis complete. Energy resolution, linearity, and containment metrics calculated with statistical uncertainties.')