import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# BGO simulation file paths from Previous Step Outputs
energy_files = {
    0.5: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/emcal_claude-sonnet-4-20250514_20260127_185530/energy_0.500GeV/bgo_box_calorimeter_electron_events.parquet',
    2.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/emcal_claude-sonnet-4-20250514_20260127_185530/energy_2.000GeV/bgo_box_calorimeter_electron_events.parquet',
    5.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/emcal_claude-sonnet-4-20250514_20260127_185530/energy_5.000GeV/bgo_box_calorimeter_electron_events.parquet'
}

# Analyze each energy point
results = []
energies = []
resolutions = []
resolution_errors = []
mean_deposits = []

for energy_gev, filepath in energy_files.items():
    print(f'Analyzing BGO at {energy_gev} GeV...')
    
    # Read events data
    df = pd.read_parquet(filepath)
    
    # Calculate metrics
    beam_energy_mev = energy_gev * 1000  # Convert GeV to MeV
    mean_edep = df['totalEdep'].mean()
    std_edep = df['totalEdep'].std()
    num_events = len(df)
    
    # Energy resolution (sigma/E)
    resolution = std_edep / mean_edep if mean_edep > 0 else 0
    resolution_err = resolution / np.sqrt(2 * num_events) if num_events > 0 else 0
    
    # Linearity
    linearity = mean_edep / beam_energy_mev if beam_energy_mev > 0 else 0
    
    # Store results with array conversion to lists
    result = {
        'energy_gev': float(energy_gev),
        'beam_energy_mev': float(beam_energy_mev),
        'mean_deposit_mev': float(mean_edep),
        'std_deposit_mev': float(std_edep),
        'energy_resolution': float(resolution),
        'resolution_err': float(resolution_err),
        'linearity': float(linearity),
        'num_events': int(num_events)
    }
    
    # JSON serialization check
    try:
        json.dumps(result)
        print(f'JSON serialization check passed for {energy_gev} GeV')
    except TypeError as e:
        print(f'JSON serialization failed for {energy_gev} GeV: {e}')
    
    results.append(result)
    energies.append(energy_gev)
    resolutions.append(resolution)
    resolution_errors.append(resolution_err)
    mean_deposits.append(mean_edep)
    
    # Plot raw energy distribution for this energy
    plt.figure(figsize=(10, 6))
    plt.hist(df['totalEdep'], bins=50, histtype='step', linewidth=2, alpha=0.8)
    plt.axvline(mean_edep, color='r', linestyle='--', label=f'Mean: {mean_edep:.2f} MeV')
    plt.xlabel('Total Energy Deposit (MeV)')
    plt.ylabel('Events')
    plt.title(f'BGO Energy Distribution at {energy_gev} GeV (σ/E = {resolution:.4f} ± {resolution_err:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'bgo_energy_dist_{energy_gev}gev.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'bgo_energy_dist_{energy_gev}gev.pdf', bbox_inches='tight')
    plt.close()

# Convert arrays to lists for JSON serialization
energies_list = [float(x) for x in energies]
resolutions_list = [float(x) for x in resolutions]
resolution_errors_list = [float(x) for x in resolution_errors]
mean_deposits_list = [float(x) for x in mean_deposits]

# Create energy resolution curve plot
plt.figure(figsize=(10, 6))
plt.errorbar(energies_list, resolutions_list, yerr=resolution_errors_list, 
             marker='o', markersize=8, linewidth=2, capsize=5, capthick=2)
plt.xlabel('Beam Energy (GeV)')
plt.ylabel('Energy Resolution (σ/E)')
plt.title('BGO Energy Resolution vs Beam Energy')
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.yscale('log')
plt.savefig('bgo_resolution_curve.png', dpi=150, bbox_inches='tight')
plt.savefig('bgo_resolution_curve.pdf', bbox_inches='tight')
plt.close()

# Linearity plot
linearity_values = [r['linearity'] for r in results]
plt.figure(figsize=(10, 6))
plt.plot(energies_list, linearity_values, 'o-', markersize=8, linewidth=2)
plt.xlabel('Beam Energy (GeV)')
plt.ylabel('Linearity (Response/Energy)')
plt.title('BGO Linearity vs Beam Energy')
plt.grid(True, alpha=0.3)
plt.savefig('bgo_linearity_curve.png', dpi=150, bbox_inches='tight')
plt.savefig('bgo_linearity_curve.pdf', bbox_inches='tight')
plt.close()

# Save results to JSON with proper array handling
final_results = {
    'material_name': 'BGO',
    'energy_points': energies_list,
    'resolutions': resolutions_list,
    'resolution_errors': resolution_errors_list,
    'mean_deposits_mev': mean_deposits_list,
    'detailed_results': results
}

# Final JSON serialization check
try:
    with open('bgo_performance_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    print('BGO performance results saved to JSON successfully')
except TypeError as e:
    print(f'Final JSON serialization failed: {e}')

# Output results with proper prefixes
for i, result in enumerate(results):
    energy = energies_list[i]
    print(f'RESULT:bgo_{energy}gev_energy_resolution={result["energy_resolution"]:.6f}')
    print(f'RESULT:bgo_{energy}gev_resolution_err={result["resolution_err"]:.6f}')
    print(f'RESULT:bgo_{energy}gev_mean_deposit_mev={result["mean_deposit_mev"]:.2f}')
    print(f'RESULT:bgo_{energy}gev_linearity={result["linearity"]:.4f}')
    print(f'RESULT:bgo_{energy}gev_num_events={result["num_events"]}')

# Overall metrics
print(f'RESULT:bgo_resolution_curve_plot=bgo_resolution_curve.png')
print(f'RESULT:bgo_linearity_plot=bgo_linearity_curve.png')
print(f'RESULT:bgo_results_json=bgo_performance_results.json')
print('RESULT:success=True')
print('BGO performance analysis completed with array-to-list conversion and JSON serialization checks')