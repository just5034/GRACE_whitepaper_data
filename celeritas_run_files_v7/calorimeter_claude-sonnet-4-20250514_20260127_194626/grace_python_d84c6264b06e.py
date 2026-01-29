import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Energy sweep file paths from Previous Step Outputs
energy_files = {
    10.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/calorimeter_claude-sonnet-4-20250514_20260127_194626/energy_10.000GeV/tungsten_barrel_geometry_retry_pip_events.parquet',
    30.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/calorimeter_claude-sonnet-4-20250514_20260127_194626/energy_30.000GeV/tungsten_barrel_geometry_retry_pip_events.parquet',
    50.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/calorimeter_claude-sonnet-4-20250514_20260127_194626/energy_50.000GeV/tungsten_barrel_geometry_retry_pip_events.parquet'
}

# Analyze each energy point
results = []
for energy_gev, filepath in energy_files.items():
    print(f'Analyzing {energy_gev} GeV data...')
    
    # Read events data
    df = pd.read_parquet(filepath)
    
    # Calculate performance metrics
    beam_energy_mev = energy_gev * 1000
    mean_deposit_mev = df['totalEdep'].mean()
    std_deposit_mev = df['totalEdep'].std()
    num_events = len(df)
    
    # Energy resolution (sigma/mean)
    energy_resolution = std_deposit_mev / mean_deposit_mev if mean_deposit_mev > 0 else 0
    resolution_err = energy_resolution / np.sqrt(2 * num_events) if num_events > 0 else 0
    
    # Response linearity (measured/expected)
    linearity = mean_deposit_mev / beam_energy_mev if beam_energy_mev > 0 else 0
    
    # Store results
    results.append({
        'energy_gev': energy_gev,
        'beam_energy_mev': beam_energy_mev,
        'mean_deposit_mev': mean_deposit_mev,
        'energy_resolution': energy_resolution,
        'resolution_err': resolution_err,
        'linearity': linearity,
        'num_events': num_events
    })
    
    # Output individual energy results
    print(f'RESULT:tungsten_{energy_gev}GeV_energy_resolution={energy_resolution:.6f}')
    print(f'RESULT:tungsten_{energy_gev}GeV_resolution_err={resolution_err:.6f}')
    print(f'RESULT:tungsten_{energy_gev}GeV_mean_deposit_mev={mean_deposit_mev:.2f}')
    print(f'RESULT:tungsten_{energy_gev}GeV_linearity={linearity:.6f}')
    print(f'RESULT:tungsten_{energy_gev}GeV_num_events={num_events}')

# Calculate overall performance metrics
resolutions = [r['energy_resolution'] for r in results]
linearities = [r['linearity'] for r in results]

overall_resolution_mean = np.mean(resolutions)
overall_resolution_std = np.std(resolutions)
linearity_deviation = np.std(linearities)

# Create performance plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Energy resolution vs energy
energies = [r['energy_gev'] for r in results]
resolution_errs = [r['resolution_err'] for r in results]
ax1.errorbar(energies, resolutions, yerr=resolution_errs, marker='o', capsize=5, linewidth=2)
ax1.set_xlabel('Beam Energy (GeV)')
ax1.set_ylabel('Energy Resolution (σ/E)')
ax1.set_title('Tungsten Barrel Energy Resolution')
ax1.grid(True, alpha=0.3)

# Linearity vs energy
ax2.plot(energies, linearities, 'o-', linewidth=2, markersize=8)
ax2.set_xlabel('Beam Energy (GeV)')
ax2.set_ylabel('Response Linearity (Measured/Expected)')
ax2.set_title('Tungsten Barrel Response Linearity')
ax2.grid(True, alpha=0.3)
ax2.axhline(1.0, color='r', linestyle='--', alpha=0.7, label='Perfect linearity')
ax2.legend()

# Energy distribution for highest energy
highest_energy_data = pd.read_parquet(energy_files[50.0])
ax3.hist(highest_energy_data['totalEdep'], bins=50, histtype='step', linewidth=2, density=True)
ax3.axvline(highest_energy_data['totalEdep'].mean(), color='r', linestyle='--', 
           label=f'Mean: {highest_energy_data["totalEdep"].mean():.1f} MeV')
ax3.set_xlabel('Energy Deposit (MeV)')
ax3.set_ylabel('Normalized Events')
ax3.set_title('Energy Distribution (50 GeV)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Mean energy deposit vs beam energy
mean_deposits = [r['mean_deposit_mev'] for r in results]
beam_energies_mev = [r['beam_energy_mev'] for r in results]
ax4.plot(beam_energies_mev, mean_deposits, 'o-', linewidth=2, markersize=8, label='Measured')
ax4.plot(beam_energies_mev, beam_energies_mev, 'r--', alpha=0.7, label='Perfect response')
ax4.set_xlabel('Beam Energy (MeV)')
ax4.set_ylabel('Mean Energy Deposit (MeV)')
ax4.set_title('Energy Response')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tungsten_performance_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('tungsten_performance_analysis.pdf', bbox_inches='tight')
plt.close()

# Output overall results
print(f'RESULT:tungsten_overall_resolution_mean={overall_resolution_mean:.6f}')
print(f'RESULT:tungsten_overall_resolution_std={overall_resolution_std:.6f}')
print(f'RESULT:tungsten_linearity_deviation={linearity_deviation:.6f}')
print('RESULT:tungsten_resolution_plot=tungsten_performance_analysis.png')
print('RESULT:tungsten_linearity_plot=tungsten_performance_analysis.png')
print('RESULT:success=True')

print('\nTungsten Barrel Calorimeter Performance Summary:')
print(f'Average Energy Resolution: {overall_resolution_mean:.4f} ± {overall_resolution_std:.4f}')
print(f'Linearity Deviation: {linearity_deviation:.4f}')
print(f'Energy range analyzed: {min(energies)}-{max(energies)} GeV')
print(f'Total events analyzed: {sum(r["num_events"] for r in results)}')