import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Energy sweep file paths from Previous Step Outputs
energy_files = {
    10.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/calorimeter_claude-sonnet-4-20250514_20260127_194626/energy_10.000GeV/projective_geometry_pip_events.parquet',
    30.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/calorimeter_claude-sonnet-4-20250514_20260127_194626/energy_30.000GeV/projective_geometry_pip_events.parquet',
    50.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/calorimeter_claude-sonnet-4-20250514_20260127_194626/energy_50.000GeV/projective_geometry_pip_events.parquet'
}

hits_files = {
    10.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/calorimeter_claude-sonnet-4-20250514_20260127_194626/energy_10.000GeV/projective_geometry_pip_hits_data.parquet',
    30.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/calorimeter_claude-sonnet-4-20250514_20260127_194626/energy_30.000GeV/projective_geometry_pip_hits_data.parquet',
    50.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/calorimeter_claude-sonnet-4-20250514_20260127_194626/energy_50.000GeV/projective_geometry_pip_hits_data.parquet'
}

# Analyze each energy point
results = []
for energy_gev in [10.0, 30.0, 50.0]:
    print(f'Analyzing {energy_gev} GeV data...')
    
    # Load events data
    events_df = pd.read_parquet(energy_files[energy_gev])
    
    # Calculate energy resolution
    mean_edep = events_df['totalEdep'].mean()
    std_edep = events_df['totalEdep'].std()
    resolution = std_edep / mean_edep if mean_edep > 0 else 0
    resolution_err = resolution / np.sqrt(2 * len(events_df))
    
    # Calculate linearity (response vs beam energy)
    beam_energy_mev = energy_gev * 1000  # Convert GeV to MeV
    linearity = mean_edep / beam_energy_mev if beam_energy_mev > 0 else 0
    
    # Store results
    results.append({
        'energy_gev': energy_gev,
        'mean_deposit_mev': mean_edep,
        'energy_resolution': resolution,
        'resolution_err': resolution_err,
        'linearity': linearity,
        'num_events': len(events_df)
    })
    
    # Output individual energy results
    print(f'RESULT:projective_{energy_gev}GeV_energy_resolution={resolution:.6f}')
    print(f'RESULT:projective_{energy_gev}GeV_resolution_err={resolution_err:.6f}')
    print(f'RESULT:projective_{energy_gev}GeV_mean_deposit_mev={mean_edep:.2f}')
    print(f'RESULT:projective_{energy_gev}GeV_linearity={linearity:.6f}')
    print(f'RESULT:projective_{energy_gev}GeV_num_events={len(events_df)}')

# Calculate overall performance metrics
resolutions = [r['energy_resolution'] for r in results]
linearities = [r['linearity'] for r in results]

overall_resolution_mean = np.mean(resolutions)
overall_resolution_std = np.std(resolutions)
linearity_deviation = np.std(linearities)

# Create performance plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Energy resolution vs energy
energies = [r['energy_gev'] for r in results]
res_values = [r['energy_resolution'] for r in results]
res_errors = [r['resolution_err'] for r in results]

ax1.errorbar(energies, res_values, yerr=res_errors, marker='o', capsize=5, linewidth=2)
ax1.set_xlabel('Beam Energy (GeV)')
ax1.set_ylabel('Energy Resolution (σ/E)')
ax1.set_title('Projective Tower Energy Resolution')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, max(res_values) * 1.2)

# Linearity plot
ax2.plot(energies, linearities, 'o-', linewidth=2, markersize=8)
ax2.axhline(1.0, color='r', linestyle='--', alpha=0.7, label='Perfect linearity')
ax2.set_xlabel('Beam Energy (GeV)')
ax2.set_ylabel('Response Linearity (Edep/Ebeam)')
ax2.set_title('Projective Tower Response Linearity')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('projective_performance_plots.png', dpi=150, bbox_inches='tight')
plt.savefig('projective_performance_plots.pdf', bbox_inches='tight')
plt.show()

# Analyze shower containment using hits data (sample first 100k hits to avoid timeout)
print('Analyzing shower containment...')
containment_results = []

for energy_gev in [10.0, 30.0, 50.0]:
    hits_df = pd.read_parquet(hits_files[energy_gev])
    
    # Sample first 100k hits to avoid timeout on large files
    if len(hits_df) > 100000:
        hits_sample = hits_df.head(100000)
        print(f'Sampling first 100k hits from {len(hits_df)} total hits at {energy_gev} GeV')
    else:
        hits_sample = hits_df
    
    # Calculate radial distance from origin
    r = np.sqrt(hits_sample['x']**2 + hits_sample['y']**2)
    
    # Calculate containment fractions
    total_edep = hits_sample['edep'].sum()
    if total_edep > 0:
        # Find 90% containment radius
        sorted_indices = np.argsort(r)
        cumulative_edep = np.cumsum(hits_sample['edep'].iloc[sorted_indices])
        containment_90_idx = np.where(cumulative_edep >= 0.9 * total_edep)[0]
        
        if len(containment_90_idx) > 0:
            r90 = r.iloc[sorted_indices[containment_90_idx[0]]]
        else:
            r90 = r.max()
    else:
        r90 = 0
    
    containment_results.append({
        'energy_gev': energy_gev,
        'r90_containment_mm': r90,
        'total_hits_analyzed': len(hits_sample)
    })
    
    print(f'RESULT:projective_{energy_gev}GeV_r90_containment_mm={r90:.1f}')

# Output overall performance metrics
print(f'RESULT:projective_overall_resolution_mean={overall_resolution_mean:.6f}')
print(f'RESULT:projective_overall_resolution_std={overall_resolution_std:.6f}')
print(f'RESULT:projective_linearity_deviation={linearity_deviation:.6f}')
print('RESULT:projective_resolution_plot=projective_performance_plots.png')
print('RESULT:projective_linearity_plot=projective_performance_plots.png')
print('RESULT:success=True')

print('\nProjective tower calorimeter performance analysis completed!')
print(f'Mean energy resolution: {overall_resolution_mean:.4f} ± {overall_resolution_std:.4f}')
print(f'Linearity variation: {linearity_deviation:.4f}')