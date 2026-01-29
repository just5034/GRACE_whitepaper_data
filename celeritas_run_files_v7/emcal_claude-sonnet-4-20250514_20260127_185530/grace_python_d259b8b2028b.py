import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Box geometry results from PbWO4 simulations (from Previous Step Outputs)
box_0_5gev_resolution = 0.033044
box_2_0gev_resolution = 0.014936
box_5_0gev_resolution = 0.019751
box_0_5gev_mean = 483.91
box_2_0gev_mean = 1934.88
box_5_0gev_mean = 4818.5
box_0_5gev_linearity = 0.9678
box_2_0gev_linearity = 0.9674
box_5_0gev_linearity = 0.9637

# Load projective tower geometry results
proj_energies = [0.5, 2.0, 5.0]
proj_files = {
    0.5: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/emcal_claude-sonnet-4-20250514_20260127_185530/energy_0.500GeV/optimal_projective_tower_calorimeter_retry_electron_events.parquet',
    2.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/emcal_claude-sonnet-4-20250514_20260127_185530/energy_2.000GeV/optimal_projective_tower_calorimeter_retry_electron_events.parquet',
    5.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/emcal_claude-sonnet-4-20250514_20260127_185530/energy_5.000GeV/optimal_projective_tower_calorimeter_retry_electron_events.parquet'
}

proj_hits_files = {
    0.5: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/emcal_claude-sonnet-4-20250514_20260127_185530/energy_0.500GeV/optimal_projective_tower_calorimeter_retry_electron_hits_data.parquet',
    2.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/emcal_claude-sonnet-4-20250514_20260127_185530/energy_2.000GeV/optimal_projective_tower_calorimeter_retry_electron_hits_data.parquet',
    5.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/emcal_claude-sonnet-4-20250514_20260127_185530/energy_5.000GeV/optimal_projective_tower_calorimeter_retry_electron_hits_data.parquet'
}

# Analyze projective tower results
proj_results = {}
for energy_gev in proj_energies:
    events_df = pd.read_parquet(proj_files[energy_gev])
    
    # Calculate energy resolution and linearity
    mean_edep = events_df['totalEdep'].mean()
    std_edep = events_df['totalEdep'].std()
    resolution = std_edep / mean_edep if mean_edep > 0 else 0
    resolution_err = resolution / np.sqrt(2 * len(events_df))
    
    beam_energy_mev = energy_gev * 1000
    linearity = mean_edep / beam_energy_mev if beam_energy_mev > 0 else 0
    
    proj_results[energy_gev] = {
        'resolution': resolution,
        'resolution_err': resolution_err,
        'mean_deposit_mev': mean_edep,
        'linearity': linearity,
        'num_events': len(events_df)
    }
    
    print(f'Projective {energy_gev} GeV: resolution={resolution:.6f}, mean={mean_edep:.2f} MeV, linearity={linearity:.4f}')

# Create comparison plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Energy resolution comparison
energies = [0.5, 2.0, 5.0]
box_resolutions = [box_0_5gev_resolution, box_2_0gev_resolution, box_5_0gev_resolution]
proj_resolutions = [proj_results[e]['resolution'] for e in energies]
box_res_errs = [0.00033, 0.000149, 0.000198]  # From previous step outputs
proj_res_errs = [proj_results[e]['resolution_err'] for e in energies]

ax1.errorbar(energies, box_resolutions, yerr=box_res_errs, marker='o', label='Box Geometry', capsize=5)
ax1.errorbar(energies, proj_resolutions, yerr=proj_res_errs, marker='s', label='Projective Tower', capsize=5)
ax1.set_xlabel('Beam Energy (GeV)')
ax1.set_ylabel('Energy Resolution (σ/E)')
ax1.set_title('Energy Resolution Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')
ax1.set_yscale('log')

# Linearity comparison
box_linearities = [box_0_5gev_linearity, box_2_0gev_linearity, box_5_0gev_linearity]
proj_linearities = [proj_results[e]['linearity'] for e in energies]

ax2.plot(energies, box_linearities, 'o-', label='Box Geometry')
ax2.plot(energies, proj_linearities, 's-', label='Projective Tower')
ax2.set_xlabel('Beam Energy (GeV)')
ax2.set_ylabel('Linearity (E_dep/E_beam)')
ax2.set_title('Linearity Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.94, 0.98)

# Mean energy deposit comparison
box_means = [box_0_5gev_mean, box_2_0gev_mean, box_5_0gev_mean]
proj_means = [proj_results[e]['mean_deposit_mev'] for e in energies]

ax3.plot(energies, box_means, 'o-', label='Box Geometry')
ax3.plot(energies, proj_means, 's-', label='Projective Tower')
ax3.set_xlabel('Beam Energy (GeV)')
ax3.set_ylabel('Mean Energy Deposit (MeV)')
ax3.set_title('Mean Energy Deposit Comparison')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Resolution improvement plot
resolution_improvements = [(box_resolutions[i] - proj_resolutions[i]) / box_resolutions[i] * 100 for i in range(3)]

ax4.bar(energies, resolution_improvements, alpha=0.7, color='green')
ax4.set_xlabel('Beam Energy (GeV)')
ax4.set_ylabel('Resolution Improvement (%)')
ax4.set_title('Resolution Improvement: Projective vs Box')
ax4.grid(True, alpha=0.3)
ax4.axhline(0, color='black', linestyle='-', alpha=0.5)

plt.tight_layout()
plt.savefig('geometry_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('geometry_comparison.pdf', bbox_inches='tight')

# Calculate average performance differences
avg_box_resolution = np.mean(box_resolutions)
avg_proj_resolution = np.mean(proj_resolutions)
avg_resolution_improvement = (avg_box_resolution - avg_proj_resolution) / avg_box_resolution * 100

avg_box_linearity = np.mean(box_linearities)
avg_proj_linearity = np.mean(proj_linearities)
linearity_difference = avg_proj_linearity - avg_box_linearity

# Summary statistics
print('\n=== GEOMETRY COMPARISON SUMMARY ===')
print(f'Average Energy Resolution:')
print(f'  Box Geometry: {avg_box_resolution:.6f}')
print(f'  Projective Tower: {avg_proj_resolution:.6f}')
print(f'  Improvement: {avg_resolution_improvement:.2f}%')
print(f'Average Linearity:')
print(f'  Box Geometry: {avg_box_linearity:.4f}')
print(f'  Projective Tower: {avg_proj_linearity:.4f}')
print(f'  Difference: {linearity_difference:.4f}')

# Best performance at 2 GeV (design energy)
resolution_improvement_2gev = (box_2_0gev_resolution - proj_results[2.0]['resolution']) / box_2_0gev_resolution * 100
linearity_improvement_2gev = proj_results[2.0]['linearity'] - box_2_0gev_linearity

print(f'\nAt 2 GeV (design energy):')
print(f'  Resolution improvement: {resolution_improvement_2gev:.2f}%')
print(f'  Linearity improvement: {linearity_improvement_2gev:.4f}')

# Save detailed results
comparison_results = {
    'box_geometry': {
        'resolutions': box_resolutions,
        'linearities': box_linearities,
        'mean_deposits': box_means,
        'avg_resolution': avg_box_resolution,
        'avg_linearity': avg_box_linearity
    },
    'projective_geometry': {
        'resolutions': proj_resolutions,
        'linearities': proj_linearities,
        'mean_deposits': proj_means,
        'avg_resolution': avg_proj_resolution,
        'avg_linearity': avg_proj_linearity
    },
    'improvements': {
        'avg_resolution_improvement_percent': avg_resolution_improvement,
        'linearity_difference': linearity_difference,
        'resolution_improvement_2gev_percent': resolution_improvement_2gev,
        'linearity_improvement_2gev': linearity_improvement_2gev
    },
    'energies_gev': energies
}

with open('geometry_comparison_results.json', 'w') as f:
    json.dump(comparison_results, f, indent=2)

# Output results for workflow
print(f'RESULT:box_avg_resolution={avg_box_resolution:.6f}')
print(f'RESULT:projective_avg_resolution={avg_proj_resolution:.6f}')
print(f'RESULT:resolution_improvement_percent={avg_resolution_improvement:.2f}')
print(f'RESULT:box_avg_linearity={avg_box_linearity:.4f}')
print(f'RESULT:projective_avg_linearity={avg_proj_linearity:.4f}')
print(f'RESULT:linearity_difference={linearity_difference:.4f}')
print(f'RESULT:resolution_improvement_2gev_percent={resolution_improvement_2gev:.2f}')
print(f'RESULT:comparison_plot=geometry_comparison.png')
print(f'RESULT:results_json=geometry_comparison_results.json')
print('RESULT:success=True')