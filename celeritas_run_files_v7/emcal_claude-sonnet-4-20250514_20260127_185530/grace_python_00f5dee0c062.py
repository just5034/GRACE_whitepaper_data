import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data from previous analysis steps (from Previous Step Outputs)
# Box geometry (PbWO4) performance
box_0_5gev_resolution = 0.033044
box_0_5gev_resolution_err = 0.00033
box_2_0gev_resolution = 0.014936
box_2_0gev_resolution_err = 0.000149
box_5_0gev_resolution = 0.019751
box_5_0gev_resolution_err = 0.000198
box_0_5gev_linearity = 0.9678
box_2_0gev_linearity = 0.9674
box_5_0gev_linearity = 0.9637

# Load projective tower data from simulation files
proj_energies = [0.5, 2.0, 5.0]
proj_resolutions = []
proj_resolution_errs = []
proj_linearities = []

proj_files = {
    0.5: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/emcal_claude-sonnet-4-20250514_20260127_185530/energy_0.500GeV/optimal_projective_tower_calorimeter_retry_electron_events.parquet',
    2.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/emcal_claude-sonnet-4-20250514_20260127_185530/energy_2.000GeV/optimal_projective_tower_calorimeter_retry_electron_events.parquet',
    5.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/emcal_claude-sonnet-4-20250514_20260127_185530/energy_5.000GeV/optimal_projective_tower_calorimeter_retry_electron_events.parquet'
}

for energy_gev, filepath in proj_files.items():
    if Path(filepath).exists():
        df = pd.read_parquet(filepath)
        beam_energy_mev = energy_gev * 1000
        mean_edep = df['totalEdep'].mean()
        std_edep = df['totalEdep'].std()
        resolution = std_edep / mean_edep
        resolution_err = resolution / np.sqrt(2 * len(df))
        linearity = mean_edep / beam_energy_mev
        
        proj_resolutions.append(resolution)
        proj_resolution_errs.append(resolution_err)
        proj_linearities.append(linearity)
        
        print(f'Projective {energy_gev} GeV: resolution={resolution:.6f}, linearity={linearity:.4f}')

# Organize data for plotting
energies = [0.5, 2.0, 5.0]
box_resolutions = [box_0_5gev_resolution, box_2_0gev_resolution, box_5_0gev_resolution]
box_resolution_errs = [box_0_5gev_resolution_err, box_2_0gev_resolution_err, box_5_0gev_resolution_err]
box_linearities = [box_0_5gev_linearity, box_2_0gev_linearity, box_5_0gev_linearity]

# Create comparison plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Energy Resolution Comparison
width = 0.35
x_pos = np.arange(len(energies))

ax1.bar(x_pos - width/2, box_resolutions, width, yerr=box_resolution_errs, 
        label='Box Geometry', alpha=0.7, capsize=5, color='blue')
ax1.bar(x_pos + width/2, proj_resolutions, width, yerr=proj_resolution_errs,
        label='Projective Tower', alpha=0.7, capsize=5, color='green')
ax1.set_xlabel('Beam Energy (GeV)')
ax1.set_ylabel('Energy Resolution (σ/E)')
ax1.set_title('Energy Resolution Comparison')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'{e}' for e in energies])
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Linearity Comparison
ax2.bar(x_pos - width/2, box_linearities, width, 
        label='Box Geometry', alpha=0.7, color='blue')
ax2.bar(x_pos + width/2, proj_linearities, width,
        label='Projective Tower', alpha=0.7, color='green')
ax2.set_xlabel('Beam Energy (GeV)')
ax2.set_ylabel('Linearity (E_dep/E_beam)')
ax2.set_title('Linearity Comparison')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'{e}' for e in energies])
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect Linearity')

# 3. Resolution vs Energy Curves
ax3.errorbar(energies, box_resolutions, yerr=box_resolution_errs, 
             marker='o', linewidth=2, capsize=5, label='Box Geometry', color='blue')
ax3.errorbar(energies, proj_resolutions, yerr=proj_resolution_errs,
             marker='s', linewidth=2, capsize=5, label='Projective Tower', color='green')
ax3.set_xlabel('Beam Energy (GeV)')
ax3.set_ylabel('Energy Resolution (σ/E)')
ax3.set_title('Resolution vs Energy')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xscale('log')
ax3.set_yscale('log')

# 4. Performance Improvement Summary
improvements = [(box_res - proj_res) / box_res * 100 
                for box_res, proj_res in zip(box_resolutions, proj_resolutions)]
ax4.bar(x_pos, improvements, color='orange', alpha=0.7)
ax4.set_xlabel('Beam Energy (GeV)')
ax4.set_ylabel('Resolution Improvement (%)')
ax4.set_title('Projective Tower Improvement over Box')
ax4.set_xticks(x_pos)
ax4.set_xticklabels([f'{e}' for e in energies])
ax4.grid(True, alpha=0.3)
for i, imp in enumerate(improvements):
    ax4.text(i, imp + 1, f'{imp:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('geometry_comparison_plots.png', dpi=150, bbox_inches='tight')
plt.savefig('geometry_comparison_plots.pdf', bbox_inches='tight')
plt.show()

# Calculate overall statistics
box_avg_resolution = np.mean(box_resolutions)
proj_avg_resolution = np.mean(proj_resolutions)
overall_improvement = (box_avg_resolution - proj_avg_resolution) / box_avg_resolution * 100

box_avg_linearity = np.mean(box_linearities)
proj_avg_linearity = np.mean(proj_linearities)
linearity_improvement = proj_avg_linearity - box_avg_linearity

# Statistical significance test (t-test approximation)
from scipy import stats
t_stat = (box_avg_resolution - proj_avg_resolution) / np.sqrt(np.mean(box_resolution_errs)**2 + np.mean(proj_resolution_errs)**2)
p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

print(f'\n=== GEOMETRY COMPARISON RESULTS ===')
print(f'Box geometry average resolution: {box_avg_resolution:.6f}')
print(f'Projective tower average resolution: {proj_avg_resolution:.6f}')
print(f'Overall resolution improvement: {overall_improvement:.1f}%')
print(f'Box geometry average linearity: {box_avg_linearity:.4f}')
print(f'Projective tower average linearity: {proj_avg_linearity:.4f}')
print(f'Linearity improvement: {linearity_improvement:.4f}')
print(f'Statistical significance (t-test): p = {p_value:.3e}')

# Output results for downstream steps
print(f'RESULT:box_avg_resolution={box_avg_resolution:.6f}')
print(f'RESULT:projective_avg_resolution={proj_avg_resolution:.6f}')
print(f'RESULT:resolution_improvement_percent={overall_improvement:.2f}')
print(f'RESULT:box_avg_linearity={box_avg_linearity:.4f}')
print(f'RESULT:projective_avg_linearity={proj_avg_linearity:.4f}')
print(f'RESULT:linearity_improvement={linearity_improvement:.4f}')
print(f'RESULT:statistical_significance_p={p_value:.3e}')
print('RESULT:comparison_plots=geometry_comparison_plots.png')
print('RESULT:success=True')