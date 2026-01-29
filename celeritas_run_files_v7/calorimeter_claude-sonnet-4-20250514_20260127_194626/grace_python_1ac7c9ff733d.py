import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Performance data from previous analysis steps (from Previous Step Outputs)
# Baseline configuration results
baseline_10_resolution = 0.095581
baseline_10_err = 0.000956
baseline_30_resolution = 0.077267
baseline_30_err = 0.000773
baseline_50_resolution = 0.070718
baseline_50_err = 0.000707
baseline_10_linearity = 0.817535
baseline_30_linearity = 0.837036
baseline_50_linearity = 0.845002

# Projective configuration results
projective_10_resolution = 1.1149
projective_10_err = 0.011149
projective_30_resolution = 1.30389
projective_30_err = 0.013039
projective_50_resolution = 1.38055
projective_50_err = 0.013806
projective_10_linearity = 0.167281
projective_30_linearity = 0.115371
projective_50_linearity = 0.096326

# Tungsten barrel configuration results
tungsten_10_resolution = 0.610701
tungsten_10_err = 0.006107
tungsten_30_resolution = 0.68318
tungsten_30_err = 0.006832
tungsten_50_resolution = 0.727988
tungsten_50_err = 0.00728
tungsten_10_linearity = 0.48054
tungsten_30_linearity = 0.432419
tungsten_50_linearity = 0.407177

# Energy points
energies = [10.0, 30.0, 50.0]

# Organize data for plotting
baseline_resolutions = [baseline_10_resolution, baseline_30_resolution, baseline_50_resolution]
baseline_errors = [baseline_10_err, baseline_30_err, baseline_50_err]
baseline_linearities = [baseline_10_linearity, baseline_30_linearity, baseline_50_linearity]

projective_resolutions = [projective_10_resolution, projective_30_resolution, projective_50_resolution]
projective_errors = [projective_10_err, projective_30_err, projective_50_err]
projective_linearities = [projective_10_linearity, projective_30_linearity, projective_50_linearity]

tungsten_resolutions = [tungsten_10_resolution, tungsten_30_resolution, tungsten_50_resolution]
tungsten_errors = [tungsten_10_err, tungsten_30_err, tungsten_50_err]
tungsten_linearities = [tungsten_10_linearity, tungsten_30_linearity, tungsten_50_linearity]

# Create comprehensive comparison plots
fig = plt.figure(figsize=(16, 12))

# Plot 1: Energy Resolution vs Energy
ax1 = plt.subplot(2, 3, 1)
plt.errorbar(energies, baseline_resolutions, yerr=baseline_errors, 
            marker='o', linewidth=2, capsize=5, label='Baseline (Fe/Scint)', color='blue')
plt.errorbar(energies, tungsten_resolutions, yerr=tungsten_errors, 
            marker='s', linewidth=2, capsize=5, label='Tungsten Barrel', color='red')
plt.errorbar(energies, projective_resolutions, yerr=projective_errors, 
            marker='^', linewidth=2, capsize=5, label='Projective Tower', color='green')
plt.xlabel('Beam Energy (GeV)')
plt.ylabel('Energy Resolution (σ/E)')
plt.title('Energy Resolution Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Plot 2: Linearity vs Energy
ax2 = plt.subplot(2, 3, 2)
plt.plot(energies, baseline_linearities, 'o-', linewidth=2, label='Baseline (Fe/Scint)', color='blue')
plt.plot(energies, tungsten_linearities, 's-', linewidth=2, label='Tungsten Barrel', color='red')
plt.plot(energies, projective_linearities, '^-', linewidth=2, label='Projective Tower', color='green')
plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect Linearity')
plt.xlabel('Beam Energy (GeV)')
plt.ylabel('Linearity (E_measured/E_beam)')
plt.title('Energy Linearity Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Average Resolution Bar Chart
ax3 = plt.subplot(2, 3, 3)
configs = ['Baseline\n(Fe/Scint)', 'Tungsten\nBarrel', 'Projective\nTower']
avg_resolutions = [np.mean(baseline_resolutions), np.mean(tungsten_resolutions), np.mean(projective_resolutions)]
avg_errors = [np.std(baseline_resolutions)/np.sqrt(3), np.std(tungsten_resolutions)/np.sqrt(3), np.std(projective_resolutions)/np.sqrt(3)]
colors = ['blue', 'red', 'green']
bars = plt.bar(configs, avg_resolutions, yerr=avg_errors, capsize=5, color=colors, alpha=0.7)
plt.ylabel('Average Energy Resolution')
plt.title('Average Resolution Comparison')
plt.yscale('log')
for i, (res, err) in enumerate(zip(avg_resolutions, avg_errors)):
    plt.text(i, res + err + 0.01, f'{res:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Compactness Analysis (Depth comparison)
ax4 = plt.subplot(2, 3, 4)
# Geometry parameters from previous steps
baseline_depth_m = 1.67022
tungsten_depth_m = 0.23175
projective_depth_m = 0.1675  # From projective geometry step

depths_m = [baseline_depth_m, tungsten_depth_m, projective_depth_m]
depths_cm = [d * 100 for d in depths_m]
bars = plt.bar(configs, depths_cm, color=colors, alpha=0.7)
plt.ylabel('Total Depth (cm)')
plt.title('Detector Compactness')
for i, depth in enumerate(depths_cm):
    plt.text(i, depth + 5, f'{depth:.1f} cm', ha='center', va='bottom', fontweight='bold')

# Plot 5: Resolution vs Compactness Trade-off
ax5 = plt.subplot(2, 3, 5)
plt.scatter(depths_cm, avg_resolutions, s=200, c=colors, alpha=0.7)
for i, config in enumerate(['Baseline', 'Tungsten', 'Projective']):
    plt.annotate(config, (depths_cm[i], avg_resolutions[i]), 
                xytext=(10, 10), textcoords='offset points', fontsize=10)
plt.xlabel('Total Depth (cm)')
plt.ylabel('Average Energy Resolution')
plt.title('Resolution vs Compactness Trade-off')
plt.yscale('log')
plt.grid(True, alpha=0.3)

# Plot 6: Performance Summary Table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('tight')
ax6.axis('off')

# Create summary table data
table_data = [
    ['Configuration', 'Avg Resolution', 'Depth (cm)', 'Compactness Factor'],
    ['Baseline (Fe/Scint)', f'{avg_resolutions[0]:.3f}', f'{depths_cm[0]:.1f}', '1.0x'],
    ['Tungsten Barrel', f'{avg_resolutions[1]:.3f}', f'{depths_cm[1]:.1f}', f'{depths_cm[0]/depths_cm[1]:.1f}x'],
    ['Projective Tower', f'{avg_resolutions[2]:.3f}', f'{depths_cm[2]:.1f}', f'{depths_cm[0]/depths_cm[2]:.1f}x']
]

table = ax6.table(cellText=table_data[1:], colLabels=table_data[0], 
                 cellLoc='center', loc='center', 
                 colColours=['lightgray']*4)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
ax6.set_title('Performance Summary', pad=20)

plt.tight_layout()
plt.savefig('tungsten_comprehensive_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('tungsten_comprehensive_comparison.pdf', bbox_inches='tight')
plt.show()

# Calculate key performance metrics
baseline_avg_resolution = np.mean(baseline_resolutions)
tungsten_avg_resolution = np.mean(tungsten_resolutions)
projective_avg_resolution = np.mean(projective_resolutions)

baseline_avg_linearity = np.mean(baseline_linearities)
tungsten_avg_linearity = np.mean(tungsten_linearities)
projective_avg_linearity = np.mean(projective_linearities)

# Compactness improvements
tungsten_compactness_factor = baseline_depth_m / tungsten_depth_m
projective_compactness_factor = baseline_depth_m / projective_depth_m

# Resolution trade-offs
tungsten_resolution_penalty = tungsten_avg_resolution / baseline_avg_resolution
projective_resolution_penalty = projective_avg_resolution / baseline_avg_resolution

# Save detailed results
results = {
    'baseline_avg_resolution': baseline_avg_resolution,
    'tungsten_avg_resolution': tungsten_avg_resolution,
    'projective_avg_resolution': projective_avg_resolution,
    'baseline_avg_linearity': baseline_avg_linearity,
    'tungsten_avg_linearity': tungsten_avg_linearity,
    'projective_avg_linearity': projective_avg_linearity,
    'tungsten_compactness_factor': tungsten_compactness_factor,
    'projective_compactness_factor': projective_compactness_factor,
    'tungsten_resolution_penalty': tungsten_resolution_penalty,
    'projective_resolution_penalty': projective_resolution_penalty,
    'baseline_depth_cm': depths_cm[0],
    'tungsten_depth_cm': depths_cm[1],
    'projective_depth_cm': depths_cm[2]
}

with open('tungsten_comparison_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print key findings
print('=== TUNGSTEN BARREL CALORIMETER COMPARISON ===')
print(f'Baseline (Fe/Scint): Resolution = {baseline_avg_resolution:.3f}, Depth = {depths_cm[0]:.1f} cm')
print(f'Tungsten Barrel: Resolution = {tungsten_avg_resolution:.3f}, Depth = {depths_cm[1]:.1f} cm')
print(f'Projective Tower: Resolution = {projective_avg_resolution:.3f}, Depth = {depths_cm[2]:.1f} cm')
print(f'\nCompactness Improvements:')
print(f'Tungsten vs Baseline: {tungsten_compactness_factor:.1f}x more compact')
print(f'Projective vs Baseline: {projective_compactness_factor:.1f}x more compact')
print(f'\nResolution Trade-offs:')
print(f'Tungsten penalty: {tungsten_resolution_penalty:.1f}x worse resolution')
print(f'Projective penalty: {projective_resolution_penalty:.1f}x worse resolution')

# Output results for workflow
print(f'RESULT:tungsten_avg_resolution={tungsten_avg_resolution:.4f}')
print(f'RESULT:tungsten_compactness_factor={tungsten_compactness_factor:.2f}')
print(f'RESULT:tungsten_resolution_penalty={tungsten_resolution_penalty:.2f}')
print(f'RESULT:comparison_plots=tungsten_comprehensive_comparison.png')
print(f'RESULT:comparison_plots_pdf=tungsten_comprehensive_comparison.pdf')
print(f'RESULT:results_json=tungsten_comparison_results.json')
print('RESULT:success=True')