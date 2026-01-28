import matplotlib
matplotlib.use('Agg')
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# Load optimized thickness simulation data
with uproot.open('optimized_thickness_detector_muon_hits.root') as f:
    events = f['events'].arrays(library='pd')

# Calculate optimized thickness performance metrics
mean_E = events['totalEdep'].mean()
std_E = events['totalEdep'].std()
resolution = std_E / mean_E if mean_E > 0 else 0
resolution_err = resolution / np.sqrt(2 * len(events)) if len(events) > 0 else 0
hit_efficiency = len(events[events['nHits'] > 0]) / len(events) if len(events) > 0 else 0
hit_efficiency_err = np.sqrt(hit_efficiency * (1 - hit_efficiency) / len(events)) if len(events) > 0 else 0

# Material budget calculation (from geometry parameters)
# Optimized: 4 layers × 0.2mm Si = 0.8mm total
# Silicon X0 = 93.7mm, so material budget = 0.8/93.7 = 0.00854 X0
material_budget_optimized = 0.00854

# Get baseline and cylindrical results from previous steps
baseline_resolution = 0.346873
baseline_efficiency = 0.999
baseline_material_budget = 0.013464
cylindrical_resolution = 0.43601
cylindrical_efficiency = 1.0
cylindrical_material_budget = 0.013464

# Calculate trade-offs
material_reduction_vs_baseline = (baseline_material_budget - material_budget_optimized) / baseline_material_budget * 100
material_reduction_vs_cylindrical = (cylindrical_material_budget - material_budget_optimized) / cylindrical_material_budget * 100
resolution_change_vs_baseline = (resolution - baseline_resolution) / baseline_resolution * 100
resolution_change_vs_cylindrical = (resolution - cylindrical_resolution) / cylindrical_resolution * 100
efficiency_change_vs_baseline = (hit_efficiency - baseline_efficiency) / baseline_efficiency * 100
efficiency_change_vs_cylindrical = (hit_efficiency - cylindrical_efficiency) / cylindrical_efficiency * 100

# Create comprehensive comparison plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Energy resolution comparison
configs = ['Baseline\nPlanar', 'Cylindrical\nBarrel', 'Optimized\nThickness']
resolutions = [baseline_resolution, cylindrical_resolution, resolution]
res_errors = [0.007756, 0.009749, resolution_err]
ax1.bar(configs, resolutions, yerr=res_errors, capsize=5, color=['blue', 'green', 'red'], alpha=0.7)
ax1.set_ylabel('Energy Resolution (σ/E)')
ax1.set_title('Energy Resolution Comparison')
ax1.grid(True, alpha=0.3)

# Hit efficiency comparison
efficiencies = [baseline_efficiency, cylindrical_efficiency, hit_efficiency]
eff_errors = [0.000999, 0, hit_efficiency_err]
ax2.bar(configs, efficiencies, yerr=eff_errors, capsize=5, color=['blue', 'green', 'red'], alpha=0.7)
ax2.set_ylabel('Hit Efficiency')
ax2.set_title('Hit Efficiency Comparison')
ax2.grid(True, alpha=0.3)

# Material budget comparison
material_budgets = [baseline_material_budget, cylindrical_material_budget, material_budget_optimized]
ax3.bar(configs, material_budgets, color=['blue', 'green', 'red'], alpha=0.7)
ax3.set_ylabel('Material Budget (X/X₀)')
ax3.set_title('Material Budget Comparison')
ax3.grid(True, alpha=0.3)

# Trade-off scatter plot
ax4.scatter([material_reduction_vs_baseline], [resolution_change_vs_baseline], 
           s=100, color='red', label='vs Baseline', alpha=0.8)
ax4.scatter([material_reduction_vs_cylindrical], [resolution_change_vs_cylindrical], 
           s=100, color='orange', label='vs Cylindrical', alpha=0.8)
ax4.axhline(0, color='black', linestyle='--', alpha=0.5)
ax4.axvline(0, color='black', linestyle='--', alpha=0.5)
ax4.set_xlabel('Material Budget Reduction (%)')
ax4.set_ylabel('Resolution Change (%)')
ax4.set_title('Material Budget vs Performance Trade-off')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('thickness_optimization_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('thickness_optimization_analysis.pdf', bbox_inches='tight')

# Raw energy distribution plot
plt.figure(figsize=(10, 6))
plt.hist(events['totalEdep'], bins=50, histtype='step', linewidth=2, label='Optimized Thickness')
plt.axvline(mean_E, color='r', linestyle='--', label=f'Mean: {mean_E:.3f} MeV')
plt.xlabel('Total Energy Deposit (MeV)')
plt.ylabel('Events')
plt.title(f'Optimized Thickness Energy Distribution (σ/E = {resolution:.4f} ± {resolution_err:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('optimized_thickness_energy_distribution.png', dpi=150, bbox_inches='tight')
plt.savefig('optimized_thickness_energy_distribution.pdf', bbox_inches='tight')

# Save detailed metrics
metrics = {
    'optimized_thickness': {
        'energy_resolution': float(resolution),
        'energy_resolution_error': float(resolution_err),
        'hit_efficiency': float(hit_efficiency),
        'hit_efficiency_error': float(hit_efficiency_err),
        'material_budget': float(material_budget_optimized),
        'mean_energy_deposit': float(mean_E)
    },
    'comparisons': {
        'material_reduction_vs_baseline_percent': float(material_reduction_vs_baseline),
        'material_reduction_vs_cylindrical_percent': float(material_reduction_vs_cylindrical),
        'resolution_change_vs_baseline_percent': float(resolution_change_vs_baseline),
        'resolution_change_vs_cylindrical_percent': float(resolution_change_vs_cylindrical),
        'efficiency_change_vs_baseline_percent': float(efficiency_change_vs_baseline),
        'efficiency_change_vs_cylindrical_percent': float(efficiency_change_vs_cylindrical)
    },
    'baseline_reference': {
        'energy_resolution': baseline_resolution,
        'hit_efficiency': baseline_efficiency,
        'material_budget': baseline_material_budget
    },
    'cylindrical_reference': {
        'energy_resolution': cylindrical_resolution,
        'hit_efficiency': cylindrical_efficiency,
        'material_budget': cylindrical_material_budget
    }
}

with open('thickness_optimization_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Print results
print(f'Optimized Thickness Performance:')
print(f'Energy Resolution: {resolution:.4f} ± {resolution_err:.4f}')
print(f'Hit Efficiency: {hit_efficiency:.3f} ± {hit_efficiency_err:.3f}')
print(f'Material Budget: {material_budget_optimized:.6f} X/X₀')
print(f'Mean Energy Deposit: {mean_E:.3f} MeV')
print(f'')
print(f'Material Budget Reduction:')
print(f'vs Baseline: {material_reduction_vs_baseline:.1f}%')
print(f'vs Cylindrical: {material_reduction_vs_cylindrical:.1f}%')
print(f'')
print(f'Performance Impact:')
print(f'Resolution change vs Baseline: {resolution_change_vs_baseline:+.1f}%')
print(f'Resolution change vs Cylindrical: {resolution_change_vs_cylindrical:+.1f}%')
print(f'Efficiency change vs Baseline: {efficiency_change_vs_baseline:+.1f}%')
print(f'Efficiency change vs Cylindrical: {efficiency_change_vs_cylindrical:+.1f}%')

# Return values for downstream steps
print(f'RESULT:energy_resolution={resolution:.6f}')
print(f'RESULT:energy_resolution_error={resolution_err:.6f}')
print(f'RESULT:hit_efficiency={hit_efficiency:.6f}')
print(f'RESULT:hit_efficiency_error={hit_efficiency_err:.6f}')
print(f'RESULT:material_budget={material_budget_optimized:.6f}')
print(f'RESULT:mean_energy_deposit={mean_E:.6f}')
print(f'RESULT:material_reduction_vs_baseline={material_reduction_vs_baseline:.2f}')
print(f'RESULT:material_reduction_vs_cylindrical={material_reduction_vs_cylindrical:.2f}')
print(f'RESULT:resolution_change_vs_baseline={resolution_change_vs_baseline:.2f}')
print(f'RESULT:resolution_change_vs_cylindrical={resolution_change_vs_cylindrical:.2f}')
print(f'RESULT:efficiency_change_vs_baseline={efficiency_change_vs_baseline:.2f}')
print(f'RESULT:efficiency_change_vs_cylindrical={efficiency_change_vs_cylindrical:.2f}')
print('RESULT:analysis_plot=thickness_optimization_analysis.png')
print('RESULT:energy_distribution_plot=optimized_thickness_energy_distribution.png')
print('RESULT:metrics_file=thickness_optimization_metrics.json')
print('RESULT:success=True')