import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

# Performance metrics from Previous Step Outputs
# Baseline results
baseline_energy_res = 3.1729
baseline_energy_res_err = 0.3173
baseline_detection_eff = 0.1
baseline_detection_eff_err = 0.0424
baseline_light_yield = 0.7

# Optimized results
optimized_energy_res = 3.0825
optimized_energy_res_err = 0.218
optimized_detection_eff = 0.17
optimized_detection_eff_err = 0.0376
optimized_light_yield = 22.61

# Calculate improvements
energy_res_improvement = ((baseline_energy_res - optimized_energy_res) / baseline_energy_res) * 100
detection_eff_improvement = ((optimized_detection_eff - baseline_detection_eff) / baseline_detection_eff) * 100
light_yield_improvement = ((optimized_light_yield - baseline_light_yield) / baseline_light_yield) * 100

# Statistical significance calculations
# Using t-test approximation: t = (mean1 - mean2) / sqrt(err1^2 + err2^2)
energy_res_significance = abs(baseline_energy_res - optimized_energy_res) / np.sqrt(baseline_energy_res_err**2 + optimized_energy_res_err**2)
detection_eff_significance = abs(optimized_detection_eff - baseline_detection_eff) / np.sqrt(baseline_detection_eff_err**2 + optimized_detection_eff_err**2)

print(f'Energy resolution improvement: {energy_res_improvement:.2f}%')
print(f'Detection efficiency improvement: {detection_eff_improvement:.2f}%')
print(f'Light yield improvement: {light_yield_improvement:.2f}%')
print(f'Energy resolution significance: {energy_res_significance:.2f} sigma')
print(f'Detection efficiency significance: {detection_eff_significance:.2f} sigma')

# Create comprehensive comparison plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Energy Resolution Comparison
configs = ['Baseline', 'Optimized']
energy_values = [baseline_energy_res, optimized_energy_res]
energy_errors = [baseline_energy_res_err, optimized_energy_res_err]
colors = ['red', 'blue']

ax1.bar(configs, energy_values, yerr=energy_errors, capsize=8, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Energy Resolution (σ/E)')
ax1.set_title(f'Energy Resolution Comparison\n({energy_res_improvement:.1f}% improvement, {energy_res_significance:.1f}σ significance)')
ax1.grid(True, alpha=0.3)
for i, (val, err) in enumerate(zip(energy_values, energy_errors)):
    ax1.text(i, val + err + 0.05, f'{val:.3f}±{err:.3f}', ha='center', fontweight='bold')

# Plot 2: Detection Efficiency Comparison
eff_values = [baseline_detection_eff, optimized_detection_eff]
eff_errors = [baseline_detection_eff_err, optimized_detection_eff_err]

ax2.bar(configs, eff_values, yerr=eff_errors, capsize=8, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Detection Efficiency')
ax2.set_title(f'Detection Efficiency Comparison\n({detection_eff_improvement:.1f}% improvement, {detection_eff_significance:.1f}σ significance)')
ax2.grid(True, alpha=0.3)
for i, (val, err) in enumerate(zip(eff_values, eff_errors)):
    ax2.text(i, val + err + 0.005, f'{val:.3f}±{err:.3f}', ha='center', fontweight='bold')

# Plot 3: Light Yield Comparison (log scale due to large difference)
light_values = [baseline_light_yield, optimized_light_yield]

ax3.bar(configs, light_values, color=colors, alpha=0.7, edgecolor='black')
ax3.set_ylabel('Light Yield (PE/MeV)')
ax3.set_yscale('log')
ax3.set_title(f'Light Yield Comparison\n({light_yield_improvement:.0f}% improvement)')
ax3.grid(True, alpha=0.3)
for i, val in enumerate(light_values):
    ax3.text(i, val * 1.5, f'{val:.2f}', ha='center', fontweight='bold')

# Plot 4: Summary improvement percentages
metrics = ['Energy\nResolution', 'Detection\nEfficiency', 'Light\nYield']
improvements = [energy_res_improvement, detection_eff_improvement, light_yield_improvement]
bar_colors = ['green' if imp > 0 else 'red' for imp in improvements]

ax4.bar(metrics, improvements, color=bar_colors, alpha=0.7, edgecolor='black')
ax4.set_ylabel('Improvement (%)')
ax4.set_title('Performance Improvements Summary')
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
for i, imp in enumerate(improvements):
    ax4.text(i, imp + (5 if imp > 0 else -15), f'{imp:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('optimization_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('optimization_comparison.pdf', bbox_inches='tight')
plt.show()

# Create detailed statistical significance plot
fig2, ax = plt.subplots(figsize=(10, 6))
significance_metrics = ['Energy Resolution', 'Detection Efficiency']
significance_values = [energy_res_significance, detection_eff_significance]
sig_colors = ['green' if sig > 2.0 else 'orange' if sig > 1.0 else 'red' for sig in significance_values]

ax.bar(significance_metrics, significance_values, color=sig_colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Statistical Significance (σ)')
ax.set_title('Statistical Significance of Improvements')
ax.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='1σ (68% confidence)')
ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='2σ (95% confidence)')
ax.axhline(y=3.0, color='green', linestyle='--', alpha=0.7, label='3σ (99.7% confidence)')
ax.grid(True, alpha=0.3)
ax.legend()
for i, sig in enumerate(significance_values):
    ax.text(i, sig + 0.05, f'{sig:.2f}σ', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('statistical_significance.png', dpi=150, bbox_inches='tight')
plt.savefig('statistical_significance.pdf', bbox_inches='tight')
plt.show()

# Save detailed comparison results
comparison_results = {
    'baseline_metrics': {
        'energy_resolution': baseline_energy_res,
        'energy_resolution_error': baseline_energy_res_err,
        'detection_efficiency': baseline_detection_eff,
        'detection_efficiency_error': baseline_detection_eff_err,
        'light_yield_pe_per_mev': baseline_light_yield
    },
    'optimized_metrics': {
        'energy_resolution': optimized_energy_res,
        'energy_resolution_error': optimized_energy_res_err,
        'detection_efficiency': optimized_detection_eff,
        'detection_efficiency_error': optimized_detection_eff_err,
        'light_yield_pe_per_mev': optimized_light_yield
    },
    'improvements': {
        'energy_resolution_improvement_percent': energy_res_improvement,
        'detection_efficiency_improvement_percent': detection_eff_improvement,
        'light_yield_improvement_percent': light_yield_improvement
    },
    'statistical_significance': {
        'energy_resolution_significance_sigma': energy_res_significance,
        'detection_efficiency_significance_sigma': detection_eff_significance
    }
}

with open('optimization_comparison_results.json', 'w') as f:
    json.dump(comparison_results, f, indent=2)

print('RESULT:comparison_plot=optimization_comparison.png')
print('RESULT:significance_plot=statistical_significance.png')
print('RESULT:results_file=optimization_comparison_results.json')
print(f'RESULT:energy_resolution_improvement={energy_res_improvement:.2f}')
print(f'RESULT:detection_efficiency_improvement={detection_eff_improvement:.2f}')
print(f'RESULT:light_yield_improvement={light_yield_improvement:.2f}')
print(f'RESULT:energy_resolution_significance={energy_res_significance:.2f}')
print(f'RESULT:detection_efficiency_significance={detection_eff_significance:.2f}')