import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

# Extract performance data from previous step outputs
# Baseline configuration
baseline_resolution = 0.346873
baseline_resolution_err = 0.007756
baseline_efficiency = 0.999
baseline_efficiency_err = 0.000999
baseline_material_budget = 0.013464
baseline_mean_edep = 0.437058

# Cylindrical configuration
cylindrical_resolution = 0.43601
cylindrical_resolution_err = 0.009749
cylindrical_efficiency = 1.0
cylindrical_efficiency_err = 0.0
cylindrical_material_budget = 0.013464
cylindrical_mean_edep = 0.554646

# Thickness optimized configuration
thickness_resolution = 0.467486
thickness_resolution_err = 0.010453
thickness_efficiency = 1.0
thickness_efficiency_err = 0.0
thickness_material_budget = 0.00854
thickness_mean_edep = 0.393419

# Final optimized configuration
final_resolution = 0.212969
final_resolution_err = 0.004762
final_efficiency = 1.0
final_efficiency_err = 0.0
final_material_budget = 0.854709
final_mean_edep = 36.2704

# Configuration labels
configs = ['Baseline\nPlanar', 'Cylindrical\nBarrel', 'Thickness\nOptimized', 'Final\nOptimized']

# Create comprehensive comparison plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Energy Resolution Comparison
resolutions = [baseline_resolution, cylindrical_resolution, thickness_resolution, final_resolution]
res_errors = [baseline_resolution_err, cylindrical_resolution_err, thickness_resolution_err, final_resolution_err]
colors = ['blue', 'green', 'orange', 'red']

ax1.bar(configs, resolutions, yerr=res_errors, capsize=5, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Energy Resolution (σ/E)')
ax1.set_title('Energy Resolution Comparison')
ax1.grid(True, alpha=0.3)
for i, (res, err) in enumerate(zip(resolutions, res_errors)):
    ax1.text(i, res + err + 0.01, f'{res:.3f}±{err:.3f}', ha='center', fontsize=9)

# 2. Detection Efficiency Comparison
efficiencies = [baseline_efficiency, cylindrical_efficiency, thickness_efficiency, final_efficiency]
eff_errors = [baseline_efficiency_err, cylindrical_efficiency_err, thickness_efficiency_err, final_efficiency_err]

ax2.bar(configs, efficiencies, yerr=eff_errors, capsize=5, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Detection Efficiency')
ax2.set_title('Detection Efficiency Comparison')
ax2.set_ylim([0.995, 1.001])
ax2.grid(True, alpha=0.3)
for i, (eff, err) in enumerate(zip(efficiencies,eff_errors)):
    ax2.text(i, eff + err + 0.0001, f'{eff:.3f}±{err:.3f}', ha='center', fontsize=9)

# 3. Material Budget Comparison
material_budgets = [baseline_material_budget, cylindrical_material_budget, thickness_material_budget, final_material_budget]

ax3.bar(configs, material_budgets, color=colors, alpha=0.7, edgecolor='black')
ax3.set_ylabel('Material Budget (X/X₀)')
ax3.set_title('Material Budget Comparison')
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3)
for i, mb in enumerate(material_budgets):
    ax3.text(i, mb * 1.2, f'{mb:.4f}', ha='center', fontsize=9)

# 4. Mean Energy Deposit Comparison
mean_edeps = [baseline_mean_edep, cylindrical_mean_edep, thickness_mean_edep, final_mean_edep]

ax4.bar(configs, mean_edeps, color=colors, alpha=0.7, edgecolor='black')
ax4.set_ylabel('Mean Energy Deposit (MeV)')
ax4.set_title('Mean Energy Deposit Comparison')
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3)
for i, edep in enumerate(mean_edeps):
    ax4.text(i, edep * 1.2, f'{edep:.2f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('performance_comparison_plots.png', dpi=150, bbox_inches='tight')
plt.savefig('performance_comparison_plots.pdf', bbox_inches='tight')
plt.show()

# Create performance summary plot
fig, ax = plt.subplots(figsize=(12, 8))

# Normalize metrics for radar-like comparison (lower is better for resolution and material budget)
norm_resolution = np.array(resolutions) / max(resolutions)
norm_efficiency = np.array(efficiencies)
norm_material = np.array(material_budgets) / max(material_budgets)

# Performance score (lower resolution and material budget is better, higher efficiency is better)
performance_scores = (1 - norm_resolution) * 0.4 + norm_efficiency * 0.3 + (1 - norm_material) * 0.3

ax.bar(configs, performance_scores, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Overall Performance Score')
ax.set_title('Overall Performance Comparison\n(Higher is Better: 40% Resolution + 30% Efficiency + 30% Material)')
ax.grid(True, alpha=0.3)
for i, score in enumerate(performance_scores):
    ax.text(i, score + 0.02, f'{score:.3f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('performance_summary_plot.png', dpi=150, bbox_inches='tight')
plt.savefig('performance_summary_plot.pdf', bbox_inches='tight')
plt.show()

# Save comprehensive results to JSON
results = {
    'configurations': {
        'baseline': {
            'energy_resolution': baseline_resolution,
            'energy_resolution_error': baseline_resolution_err,
            'detection_efficiency': baseline_efficiency,
            'detection_efficiency_error': baseline_efficiency_err,
            'material_budget': baseline_material_budget,
            'mean_energy_deposit': baseline_mean_edep
        },
        'cylindrical': {
            'energy_resolution': cylindrical_resolution,
            'energy_resolution_error': cylindrical_resolution_err,
            'detection_efficiency': cylindrical_efficiency,
            'detection_efficiency_error': cylindrical_efficiency_err,
            'material_budget': cylindrical_material_budget,
            'mean_energy_deposit': cylindrical_mean_edep
        },
        'thickness_optimized': {
            'energy_resolution': thickness_resolution,
            'energy_resolution_error': thickness_resolution_err,
            'detection_efficiency': thickness_efficiency,
            'detection_efficiency_error': thickness_efficiency_err,
            'material_budget': thickness_material_budget,
            'mean_energy_deposit': thickness_mean_edep
        },
        'final_optimized': {
            'energy_resolution': final_resolution,
            'energy_resolution_error': final_resolution_err,
            'detection_efficiency': final_efficiency,
            'detection_efficiency_error': final_efficiency_err,
            'material_budget': final_material_budget,
            'mean_energy_deposit': final_mean_edep
        }
    },
    'performance_scores': performance_scores.tolist(),
    'best_configuration': configs[np.argmax(performance_scores)]
}

with open('comprehensive_performance_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)

print('RESULT:comparison_plots=performance_comparison_plots.png')
print('RESULT:summary_plot=performance_summary_plot.png')
print('RESULT:results_file=comprehensive_performance_comparison.json')
print(f'RESULT:best_overall_config={configs[np.argmax(performance_scores)]}')
print(f'RESULT:best_resolution_config=Final Optimized')
print(f'RESULT:best_efficiency_config=Cylindrical')
print(f'RESULT:lowest_material_budget_config=Thickness Optimized')
print('RESULT:success=True')