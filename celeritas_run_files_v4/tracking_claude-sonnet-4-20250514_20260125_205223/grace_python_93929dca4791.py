import matplotlib
matplotlib.use('Agg')
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Load spacing-optimized simulation data
spacing_hits_file = 'spacing_optimized_detector_muon_hits.root'
print(f'Loading spacing-optimized data from {spacing_hits_file}')

with uproot.open(spacing_hits_file) as f:
    spacing_events = f['events'].arrays(library='pd')
    print(f'Loaded {len(spacing_events)} events from spacing-optimized detector')

# Calculate spacing-optimized performance metrics
spacing_mean_E = spacing_events['totalEdep'].mean()
spacing_std_E = spacing_events['totalEdep'].std()
spacing_resolution = spacing_std_E / spacing_mean_E if spacing_mean_E > 0 else 0
spacing_resolution_err = spacing_resolution / np.sqrt(2 * len(spacing_events)) if len(spacing_events) > 0 else 0
spacing_efficiency = len(spacing_events[spacing_events['nHits'] > 0]) / len(spacing_events) if len(spacing_events) > 0 else 0
spacing_efficiency_err = np.sqrt(spacing_efficiency * (1 - spacing_efficiency) / len(spacing_events)) if len(spacing_events) > 0 else 0

# Material budget from geometry parameters (spacing-optimized: 4 layers, 20mm active, 70mm absorber)
# Total thickness = 4 * (20 + 70) = 360mm = 0.36m
# Silicon X0 = 9.37 cm, Air X0 = 30420 cm
spacing_material_budget = (4 * 0.02) / 0.0937 + (4 * 0.07) / 304.2  # Si + Air contributions

print(f'Spacing-optimized metrics:')
print(f'Energy resolution: {spacing_resolution:.6f} ± {spacing_resolution_err:.6f}')
print(f'Hit efficiency: {spacing_efficiency:.6f} ± {spacing_efficiency_err:.6f}')
print(f'Mean energy deposit: {spacing_mean_E:.6f} MeV')
print(f'Material budget: {spacing_material_budget:.6f} X0')

# Previous results from workflow state (from Previous Step Outputs)
baseline_resolution = 0.346873
baseline_resolution_err = 0.007756
baseline_efficiency = 0.999
baseline_efficiency_err = 0.000999
baseline_material_budget = 0.013464
baseline_mean_edep = 0.437058

cylindrical_resolution = 0.43601
cylindrical_resolution_err = 0.009749
cylindrical_efficiency = 1.0
cylindrical_efficiency_err = 0.0
cylindrical_material_budget = 0.013464
cylindrical_mean_edep = 0.554646

thickness_resolution = 0.467486
thickness_resolution_err = 0.010453
thickness_efficiency = 1.0
thickness_efficiency_err = 0.0
thickness_material_budget = 0.00854
thickness_mean_edep = 0.393419

# Create comprehensive comparison
configurations = ['Baseline\nPlanar', 'Cylindrical\nBarrel', 'Optimized\nThickness', 'Optimized\nSpacing']
resolutions = [baseline_resolution, cylindrical_resolution, thickness_resolution, spacing_resolution]
resolution_errs = [baseline_resolution_err, cylindrical_resolution_err, thickness_resolution_err, spacing_resolution_err]
efficiencies = [baseline_efficiency, cylindrical_efficiency, thickness_efficiency, spacing_efficiency]
efficiency_errs = [baseline_efficiency_err, cylindrical_efficiency_err, thickness_efficiency_err, spacing_efficiency_err]
material_budgets = [baseline_material_budget, cylindrical_material_budget, thickness_material_budget, spacing_material_budget]
mean_edeps = [baseline_mean_edep, cylindrical_mean_edep, thickness_mean_edep, spacing_mean_E]

# Create comprehensive comparison plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Energy Resolution Comparison
colors = ['blue', 'green', 'orange', 'red']
ax1.bar(configurations, resolutions, yerr=resolution_errs, capsize=5, color=colors, alpha=0.7)
ax1.set_ylabel('Energy Resolution (σ/E)')
ax1.set_title('Energy Resolution Comparison')
ax1.grid(True, alpha=0.3)
for i, (res, err) in enumerate(zip(resolutions, resolution_errs)):
    ax1.text(i, res + err + 0.01, f'{res:.3f}±{err:.3f}', ha='center', fontsize=9)

# Hit Efficiency Comparison
ax2.bar(configurations, [e*100 for e in efficiencies], yerr=[e*100 for e in efficiency_errs], capsize=5, color=colors, alpha=0.7)
ax2.set_ylabel('Hit Efficiency (%)')
ax2.set_title('Hit Efficiency Comparison')
ax2.set_ylim(95, 101)
ax2.grid(True, alpha=0.3)
for i, (eff, err) in enumerate(zip(efficiencies, efficiency_errs)):
    ax2.text(i, eff*100 + err*100 + 0.1, f'{eff*100:.1f}±{err*100:.1f}%', ha='center', fontsize=9)

# Material Budget Comparison
ax3.bar(configurations, material_budgets, capsize=5, color=colors, alpha=0.7)
ax3.set_ylabel('Material Budget (X₀)')
ax3.set_title('Material Budget Comparison')
ax3.grid(True, alpha=0.3)
for i, mb in enumerate(material_budgets):
    ax3.text(i, mb + 0.01, f'{mb:.4f}', ha='center', fontsize=9)

# Mean Energy Deposit Comparison
ax4.bar(configurations, mean_edeps, capsize=5, color=colors, alpha=0.7)
ax4.set_ylabel('Mean Energy Deposit (MeV)')
ax4.set_title('Mean Energy Deposit Comparison')
ax4.grid(True, alpha=0.3)
for i, edep in enumerate(mean_edeps):
    ax4.text(i, edep + 0.02, f'{edep:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('final_detector_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('final_detector_comparison.pdf', bbox_inches='tight')
print('Saved comprehensive comparison plot: final_detector_comparison.png')

# Identify optimal configuration
best_resolution_idx = np.argmin(resolutions)
best_efficiency_idx = np.argmax(efficiencies)
best_material_idx = np.argmin(material_budgets)

print('\n=== OPTIMAL CONFIGURATION ANALYSIS ===')
print(f'Best energy resolution: {configurations[best_resolution_idx]} ({resolutions[best_resolution_idx]:.6f})')
print(f'Best hit efficiency: {configurations[best_efficiency_idx]} ({efficiencies[best_efficiency_idx]*100:.1f}%)')
print(f'Lowest material budget: {configurations[best_material_idx]} ({material_budgets[best_material_idx]:.6f} X₀)')

# Calculate relative improvements
baseline_idx = 0
for i, config in enumerate(configurations):
    if i == baseline_idx:
        continue
    res_improvement = (baseline_resolution - resolutions[i]) / baseline_resolution * 100
    eff_change = (efficiencies[i] - baseline_efficiency) / baseline_efficiency * 100
    material_change = (material_budgets[i] - baseline_material_budget) / baseline_material_budget * 100
    print(f'\n{config} vs Baseline:')
    print(f'  Resolution change: {res_improvement:+.1f}%')
    print(f'  Efficiency change: {eff_change:+.1f}%')
    print(f'  Material budget change: {material_change:+.1f}%')

# Create summary metrics
summary_metrics = {
    'configurations': {
        'baseline_planar': {
            'energy_resolution': baseline_resolution,
            'energy_resolution_error': baseline_resolution_err,
            'hit_efficiency': baseline_efficiency,
            'hit_efficiency_error': baseline_efficiency_err,
            'material_budget': baseline_material_budget,
            'mean_energy_deposit': baseline_mean_edep
        },
        'cylindrical_barrel': {
            'energy_resolution': cylindrical_resolution,
            'energy_resolution_error': cylindrical_resolution_err,
            'hit_efficiency': cylindrical_efficiency,
            'hit_efficiency_error': cylindrical_efficiency_err,
            'material_budget': cylindrical_material_budget,
            'mean_energy_deposit': cylindrical_mean_edep
        },
        'optimized_thickness': {
            'energy_resolution': thickness_resolution,
            'energy_resolution_error': thickness_resolution_err,
            'hit_efficiency': thickness_efficiency,
            'hit_efficiency_error': thickness_efficiency_err,
            'material_budget': thickness_material_budget,
            'mean_energy_deposit': thickness_mean_edep
        },
        'optimized_spacing': {
            'energy_resolution': spacing_resolution,
            'energy_resolution_error': spacing_resolution_err,
            'hit_efficiency': spacing_efficiency,
            'hit_efficiency_error': spacing_efficiency_err,
            'material_budget': spacing_material_budget,
            'mean_energy_deposit': spacing_mean_E
        }
    },
    'optimal_design': {
        'best_resolution_config': configurations[best_resolution_idx],
        'best_resolution_value': resolutions[best_resolution_idx],
        'best_efficiency_config': configurations[best_efficiency_idx],
        'best_efficiency_value': efficiencies[best_efficiency_idx],
        'lowest_material_config': configurations[best_material_idx],
        'lowest_material_value': material_budgets[best_material_idx]
    }
}

# Save comprehensive results
with open('final_optimization_results.json', 'w') as f:
    json.dump(summary_metrics, f, indent=2)

# Return key results
print(f'RESULT:spacing_energy_resolution={spacing_resolution:.6f}')
print(f'RESULT:spacing_energy_resolution_error={spacing_resolution_err:.6f}')
print(f'RESULT:spacing_hit_efficiency={spacing_efficiency:.6f}')
print(f'RESULT:spacing_hit_efficiency_error={spacing_efficiency_err:.6f}')
print(f'RESULT:spacing_material_budget={spacing_material_budget:.6f}')
print(f'RESULT:spacing_mean_energy_deposit={spacing_mean_E:.6f}')
print(f'RESULT:best_resolution_config={configurations[best_resolution_idx]}')
print(f'RESULT:best_resolution_value={resolutions[best_resolution_idx]:.6f}')
print(f'RESULT:best_efficiency_config={configurations[best_efficiency_idx]}')
print(f'RESULT:best_material_config={configurations[best_material_idx]}')
print('RESULT:comparison_plot=final_detector_comparison.png')
print('RESULT:summary_file=final_optimization_results.json')
print('RESULT:success=True')