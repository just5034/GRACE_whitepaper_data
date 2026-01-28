import matplotlib
matplotlib.use('Agg')
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# Load cylindrical barrel simulation results
with uproot.open('cylindrical_barrel_silicon_muon_hits.root') as f:
    cyl_events = f['events'].arrays(library='pd')
    print(f'Loaded {len(cyl_events)} cylindrical events')

# Get baseline results from previous step outputs
baseline_energy_resolution = 0.346873
baseline_energy_resolution_error = 0.007756
baseline_hit_efficiency = 0.999
baseline_hit_efficiency_error = 0.000999
baseline_mean_energy_deposit = 0.437058
baseline_material_budget = 0.013464

# Calculate cylindrical detector performance metrics
cyl_mean_E = cyl_events['totalEdep'].mean()
cyl_std_E = cyl_events['totalEdep'].std()
cyl_energy_resolution = cyl_std_E / cyl_mean_E if cyl_mean_E > 0 else 0
cyl_energy_resolution_error = cyl_energy_resolution / np.sqrt(2 * len(cyl_events)) if len(cyl_events) > 0 else 0

# Hit efficiency (events with at least one hit)
events_with_hits = len(cyl_events[cyl_events['nHits'] > 0])
total_events = len(cyl_events)
cyl_hit_efficiency = events_with_hits / total_events if total_events > 0 else 0
cyl_hit_efficiency_error = np.sqrt(cyl_hit_efficiency * (1 - cyl_hit_efficiency) / total_events) if total_events > 0 else 0

# Material budget (same geometry parameters, so same as baseline)
cyl_material_budget = baseline_material_budget

print(f'Cylindrical Performance:')
print(f'Energy Resolution: {cyl_energy_resolution:.6f} ± {cyl_energy_resolution_error:.6f}')
print(f'Hit Efficiency: {cyl_hit_efficiency:.3f} ± {cyl_hit_efficiency_error:.6f}')
print(f'Mean Energy Deposit: {cyl_mean_E:.6f} MeV')
print(f'Material Budget: {cyl_material_budget:.6f} X0')

# Calculate comparison metrics
resolution_improvement = (baseline_energy_resolution - cyl_energy_resolution) / baseline_energy_resolution * 100
efficiency_change = (cyl_hit_efficiency - baseline_hit_efficiency) / baseline_hit_efficiency * 100
energy_change = (cyl_mean_E - baseline_mean_energy_deposit) / baseline_mean_energy_deposit * 100

print(f'\nComparison with Baseline:')
print(f'Energy Resolution Change: {resolution_improvement:.2f}%')
print(f'Hit Efficiency Change: {efficiency_change:.2f}%')
print(f'Mean Energy Deposit Change: {energy_change:.2f}%')

# MANDATORY: Raw energy distribution plot
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.hist(cyl_events['totalEdep'], bins=50, histtype='step', linewidth=2, label='Cylindrical', color='green')
plt.axvline(cyl_mean_E, color='green', linestyle='--', label=f'Mean: {cyl_mean_E:.3f} MeV')
plt.xlabel('Total Energy Deposit (MeV)')
plt.ylabel('Events')
plt.title(f'Cylindrical Energy Distribution (σ/E = {cyl_energy_resolution:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)

# Comparison bar chart
plt.subplot(2, 2, 2)
configs = ['Baseline', 'Cylindrical']
resolutions = [baseline_energy_resolution, cyl_energy_resolution]
res_errors = [baseline_energy_resolution_error, cyl_energy_resolution_error]
plt.bar(configs, resolutions, yerr=res_errors, capsize=5, color=['blue', 'green'], alpha=0.7)
plt.ylabel('Energy Resolution (σ/E)')
plt.title('Energy Resolution Comparison')
plt.grid(True, alpha=0.3)

# Efficiency comparison
plt.subplot(2, 2, 3)
efficiencies = [baseline_hit_efficiency, cyl_hit_efficiency]
eff_errors = [baseline_hit_efficiency_error, cyl_hit_efficiency_error]
plt.bar(configs, efficiencies, yerr=eff_errors, capsize=5, color=['blue', 'green'], alpha=0.7)
plt.ylabel('Hit Efficiency')
plt.title('Hit Efficiency Comparison')
plt.grid(True, alpha=0.3)

# Energy deposit comparison
plt.subplot(2, 2, 4)
energy_deposits = [baseline_mean_energy_deposit, cyl_mean_E]
plt.bar(configs, energy_deposits, color=['blue', 'green'], alpha=0.7)
plt.ylabel('Mean Energy Deposit (MeV)')
plt.title('Mean Energy Deposit Comparison')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cylindrical_performance_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('cylindrical_performance_analysis.pdf', bbox_inches='tight')
plt.show()

# Save detailed metrics to JSON
metrics = {
    'cylindrical_performance': {
        'energy_resolution': float(cyl_energy_resolution),
        'energy_resolution_error': float(cyl_energy_resolution_error),
        'hit_efficiency': float(cyl_hit_efficiency),
        'hit_efficiency_error': float(cyl_hit_efficiency_error),
        'mean_energy_deposit': float(cyl_mean_E),
        'material_budget': float(cyl_material_budget)
    },
    'baseline_performance': {
        'energy_resolution': float(baseline_energy_resolution),
        'energy_resolution_error': float(baseline_energy_resolution_error),
        'hit_efficiency': float(baseline_hit_efficiency),
        'hit_efficiency_error': float(baseline_hit_efficiency_error),
        'mean_energy_deposit': float(baseline_mean_energy_deposit),
        'material_budget': float(baseline_material_budget)
    },
    'comparison': {
        'resolution_improvement_percent': float(resolution_improvement),
        'efficiency_change_percent': float(efficiency_change),
        'energy_change_percent': float(energy_change)
    }
}

with open('cylindrical_performance_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Return values for downstream steps
print(f'RESULT:energy_resolution={cyl_energy_resolution:.6f}')
print(f'RESULT:energy_resolution_error={cyl_energy_resolution_error:.6f}')
print(f'RESULT:hit_efficiency={cyl_hit_efficiency:.3f}')
print(f'RESULT:hit_efficiency_error={cyl_hit_efficiency_error:.6f}')
print(f'RESULT:mean_energy_deposit={cyl_mean_E:.6f}')
print(f'RESULT:material_budget={cyl_material_budget:.6f}')
print(f'RESULT:resolution_improvement={resolution_improvement:.2f}')
print(f'RESULT:efficiency_change={efficiency_change:.2f}')
print(f'RESULT:energy_change={energy_change:.2f}')
print('RESULT:analysis_plot=cylindrical_performance_analysis.png')
print('RESULT:metrics_file=cylindrical_performance_metrics.json')
print('RESULT:success=True')