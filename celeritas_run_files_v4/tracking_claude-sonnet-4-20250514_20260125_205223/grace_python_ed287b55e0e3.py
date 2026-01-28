import matplotlib
matplotlib.use('Agg')
import uproot
import numpy as np
import matplotlib.pyplot as plt
import json

# Load simulation data from ROOT file
with uproot.open('/u/jhill5/grace/work/benchmarks/celeritas_20260125_192529/tracking_claude-sonnet-4-20250514_20260125_205223/baseline_planar_detector_muon_hits.root') as f:
    events = f['events'].arrays(library='pd')
    print(f'Loaded {len(events)} events')

# Extract geometry parameters from previous step outputs
num_layers = 4
active_thickness_mm = 0.3
absorber_thickness_mm = 50.0
expected_energy_deposit = 0.1  # MeV from input
material_budget_per_layer = 0.032  # from input

# 1. Energy deposit analysis
mean_energy = events['totalEdep'].mean()
std_energy = events['totalEdep'].std()
energy_resolution = std_energy / mean_energy if mean_energy > 0 else 0
energy_resolution_err = energy_resolution / np.sqrt(2 * len(events)) if len(events) > 0 else 0

print(f'Mean energy deposit: {mean_energy:.4f} MeV')
print(f'Energy resolution (sigma/E): {energy_resolution:.4f} +/- {energy_resolution_err:.4f}')

# 2. Hit detection efficiency
hit_events = events[events['nHits'] > 0]
hit_efficiency = len(hit_events) / len(events) if len(events) > 0 else 0
hit_efficiency_err = np.sqrt(hit_efficiency * (1 - hit_efficiency) / len(events)) if len(events) > 0 else 0

print(f'Hit detection efficiency: {hit_efficiency:.4f} +/- {hit_efficiency_err:.4f}')

# 3. Material budget calculation
# Silicon X0 = 9.37 cm, Air X0 = 30420 cm
silicon_x0_cm = 9.37
air_x0_cm = 30420

# Total material budget (X/X0)
silicon_material_budget = (num_layers * active_thickness_mm / 10) / silicon_x0_cm  # Convert mm to cm
air_material_budget = (num_layers * absorber_thickness_mm / 10) / air_x0_cm
total_material_budget = silicon_material_budget + air_material_budget

print(f'Silicon material budget: {silicon_material_budget:.6f} X0')
print(f'Air material budget: {air_material_budget:.6f} X0')
print(f'Total material budget: {total_material_budget:.6f} X0')

# 4. Generate plots
# Raw energy distribution
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.hist(events['totalEdep'], bins=50, histtype='step', linewidth=2, alpha=0.8)
plt.axvline(mean_energy, color='r', linestyle='--', label=f'Mean: {mean_energy:.3f} MeV')
plt.xlabel('Total Energy Deposit (MeV)')
plt.ylabel('Events')
plt.title(f'Energy Distribution (σ/E = {energy_resolution:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)

# Hit multiplicity
plt.subplot(2, 2, 2)
plt.hist(events['nHits'], bins=range(0, max(events['nHits'])+2), histtype='step', linewidth=2, alpha=0.8)
plt.xlabel('Number of Hits')
plt.ylabel('Events')
plt.title(f'Hit Multiplicity (Efficiency = {hit_efficiency:.3f})')
plt.grid(True, alpha=0.3)

# Energy vs hits correlation
plt.subplot(2, 2, 3)
plt.scatter(events['nHits'], events['totalEdep'], alpha=0.6, s=10)
plt.xlabel('Number of Hits')
plt.ylabel('Total Energy Deposit (MeV)')
plt.title('Energy vs Hit Correlation')
plt.grid(True, alpha=0.3)

# Material budget comparison
plt.subplot(2, 2, 4)
layers = ['Silicon', 'Air', 'Total']
budgets = [silicon_material_budget, air_material_budget, total_material_budget]
colors = ['blue', 'orange', 'green']
plt.bar(layers, budgets, color=colors, alpha=0.7)
plt.ylabel('Material Budget (X/X0)')
plt.title('Material Budget Breakdown')
plt.yscale('log')
for i, v in enumerate(budgets):
    plt.text(i, v*1.1, f'{v:.1e}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('baseline_performance_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('baseline_performance_analysis.pdf', bbox_inches='tight')
plt.show()

# 5. Save results to JSON
results = {
    'energy_deposit_mean_mev': float(mean_energy),
    'energy_deposit_std_mev': float(std_energy),
    'energy_resolution': float(energy_resolution),
    'energy_resolution_error': float(energy_resolution_err),
    'hit_detection_efficiency': float(hit_efficiency),
    'hit_efficiency_error': float(hit_efficiency_err),
    'silicon_material_budget_x0': float(silicon_material_budget),
    'air_material_budget_x0': float(air_material_budget),
    'total_material_budget_x0': float(total_material_budget),
    'num_events_analyzed': int(len(events)),
    'geometry_parameters': {
        'num_layers': num_layers,
        'active_thickness_mm': active_thickness_mm,
        'absorber_thickness_mm': absorber_thickness_mm
    }
}

with open('baseline_performance_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print results for workflow
print(f'RESULT:energy_resolution={energy_resolution:.6f}')
print(f'RESULT:energy_resolution_error={energy_resolution_err:.6f}')
print(f'RESULT:hit_efficiency={hit_efficiency:.6f}')
print(f'RESULT:hit_efficiency_error={hit_efficiency_err:.6f}')
print(f'RESULT:total_material_budget={total_material_budget:.6f}')
print(f'RESULT:mean_energy_deposit={mean_energy:.6f}')
print('RESULT:analysis_plot=baseline_performance_analysis.png')
print('RESULT:metrics_file=baseline_performance_metrics.json')

print('\n=== BASELINE PERFORMANCE SUMMARY ===')
print(f'Energy Resolution: {energy_resolution:.4f} ± {energy_resolution_err:.4f}')
print(f'Hit Detection Efficiency: {hit_efficiency*100:.2f}% ± {hit_efficiency_err*100:.2f}%')
print(f'Total Material Budget: {total_material_budget:.6f} X0')
print(f'Mean Energy Deposit: {mean_energy:.4f} MeV')