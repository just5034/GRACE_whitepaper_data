import matplotlib
matplotlib.use('Agg')
import uproot
import matplotlib.pyplot as plt
import numpy as np
import json

# Load analysis results from previous step outputs
energy_resolution = 0.346873
energy_resolution_error = 0.007756
hit_efficiency = 0.999
hit_efficiency_error = 0.000999
total_material_budget = 0.013464
mean_energy_deposit = 0.437058

# Load simulation data
with uproot.open('baseline_planar_detector_muon_hits.root') as f:
    events = f['events'].arrays(library='pd')
    print(f'Loaded {len(events)} events')

# Create figure with subplots
fig = plt.figure(figsize=(15, 12))

# Plot 1: Energy deposit histogram
ax1 = plt.subplot(2, 3, 1)
plt.hist(events['totalEdep'], bins=50, histtype='step', linewidth=2, alpha=0.8, color='blue')
plt.axvline(mean_energy_deposit, color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {mean_energy_deposit:.3f} MeV')
plt.xlabel('Total Energy Deposit (MeV)')
plt.ylabel('Events')
plt.title(f'Energy Deposits\n(σ/E = {energy_resolution:.3f} ± {energy_resolution_error:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Hit efficiency (single point with error bar)
ax2 = plt.subplot(2, 3, 2)
momentum_points = [10.0]  # Single momentum point from simulation
efficiency_points = [hit_efficiency]
efficiency_errors = [hit_efficiency_error]
plt.errorbar(momentum_points, efficiency_points, yerr=efficiency_errors, 
            fmt='o', markersize=8, capsize=5, color='green', linewidth=2)
plt.xlabel('Momentum (GeV/c)')
plt.ylabel('Hit Efficiency')
plt.title('Detection Efficiency vs Momentum')
plt.grid(True, alpha=0.3)
plt.ylim(0.95, 1.005)

# Plot 3: Material budget visualization
ax3 = plt.subplot(2, 3, 3)
# Geometry parameters from previous steps
num_layers = 4
active_thickness_mm = 0.3
absorber_thickness_mm = 50.0

# Calculate cumulative material budget
layer_positions = np.arange(1, num_layers + 1)
material_per_layer = total_material_budget / num_layers
cumulative_material = np.cumsum([material_per_layer] * num_layers)

plt.step(layer_positions, cumulative_material, where='post', linewidth=3, color='purple')
plt.scatter(layer_positions, cumulative_material, s=50, color='purple', zorder=5)
plt.xlabel('Layer Number')
plt.ylabel('Cumulative Material Budget (X/X₀)')
plt.title(f'Material Budget\n(Total: {total_material_budget:.4f} X₀)')
plt.grid(True, alpha=0.3)

# Plot 4: Energy resolution vs statistics (bootstrap)
ax4 = plt.subplot(2, 3, 4)
n_bootstrap = 20
sample_sizes = np.logspace(1, np.log10(len(events)), 10).astype(int)
resolutions = []
for n in sample_sizes:
    bootstrap_resolutions = []
    for _ in range(n_bootstrap):
        sample = events['totalEdep'].sample(n=min(n, len(events)), replace=True)
        res = sample.std() / sample.mean()
        bootstrap_resolutions.append(res)
    resolutions.append(np.mean(bootstrap_resolutions))

plt.loglog(sample_sizes, resolutions, 'o-', linewidth=2, markersize=6, color='orange')
plt.axhline(energy_resolution, color='red', linestyle='--', 
           label=f'Full dataset: {energy_resolution:.3f}')
plt.xlabel('Number of Events')
plt.ylabel('Energy Resolution (σ/E)')
plt.title('Resolution vs Statistics')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: Detector geometry schematic
ax5 = plt.subplot(2, 3, 5)
# Simple schematic of detector layers
z_positions = np.array([0, 50.3, 100.6, 150.9, 201.2])  # Layer boundaries
for i in range(num_layers):
    # Active layers (silicon)
    plt.barh(0, active_thickness_mm, left=z_positions[i], height=0.5, 
            color='blue', alpha=0.7, label='Silicon' if i == 0 else '')
    # Air gaps
    if i < num_layers - 1:
        plt.barh(0, absorber_thickness_mm - active_thickness_mm, 
                left=z_positions[i] + active_thickness_mm, height=0.5, 
                color='lightgray', alpha=0.5, label='Air' if i == 0 else '')

plt.xlabel('Z Position (mm)')
plt.ylabel('')
plt.title('Detector Layout (Side View)')
plt.legend()
plt.ylim(-0.5, 0.5)
plt.yticks([])

# Plot 6: Performance summary
ax6 = plt.subplot(2, 3, 6)
metrics = ['Energy\nResolution', 'Hit\nEfficiency', 'Material\nBudget\n(×100)']
values = [energy_resolution, hit_efficiency, total_material_budget * 100]
errors = [energy_resolution_error, hit_efficiency_error, 0.001]
colors = ['blue', 'green', 'purple']

bars = plt.bar(metrics, values, yerr=errors, capsize=5, color=colors, alpha=0.7)
plt.ylabel('Value')
plt.title('Performance Summary')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val, err in zip(bars, values, errors):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + err + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('baseline_detector_performance_plots.png', dpi=150, bbox_inches='tight')
plt.savefig('baseline_detector_performance_plots.pdf', bbox_inches='tight')
plt.show()

# Save summary metrics
summary = {
    'energy_resolution': energy_resolution,
    'energy_resolution_error': energy_resolution_error,
    'hit_efficiency': hit_efficiency,
    'hit_efficiency_error': hit_efficiency_error,
    'material_budget': total_material_budget,
    'mean_energy_deposit_mev': mean_energy_deposit,
    'num_events_analyzed': len(events),
    'detector_layers': num_layers,
    'silicon_thickness_um': active_thickness_mm * 1000,
    'layer_spacing_mm': absorber_thickness_mm
}

with open('baseline_performance_plots_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('RESULT:plots_file=baseline_detector_performance_plots.png')
print('RESULT:plots_pdf=baseline_detector_performance_plots.pdf')
print('RESULT:summary_file=baseline_performance_plots_summary.json')
print(f'RESULT:energy_resolution_plot={energy_resolution:.4f}')
print(f'RESULT:efficiency_plot={hit_efficiency:.4f}')
print(f'RESULT:material_budget_plot={total_material_budget:.6f}')
print('Baseline detector performance plots generated successfully!')