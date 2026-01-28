import matplotlib
matplotlib.use('Agg')
import uproot
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Load projective calorimeter simulation data
proj_hits_file = 'projective_pbwo4_calorimeter_electron_hits.root'

print(f'Analyzing projective calorimeter: {proj_hits_file}')

# Read events data for energy resolution and linearity
with uproot.open(proj_hits_file) as f:
    events = f['events'].arrays(library='pd')
    print(f'Loaded {len(events)} events from projective calorimeter')

# Calculate energy resolution
mean_E = events['totalEdep'].mean()
std_E = events['totalEdep'].std()
proj_energy_resolution = std_E / mean_E if mean_E > 0 else 0
proj_energy_resolution_error = proj_energy_resolution / np.sqrt(2 * len(events)) if len(events) > 0 else 0

# Calculate linearity (mean reconstructed energy / true energy)
# Assuming 1 GeV true energy based on typical simulation setup
true_energy_gev = 1.0  # GeV
mean_E_gev = mean_E / 1000.0  # Convert MeV to GeV
proj_linearity = mean_E_gev / true_energy_gev if true_energy_gev > 0 else 0
proj_linearity_error = (std_E / 1000.0) / (true_energy_gev * np.sqrt(len(events))) if len(events) > 0 else 0

print(f'Projective Energy Resolution: {proj_energy_resolution:.6f} ± {proj_energy_resolution_error:.6f}')
print(f'Projective Linearity: {proj_linearity:.6f} ± {proj_linearity_error:.6f}')

# Calculate containment - sample first 500k hits to avoid timeout
print('Calculating containment (sampling hits for performance)...')
with uproot.open(proj_hits_file) as f:
    hits = f['hits'].arrays(['x', 'y', 'edep'], library='np', entry_stop=500000)

r = np.sqrt(hits['x']**2 + hits['y']**2)
total_edep = np.sum(hits['edep'])

# Calculate containment at different radii
radius_values = np.linspace(5, 100, 100)  # mm
containment_fractions = []
for r_cut in radius_values:
    contained_edep = np.sum(hits['edep'][r < r_cut])
    frac = contained_edep / total_edep if total_edep > 0 else 0
    containment_fractions.append(frac)

containment_fractions = np.array(containment_fractions)

# Find 90% and 95% containment radii
idx_90 = np.where(containment_fractions >= 0.9)[0]
idx_95 = np.where(containment_fractions >= 0.95)[0]
proj_containment_90_radius = radius_values[idx_90[0]] if len(idx_90) > 0 else radius_values[-1]
proj_containment_95_radius = radius_values[idx_95[0]] if len(idx_95) > 0 else radius_values[-1]

print(f'Projective 90% Containment Radius: {proj_containment_90_radius:.2f} mm')
print(f'Projective 95% Containment Radius: {proj_containment_95_radius:.2f} mm')

# Baseline values from previous step outputs
baseline_energy_resolution = 0.034363
baseline_energy_resolution_error = 0.000768
baseline_linearity = 0.467007
baseline_linearity_error = 0.000507
baseline_containment_90_radius = 40.82
baseline_containment_95_radius = 53.06

# Calculate improvements/degradations
resolution_change = (proj_energy_resolution - baseline_energy_resolution) / baseline_energy_resolution * 100
linearity_change = (proj_linearity - baseline_linearity) / baseline_linearity * 100
containment_90_change = (proj_containment_90_radius - baseline_containment_90_radius) / baseline_containment_90_radius * 100

print('\n=== PERFORMANCE COMPARISON ===')
print(f'Energy Resolution: {baseline_energy_resolution:.6f} → {proj_energy_resolution:.6f} ({resolution_change:+.1f}%)')
print(f'Linearity: {baseline_linearity:.6f} → {proj_linearity:.6f} ({linearity_change:+.1f}%)')
print(f'90% Containment: {baseline_containment_90_radius:.1f} → {proj_containment_90_radius:.1f} mm ({containment_90_change:+.1f}%)')

# Generate comparison plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Energy distribution comparison
ax1.hist(events['totalEdep'], bins=50, alpha=0.7, label='Projective PbWO4', color='red', histtype='step', linewidth=2)
ax1.axvline(mean_E, color='red', linestyle='--', label=f'Proj Mean: {mean_E:.1f} MeV')
ax1.set_xlabel('Total Energy Deposit (MeV)')
ax1.set_ylabel('Events')
ax1.set_title('Energy Distribution Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Resolution comparison
configs = ['Baseline\n(CsI Box)', 'Projective\n(PbWO4 Tower)']
resolutions = [baseline_energy_resolution, proj_energy_resolution]
res_errors = [baseline_energy_resolution_error, proj_energy_resolution_error]
ax2.bar(configs, resolutions, yerr=res_errors, capsize=5, color=['blue', 'red'], alpha=0.7)
ax2.set_ylabel('Energy Resolution (σ/E)')
ax2.set_title('Energy Resolution Comparison')
ax2.grid(True, alpha=0.3)

# Plot 3: Linearity comparison
linearities = [baseline_linearity, proj_linearity]
lin_errors = [baseline_linearity_error, proj_linearity_error]
ax3.bar(configs, linearities, yerr=lin_errors, capsize=5, color=['blue', 'red'], alpha=0.7)
ax3.set_ylabel('Linearity (Reconstructed/True)')
ax3.set_title('Linearity Comparison')
ax3.grid(True, alpha=0.3)

# Plot 4: Containment comparison
ax4.plot(radius_values, containment_fractions * 100, 'r-', linewidth=2, label='Projective PbWO4')
ax4.axhline(90, color='gray', linestyle='--', alpha=0.7)
ax4.axvline(baseline_containment_90_radius, color='blue', linestyle=':', label=f'Baseline 90%: {baseline_containment_90_radius:.1f} mm')
ax4.axvline(proj_containment_90_radius, color='red', linestyle=':', label=f'Projective 90%: {proj_containment_90_radius:.1f} mm')
ax4.set_xlabel('Radius (mm)')
ax4.set_ylabel('Containment (%)')
ax4.set_title('Radial Containment Comparison')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('projective_performance_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('projective_performance_comparison.pdf', bbox_inches='tight')
plt.show()

# Save detailed metrics to JSON
metrics = {
    'projective_metrics': {
        'energy_resolution': float(proj_energy_resolution),
        'energy_resolution_error': float(proj_energy_resolution_error),
        'linearity': float(proj_linearity),
        'linearity_error': float(proj_linearity_error),
        'containment_90_radius': float(proj_containment_90_radius),
        'containment_95_radius': float(proj_containment_95_radius)
    },
    'baseline_metrics': {
        'energy_resolution': baseline_energy_resolution,
        'energy_resolution_error': baseline_energy_resolution_error,
        'linearity': baseline_linearity,
        'linearity_error': baseline_linearity_error,
        'containment_90_radius': baseline_containment_90_radius,
        'containment_95_radius': baseline_containment_95_radius
    },
    'performance_changes': {
        'resolution_change_percent': float(resolution_change),
        'linearity_change_percent': float(linearity_change),
        'containment_90_change_percent': float(containment_90_change)
    }
}

with open('projective_performance_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Output results for workflow
print(f'RESULT:energy_resolution={proj_energy_resolution:.6f}')
print(f'RESULT:energy_resolution_error={proj_energy_resolution_error:.6f}')
print(f'RESULT:linearity={proj_linearity:.6f}')
print(f'RESULT:linearity_error={proj_linearity_error:.6f}')
print(f'RESULT:containment_90_radius={proj_containment_90_radius:.2f}')
print(f'RESULT:containment_95_radius={proj_containment_95_radius:.2f}')
print(f'RESULT:resolution_change_percent={resolution_change:.2f}')
print(f'RESULT:linearity_change_percent={linearity_change:.2f}')
print(f'RESULT:containment_change_percent={containment_90_change:.2f}')
print('RESULT:comparison_plot=projective_performance_comparison.png')
print('RESULT:metrics_file=projective_performance_metrics.json')
print('RESULT:success=True')