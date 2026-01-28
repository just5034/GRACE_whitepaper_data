import matplotlib
matplotlib.use('Agg')
import uproot
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Set publication style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

# Load validation data from optimal design (Accordion CsI)
validation_file = 'accordion_csi_calorimeter_electron_hits.root'

with uproot.open(validation_file) as f:
    events = f['events'].arrays(library='pd')
    print(f"Loaded {len(events)} events from validation simulation")

# Calculate key performance metrics
mean_energy = events['totalEdep'].mean()
std_energy = events['totalEdep'].std()
resolution = std_energy / mean_energy
resolution_err = resolution / np.sqrt(2 * len(events))

# Energy linearity (assuming 50 GeV input)
true_energy = 50000  # MeV
linearity = mean_energy / true_energy
linearity_err = std_energy / (true_energy * np.sqrt(len(events)))

print(f"Final Performance Metrics:")
print(f"Energy Resolution: {resolution:.4f} ± {resolution_err:.4f}")
print(f"Energy Linearity: {linearity:.4f} ± {linearity_err:.4f}")
print(f"Mean Energy: {mean_energy:.1f} MeV")

# Create figure with 4 subplots for final performance documentation
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Optimal Calorimeter Design: Accordion CsI Performance', fontsize=20, fontweight='bold')

# Plot 1: Final Resolution Curve (Energy Distribution)
ax1 = plt.subplot(2, 2, 1)
bins = np.linspace(events['totalEdep'].min(), events['totalEdep'].max(), 50)
n, bins_edges, patches = ax1.hist(events['totalEdep'], bins=bins, histtype='step', 
                                  linewidth=2, color='darkblue', label='Energy Deposits')
ax1.axvline(mean_energy, color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {mean_energy:.0f} MeV')
ax1.fill_between([mean_energy-std_energy, mean_energy+std_energy], 
                0, ax1.get_ylim()[1], alpha=0.2, color='red',
                label=f'±1σ: {std_energy:.0f} MeV')
ax1.set_xlabel('Total Energy Deposit (MeV)')
ax1.set_ylabel('Events')
ax1.set_title(f'Final Resolution: σ/E = {resolution:.3f} ± {resolution_err:.3f}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Energy Linearity
ax2 = plt.subplot(2, 2, 2)
# Show linearity as response vs true energy
response = mean_energy / 1000  # Convert to GeV for display
true_energy_gev = true_energy / 1000
response_err = std_energy / (1000 * np.sqrt(len(events)))

ax2.errorbar([true_energy_gev], [response], yerr=[response_err], 
            fmt='o', markersize=8, color='darkgreen', capsize=5, capthick=2)
ax2.plot([0, true_energy_gev*1.2], [0, true_energy_gev*1.2], 'k--', alpha=0.5, label='Perfect Linearity')
ax2.plot([true_energy_gev], [response], 'o', markersize=8, color='darkgreen', 
        label=f'Accordion CsI\nLinearity = {linearity:.3f}')
ax2.set_xlabel('True Energy (GeV)')
ax2.set_ylabel('Reconstructed Energy (GeV)')
ax2.set_title('Energy Linearity Performance')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, true_energy_gev*1.1)
ax2.set_ylim(0, max(response*1.1, true_energy_gev*1.1))

# Plot 3: Shower Profiles (Longitudinal)
ax3 = plt.subplot(2, 2, 3)
# Sample hits for shower profile to avoid memory issues
z_bins = np.linspace(-500, 500, 50)
z_centers = (z_bins[:-1] + z_bins[1:]) / 2
z_hist = np.zeros(len(z_bins)-1)

# Process hits in chunks to avoid memory issues
for batch in uproot.iterate(f'{validation_file}:hits', ['z', 'edep'], step_size='100 MB'):
    z_hist += np.histogram(batch['z'], bins=z_bins, weights=batch['edep'])[0]

# Normalize shower profile
z_hist = z_hist / np.sum(z_hist) if np.sum(z_hist) > 0 else z_hist

ax3.step(z_centers, z_hist, where='mid', linewidth=2, color='purple', label='Longitudinal Profile')
ax3.fill_between(z_centers, 0, z_hist, alpha=0.3, color='purple')
ax3.set_xlabel('Z Position (mm)')
ax3.set_ylabel('Normalized Energy Fraction')
ax3.set_title('Longitudinal Shower Profile')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Containment Efficiency
ax4 = plt.subplot(2, 2, 4)
radius_bins = np.linspace(0, 300, 50)
radius_centers = (radius_bins[:-1] + radius_bins[1:]) / 2

# Sample first 100k hits for containment analysis
with uproot.open(validation_file) as f:
    hits = f['hits'].arrays(['x', 'y', 'edep'], library='np', entry_stop=100000)

r = np.sqrt(hits['x']**2 + hits['y']**2)
total_edep = np.sum(hits['edep'])

contained = np.array([np.sum(hits['edep'][r < r_cut]) for r_cut in radius_bins])
containment_fractions = contained / total_edep if total_edep > 0 else contained

ax4.plot(radius_bins, containment_fractions * 100, linewidth=2, color='orange', label='Radial Containment')
ax4.axhline(90, color='red', linestyle='--', alpha=0.7, label='90% Target')
ax4.axhline(95, color='darkred', linestyle=':', alpha=0.7, label='95% Target')

# Find containment radii
idx_90 = np.where(containment_fractions >= 0.9)[0]
idx_95 = np.where(containment_fractions >= 0.95)[0]
r90 = radius_bins[idx_90[0]] if len(idx_90) > 0 else radius_bins[-1]
r95 = radius_bins[idx_95[0]] if len(idx_95) > 0 else radius_bins[-1]

ax4.axvline(r90, color='red', linestyle='--', alpha=0.5)
ax4.axvline(r95, color='darkred', linestyle=':', alpha=0.5)

ax4.set_xlabel('Radius (mm)')
ax4.set_ylabel('Containment (%)')
ax4.set_title(f'Containment: R90={r90:.0f}mm, R95={r95:.0f}mm')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0, 105)

plt.tight_layout()
plt.savefig('final_optimal_performance_plots.png', dpi=300, bbox_inches='tight')
plt.savefig('final_optimal_performance_plots.pdf', bbox_inches='tight')
print("RESULT:final_performance_plots=final_optimal_performance_plots.png")

# Save comprehensive performance summary
performance_summary = {
    'optimal_configuration': 'Accordion CsI',
    'energy_resolution': float(resolution),
    'energy_resolution_error': float(resolution_err),
    'energy_linearity': float(linearity),
    'energy_linearity_error': float(linearity_err),
    'containment_90_radius_mm': float(r90),
    'containment_95_radius_mm': float(r95),
    'mean_energy_mev': float(mean_energy),
    'std_energy_mev': float(std_energy),
    'validation_events': len(events),
    'meets_resolution_target': resolution < 0.03,
    'meets_linearity_target': abs(linearity - 1.0) < 0.05,
    'meets_containment_target': r90 < 250
}

with open('final_performance_summary.json', 'w') as f:
    json.dump(performance_summary, f, indent=2)

print("\n=== FINAL OPTIMAL DESIGN SUMMARY ===")
print(f"Configuration: {performance_summary['optimal_configuration']}")
print(f"Energy Resolution: {resolution:.3f} ± {resolution_err:.3f} ({'PASS' if performance_summary['meets_resolution_target'] else 'FAIL'})")
print(f"Energy Linearity: {linearity:.3f} ± {linearity_err:.3f} ({'PASS' if performance_summary['meets_linearity_target'] else 'FAIL'})")
print(f"90% Containment: {r90:.0f} mm ({'PASS' if performance_summary['meets_containment_target'] else 'FAIL'})")
print(f"95% Containment: {r95:.0f} mm")

print(f"\nRESULT:energy_resolution={resolution:.4f}")
print(f"RESULT:energy_resolution_error={resolution_err:.4f}")
print(f"RESULT:energy_linearity={linearity:.4f}")
print(f"RESULT:containment_90_radius={r90:.1f}")
print(f"RESULT:containment_95_radius={r95:.1f}")
print(f"RESULT:performance_summary_file=final_performance_summary.json")
print("RESULT:success=True")