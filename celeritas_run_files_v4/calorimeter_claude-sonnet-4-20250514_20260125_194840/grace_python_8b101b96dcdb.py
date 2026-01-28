import matplotlib
matplotlib.use('Agg')
import uproot
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Load projective simulation data
with uproot.open('projective_calorimeter_pip_hits.root') as f:
    proj_events = f['events'].arrays(library='pd')
    print(f"Loaded {len(proj_events)} projective events")

# Load baseline results from previous step outputs
baseline_resolution = 0.0771  # From plot_baseline_distributions
baseline_linearity_deviation = 650.01  # From analyze_baseline_performance

# Calculate projective performance metrics
proj_mean_E = proj_events['totalEdep'].mean()
proj_std_E = proj_events['totalEdep'].std()
proj_resolution = proj_std_E / proj_mean_E if proj_mean_E > 0 else 0
proj_resolution_err = proj_resolution / np.sqrt(2 * len(proj_events)) if len(proj_events) > 0 else 0

print(f"Projective resolution: {proj_resolution:.4f} ± {proj_resolution_err:.4f}")
print(f"Baseline resolution: {baseline_resolution:.4f}")

# Calculate topology comparison metrics
resolution_improvement = (baseline_resolution - proj_resolution) / baseline_resolution * 100 if baseline_resolution > 0 else 0

# Analyze crack effects by examining hit distributions
with uproot.open('projective_calorimeter_pip_hits.root') as f:
    # Sample first 100k hits to avoid timeout on large files
    proj_hits = f['hits'].arrays(['x', 'y', 'z', 'edep'], library='np', entry_stop=100000)

# Convert to numpy arrays explicitly to fix array type conversion
proj_x = np.array(proj_hits['x'])
proj_y = np.array(proj_hits['y'])
proj_z = np.array(proj_hits['z'])
proj_edep = np.array(proj_hits['edep'])

# Calculate angular response (eta-phi distribution)
proj_r = np.sqrt(proj_x**2 + proj_y**2)
proj_eta = -np.log(np.tan(0.5 * np.arctan2(proj_r, proj_z + 1500)))  # +1500mm for inner radius offset
proj_phi = np.arctan2(proj_y, proj_x)

# Analyze crack effects - look for non-uniformities in phi distribution
phi_bins = np.linspace(-np.pi, np.pi, 64)  # Match tower_phi_segments from geometry
phi_hist, _ = np.histogram(proj_phi, bins=phi_bins, weights=proj_edep)
phi_uniformity = np.std(phi_hist) / np.mean(phi_hist) if np.mean(phi_hist) > 0 else 0

# Create comparison plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Energy resolution comparison
configs = ['Baseline\n(Box)', 'Projective\n(Tower)']
resolutions = [baseline_resolution, proj_resolution]
errors = [baseline_resolution * 0.05, proj_resolution_err]  # Estimate baseline error
ax1.bar(configs, resolutions, yerr=errors, capsize=5, color=['blue', 'green'], alpha=0.7)
ax1.set_ylabel('Energy Resolution (σ/E)')
ax1.set_title('Topology Comparison: Energy Resolution')
ax1.grid(True, alpha=0.3)

# Plot 2: Raw energy distributions
ax2.hist(proj_events['totalEdep'], bins=50, histtype='step', linewidth=2, label='Projective', color='green')
ax2.axvline(proj_mean_E, color='g', linestyle='--', label=f'Proj Mean: {proj_mean_E:.1f} MeV')
ax2.set_xlabel('Total Energy Deposit (MeV)')
ax2.set_ylabel('Events')
ax2.set_title('Energy Distribution Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Angular response (eta-phi map)
hist_2d, eta_edges, phi_edges = np.histogram2d(proj_eta, proj_phi, bins=[25, 32], weights=proj_edep)
eta_centers = (eta_edges[:-1] + eta_edges[1:]) / 2
phi_centers = (phi_edges[:-1] + phi_edges[1:]) / 2
im = ax3.imshow(hist_2d.T, extent=[eta_edges[0], eta_edges[-1], phi_edges[0], phi_edges[-1]], 
                aspect='auto', origin='lower', cmap='viridis')
ax3.set_xlabel('Pseudorapidity (η)')
ax3.set_ylabel('Azimuthal angle (φ)')
ax3.set_title('Angular Response (η-φ map)')
plt.colorbar(im, ax=ax3, label='Energy Deposit (MeV)')

# Plot 4: Crack effects (phi uniformity)
ax4.step(phi_centers, phi_hist, where='mid', linewidth=2, color='red')
ax4.axhline(np.mean(phi_hist), color='k', linestyle='--', label=f'Mean: {np.mean(phi_hist):.1f}')
ax4.set_xlabel('Azimuthal angle φ (rad)')
ax4.set_ylabel('Energy Deposit (MeV)')
ax4.set_title(f'Crack Effects: φ Uniformity (σ/μ = {phi_uniformity:.3f})')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('projective_performance_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('projective_performance_analysis.pdf', bbox_inches='tight')
plt.show()

# Save analysis results
analysis_results = {
    'projective_resolution': float(proj_resolution),
    'projective_resolution_error': float(proj_resolution_err),
    'baseline_resolution': float(baseline_resolution),
    'resolution_improvement_percent': float(resolution_improvement),
    'phi_uniformity': float(phi_uniformity),
    'mean_energy_deposit': float(proj_mean_E),
    'num_events_analyzed': int(len(proj_events)),
    'num_hits_sampled': int(len(proj_x)),
    'topology': 'projective_tower',
    'analysis_focus': ['topology_comparison', 'crack_effects', 'angular_response']
}

with open('projective_performance_analysis.json', 'w') as f:
    json.dump(analysis_results, f, indent=2)

# Print results for workflow
print(f"RESULT:projective_resolution={proj_resolution:.4f}")
print(f"RESULT:resolution_improvement={resolution_improvement:.2f}")
print(f"RESULT:phi_uniformity={phi_uniformity:.4f}")
print(f"RESULT:mean_energy_deposit={proj_mean_E:.1f}")
print(f"RESULT:analysis_plot=projective_performance_analysis.png")
print(f"RESULT:analysis_file=projective_performance_analysis.json")
print(f"RESULT:topology_comparison_completed=True")
print(f"RESULT:crack_effects_analyzed=True")
print(f"RESULT:angular_response_mapped=True")

print("\n=== PROJECTIVE PERFORMANCE ANALYSIS COMPLETE ===")
print(f"Resolution improvement: {resolution_improvement:.1f}%")
print(f"Phi uniformity (crack metric): {phi_uniformity:.3f}")
print(f"Angular coverage mapped in η-φ space")