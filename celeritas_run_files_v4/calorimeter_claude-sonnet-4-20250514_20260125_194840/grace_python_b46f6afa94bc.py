import matplotlib
matplotlib.use('Agg')
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Load projective simulation data
proj_hits_file = '/u/jhill5/grace/work/benchmarks/celeritas_20260125_192529/calorimeter_claude-sonnet-4-20250514_20260125_194840/projective_calorimeter_pip_hits.root'
baseline_hits_file = '/u/jhill5/grace/work/benchmarks/celeritas_20260125_192529/calorimeter_claude-sonnet-4-20250514_20260125_194840/baseline_calorimeter_pip_hits.root'

# Load baseline results from previous step outputs
baseline_resolution = 0.0771
baseline_linearity_deviation = 650.01

print('Loading projective calorimeter data...')
with uproot.open(proj_hits_file) as f:
    proj_events = f['events'].arrays(library='pd')
    print(f'Loaded {len(proj_events)} projective events')

print('Loading baseline calorimeter data...')
with uproot.open(baseline_hits_file) as f:
    baseline_events = f['events'].arrays(library='pd')
    print(f'Loaded {len(baseline_events)} baseline events')

# Calculate projective performance metrics
proj_mean_E = proj_events['totalEdep'].mean()
proj_std_E = proj_events['totalEdep'].std()
proj_resolution = proj_std_E / proj_mean_E if proj_mean_E > 0 else 0
proj_resolution_err = proj_resolution / np.sqrt(2 * len(proj_events)) if len(proj_events) > 0 else 0

# Calculate baseline performance for comparison
base_mean_E = baseline_events['totalEdep'].mean()
base_std_E = baseline_events['totalEdep'].std()
base_resolution = base_std_E / base_mean_E if base_mean_E > 0 else 0
base_resolution_err = base_resolution / np.sqrt(2 * len(baseline_events)) if len(baseline_events) > 0 else 0

print(f'Projective resolution: {proj_resolution:.4f} ± {proj_resolution_err:.4f}')
print(f'Baseline resolution: {base_resolution:.4f} ± {base_resolution_err:.4f}')

# Topology comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Energy distributions
ax1.hist(baseline_events['totalEdep'], bins=50, alpha=0.7, label='Baseline (Box)', density=True, histtype='step', linewidth=2)
ax1.hist(proj_events['totalEdep'], bins=50, alpha=0.7, label='Projective (Tower)', density=True, histtype='step', linewidth=2)
ax1.set_xlabel('Total Energy Deposit (MeV)')
ax1.set_ylabel('Normalized Events')
ax1.set_title('Energy Distribution Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Resolution comparison with error bars
configs = ['Baseline\n(Box)', 'Projective\n(Tower)']
resolutions = [base_resolution, proj_resolution]
errors = [base_resolution_err, proj_resolution_err]
colors = ['blue', 'green']
ax2.bar(configs, resolutions, yerr=errors, capsize=5, color=colors, alpha=0.7)
ax2.set_ylabel('Energy Resolution (σ/E)')
ax2.set_title('Resolution Comparison')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('projective_topology_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('projective_topology_comparison.pdf', bbox_inches='tight')
plt.close()

# Angular response analysis for projective geometry
print('Analyzing angular response...')
with uproot.open(proj_hits_file) as f:
    # Sample first 100k hits to avoid timeout
    proj_hits = f['hits'].arrays(['x', 'y', 'z', 'edep'], library='np', entry_stop=100000)

# Convert to cylindrical coordinates for projective analysis
r = np.sqrt(proj_hits['x']**2 + proj_hits['y']**2)
phi = np.arctan2(proj_hits['y'], proj_hits['x'])
z = proj_hits['z']

# Calculate pseudorapidity (eta) for projective towers
theta = np.arctan2(r, z)
eta = -np.log(np.tan(theta / 2))

# Create angular response histograms with proper binning
phi_bins = np.linspace(-np.pi, np.pi, 32)
eta_bins = np.linspace(-2, 2, 32)

# Fix histogram binning - ensure matching dimensions
phi_hist, phi_edges = np.histogram(phi, bins=phi_bins, weights=proj_hits['edep'])
eta_hist, eta_edges = np.histogram(eta, bins=eta_bins, weights=proj_hits['edep'])

# Use proper bin centers calculation
phi_centers = (phi_edges[:-1] + phi_edges[1:]) / 2
eta_centers = (eta_edges[:-1] + eta_edges[1:]) / 2

# Verify array shapes before plotting
print(f'Phi histogram shape: {phi_hist.shape}, centers shape: {phi_centers.shape}')
print(f'Eta histogram shape: {eta_hist.shape}, centers shape: {eta_centers.shape}')

# Angular response plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Phi distribution
ax1.step(phi_centers, phi_hist, where='mid', linewidth=2)
ax1.set_xlabel('Azimuthal Angle φ (rad)')
ax1.set_ylabel('Energy Deposit (MeV)')
ax1.set_title('Azimuthal Response (Projective Towers)')
ax1.grid(True, alpha=0.3)

# Eta distribution
ax2.step(eta_centers, eta_hist, where='mid', linewidth=2)
ax2.set_xlabel('Pseudorapidity η')
ax2.set_ylabel('Energy Deposit (MeV)')
ax2.set_title('Pseudorapidity Response')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('projective_angular_response.png', dpi=150, bbox_inches='tight')
plt.savefig('projective_angular_response.pdf', bbox_inches='tight')
plt.close()

# Calculate improvement metrics
resolution_improvement = (base_resolution - proj_resolution) / base_resolution * 100 if base_resolution > 0 else 0

# Save analysis results
analysis_results = {
    'projective_resolution': float(proj_resolution),
    'projective_resolution_error': float(proj_resolution_err),
    'baseline_resolution': float(base_resolution),
    'baseline_resolution_error': float(base_resolution_err),
    'resolution_improvement_percent': float(resolution_improvement),
    'topology_comparison_completed': True,
    'angular_response_analyzed': True,
    'crack_effects_minimized': proj_resolution < base_resolution
}

with open('projective_performance_analysis.json', 'w') as f:
    json.dump(analysis_results, f, indent=2)

print(f'RESULT:projective_resolution={proj_resolution:.4f}')
print(f'RESULT:resolution_improvement={resolution_improvement:.2f}')
print(f'RESULT:topology_comparison_plot=projective_topology_comparison.png')
print(f'RESULT:angular_response_plot=projective_angular_response.png')
print(f'RESULT:analysis_file=projective_performance_analysis.json')
print(f'RESULT:projective_analysis_completed=True')
print('Projective performance analysis completed successfully!')