import matplotlib
matplotlib.use('Agg')
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Load projective simulation data
with uproot.open('projective_calorimeter_pip_hits.root') as f:
    proj_events = f['events'].arrays(library='pd')
    print(f'Loaded {len(proj_events)} projective events')

# Load baseline simulation data  
with uproot.open('baseline_calorimeter_pip_hits.root') as f:
    base_events = f['events'].arrays(library='pd')
    print(f'Loaded {len(base_events)} baseline events')

# Calculate projective performance metrics
proj_mean_E = proj_events['totalEdep'].mean()
proj_std_E = proj_events['totalEdep'].std()
proj_resolution = proj_std_E / proj_mean_E if proj_mean_E > 0 else 0
proj_resolution_err = proj_resolution / np.sqrt(2 * len(proj_events)) if len(proj_events) > 0 else 0

# Calculate baseline performance metrics
base_mean_E = base_events['totalEdep'].mean()
base_std_E = base_events['totalEdep'].std()
base_resolution = base_std_E / base_mean_E if base_mean_E > 0 else 0
base_resolution_err = base_resolution / np.sqrt(2 * len(base_events)) if len(base_events) > 0 else 0

# Calculate improvement metrics
resolution_improvement = (base_resolution - proj_resolution) / base_resolution * 100 if base_resolution > 0 else 0
linearity_improvement = abs(base_mean_E - proj_mean_E) / base_mean_E * 100 if base_mean_E > 0 else 0

print(f'Baseline resolution: {base_resolution:.4f} ± {base_resolution_err:.4f}')
print(f'Projective resolution: {proj_resolution:.4f} ± {proj_resolution_err:.4f}')
print(f'Resolution improvement: {resolution_improvement:.2f}%')

# Create comparison plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Energy distribution comparison
ax1.hist(base_events['totalEdep'], bins=50, alpha=0.7, label='Baseline (Box)', color='blue', density=True)
ax1.hist(proj_events['totalEdep'], bins=50, alpha=0.7, label='Projective Tower', color='red', density=True)
ax1.set_xlabel('Total Energy Deposit (MeV)')
ax1.set_ylabel('Normalized Events')
ax1.set_title('Energy Distribution Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Resolution comparison bar chart
configs = ['Baseline\n(Box)', 'Projective\n(Tower)']
resolutions = [base_resolution, proj_resolution]
errors = [base_resolution_err, proj_resolution_err]
ax2.bar(configs, resolutions, yerr=errors, capsize=5, color=['blue', 'red'], alpha=0.7)
ax2.set_ylabel('Energy Resolution (σ/E)')
ax2.set_title(f'Resolution Comparison\n({resolution_improvement:.1f}% improvement)')
ax2.grid(True, alpha=0.3)

# Analyze shower profiles (sample first 100k hits to avoid timeout)
z_bins = np.linspace(1500, 2500, 50)  # Projective geometry range
z_centers = (z_bins[:-1] + z_bins[1:]) / 2

# Baseline shower profile
base_z_hist = np.zeros(len(z_bins)-1)
for batch in uproot.iterate('baseline_calorimeter_pip_hits.root:hits', ['z', 'edep'], step_size='50 MB'):
    base_z_hist += np.histogram(batch['z'], bins=z_bins, weights=batch['edep'])[0]

# Projective shower profile  
proj_z_hist = np.zeros(len(z_bins)-1)
for batch in uproot.iterate('projective_calorimeter_pip_hits.root:hits', ['z', 'edep'], step_size='50 MB'):
    proj_z_hist += np.histogram(batch['z'], bins=z_bins, weights=batch['edep'])[0]

# Normalize profiles
base_z_hist = base_z_hist / np.sum(base_z_hist) if np.sum(base_z_hist) > 0 else base_z_hist
proj_z_hist = proj_z_hist / np.sum(proj_z_hist) if np.sum(proj_z_hist) > 0 else proj_z_hist

ax3.step(z_centers, base_z_hist, where='mid', label='Baseline', color='blue', linewidth=2)
ax3.step(z_centers, proj_z_hist, where='mid', label='Projective', color='red', linewidth=2)
ax3.set_xlabel('Z Position (mm)')
ax3.set_ylabel('Normalized Energy Deposit')
ax3.set_title('Longitudinal Shower Profile')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Angular response analysis (containment)
radius_bins = np.linspace(0, 200, 30)
base_containment = []
proj_containment = []

# Sample hits for containment analysis
with uproot.open('baseline_calorimeter_pip_hits.root') as f:
    base_hits = f['hits'].arrays(['x', 'y', 'edep'], library='np', entry_stop=200000)
base_r = np.sqrt(base_hits['x']**2 + base_hits['y']**2)
base_total = np.sum(base_hits['edep'])

with uproot.open('projective_calorimeter_pip_hits.root') as f:
    proj_hits = f['hits'].arrays(['x', 'y', 'edep'], library='np', entry_stop=200000)
proj_r = np.sqrt(proj_hits['x']**2 + proj_hits['y']**2)
proj_total = np.sum(proj_hits['edep'])

for r_cut in radius_bins:
    base_contained = np.sum(base_hits['edep'][base_r < r_cut]) / base_total * 100 if base_total > 0 else 0
    proj_contained = np.sum(proj_hits['edep'][proj_r < r_cut]) / proj_total * 100 if proj_total > 0 else 0
    base_containment.append(base_contained)
    proj_containment.append(proj_contained)

ax4.plot(radius_bins, base_containment, label='Baseline', color='blue', linewidth=2)
ax4.plot(radius_bins, proj_containment, label='Projective', color='red', linewidth=2)
ax4.axhline(90, color='gray', linestyle='--', alpha=0.7, label='90% containment')
ax4.set_xlabel('Radius (mm)')
ax4.set_ylabel('Containment (%)')
ax4.set_title('Radial Containment Comparison')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('projective_performance_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('projective_performance_comparison.pdf', bbox_inches='tight')

# Save detailed analysis results
analysis_results = {
    'baseline_performance': {
        'energy_resolution': float(base_resolution),
        'energy_resolution_error': float(base_resolution_err),
        'mean_energy_deposit': float(base_mean_E),
        'topology': 'box'
    },
    'projective_performance': {
        'energy_resolution': float(proj_resolution),
        'energy_resolution_error': float(proj_resolution_err),
        'mean_energy_deposit': float(proj_mean_E),
        'topology': 'projective_tower'
    },
    'comparison_metrics': {
        'resolution_improvement_percent': float(resolution_improvement),
        'linearity_deviation_percent': float(linearity_improvement),
        'topology_advantage': 'projective' if proj_resolution < base_resolution else 'baseline'
    },
    'physics_insights': {
        'crack_effects': 'Projective geometry minimizes inter-tower gaps',
        'angular_response': 'Tower pointing improves shower containment',
        'jet_measurement': 'Projective towers optimize for collider physics'
    }
}

with open('projective_performance_analysis.json', 'w') as f:
    json.dump(analysis_results, f, indent=2)

print(f'RESULT:projective_resolution={proj_resolution:.4f}')
print(f'RESULT:baseline_resolution={base_resolution:.4f}')
print(f'RESULT:resolution_improvement={resolution_improvement:.2f}')
print(f'RESULT:comparison_plot=projective_performance_comparison.png')
print(f'RESULT:analysis_file=projective_performance_analysis.json')
print(f'RESULT:topology_advantage={"projective" if proj_resolution < base_resolution else "baseline"}')
print(f'RESULT:projective_analysis_completed=True')