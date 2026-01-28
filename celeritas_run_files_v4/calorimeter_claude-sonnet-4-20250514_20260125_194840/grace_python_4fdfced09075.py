import matplotlib
matplotlib.use('Agg')
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Load cylindrical simulation data
cylindrical_hits_file = 'cylindrical_calorimeter_pip_hits.root'
baseline_hits_file = 'baseline_calorimeter_pip_hits.root'

print('Loading cylindrical simulation results...')
with uproot.open(cylindrical_hits_file) as f:
    cyl_events = f['events'].arrays(library='pd')
    print(f'Cylindrical events loaded: {len(cyl_events)} events')

print('Loading baseline simulation results...')
with uproot.open(baseline_hits_file) as f:
    base_events = f['events'].arrays(library='pd')
    print(f'Baseline events loaded: {len(base_events)} events')

# Calculate energy resolution for both geometries
cyl_mean_E = cyl_events['totalEdep'].mean()
cyl_std_E = cyl_events['totalEdep'].std()
cyl_resolution = cyl_std_E / cyl_mean_E if cyl_mean_E > 0 else 0
cyl_resolution_err = cyl_resolution / np.sqrt(2 * len(cyl_events)) if len(cyl_events) > 0 else 0

base_mean_E = base_events['totalEdep'].mean()
base_std_E = base_events['totalEdep'].std()
base_resolution = base_std_E / base_mean_E if base_mean_E > 0 else 0
base_resolution_err = base_resolution / np.sqrt(2 * len(base_events)) if len(base_events) > 0 else 0

print(f'Cylindrical (brass) resolution: {cyl_resolution:.4f} ± {cyl_resolution_err:.4f}')
print(f'Baseline (steel) resolution: {base_resolution:.4f} ± {base_resolution_err:.4f}')

# Material effect analysis (brass vs steel)
material_improvement = (base_resolution - cyl_resolution) / base_resolution * 100 if base_resolution > 0 else 0
print(f'Material effect (brass vs steel): {material_improvement:.2f}% improvement')

# Analyze hermeticity and coverage uniformity using hit-level data
print('Analyzing coverage uniformity and hermeticity...')

# Sample first 100k hits to avoid timeout on large files
with uproot.open(cylindrical_hits_file) as f:
    cyl_hits = f['hits'].arrays(['x', 'y', 'z', 'edep'], library='np', entry_stop=100000)

with uproot.open(baseline_hits_file) as f:
    base_hits = f['hits'].arrays(['x', 'y', 'z', 'edep'], library='np', entry_stop=100000)

# Calculate radial distributions for coverage analysis
cyl_r = np.sqrt(cyl_hits['x']**2 + cyl_hits['y']**2)
base_r = np.sqrt(base_hits['x']**2 + base_hits['y']**2)

# Hermeticity analysis - radial containment
radius_bins = np.linspace(0, 3000, 50)  # mm
cyl_total_edep = np.sum(cyl_hits['edep'])
base_total_edep = np.sum(base_hits['edep'])

cyl_contained = np.array([np.sum(cyl_hits['edep'][cyl_r < r_cut]) for r_cut in radius_bins])
base_contained = np.array([np.sum(base_hits['edep'][base_r < r_cut]) for r_cut in radius_bins])

cyl_containment = cyl_contained / cyl_total_edep if cyl_total_edep > 0 else cyl_contained
base_containment = base_contained / base_total_edep if base_total_edep > 0 else base_contained

# Find 90% containment radii
cyl_idx_90 = np.where(cyl_containment >= 0.9)[0]
base_idx_90 = np.where(base_containment >= 0.9)[0]
cyl_r90 = radius_bins[cyl_idx_90[0]] if len(cyl_idx_90) > 0 else radius_bins[-1]
base_r90 = radius_bins[base_idx_90[0]] if len(base_idx_90) > 0 else radius_bins[-1]

print(f'Cylindrical 90% containment radius: {cyl_r90:.1f} mm')
print(f'Baseline 90% containment radius: {base_r90:.1f} mm')

hermeticity_improvement = (base_r90 - cyl_r90) / base_r90 * 100 if base_r90 > 0 else 0
print(f'Hermeticity improvement: {hermeticity_improvement:.2f}% (smaller containment radius is better)')

# Coverage uniformity analysis - azimuthal distribution
cyl_phi = np.arctan2(cyl_hits['y'], cyl_hits['x'])
base_phi = np.arctan2(base_hits['y'], base_hits['x'])

phi_bins = np.linspace(-np.pi, np.pi, 36)  # 10-degree bins
cyl_phi_hist = np.histogram(cyl_phi, bins=phi_bins, weights=cyl_hits['edep'])[0]
base_phi_hist = np.histogram(base_phi, bins=phi_bins, weights=base_hits['edep'])[0]

# Calculate uniformity (coefficient of variation)
cyl_uniformity = np.std(cyl_phi_hist) / np.mean(cyl_phi_hist) if np.mean(cyl_phi_hist) > 0 else 0
base_uniformity = np.std(base_phi_hist) / np.mean(base_phi_hist) if np.mean(base_phi_hist) > 0 else 0

print(f'Cylindrical azimuthal uniformity (CV): {cyl_uniformity:.4f}')
print(f'Baseline azimuthal uniformity (CV): {base_uniformity:.4f}')

uniformity_improvement = (base_uniformity - cyl_uniformity) / base_uniformity * 100 if base_uniformity > 0 else 0
print(f'Coverage uniformity improvement: {uniformity_improvement:.2f}%')

# Generate comparison plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Energy resolution comparison
configs = ['Baseline\n(Steel)', 'Cylindrical\n(Brass)']
resolutions = [base_resolution, cyl_resolution]
errors = [base_resolution_err, cyl_resolution_err]
ax1.bar(configs, resolutions, yerr=errors, capsize=5, color=['blue', 'green'], alpha=0.7)
ax1.set_ylabel('Energy Resolution (σ/E)')
ax1.set_title('Energy Resolution Comparison')
ax1.grid(True, alpha=0.3)

# Containment comparison
ax2.plot(radius_bins, base_containment * 100, 'b-', linewidth=2, label='Baseline (Steel)')
ax2.plot(radius_bins, cyl_containment * 100, 'g-', linewidth=2, label='Cylindrical (Brass)')
ax2.axhline(90, color='r', linestyle='--', alpha=0.7, label='90% containment')
ax2.set_xlabel('Radius (mm)')
ax2.set_ylabel('Containment (%)')
ax2.set_title('Radial Containment (Hermeticity)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Azimuthal uniformity
phi_centers = (phi_bins[:-1] + phi_bins[1:]) / 2
ax3.plot(phi_centers * 180/np.pi, base_phi_hist / np.max(base_phi_hist), 'b-', linewidth=2, label='Baseline (Steel)')
ax3.plot(phi_centers * 180/np.pi, cyl_phi_hist / np.max(cyl_phi_hist), 'g-', linewidth=2, label='Cylindrical (Brass)')
ax3.set_xlabel('Azimuthal Angle (degrees)')
ax3.set_ylabel('Normalized Energy Deposit')
ax3.set_title('Azimuthal Coverage Uniformity')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Summary metrics
metrics = ['Energy\nResolution', 'Containment\nRadius (mm)', 'Azimuthal\nUniformity (CV)']
base_values = [base_resolution, base_r90, base_uniformity]
cyl_values = [cyl_resolution, cyl_r90, cyl_uniformity]

x_pos = np.arange(len(metrics))
width = 0.35
ax4.bar(x_pos - width/2, base_values, width, label='Baseline (Steel)', color='blue', alpha=0.7)
ax4.bar(x_pos + width/2, cyl_values, width, label='Cylindrical (Brass)', color='green', alpha=0.7)
ax4.set_xlabel('Performance Metrics')
ax4.set_ylabel('Value')
ax4.set_title('Performance Summary')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(metrics)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cylindrical_performance_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('cylindrical_performance_analysis.pdf', bbox_inches='tight')

# Save analysis results
analysis_results = {
    'cylindrical_geometry': {
        'topology': 'cylinder_barrel',
        'absorber_material': 'brass',
        'energy_resolution': float(cyl_resolution),
        'energy_resolution_error': float(cyl_resolution_err),
        'containment_90_radius_mm': float(cyl_r90),
        'azimuthal_uniformity_cv': float(cyl_uniformity)
    },
    'baseline_geometry': {
        'topology': 'box',
        'absorber_material': 'steel',
        'energy_resolution': float(base_resolution),
        'energy_resolution_error': float(base_resolution_err),
        'containment_90_radius_mm': float(base_r90),
        'azimuthal_uniformity_cv': float(base_uniformity)
    },
    'comparisons': {
        'material_effect_improvement_percent': float(material_improvement),
        'hermeticity_improvement_percent': float(hermeticity_improvement),
        'uniformity_improvement_percent': float(uniformity_improvement)
    },
    'analysis_focus': ['hermeticity', 'material_effects', 'coverage_uniformity']
}

with open('cylindrical_performance_analysis.json', 'w') as f:
    json.dump(analysis_results, f, indent=2)

print('\n=== CYLINDRICAL PERFORMANCE ANALYSIS SUMMARY ===')
print(f'Material Effect (Brass vs Steel): {material_improvement:.2f}% resolution improvement')
print(f'Hermeticity: {hermeticity_improvement:.2f}% containment improvement')
print(f'Coverage Uniformity: {uniformity_improvement:.2f}% uniformity improvement')
print('Analysis plots saved to cylindrical_performance_analysis.png')
print('Detailed results saved to cylindrical_performance_analysis.json')

# Return values for downstream steps
print(f'RESULT:cylindrical_resolution={cyl_resolution:.4f}')
print(f'RESULT:baseline_resolution={base_resolution:.4f}')
print(f'RESULT:material_improvement={material_improvement:.2f}')
print(f'RESULT:hermeticity_improvement={hermeticity_improvement:.2f}')
print(f'RESULT:uniformity_improvement={uniformity_improvement:.2f}')
print(f'RESULT:containment_90_radius={cyl_r90:.1f}')
print('RESULT:performance_plot=cylindrical_performance_analysis.png')
print('RESULT:analysis_file=cylindrical_performance_analysis.json')
print('RESULT:cylindrical_analysis_completed=True')