import matplotlib
matplotlib.use('Agg')
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# First analyze accordion calorimeter performance
print('Analyzing accordion calorimeter performance...')

# Read accordion simulation data
with uproot.open('accordion_csi_calorimeter_electron_hits.root') as f:
    events = f['events'].arrays(library='pd')

# Calculate accordion performance metrics
mean_E = events['totalEdep'].mean()
std_E = events['totalEdep'].std()
accordion_resolution = std_E / mean_E if mean_E > 0 else 0
accordion_resolution_err = accordion_resolution / np.sqrt(2 * len(events)) if len(events) > 0 else 0

# Calculate linearity (ratio of measured to expected energy)
# Assuming 1 GeV input energy based on simulation setup
expected_energy = 1000.0  # MeV
accordion_linearity = mean_E / expected_energy if expected_energy > 0 else 0
accordion_linearity_err = std_E / expected_energy if expected_energy > 0 else 0

# Calculate containment (sample first 500k hits to avoid timeout)
print('Calculating accordion containment...')
with uproot.open('accordion_csi_calorimeter_electron_hits.root') as f:
    hits = f['hits'].arrays(['x', 'y', 'edep'], library='np', entry_stop=500000)

r = np.sqrt(hits['x']**2 + hits['y']**2)
total_edep = np.sum(hits['edep'])
radius_bins = np.linspace(0, 200, 100)
contained = np.array([np.sum(hits['edep'][r < r_cut]) for r_cut in radius_bins])
containment_fractions = contained / total_edep if total_edep > 0 else contained

# Find 90% and 95% containment radii
idx_90 = np.where(containment_fractions >= 0.9)[0]
idx_95 = np.where(containment_fractions >= 0.95)[0]
accordion_r90 = radius_bins[idx_90[0]] if len(idx_90) > 0 else radius_bins[-1]
accordion_r95 = radius_bins[idx_95[0]] if len(idx_95) > 0 else radius_bins[-1]

print(f'Accordion Energy Resolution: {accordion_resolution:.6f} ± {accordion_resolution_err:.6f}')
print(f'Accordion Linearity: {accordion_linearity:.6f} ± {accordion_linearity_err:.6f}')
print(f'Accordion 90% Containment: {accordion_r90:.1f} mm')
print(f'Accordion 95% Containment: {accordion_r95:.1f} mm')

# Compile all results from previous analyses (from Previous Step Outputs)
results = {
    'Baseline (CsI Box)': {
        'energy_resolution': 0.034363,
        'energy_resolution_error': 0.000768,
        'linearity': 0.467007,
        'linearity_error': 0.000507,
        'containment_90_radius': 40.82,
        'containment_95_radius': 53.06,
        'material': 'CsI',
        'topology': 'box'
    },
    'Projective (PbWO4)': {
        'energy_resolution': 0.021387,
        'energy_resolution_error': 0.000478,
        'linearity': 0.978699,
        'linearity_error': 0.000662,
        'containment_90_radius': 100.0,
        'containment_95_radius': 100.0,
        'material': 'PbWO4',
        'topology': 'projective_tower'
    },
    'Shashlik (BGO)': {
        'energy_resolution': 0.036765,
        'energy_resolution_error': 0.000822,
        'linearity': 4.72437,
        'linearity_error': 0.005493,
        'containment_90_radius': 24.2,
        'containment_95_radius': 36.4,
        'material': 'BGO',
        'topology': 'shashlik'
    },
    'Accordion (CsI)': {
        'energy_resolution': accordion_resolution,
        'energy_resolution_error': accordion_resolution_err,
        'linearity': accordion_linearity,
        'linearity_error': accordion_linearity_err,
        'containment_90_radius': accordion_r90,
        'containment_95_radius': accordion_r95,
        'material': 'CsI',
        'topology': 'accordion'
    }
}

# Create comprehensive performance comparison matrix
print('\n=== COMPREHENSIVE PERFORMANCE COMPARISON MATRIX ===')
print('Configuration\t\tResolution\t\tLinearity\t\tContainment 90%')
print('=' * 80)
for config, metrics in results.items():
    res = metrics['energy_resolution']
    res_err = metrics['energy_resolution_error']
    lin = metrics['linearity']
    lin_err = metrics['linearity_error']
    cont = metrics['containment_90_radius']
    print(f'{config:<20}\t{res:.4f}±{res_err:.4f}\t\t{lin:.3f}±{lin_err:.3f}\t\t{cont:.1f} mm')

# Find best configuration for each metric
best_resolution = min(results.items(), key=lambda x: x[1]['energy_resolution'])
best_linearity = min(results.items(), key=lambda x: abs(x[1]['linearity'] - 1.0))
best_containment = min(results.items(), key=lambda x: x[1]['containment_90_radius'])

print(f'\nBest Energy Resolution: {best_resolution[0]} ({best_resolution[1]["energy_resolution"]:.4f})')
print(f'Best Linearity: {best_linearity[0]} (deviation from 1.0: {abs(best_linearity[1]["linearity"] - 1.0):.3f})')
print(f'Best Containment: {best_containment[0]} ({best_containment[1]["containment_90_radius"]:.1f} mm)')

# Create comprehensive comparison plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

configs = list(results.keys())
colors = ['blue', 'green', 'red', 'orange']

# Energy Resolution comparison
resolutions = [results[c]['energy_resolution'] for c in configs]
res_errors = [results[c]['energy_resolution_error'] for c in configs]
ax1.bar(range(len(configs)), resolutions, yerr=res_errors, capsize=5, color=colors, alpha=0.7)
ax1.set_xlabel('Configuration')
ax1.set_ylabel('Energy Resolution (σ/E)')
ax1.set_title('Energy Resolution Comparison')
ax1.set_xticks(range(len(configs)))
ax1.set_xticklabels([c.split()[0] for c in configs], rotation=45)
ax1.grid(True, alpha=0.3)

# Linearity comparison (log scale due to large range)
linearities = [results[c]['linearity'] for c in configs]
lin_errors = [results[c]['linearity_error'] for c in configs]
ax2.bar(range(len(configs)), linearities, yerr=lin_errors, capsize=5, color=colors, alpha=0.7)
ax2.set_xlabel('Configuration')
ax2.set_ylabel('Linearity (Measured/Expected)')
ax2.set_title('Linearity Comparison')
ax2.set_xticks(range(len(configs)))
ax2.set_xticklabels([c.split()[0] for c in configs], rotation=45)
ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect Linearity')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Containment comparison
containments_90 = [results[c]['containment_90_radius'] for c in configs]
containments_95 = [results[c]['containment_95_radius'] for c in configs]
width = 0.35
x = np.arange(len(configs))
ax3.bar(x - width/2, containments_90, width, label='90% Containment', color=colors, alpha=0.7)
ax3.bar(x + width/2, containments_95, width, label='95% Containment', color=colors, alpha=0.5)
ax3.set_xlabel('Configuration')
ax3.set_ylabel('Containment Radius (mm)')
ax3.set_title('Lateral Containment Comparison')
ax3.set_xticks(x)
ax3.set_xticklabels([c.split()[0] for c in configs], rotation=45)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Performance ranking matrix
ranking_data = np.zeros((len(configs), 3))
for i, config in enumerate(configs):
    # Rank by resolution (lower is better)
    ranking_data[i, 0] = sorted(resolutions).index(resolutions[i]) + 1
    # Rank by linearity deviation from 1.0 (lower deviation is better)
    deviations = [abs(lin - 1.0) for lin in linearities]
    ranking_data[i, 1] = sorted(deviations).index(deviations[i]) + 1
    # Rank by containment (lower radius is better)
    ranking_data[i, 2] = sorted(containments_90).index(containments_90[i]) + 1

im = ax4.imshow(ranking_data, cmap='RdYlGn_r', aspect='auto')
ax4.set_xlabel('Performance Metric')
ax4.set_ylabel('Configuration')
ax4.set_title('Performance Ranking Matrix\n(1=Best, 4=Worst)')
ax4.set_xticks([0, 1, 2])
ax4.set_xticklabels(['Resolution', 'Linearity', 'Containment'])
ax4.set_yticks(range(len(configs)))
ax4.set_yticklabels([c.split()[0] for c in configs])

# Add ranking numbers to cells
for i in range(len(configs)):
    for j in range(3):
        ax4.text(j, i, f'{int(ranking_data[i, j])}', ha='center', va='center', fontweight='bold')

plt.colorbar(im, ax=ax4)
plt.tight_layout()
plt.savefig('comprehensive_performance_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('comprehensive_performance_comparison.pdf', bbox_inches='tight')

# Calculate overall performance score (weighted average of rankings)
weights = [0.4, 0.4, 0.2]  # Resolution and linearity more important than containment
overall_scores = {}
for i, config in enumerate(configs):
    score = sum(ranking_data[i, j] * weights[j] for j in range(3))
    overall_scores[config] = score

best_overall = min(overall_scores.items(), key=lambda x: x[1])
print(f'\nOverall Best Configuration: {best_overall[0]} (score: {best_overall[1]:.2f})')

# Save comprehensive results
final_results = {
    'performance_matrix': results,
    'best_resolution': best_resolution[0],
    'best_linearity': best_linearity[0],
    'best_containment': best_containment[0],
    'best_overall': best_overall[0],
    'overall_scores': overall_scores,
    'ranking_matrix': ranking_data.tolist()
}

with open('comprehensive_performance_matrix.json', 'w') as f:
    json.dump(final_results, f, indent=2)

print(f'RESULT:energy_resolution={accordion_resolution:.6f}')
print(f'RESULT:energy_resolution_error={accordion_resolution_err:.6f}')
print(f'RESULT:linearity={accordion_linearity:.6f}')
print(f'RESULT:linearity_error={accordion_linearity_err:.6f}')
print(f'RESULT:containment_90_radius={accordion_r90:.1f}')
print(f'RESULT:containment_95_radius={accordion_r95:.1f}')
print(f'RESULT:best_overall_configuration={best_overall[0]}')
print(f'RESULT:best_resolution_configuration={best_resolution[0]}')
print(f'RESULT:best_linearity_configuration={best_linearity[0]}')
print(f'RESULT:best_containment_configuration={best_containment[0]}')
print('RESULT:comparison_plot=comprehensive_performance_comparison.png')
print('RESULT:metrics_file=comprehensive_performance_matrix.json')
print('RESULT:success=True')