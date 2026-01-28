import matplotlib
matplotlib.use('Agg')
import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

# Load shashlik simulation data
with uproot.open('shashlik_bgo_calorimeter_electron_hits.root') as f:
    events = f['events'].arrays(library='pd')
    print(f'Loaded {len(events)} shashlik events')

# Calculate shashlik performance metrics
mean_E = events['totalEdep'].mean()
std_E = events['totalEdep'].std()
shashlik_resolution = std_E / mean_E if mean_E > 0 else 0
shashlik_resolution_err = shashlik_resolution / np.sqrt(2 * len(events)) if len(events) > 0 else 0

# Calculate linearity (ratio of measured to expected energy)
# Assuming 1 GeV input energy based on typical simulation setup
expected_energy = 1000.0  # MeV
shashlik_linearity = mean_E / expected_energy if expected_energy > 0 else 0
shashlik_linearity_err = std_E / expected_energy / np.sqrt(len(events)) if len(events) > 0 else 0

# Calculate containment (sample first 500k hits to avoid timeout)
with uproot.open('shashlik_bgo_calorimeter_electron_hits.root') as f:
    hits = f['hits'].arrays(['x', 'y', 'edep'], library='np', entry_stop=500000)

r = np.sqrt(hits['x']**2 + hits['y']**2)
total_edep = np.sum(hits['edep'])
radius_bins = np.linspace(0, 200, 100)
contained = np.array([np.sum(hits['edep'][r < r_cut]) for r_cut in radius_bins])
containment_fractions = contained / total_edep if total_edep > 0 else contained

# Find 90% and 95% containment radii
idx_90 = np.where(containment_fractions >= 0.9)[0]
idx_95 = np.where(containment_fractions >= 0.95)[0]
shashlik_r90 = radius_bins[idx_90[0]] if len(idx_90) > 0 else radius_bins[-1]
shashlik_r95 = radius_bins[idx_95[0]] if len(idx_95) > 0 else radius_bins[-1]

print(f'Shashlik Performance:')
print(f'Energy Resolution: {shashlik_resolution:.6f} ± {shashlik_resolution_err:.6f}')
print(f'Linearity: {shashlik_linearity:.6f} ± {shashlik_linearity_err:.6f}')
print(f'90% Containment: {shashlik_r90:.1f} mm')
print(f'95% Containment: {shashlik_r95:.1f} mm')

# Previous step results (from workflow state)
baseline_resolution = 0.034363
baseline_linearity = 0.467007
baseline_r90 = 40.82

projective_resolution = 0.021387
projective_linearity = 0.978699
projective_r90 = 100.0

# Calculate performance changes relative to baseline
resolution_change_vs_baseline = (shashlik_resolution - baseline_resolution) / baseline_resolution * 100
linearity_change_vs_baseline = (shashlik_linearity - baseline_linearity) / baseline_linearity * 100
containment_change_vs_baseline = (shashlik_r90 - baseline_r90) / baseline_r90 * 100

# Calculate performance changes relative to projective
resolution_change_vs_projective = (shashlik_resolution - projective_resolution) / projective_resolution * 100
linearity_change_vs_projective = (shashlik_linearity - projective_linearity) / projective_linearity * 100
containment_change_vs_projective = (shashlik_r90 - projective_r90) / projective_r90 * 100

print(f'\nComparison vs Baseline:')
print(f'Resolution change: {resolution_change_vs_baseline:.1f}%')
print(f'Linearity change: {linearity_change_vs_baseline:.1f}%')
print(f'Containment change: {containment_change_vs_baseline:.1f}%')

print(f'\nComparison vs Projective:')
print(f'Resolution change: {resolution_change_vs_projective:.1f}%')
print(f'Linearity change: {linearity_change_vs_projective:.1f}%')
print(f'Containment change: {containment_change_vs_projective:.1f}%')

# Create comprehensive comparison plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Energy resolution comparison
configs = ['Baseline\n(CsI)', 'Projective\n(PbWO4)', 'Shashlik\n(BGO)']
resolutions = [baseline_resolution, projective_resolution, shashlik_resolution]
colors = ['blue', 'green', 'red']
ax1.bar(configs, resolutions, color=colors, alpha=0.7)
ax1.set_ylabel('Energy Resolution (σ/E)')
ax1.set_title('Energy Resolution Comparison')
ax1.grid(True, alpha=0.3)

# Linearity comparison
linearities = [baseline_linearity, projective_linearity, shashlik_linearity]
ax2.bar(configs, linearities, color=colors, alpha=0.7)
ax2.set_ylabel('Linearity (Measured/Expected)')
ax2.set_title('Linearity Comparison')
ax2.grid(True, alpha=0.3)

# Containment comparison
containments = [baseline_r90, projective_r90, shashlik_r90]
ax3.bar(configs, containments, color=colors, alpha=0.7)
ax3.set_ylabel('90% Containment Radius (mm)')
ax3.set_title('Lateral Containment Comparison')
ax3.grid(True, alpha=0.3)

# Performance ranking (lower resolution = better, higher linearity = better, lower containment = better)
# Normalize metrics for ranking (0-1 scale)
res_norm = [(max(resolutions) - r) / (max(resolutions) - min(resolutions)) for r in resolutions]
lin_norm = [(l - min(linearities)) / (max(linearities) - min(linearities)) for l in linearities]
cont_norm = [(max(containments) - c) / (max(containments) - min(containments)) for c in containments]

# Overall score (equal weighting)
overall_scores = [(r + l + c) / 3 for r, l, c in zip(res_norm, lin_norm, cont_norm)]
ax4.bar(configs, overall_scores, color=colors, alpha=0.7)
ax4.set_ylabel('Overall Performance Score')
ax4.set_title('Overall Performance Ranking')
ax4.set_ylim(0, 1)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('shashlik_performance_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('shashlik_performance_comparison.pdf', bbox_inches='tight')
plt.show()

# Determine ranking
ranked_indices = np.argsort(overall_scores)[::-1]  # Descending order
ranking = [configs[i] for i in ranked_indices]
ranking_scores = [overall_scores[i] for i in ranked_indices]

print(f'\nPerformance Ranking (best to worst):')
for i, (config, score) in enumerate(zip(ranking, ranking_scores)):
    print(f'{i+1}. {config}: {score:.3f}')

# Save detailed metrics
metrics = {
    'shashlik_energy_resolution': float(shashlik_resolution),
    'shashlik_energy_resolution_error': float(shashlik_resolution_err),
    'shashlik_linearity': float(shashlik_linearity),
    'shashlik_linearity_error': float(shashlik_linearity_err),
    'shashlik_containment_90_radius': float(shashlik_r90),
    'shashlik_containment_95_radius': float(shashlik_r95),
    'resolution_change_vs_baseline_percent': float(resolution_change_vs_baseline),
    'linearity_change_vs_baseline_percent': float(linearity_change_vs_baseline),
    'containment_change_vs_baseline_percent': float(containment_change_vs_baseline),
    'resolution_change_vs_projective_percent': float(resolution_change_vs_projective),
    'linearity_change_vs_projective_percent': float(linearity_change_vs_projective),
    'containment_change_vs_projective_percent': float(containment_change_vs_projective),
    'performance_ranking': ranking,
    'ranking_scores': [float(s) for s in ranking_scores],
    'overall_scores': [float(s) for s in overall_scores]
}

with open('shashlik_performance_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Output results for downstream steps
print(f'RESULT:energy_resolution={shashlik_resolution:.6f}')
print(f'RESULT:energy_resolution_error={shashlik_resolution_err:.6f}')
print(f'RESULT:linearity={shashlik_linearity:.6f}')
print(f'RESULT:linearity_error={shashlik_linearity_err:.6f}')
print(f'RESULT:containment_90_radius={shashlik_r90:.1f}')
print(f'RESULT:containment_95_radius={shashlik_r95:.1f}')
print(f'RESULT:resolution_change_vs_baseline_percent={resolution_change_vs_baseline:.1f}')
print(f'RESULT:linearity_change_vs_baseline_percent={linearity_change_vs_baseline:.1f}')
print(f'RESULT:containment_change_vs_baseline_percent={containment_change_vs_baseline:.1f}')
print(f'RESULT:resolution_change_vs_projective_percent={resolution_change_vs_projective:.1f}')
print(f'RESULT:linearity_change_vs_projective_percent={linearity_change_vs_projective:.1f}')
print(f'RESULT:containment_change_vs_projective_percent={containment_change_vs_projective:.1f}')
print(f'RESULT:best_configuration={ranking[0]}')
print(f'RESULT:comparison_plot=shashlik_performance_comparison.png')
print(f'RESULT:metrics_file=shashlik_performance_metrics.json')
print('RESULT:success=True')