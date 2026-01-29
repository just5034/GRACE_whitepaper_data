import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

# Performance metrics from Previous Step Outputs
# Baseline Configuration
baseline_resolution_10gev = 0.095581
baseline_resolution_30gev = 0.077267
baseline_resolution_50gev = 0.070718
baseline_mean_resolution = 0.081189
baseline_linearity_10gev = 0.817535
baseline_linearity_30gev = 0.837036
baseline_linearity_50gev = 0.845002
baseline_mean_linearity = 0.833191
baseline_total_depth_m = 1.67
baseline_lateral_size_m = 0.504
baseline_absorber_material = 'G4_Fe'

# Projective Configuration
projective_resolution_10gev = 1.1149
projective_resolution_30gev = 1.30389
projective_resolution_50gev = 1.38055
projective_mean_resolution = 1.26645
projective_linearity_10gev = 0.167281
projective_linearity_30gev = 0.115371
projective_linearity_50gev = 0.096326
projective_mean_linearity = 0.126326
projective_total_depth_m = 0.1675
projective_inner_radius_m = 0.252
projective_outer_radius_m = 0.4195
projective_absorber_material = 'G4_Fe'

# Tungsten Configuration
tungsten_resolution_10gev = 0.610701
tungsten_resolution_30gev = 0.68318
tungsten_resolution_50gev = 0.727988
tungsten_mean_resolution = 0.673957
tungsten_linearity_10gev = 0.48054
tungsten_linearity_30gev = 0.432419
tungsten_linearity_50gev = 0.407177
tungsten_mean_linearity = 0.439712
tungsten_total_depth_m = 0.23175
tungsten_inner_radius_m = 0.05
tungsten_outer_radius_m = 0.28175
tungsten_absorber_material = 'G4_W'

# Calculate spatial compactness metrics
baseline_volume = np.pi * (baseline_lateral_size_m/2)**2 * baseline_total_depth_m
projective_volume = np.pi * (projective_outer_radius_m**2 - projective_inner_radius_m**2) * 2 * projective_total_depth_m
tungsten_volume = np.pi * (tungsten_outer_radius_m**2 - tungsten_inner_radius_m**2) * 2 * tungsten_total_depth_m

# Relative cost factors (tungsten >> steel/iron)
cost_factors = {'G4_Fe': 1.0, 'G4_W': 15.0}  # Tungsten ~15x more expensive than iron
baseline_cost_factor = cost_factors[baseline_absorber_material] * baseline_volume
projective_cost_factor = cost_factors[projective_absorber_material] * projective_volume
tungsten_cost_factor = cost_factors[tungsten_absorber_material] * tungsten_volume

# Normalize metrics for comparison (lower is better for resolution, higher for linearity)
# Energy resolution (lower is better)
resolutions = [baseline_mean_resolution, projective_mean_resolution, tungsten_mean_resolution]
min_res, max_res = min(resolutions), max(resolutions)
resolution_scores = [(max_res - r) / (max_res - min_res) for r in resolutions]

# Linearity (higher is better)
linearities = [baseline_mean_linearity, projective_mean_linearity, tungsten_mean_linearity]
min_lin, max_lin = min(linearities), max(linearities)
linearity_scores = [(l - min_lin) / (max_lin - min_lin) for l in linearities]

# Spatial compactness (smaller volume is better)
volumes = [baseline_volume, projective_volume, tungsten_volume]
min_vol, max_vol = min(volumes), max(volumes)
compactness_scores = [(max_vol - v) / (max_vol - min_vol) for v in volumes]

# Cost effectiveness (lower cost is better)
costs = [baseline_cost_factor, projective_cost_factor, tungsten_cost_factor]
min_cost, max_cost = min(costs), max(costs)
cost_scores = [(max_cost - c) / (max_cost - min_cost) for c in costs]

# Weighted scoring (weights sum to 1.0)
weights = {'energy_resolution': 0.4, 'linearity': 0.2, 'compactness': 0.2, 'cost': 0.2}

# Calculate total scores
configs = ['Baseline (Fe)', 'Projective (Fe)', 'Tungsten (W)']
total_scores = []
for i in range(3):
    score = (weights['energy_resolution'] * resolution_scores[i] + 
             weights['linearity'] * linearity_scores[i] + 
             weights['compactness'] * compactness_scores[i] + 
             weights['cost'] * cost_scores[i])
    total_scores.append(score)

# Find optimal configuration
optimal_idx = np.argmax(total_scores)
optimal_config = configs[optimal_idx]

print(f'OPTIMIZATION RESULTS:')
print(f'===================')
for i, config in enumerate(configs):
    print(f'{config}:')
    print(f'  Energy Resolution: {resolutions[i]:.4f} (score: {resolution_scores[i]:.3f})')
    print(f'  Linearity: {linearities[i]:.4f} (score: {linearity_scores[i]:.3f})')
    print(f'  Volume: {volumes[i]:.3f} m³ (score: {compactness_scores[i]:.3f})')
    print(f'  Cost Factor: {costs[i]:.1f} (score: {cost_scores[i]:.3f})')
    print(f'  TOTAL SCORE: {total_scores[i]:.3f}')
    print()

print(f'OPTIMAL CONFIGURATION: {optimal_config}')
print(f'Total Score: {total_scores[optimal_idx]:.3f}')

# Create comprehensive comparison plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Energy resolution comparison
ax1.bar(configs, resolutions, color=['blue', 'orange', 'green'], alpha=0.7)
ax1.set_ylabel('Energy Resolution (σ/E)')
ax1.set_title('Energy Resolution Comparison')
ax1.tick_params(axis='x', rotation=45)
for i, v in enumerate(resolutions):
    ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

# Linearity comparison
ax2.bar(configs, linearities, color=['blue', 'orange', 'green'], alpha=0.7)
ax2.set_ylabel('Linearity (Response/Energy)')
ax2.set_title('Energy Linearity Comparison')
ax2.tick_params(axis='x', rotation=45)
for i, v in enumerate(linearities):
    ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

# Volume comparison
ax3.bar(configs, volumes, color=['blue', 'orange', 'green'], alpha=0.7)
ax3.set_ylabel('Detector Volume (m³)')
ax3.set_title('Spatial Compactness Comparison')
ax3.tick_params(axis='x', rotation=45)
for i, v in enumerate(volumes):
    ax3.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom')

# Overall scores
colors = ['red' if i == optimal_idx else 'lightblue' for i in range(3)]
ax4.bar(configs, total_scores, color=colors, alpha=0.8)
ax4.set_ylabel('Weighted Total Score')
ax4.set_title('Overall Performance Score')
ax4.tick_params(axis='x', rotation=45)
for i, v in enumerate(total_scores):
    ax4.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', 
             weight='bold' if i == optimal_idx else 'normal')

plt.tight_layout()
plt.savefig('design_optimization_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('design_optimization_comparison.pdf', bbox_inches='tight')
plt.show()

# Save detailed results to JSON
results = {
    'optimization_criteria': {
        'energy_resolution_weight': weights['energy_resolution'],
        'linearity_weight': weights['linearity'], 
        'compactness_weight': weights['compactness'],
        'cost_weight': weights['cost']
    },
    'configurations': {
        'baseline': {
            'energy_resolution': baseline_mean_resolution,
            'linearity': baseline_mean_linearity,
            'volume_m3': baseline_volume,
            'cost_factor': baseline_cost_factor,
            'total_score': total_scores[0]
        },
        'projective': {
            'energy_resolution': projective_mean_resolution,
            'linearity': projective_mean_linearity,
            'volume_m3': projective_volume,
            'cost_factor': projective_cost_factor,
            'total_score': total_scores[1]
        },
        'tungsten': {
            'energy_resolution': tungsten_mean_resolution,
            'linearity': tungsten_mean_linearity,
            'volume_m3': tungsten_volume,
            'cost_factor': tungsten_cost_factor,
            'total_score': total_scores[2]
        }
    },
    'optimal_configuration': optimal_config,
    'optimal_score': total_scores[optimal_idx],
    'justification': f'Selected based on weighted multi-criteria analysis considering energy resolution (40%), linearity (20%), compactness (20%), and cost (20%)'
}

with open('design_optimization_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print final recommendation
print(f'\nFINAL RECOMMENDATION:')
print(f'=====================')
print(f'Optimal Design: {optimal_config}')
print(f'Justification: Highest weighted score ({total_scores[optimal_idx]:.3f}) based on:')
print(f'- Energy Resolution: {resolutions[optimal_idx]:.4f} (weight: {weights["energy_resolution"]})')
print(f'- Energy Linearity: {linearities[optimal_idx]:.4f} (weight: {weights["linearity"]})')
print(f'- Spatial Compactness: {volumes[optimal_idx]:.2f} m³ (weight: {weights["compactness"]})')
print(f'- Cost Effectiveness: {costs[optimal_idx]:.1f} relative cost (weight: {weights["cost"]})')

# Output results for downstream steps
print(f'RESULT:optimal_configuration={optimal_config}')
print(f'RESULT:optimal_score={total_scores[optimal_idx]:.4f}')
print(f'RESULT:baseline_score={total_scores[0]:.4f}')
print(f'RESULT:projective_score={total_scores[1]:.4f}')
print(f'RESULT:tungsten_score={total_scores[2]:.4f}')
print(f'RESULT:optimization_plot=design_optimization_comparison.png')
print(f'RESULT:results_json=design_optimization_results.json')
print('RESULT:success=True')