import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

# Extract results from previous analysis steps (from Previous Step Outputs)
baseline_efficiency = 1.0
baseline_efficiency_err = 0.0
baseline_uniformity = 0.7841
baseline_events = 100

endcap_efficiency = 1.0
endcap_efficiency_err = 0.0
endcap_uniformity = 3.2419
endcap_events = 100

barrel_efficiency = 0.0
barrel_efficiency_err = 0.0
barrel_uniformity = 0.0
barrel_events = 0

# Calculate error bars for uniformity (assuming Poisson statistics)
baseline_uniformity_err = baseline_uniformity / np.sqrt(baseline_events) if baseline_events > 0 else 0
endcap_uniformity_err = endcap_uniformity / np.sqrt(endcap_events) if endcap_events > 0 else 0
barrel_uniformity_err = 0  # No valid data

# Calculate cost effectiveness (inverse of uniformity - lower uniformity = better performance)
# Note: barrel configuration failed validation, so exclude from cost analysis
baseline_cost_eff = 1.0 / baseline_uniformity if baseline_uniformity > 0 else 0
endcap_cost_eff = 1.0 / endcap_uniformity if endcap_uniformity > 0 else 0
barrel_cost_eff = 0  # Failed configuration

baseline_cost_eff_err = baseline_cost_eff * (baseline_uniformity_err / baseline_uniformity) if baseline_uniformity > 0 else 0
endcap_cost_eff_err = endcap_cost_eff * (endcap_uniformity_err / endcap_uniformity) if endcap_uniformity > 0 else 0
barrel_cost_eff_err = 0

# Create comprehensive comparison plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
configs = ['Baseline\n(Uniform)', 'Endcap\n(Heavy)', 'Barrel\n(Optimized)']
colors = ['blue', 'green', 'red']

# Plot 1: Light Collection Efficiency
efficiencies = [baseline_efficiency, endcap_efficiency, barrel_efficiency]
eff_errors = [baseline_efficiency_err, endcap_efficiency_err, barrel_efficiency_err]
ax1.bar(configs, efficiencies, yerr=eff_errors, capsize=5, color=colors, alpha=0.7)
ax1.set_ylabel('Light Collection Efficiency')
ax1.set_title('Light Collection Efficiency Comparison')
ax1.set_ylim(0, 1.2)
for i, (eff, err) in enumerate(zip(efficiencies,eff_errors)):
    ax1.text(i, eff + err + 0.05, f'{eff:.3f}±{err:.3f}', ha='center', fontsize=10)

# Plot 2: Spatial Uniformity (lower is better)
uniformities = [baseline_uniformity, endcap_uniformity, barrel_uniformity]
unif_errors = [baseline_uniformity_err, endcap_uniformity_err, barrel_uniformity_err]
valid_configs = [i for i, u in enumerate(uniformities) if u > 0]
valid_uniformities = [uniformities[i] for i in valid_configs]
valid_errors = [unif_errors[i] for i in valid_configs]
valid_colors = [colors[i] for i in valid_configs]
valid_labels = [configs[i] for i in valid_configs]

ax2.bar(valid_labels, valid_uniformities, yerr=valid_errors, capsize=5, color=valid_colors, alpha=0.7)
ax2.set_ylabel('Spatial Uniformity (lower is better)')
ax2.set_title('Spatial Uniformity Comparison')
for i, (unif, err) in enumerate(zip(valid_uniformities, valid_errors)):
    ax2.text(i, unif + err + 0.1, f'{unif:.3f}±{err:.3f}', ha='center', fontsize=10)

# Plot 3: Cost Effectiveness (higher is better)
cost_effs = [baseline_cost_eff, endcap_cost_eff, barrel_cost_eff]
cost_eff_errors = [baseline_cost_eff_err, endcap_cost_eff_err, barrel_cost_eff_err]
valid_cost_configs = [i for i, c in enumerate(cost_effs) if c > 0]
valid_cost_effs = [cost_effs[i] for i in valid_cost_configs]
valid_cost_errors = [cost_eff_errors[i] for i in valid_cost_configs]
valid_cost_colors = [colors[i] for i in valid_cost_configs]
valid_cost_labels = [configs[i] for i in valid_cost_configs]

ax3.bar(valid_cost_labels, valid_cost_effs, yerr=valid_cost_errors, capsize=5, color=valid_cost_colors, alpha=0.7)
ax3.set_ylabel('Cost Effectiveness (1/uniformity)')
ax3.set_title('Cost Effectiveness Comparison')
for i, (cost, err) in enumerate(zip(valid_cost_effs, valid_cost_errors)):
    ax3.text(i, cost + err + 0.05, f'{cost:.3f}±{err:.3f}', ha='center', fontsize=10)

# Plot 4: Statistical significance testing
# Compare baseline vs endcap uniformity using t-test approximation
if baseline_uniformity > 0 and endcap_uniformity > 0:
    diff = abs(endcap_uniformity - baseline_uniformity)
    combined_err = np.sqrt(baseline_uniformity_err**2 + endcap_uniformity_err**2)
    t_stat = diff / combined_err if combined_err > 0 else 0
    significance = 'Significant' if t_stat > 2.0 else 'Not Significant'
    
    ax4.text(0.1, 0.8, 'Statistical Significance Test', fontsize=14, fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.1, 0.6, f'Baseline vs Endcap Uniformity:', fontsize=12, transform=ax4.transAxes)
    ax4.text(0.1, 0.5, f'Difference: {diff:.3f}', fontsize=11, transform=ax4.transAxes)
    ax4.text(0.1, 0.4, f'Combined Error: {combined_err:.3f}', fontsize=11, transform=ax4.transAxes)
    ax4.text(0.1, 0.3, f'T-statistic: {t_stat:.2f}', fontsize=11, transform=ax4.transAxes)
    ax4.text(0.1, 0.2, f'Result: {significance} (t > 2.0)', fontsize=11, transform=ax4.transAxes)
    ax4.text(0.1, 0.05, 'Note: Barrel configuration failed validation', fontsize=10, style='italic', transform=ax4.transAxes)
else:
    ax4.text(0.1, 0.5, 'Statistical analysis limited due to\nbarrel configuration failure', fontsize=12, transform=ax4.transAxes)

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')

plt.tight_layout()
plt.savefig('comprehensive_detector_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('comprehensive_detector_comparison.pdf', bbox_inches='tight')
plt.show()

# Summary statistics
print('=== COMPREHENSIVE DETECTOR COMPARISON ===')
print(f'Baseline Configuration:')
print(f'  Light Collection Efficiency: {baseline_efficiency:.3f} ± {baseline_efficiency_err:.3f}')
print(f'  Spatial Uniformity: {baseline_uniformity:.3f} ± {baseline_uniformity_err:.3f}')
print(f'  Cost Effectiveness: {baseline_cost_eff:.3f} ± {baseline_cost_eff_err:.3f}')
print(f'\nEndcap Heavy Configuration:')
print(f'  Light Collection Efficiency: {endcap_efficiency:.3f} ± {endcap_efficiency_err:.3f}')
print(f'  Spatial Uniformity: {endcap_uniformity:.3f} ± {endcap_uniformity_err:.3f}')
print(f'  Cost Effectiveness: {endcap_cost_eff:.3f} ± {endcap_cost_eff_err:.3f}')
print(f'\nBarrel Optimized Configuration:')
print(f'  Status: FAILED VALIDATION - No valid data')

# Determine optimal configuration
if baseline_cost_eff > endcap_cost_eff:
    optimal_config = 'Baseline (Uniform)'
    optimal_reason = f'Higher cost effectiveness ({baseline_cost_eff:.3f} vs {endcap_cost_eff:.3f})'
else:
    optimal_config = 'Endcap (Heavy)'
    optimal_reason = f'Higher cost effectiveness ({endcap_cost_eff:.3f} vs {baseline_cost_eff:.3f})'

print(f'\n=== RECOMMENDATION ===')
print(f'Optimal Configuration: {optimal_config}')
print(f'Reason: {optimal_reason}')

# Return values for workflow
print(f'RESULT:optimal_configuration={optimal_config}')
print(f'RESULT:baseline_efficiency={baseline_efficiency:.4f}')
print(f'RESULT:endcap_efficiency={endcap_efficiency:.4f}')
print(f'RESULT:barrel_efficiency={barrel_efficiency:.4f}')
print(f'RESULT:baseline_uniformity={baseline_uniformity:.4f}')
print(f'RESULT:endcap_uniformity={endcap_uniformity:.4f}')
print(f'RESULT:statistical_significance={significance if "significance" in locals() else "N/A"}')
print(f'RESULT:comparison_plots=comprehensive_detector_comparison.png')
print(f'RESULT:analysis_complete=true')

# Save detailed results to JSON
results = {
    'configurations': {
        'baseline': {
            'light_collection_efficiency': baseline_efficiency,
            'light_collection_efficiency_error': baseline_efficiency_err,
            'spatial_uniformity': baseline_uniformity,
            'spatial_uniformity_error': baseline_uniformity_err,
            'cost_effectiveness': baseline_cost_eff,
            'cost_effectiveness_error': baseline_cost_eff_err,
            'total_events': baseline_events
        },
        'endcap': {
            'light_collection_efficiency': endcap_efficiency,
            'light_collection_efficiency_error': endcap_efficiency_err,
            'spatial_uniformity': endcap_uniformity,
            'spatial_uniformity_error': endcap_uniformity_err,
            'cost_effectiveness': endcap_cost_eff,
            'cost_effectiveness_error': endcap_cost_eff_err,
            'total_events': endcap_events
        },
        'barrel': {
            'status': 'FAILED_VALIDATION',
            'light_collection_efficiency': barrel_efficiency,
            'spatial_uniformity': barrel_uniformity,
            'total_events': barrel_events
        }
    },
    'optimal_configuration': optimal_config,
    'statistical_analysis': {
        'significance_test': significance if 'significance' in locals() else 'N/A',
        't_statistic': t_stat if 't_stat' in locals() else 0
    }
}

with open('detector_comparison_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\nDetailed results saved to detector_comparison_results.json')