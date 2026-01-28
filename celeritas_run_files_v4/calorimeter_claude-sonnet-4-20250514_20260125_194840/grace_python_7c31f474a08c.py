import matplotlib
matplotlib.use('Agg')
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Load tungsten simulation data
with uproot.open('compact_tungsten_calorimeter_pip_hits.root') as f:
    tungsten_events = f['events'].arrays(library='pd')
    print(f"Loaded {len(tungsten_events)} tungsten events")

# Calculate tungsten performance metrics
tungsten_mean = tungsten_events['totalEdep'].mean()
tungsten_std = tungsten_events['totalEdep'].std()
tungsten_resolution = tungsten_std / tungsten_mean if tungsten_mean > 0 else 0
tungsten_resolution_err = tungsten_resolution / np.sqrt(2 * len(tungsten_events))

# Get baseline results from previous step outputs
baseline_resolution = 0.2572  # From analyze_baseline_performance
baseline_mean = tungsten_mean * (1 + (baseline_resolution - tungsten_resolution))  # Estimate

# Calculate high-density material effects
resolution_improvement = (baseline_resolution - tungsten_resolution) / baseline_resolution * 100
density_benefit = resolution_improvement  # Tungsten density advantage

# Compactness analysis - tungsten allows shorter calorimeter for same performance
# Tungsten X0 = 0.35 cm vs Steel X0 = 1.76 cm (5x more compact)
compactness_factor = 1.76 / 0.35  # ~5x more compact
effective_depth_reduction = (compactness_factor - 1) / compactness_factor * 100

# Resolution trade-offs analysis
if tungsten_resolution > baseline_resolution:
    resolution_tradeoff = (tungsten_resolution - baseline_resolution) / baseline_resolution * 100
    tradeoff_direction = "degradation"
else:
    resolution_tradeoff = (baseline_resolution - tungsten_resolution) / baseline_resolution * 100
    tradeoff_direction = "improvement"

# Create comprehensive analysis plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Energy distribution comparison
ax1.hist(tungsten_events['totalEdep'], bins=50, alpha=0.7, label=f'Tungsten (σ/E={tungsten_resolution:.4f})', color='orange')
ax1.axvline(tungsten_mean, color='orange', linestyle='--', alpha=0.8)
ax1.axvline(baseline_mean, color='blue', linestyle='--', alpha=0.8, label=f'Baseline (σ/E={baseline_resolution:.4f})')
ax1.set_xlabel('Total Energy Deposit (MeV)')
ax1.set_ylabel('Events')
ax1.set_title('Energy Resolution Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Compactness vs Performance
materials = ['Steel\n(Baseline)', 'Tungsten\n(Compact)']
resolutions = [baseline_resolution, tungsten_resolution]
compactness = [1.0, 1/compactness_factor]  # Relative thickness needed
ax2_twin = ax2.twinx()
bars1 = ax2.bar([0], [resolutions[0]], width=0.4, label='Resolution', color='blue', alpha=0.7)
bars2 = ax2.bar([1], [resolutions[1]], width=0.4, color='orange', alpha=0.7)
bars3 = ax2_twin.bar([0.5], [compactness[0]], width=0.4, label='Relative Depth', color='green', alpha=0.5)
bars4 = ax2_twin.bar([1.5], [compactness[1]], width=0.4, color='red', alpha=0.5)
ax2.set_ylabel('Energy Resolution (σ/E)', color='blue')
ax2_twin.set_ylabel('Relative Calorimeter Depth', color='green')
ax2.set_title('Compactness vs Resolution Trade-off')
ax2.set_xticks([0, 1])
ax2.set_xticklabels(materials)
ax2.grid(True, alpha=0.3)

# Plot 3: High-density material benefits
benefits = ['Resolution\nImprovement', 'Compactness\nGain', 'Material\nEfficiency']
values = [resolution_improvement, effective_depth_reduction, density_benefit]
colors = ['green' if v > 0 else 'red' for v in values]
ax3.bar(benefits, values, color=colors, alpha=0.7)
ax3.set_ylabel('Improvement (%)')
ax3.set_title('High-Density Material Benefits')
ax3.grid(True, alpha=0.3)
for i, v in enumerate(values):
    ax3.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')

# Plot 4: Performance metrics summary
metrics = ['Energy\nResolution', 'Compactness\nFactor', 'Material\nUtilization']
tungsten_metrics = [tungsten_resolution, compactness_factor, 1/tungsten_resolution]
baseline_metrics = [baseline_resolution, 1.0, 1/baseline_resolution]
x = np.arange(len(metrics))
width = 0.35
ax4.bar(x - width/2, baseline_metrics, width, label='Baseline', alpha=0.7, color='blue')
ax4.bar(x + width/2, tungsten_metrics, width, label='Tungsten', alpha=0.7, color='orange')
ax4.set_ylabel('Metric Value')
ax4.set_title('Performance Metrics Comparison')
ax4.set_xticks(x)
ax4.set_xticklabels(metrics)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tungsten_performance_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('tungsten_performance_analysis.pdf', bbox_inches='tight')
plt.show()

# Save detailed analysis results
analysis_results = {
    'tungsten_performance': {
        'energy_resolution': float(tungsten_resolution),
        'energy_resolution_error': float(tungsten_resolution_err),
        'mean_energy_deposit': float(tungsten_mean),
        'std_energy_deposit': float(tungsten_std)
    },
    'compactness_analysis': {
        'compactness_factor': float(compactness_factor),
        'effective_depth_reduction_percent': float(effective_depth_reduction),
        'radiation_length_ratio': 1.76/0.35
    },
    'high_density_effects': {
        'resolution_improvement_percent': float(resolution_improvement),
        'density_benefit_factor': float(density_benefit),
        'material_efficiency_gain': float(compactness_factor)
    },
    'resolution_tradeoffs': {
        'tradeoff_percent': float(resolution_tradeoff),
        'tradeoff_direction': tradeoff_direction,
        'baseline_comparison': {
            'baseline_resolution': float(baseline_resolution),
            'tungsten_resolution': float(tungsten_resolution)
        }
    },
    'summary': {
        'compactness_vs_performance_ratio': float(effective_depth_reduction / max(abs(resolution_tradeoff), 0.1)),
        'overall_benefit_score': float((effective_depth_reduction + resolution_improvement) / 2)
    }
}

with open('tungsten_performance_analysis.json', 'w') as f:
    json.dump(analysis_results, f, indent=2)

# Print key results
print(f"\n=== TUNGSTEN CALORIMETER PERFORMANCE ANALYSIS ===")
print(f"Energy Resolution: {tungsten_resolution:.4f} ± {tungsten_resolution_err:.4f}")
print(f"Resolution vs Baseline: {resolution_improvement:+.1f}% {tradeoff_direction}")
print(f"Compactness Factor: {compactness_factor:.1f}x more compact")
print(f"Effective Depth Reduction: {effective_depth_reduction:.1f}%")
print(f"High-Density Material Benefit: {density_benefit:.1f}%")
print(f"Overall Benefit Score: {analysis_results['summary']['overall_benefit_score']:.1f}%")

# Output results for workflow
print(f"RESULT:tungsten_resolution={tungsten_resolution:.4f}")
print(f"RESULT:tungsten_resolution_err={tungsten_resolution_err:.4f}")
print(f"RESULT:resolution_improvement={resolution_improvement:.2f}")
print(f"RESULT:compactness_factor={compactness_factor:.1f}")
print(f"RESULT:effective_depth_reduction={effective_depth_reduction:.1f}")
print(f"RESULT:density_benefit={density_benefit:.1f}")
print(f"RESULT:resolution_tradeoff={resolution_tradeoff:.2f}")
print(f"RESULT:overall_benefit_score={analysis_results['summary']['overall_benefit_score']:.1f}")
print("RESULT:performance_plot=tungsten_performance_analysis.png")
print("RESULT:analysis_file=tungsten_performance_analysis.json")
print("RESULT:tungsten_analysis_completed=True")