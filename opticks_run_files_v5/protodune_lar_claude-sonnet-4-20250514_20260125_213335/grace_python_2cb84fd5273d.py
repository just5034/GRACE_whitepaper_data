import matplotlib
matplotlib.use('Agg')
import uproot
import numpy as np
import matplotlib.pyplot as plt
import json

# Load optimized detector simulation data
with uproot.open('optimized_protodune_electron_hits.root') as f:
    opt_events = f['events'].arrays(library='pd')
    print(f"Loaded {len(opt_events)} optimized events")

# Calculate optimized performance metrics
opt_mean_energy = opt_events['totalEdep'].mean()
opt_std_energy = opt_events['totalEdep'].std()
opt_energy_resolution = opt_std_energy / opt_mean_energy if opt_mean_energy > 0 else 0
opt_energy_resolution_err = opt_energy_resolution / np.sqrt(2 * len(opt_events)) if len(opt_events) > 0 else 0

# Detection efficiency (fraction of events with hits)
opt_detection_efficiency = len(opt_events[opt_events['nHits'] > 0]) / len(opt_events) if len(opt_events) > 0 else 0
opt_detection_efficiency_err = np.sqrt(opt_detection_efficiency * (1 - opt_detection_efficiency) / len(opt_events)) if len(opt_events) > 0 else 0

# Light yield (hits per MeV)
opt_light_yield = opt_events['nHits'].mean() / opt_mean_energy if opt_mean_energy > 0 else 0

# Baseline metrics from previous step outputs
baseline_energy_resolution = 3.1729
baseline_energy_resolution_err = 0.3173
baseline_detection_efficiency = 0.1
baseline_detection_efficiency_err = 0.0424
baseline_light_yield = 0.7
baseline_mean_energy = 0.017

# Calculate improvements
resolution_improvement = (baseline_energy_resolution - opt_energy_resolution) / baseline_energy_resolution * 100
efficiency_improvement = (opt_detection_efficiency - baseline_detection_efficiency) / baseline_detection_efficiency * 100
light_yield_improvement = (opt_light_yield - baseline_light_yield) / baseline_light_yield * 100 if baseline_light_yield > 0 else 0

# Statistical significance of improvements
resolution_significance = abs(baseline_energy_resolution - opt_energy_resolution) / np.sqrt(baseline_energy_resolution_err**2 + opt_energy_resolution_err**2)
efficiency_significance = abs(baseline_detection_efficiency - opt_detection_efficiency) / np.sqrt(baseline_detection_efficiency_err**2 + opt_detection_efficiency_err**2)

print(f"\n=== OPTIMIZED PERFORMANCE METRICS ===")
print(f"Energy Resolution: {opt_energy_resolution:.4f} ± {opt_energy_resolution_err:.4f}")
print(f"Detection Efficiency: {opt_detection_efficiency:.4f} ± {opt_detection_efficiency_err:.4f}")
print(f"Light Yield: {opt_light_yield:.2f} hits/MeV")
print(f"Mean Energy Deposit: {opt_mean_energy:.4f} MeV")

print(f"\n=== QUANTITATIVE IMPROVEMENTS ===")
print(f"Energy Resolution: {resolution_improvement:+.1f}% improvement")
print(f"Detection Efficiency: {efficiency_improvement:+.1f}% improvement")
print(f"Light Yield: {light_yield_improvement:+.1f}% improvement")
print(f"Resolution Significance: {resolution_significance:.1f} sigma")
print(f"Efficiency Significance: {efficiency_significance:.1f} sigma")

# Create comparison plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Energy resolution comparison
configs = ['Baseline', 'Optimized']
resolutions = [baseline_energy_resolution, opt_energy_resolution]
res_errors = [baseline_energy_resolution_err, opt_energy_resolution_err]
ax1.bar(configs, resolutions, yerr=res_errors, capsize=5, color=['red', 'green'], alpha=0.7)
ax1.set_ylabel('Energy Resolution (σ/E)')
ax1.set_title(f'Energy Resolution\n({resolution_improvement:+.1f}% improvement)')
ax1.grid(True, alpha=0.3)

# Detection efficiency comparison
efficiencies = [baseline_detection_efficiency, opt_detection_efficiency]
eff_errors = [baseline_detection_efficiency_err, opt_detection_efficiency_err]
ax2.bar(configs, efficiencies, yerr=eff_errors, capsize=5, color=['red', 'green'], alpha=0.7)
ax2.set_ylabel('Detection Efficiency')
ax2.set_title(f'Detection Efficiency\n({efficiency_improvement:+.1f}% improvement)')
ax2.grid(True, alpha=0.3)

# Light yield comparison
light_yields = [baseline_light_yield, opt_light_yield]
ax3.bar(configs, light_yields, color=['red', 'green'], alpha=0.7)
ax3.set_ylabel('Light Yield (hits/MeV)')
ax3.set_title(f'Light Yield\n({light_yield_improvement:+.1f}% improvement)')
ax3.grid(True, alpha=0.3)

# Energy distribution comparison
ax4.hist(opt_events['totalEdep'], bins=30, alpha=0.7, label='Optimized', color='green', density=True)
ax4.axvline(baseline_mean_energy, color='red', linestyle='--', linewidth=2, label=f'Baseline Mean: {baseline_mean_energy:.3f} MeV')
ax4.axvline(opt_mean_energy, color='green', linestyle='--', linewidth=2, label=f'Optimized Mean: {opt_mean_energy:.3f} MeV')
ax4.set_xlabel('Total Energy Deposit (MeV)')
ax4.set_ylabel('Normalized Events')
ax4.set_title('Energy Distribution Comparison')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimized_performance_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('optimized_performance_comparison.pdf', bbox_inches='tight')

# Save detailed results to JSON
results = {
    'optimized_metrics': {
        'energy_resolution': float(opt_energy_resolution),
        'energy_resolution_err': float(opt_energy_resolution_err),
        'detection_efficiency': float(opt_detection_efficiency),
        'detection_efficiency_err': float(opt_detection_efficiency_err),
        'light_yield_pe_per_mev': float(opt_light_yield),
        'mean_energy_deposit': float(opt_mean_energy)
    },
    'improvements': {
        'energy_resolution_improvement_percent': float(resolution_improvement),
        'detection_efficiency_improvement_percent': float(efficiency_improvement),
        'light_yield_improvement_percent': float(light_yield_improvement)
    },
    'statistical_significance': {
        'resolution_significance_sigma': float(resolution_significance),
        'efficiency_significance_sigma': float(efficiency_significance)
    },
    'baseline_comparison': {
        'baseline_energy_resolution': baseline_energy_resolution,
        'baseline_detection_efficiency': baseline_detection_efficiency,
        'baseline_light_yield': baseline_light_yield
    }
}

with open('optimized_performance_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

# Output results for downstream steps
print(f"RESULT:optimized_energy_resolution={opt_energy_resolution:.4f}")
print(f"RESULT:optimized_energy_resolution_err={opt_energy_resolution_err:.4f}")
print(f"RESULT:optimized_detection_efficiency={opt_detection_efficiency:.4f}")
print(f"RESULT:optimized_detection_efficiency_err={opt_detection_efficiency_err:.4f}")
print(f"RESULT:optimized_light_yield_pe_per_mev={opt_light_yield:.2f}")
print(f"RESULT:optimized_mean_energy_deposit={opt_mean_energy:.4f}")
print(f"RESULT:energy_resolution_improvement={resolution_improvement:.2f}")
print(f"RESULT:detection_efficiency_improvement={efficiency_improvement:.2f}")
print(f"RESULT:light_yield_improvement={light_yield_improvement:.2f}")
print(f"RESULT:resolution_significance={resolution_significance:.2f}")
print(f"RESULT:efficiency_significance={efficiency_significance:.2f}")
print("RESULT:comparison_plot=optimized_performance_comparison.png")
print("RESULT:analysis_file=optimized_performance_analysis.json")