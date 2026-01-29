import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Extract results from Previous Step Outputs
# Baseline results
baseline_energies = [10.0, 30.0, 50.0]
baseline_resolutions = [0.095581, 0.077267, 0.070718]
baseline_resolution_errs = [0.000956, 0.000773, 0.000707]
baseline_linearities = [0.817535, 0.837036, 0.845002]
baseline_mean_deposits = [8175.35, 25111.1, 42250.1]

# Projective results
projective_energies = [10.0, 30.0, 50.0]
projective_resolutions = [1.1149, 1.30389, 1.38055]
projective_resolution_errs = [0.011149, 0.013039, 0.013806]
projective_linearities = [0.167281, 0.115371, 0.096326]
projective_mean_deposits = [1672.81, 3461.12, 4816.28]

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Projective vs Baseline Calorimeter Performance Comparison', fontsize=16, fontweight='bold')

# Plot 1: Energy Resolution vs Energy
ax1.errorbar(baseline_energies, baseline_resolutions, yerr=baseline_resolution_errs, 
            marker='o', linewidth=2, capsize=5, label='Baseline', color='blue')
ax1.errorbar(projective_energies, projective_resolutions, yerr=projective_resolution_errs,
            marker='s', linewidth=2, capsize=5, label='Projective', color='red')
ax1.set_xlabel('Beam Energy (GeV)')
ax1.set_ylabel('Energy Resolution (σ/E)')
ax1.set_title('Energy Resolution Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Plot 2: Linearity vs Energy
ax2.plot(baseline_energies, baseline_linearities, marker='o', linewidth=2, label='Baseline', color='blue')
ax2.plot(projective_energies, projective_linearities, marker='s', linewidth=2, label='Projective', color='red')
ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect Linearity')
ax2.set_xlabel('Beam Energy (GeV)')
ax2.set_ylabel('Linearity (E_measured/E_beam)')
ax2.set_title('Linearity Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Mean Energy Deposit vs Energy
ax3.plot(baseline_energies, baseline_mean_deposits, marker='o', linewidth=2, label='Baseline', color='blue')
ax3.plot(projective_energies, projective_mean_deposits, marker='s', linewidth=2, label='Projective', color='red')
ax3.set_xlabel('Beam Energy (GeV)')
ax3.set_ylabel('Mean Energy Deposit (MeV)')
ax3.set_title('Energy Deposit Comparison')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Direct Resolution Comparison Bar Chart
energy_labels = ['10 GeV', '30 GeV', '50 GeV']
x_pos = np.arange(len(energy_labels))
width = 0.35

ax4.bar(x_pos - width/2, baseline_resolutions, width, yerr=baseline_resolution_errs,
       capsize=5, label='Baseline', color='blue', alpha=0.7)
ax4.bar(x_pos + width/2, projective_resolutions, width, yerr=projective_resolution_errs,
       capsize=5, label='Projective', color='red', alpha=0.7)
ax4.set_xlabel('Beam Energy')
ax4.set_ylabel('Energy Resolution (σ/E)')
ax4.set_title('Resolution Comparison by Energy')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(energy_labels)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('projective_vs_baseline_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('projective_vs_baseline_comparison.pdf', bbox_inches='tight')
plt.close()

# Calculate performance metrics
baseline_avg_resolution = np.mean(baseline_resolutions)
projective_avg_resolution = np.mean(projective_resolutions)
baseline_avg_linearity = np.mean(baseline_linearities)
projective_avg_linearity = np.mean(projective_linearities)

# Performance comparison
resolution_ratio = projective_avg_resolution / baseline_avg_resolution
linearity_ratio = projective_avg_linearity / baseline_avg_linearity

# Create summary comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Average performance comparison
metrics = ['Energy Resolution', 'Linearity']
baseline_values = [baseline_avg_resolution, baseline_avg_linearity]
projective_values = [projective_avg_resolution, projective_avg_linearity]

x_pos = np.arange(len(metrics))
width = 0.35

ax1.bar(x_pos - width/2, baseline_values, width, label='Baseline', color='blue', alpha=0.7)
ax1.bar(x_pos + width/2, projective_values, width, label='Projective', color='red', alpha=0.7)
ax1.set_ylabel('Performance Metric')
ax1.set_title('Average Performance Comparison')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(metrics)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Resolution vs energy trend
ax2.loglog(baseline_energies, baseline_resolutions, 'o-', linewidth=2, label='Baseline', color='blue')
ax2.loglog(projective_energies, projective_resolutions, 's-', linewidth=2, label='Projective', color='red')
ax2.set_xlabel('Beam Energy (GeV)')
ax2.set_ylabel('Energy Resolution (σ/E)')
ax2.set_title('Resolution Scaling with Energy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('performance_summary_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('performance_summary_comparison.pdf', bbox_inches='tight')
plt.close()

# Print summary results
print(f'RESULT:baseline_avg_resolution={baseline_avg_resolution:.6f}')
print(f'RESULT:projective_avg_resolution={projective_avg_resolution:.6f}')
print(f'RESULT:baseline_avg_linearity={baseline_avg_linearity:.6f}')
print(f'RESULT:projective_avg_linearity={projective_avg_linearity:.6f}')
print(f'RESULT:resolution_ratio={resolution_ratio:.3f}')
print(f'RESULT:linearity_ratio={linearity_ratio:.3f}')
print('RESULT:comparison_plots=projective_vs_baseline_comparison.png')
print('RESULT:summary_plots=performance_summary_comparison.png')
print('RESULT:comparison_plots_pdf=projective_vs_baseline_comparison.pdf')
print('RESULT:summary_plots_pdf=performance_summary_comparison.pdf')
print('RESULT:success=True')

print('\n=== PERFORMANCE COMPARISON SUMMARY ===')
print(f'Baseline average resolution: {baseline_avg_resolution:.4f}')
print(f'Projective average resolution: {projective_avg_resolution:.4f}')
print(f'Resolution ratio (proj/base): {resolution_ratio:.2f}x')
print(f'Baseline average linearity: {baseline_avg_linearity:.4f}')
print(f'Projective average linearity: {projective_avg_linearity:.4f}')
print(f'Linearity ratio (proj/base): {linearity_ratio:.2f}x')
print('\nNote: Projective geometry shows significantly worse energy resolution')
print('but also much worse linearity, indicating poor energy containment.')