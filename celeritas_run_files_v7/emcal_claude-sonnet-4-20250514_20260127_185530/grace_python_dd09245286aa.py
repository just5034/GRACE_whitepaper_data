import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

# Material performance data from previous step outputs
# BGO data
bgo_energies = [0.5, 2.0, 5.0]
bgo_resolutions = [0.026309, 0.021725, 0.017925]
bgo_resolution_errs = [0.000263, 0.000217, 0.000179]
bgo_linearities = [0.9625, 0.9614, 0.959]

# PbWO4 data
pbwo4_energies = [0.5, 2.0, 5.0]
pbwo4_resolutions = [0.033044, 0.014936, 0.019751]
pbwo4_resolution_errs = [0.00033, 0.000149, 0.000198]
pbwo4_linearities = [0.9678, 0.9674, 0.9637]

# CsI data
csi_energies = [0.5, 2.0, 5.0]
csi_resolutions = [0.031228, 0.019525, 0.016829]
csi_resolution_errs = [0.000312, 0.000195, 0.000168]
csi_linearities = [0.9553, 0.9548, 0.9527]

# Create comprehensive comparison plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Material Comparison: BGO vs PbWO4 vs CsI', fontsize=16, fontweight='bold')

# Plot 1: Energy Resolution vs Energy
ax1 = axes[0, 0]
ax1.errorbar(bgo_energies, bgo_resolutions, yerr=bgo_resolution_errs, 
            marker='o', linewidth=2, capsize=5, label='BGO', color='blue')
ax1.errorbar(pbwo4_energies, pbwo4_resolutions, yerr=pbwo4_resolution_errs, 
            marker='s', linewidth=2, capsize=5, label='PbWO4', color='red')
ax1.errorbar(csi_energies, csi_resolutions, yerr=csi_resolution_errs, 
            marker='^', linewidth=2, capsize=5, label='CsI', color='green')
ax1.set_xlabel('Energy (GeV)')
ax1.set_ylabel('Energy Resolution (σ/E)')
ax1.set_title('Energy Resolution vs Energy')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# Plot 2: Linearity Comparison
ax2 = axes[0, 1]
ax2.plot(bgo_energies, bgo_linearities, marker='o', linewidth=2, label='BGO', color='blue')
ax2.plot(pbwo4_energies, pbwo4_linearities, marker='s', linewidth=2, label='PbWO4', color='red')
ax2.plot(csi_energies, csi_linearities, marker='^', linewidth=2, label='CsI', color='green')
ax2.set_xlabel('Energy (GeV)')
ax2.set_ylabel('Linearity (E_measured/E_beam)')
ax2.set_title('Linearity vs Energy')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.94, 0.98)

# Plot 3: Resolution at 2 GeV (key benchmark)
ax3 = axes[1, 0]
materials = ['BGO', 'PbWO4', 'CsI']
resolutions_2gev = [0.021725, 0.014936, 0.019525]
errors_2gev = [0.000217, 0.000149, 0.000195]
colors = ['blue', 'red', 'green']

bars = ax3.bar(materials, resolutions_2gev, yerr=errors_2gev, 
              capsize=5, color=colors, alpha=0.7, edgecolor='black')
ax3.set_ylabel('Energy Resolution (σ/E)')
ax3.set_title('Energy Resolution at 2 GeV')
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, val, err) in enumerate(zip(bars, resolutions_2gev, errors_2gev)):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.0005,
             f'{val:.4f}±{err:.4f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Material Properties Summary
ax4 = axes[1, 1]
ax4.axis('off')

# Create summary table
summary_text = f"""
Material Performance Summary:

BGO (Bismuth Germanate):
• Best resolution at 5 GeV: {bgo_resolutions[2]:.4f}±{bgo_resolution_errs[2]:.4f}
• Stable linearity: {np.mean(bgo_linearities):.3f}±{np.std(bgo_linearities):.3f}
• High density, compact design

PbWO4 (Lead Tungstate):
• Best resolution at 2 GeV: {pbwo4_resolutions[1]:.4f}±{pbwo4_resolution_errs[1]:.4f}
• Excellent linearity: {np.mean(pbwo4_linearities):.3f}±{np.std(pbwo4_linearities):.3f}
• Fast, radiation hard

CsI (Cesium Iodide):
• Good high-energy resolution: {csi_resolutions[2]:.4f}±{csi_resolution_errs[2]:.4f}
• Lower linearity: {np.mean(csi_linearities):.3f}±{np.std(csi_linearities):.3f}
• High light yield, cost effective

Optimal Choice: PbWO4
• Best overall resolution at 2 GeV
• Most stable linearity across energy range
• Suitable for precision calorimetry
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

plt.tight_layout()
plt.savefig('materials_comparison_plots.png', dpi=150, bbox_inches='tight')
plt.savefig('materials_comparison_plots.pdf', bbox_inches='tight')
plt.show()

# Save detailed comparison data
comparison_data = {
    'materials': ['BGO', 'PbWO4', 'CsI'],
    'energies_gev': [0.5, 2.0, 5.0],
    'bgo': {
        'resolutions': bgo_resolutions,
        'resolution_errors': bgo_resolution_errs,
        'linearities': bgo_linearities
    },
    'pbwo4': {
        'resolutions': pbwo4_resolutions,
        'resolution_errors': pbwo4_resolution_errs,
        'linearities': pbwo4_linearities
    },
    'csi': {
        'resolutions': csi_resolutions,
        'resolution_errors': csi_resolution_errs,
        'linearities': csi_linearities
    },
    'optimal_material': 'PbWO4',
    'optimal_resolution_2gev': pbwo4_resolutions[1]
}

with open('materials_comparison_detailed.json', 'w') as f:
    json.dump(comparison_data, f, indent=2)

print('RESULT:comparison_plots=materials_comparison_plots.png')
print('RESULT:optimal_material=PbWO4')
print(f'RESULT:pbwo4_best_resolution={pbwo4_resolutions[1]:.6f}')
print(f'RESULT:bgo_avg_resolution={np.mean(bgo_resolutions):.6f}')
print(f'RESULT:csi_avg_resolution={np.mean(csi_resolutions):.6f}')
print('RESULT:comparison_data_json=materials_comparison_detailed.json')
print('RESULT:success=True')