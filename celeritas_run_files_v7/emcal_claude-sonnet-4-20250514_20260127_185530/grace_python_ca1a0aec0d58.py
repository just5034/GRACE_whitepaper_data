import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

# Material performance data from Previous Step Outputs
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
fig.suptitle('Calorimeter Material Performance Comparison', fontsize=16, fontweight='bold')

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
ax1.set_ylim(0, 0.04)

# Plot 2: Linearity Comparison
ax2 = axes[0, 1]
ax2.plot(bgo_energies, bgo_linearities, marker='o', linewidth=2, label='BGO', color='blue')
ax2.plot(pbwo4_energies, pbwo4_linearities, marker='s', linewidth=2, label='PbWO4', color='red')
ax2.plot(csi_energies, csi_linearities, marker='^', linewidth=2, label='CsI', color='green')
ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect Linearity')
ax2.set_xlabel('Energy (GeV)')
ax2.set_ylabel('Linearity (Measured/Expected)')
ax2.set_title('Energy Linearity vs Energy')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.95, 0.97)

# Plot 3: Average Resolution Comparison (bar chart)
ax3 = axes[1, 0]
materials = ['BGO', 'PbWO4', 'CsI']
avg_resolutions = [np.mean(bgo_resolutions), np.mean(pbwo4_resolutions), np.mean(csi_resolutions)]
avg_resolution_errs = [np.sqrt(np.sum(np.array(bgo_resolution_errs)**2))/3, 
                       np.sqrt(np.sum(np.array(pbwo4_resolution_errs)**2))/3,
                       np.sqrt(np.sum(np.array(csi_resolution_errs)**2))/3]
colors = ['blue', 'red', 'green']
bars = ax3.bar(materials, avg_resolutions, yerr=avg_resolution_errs, 
               capsize=5, color=colors, alpha=0.7, edgecolor='black')
ax3.set_ylabel('Average Energy Resolution')
ax3.set_title('Average Energy Resolution Comparison')
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val, err in zip(bars, avg_resolutions, avg_resolution_errs):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.0005,
             f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Average Linearity Comparison (bar chart)
ax4 = axes[1, 1]
avg_linearities = [np.mean(bgo_linearities), np.mean(pbwo4_linearities), np.mean(csi_linearities)]
bars2 = ax4.bar(materials, avg_linearities, color=colors, alpha=0.7, edgecolor='black')
ax4.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect Linearity')
ax4.set_ylabel('Average Linearity')
ax4.set_title('Average Linearity Comparison')
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim(0.95, 0.97)

# Add value labels on bars
for bar, val in zip(bars2, avg_linearities):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
             f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('material_comparison_plots.png', dpi=150, bbox_inches='tight')
plt.savefig('material_comparison_plots.pdf', bbox_inches='tight')
plt.show()

# Create individual comparison plots
# Resolution comparison plot
fig2, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(bgo_energies, bgo_resolutions, yerr=bgo_resolution_errs, 
            marker='o', linewidth=3, capsize=5, label='BGO', color='blue', markersize=8)
ax.errorbar(pbwo4_energies, pbwo4_resolutions, yerr=pbwo4_resolution_errs, 
            marker='s', linewidth=3, capsize=5, label='PbWO4', color='red', markersize=8)
ax.errorbar(csi_energies, csi_resolutions, yerr=csi_resolution_errs, 
            marker='^', linewidth=3, capsize=5, label='CsI', color='green', markersize=8)
ax.set_xlabel('Energy (GeV)', fontsize=14)
ax.set_ylabel('Energy Resolution (σ/E)', fontsize=14)
ax.set_title('Energy Resolution Comparison', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('resolution_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('resolution_comparison.pdf', bbox_inches='tight')
plt.show()

# Linearity comparison plot
fig3, ax = plt.subplots(figsize=(10, 6))
ax.plot(bgo_energies, bgo_linearities, marker='o', linewidth=3, label='BGO', color='blue', markersize=8)
ax.plot(pbwo4_energies, pbwo4_linearities, marker='s', linewidth=3, label='PbWO4', color='red', markersize=8)
ax.plot(csi_energies, csi_linearities, marker='^', linewidth=3, label='CsI', color='green', markersize=8)
ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect Linearity')
ax.set_xlabel('Energy (GeV)', fontsize=14)
ax.set_ylabel('Linearity (Measured/Expected)', fontsize=14)
ax.set_title('Energy Linearity Comparison', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('linearity_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('linearity_comparison.pdf', bbox_inches='tight')
plt.show()

# Summary statistics
print('Material Performance Summary:')
print('='*50)
print(f'BGO - Avg Resolution: {np.mean(bgo_resolutions):.4f} ± {np.std(bgo_resolutions):.4f}')
print(f'BGO - Avg Linearity: {np.mean(bgo_linearities):.4f} ± {np.std(bgo_linearities):.4f}')
print(f'PbWO4 - Avg Resolution: {np.mean(pbwo4_resolutions):.4f} ± {np.std(pbwo4_resolutions):.4f}')
print(f'PbWO4 - Avg Linearity: {np.mean(pbwo4_linearities):.4f} ± {np.std(pbwo4_linearities):.4f}')
print(f'CsI - Avg Resolution: {np.mean(csi_resolutions):.4f} ± {np.std(csi_resolutions):.4f}')
print(f'CsI - Avg Linearity: {np.mean(csi_linearities):.4f} ± {np.std(csi_linearities):.4f}')

# Save results
comparison_results = {
    'materials': ['BGO', 'PbWO4', 'CsI'],
    'energies_gev': [0.5, 2.0, 5.0],
    'bgo': {
        'resolutions': bgo_resolutions,
        'resolution_errors': bgo_resolution_errs,
        'linearities': bgo_linearities,
        'avg_resolution': float(np.mean(bgo_resolutions)),
        'avg_linearity': float(np.mean(bgo_linearities))
    },
    'pbwo4': {
        'resolutions': pbwo4_resolutions,
        'resolution_errors': pbwo4_resolution_errs,
        'linearities': pbwo4_linearities,
        'avg_resolution': float(np.mean(pbwo4_resolutions)),
        'avg_linearity': float(np.mean(pbwo4_linearities))
    },
    'csi': {
        'resolutions': csi_resolutions,
        'resolution_errors': csi_resolution_errs,
        'linearities': csi_linearities,
        'avg_resolution': float(np.mean(csi_resolutions)),
        'avg_linearity': float(np.mean(csi_linearities))
    }
}

with open('material_comparison_plots.json', 'w') as f:
    json.dump(comparison_results, f, indent=2)

print('RESULT:comparison_plots=material_comparison_plots.png')
print('RESULT:resolution_comparison_plot=resolution_comparison.png')
print('RESULT:linearity_comparison_plot=linearity_comparison.png')
print('RESULT:results_json=material_comparison_plots.json')
print('RESULT:success=True')