import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import json

# Material performance data from Previous Step Outputs
# BGO results
bgo_resolutions = [0.026309, 0.021725, 0.017925]  # 0.5, 2.0, 5.0 GeV
bgo_energies = [0.5, 2.0, 5.0]
bgo_linearity = [0.9625, 0.9614, 0.959]
bgo_stochastic = 0.82  # percent
bgo_constant = 1.496  # percent

# PbWO4 results
pbwo4_resolutions = [0.033044, 0.014936, 0.019751]  # 0.5, 2.0, 5.0 GeV
pbwo4_energies = [0.5, 2.0, 5.0]
pbwo4_linearity = [0.9678, 0.9674, 0.9637]
pbwo4_stochastic = 1.62  # percent
pbwo4_constant = 0.87  # percent

# CsI results
csi_resolutions = [0.031228, 0.019525, 0.016829]  # 0.5, 2.0, 5.0 GeV
csi_energies = [0.5, 2.0, 5.0]
csi_linearity = [0.9553, 0.9548, 0.9527]
csi_stochastic = 1.48  # percent
csi_constant = 0.98  # percent

# Target resolution: 1-3% / sqrt(E)
target_min = 0.01  # 1%
target_max = 0.03  # 3%

# Calculate target resolution at each energy
energy_points = np.array([0.5, 1.0, 2.0, 5.0])
target_res_min = target_min / np.sqrt(energy_points)
target_res_max = target_max / np.sqrt(energy_points)

# Create comprehensive comparison plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Energy Resolution vs Energy
ax1.plot(bgo_energies, bgo_resolutions, 'bo-', label='BGO', linewidth=2, markersize=8)
ax1.plot(pbwo4_energies, pbwo4_resolutions, 'rs-', label='PbWO4', linewidth=2, markersize=8)
ax1.plot(csi_energies, csi_resolutions, 'g^-', label='CsI', linewidth=2, markersize=8)
ax1.fill_between(energy_points, target_res_min, target_res_max, alpha=0.2, color='gray', label='Target Range (1-3%/√E)')
ax1.set_xlabel('Energy (GeV)')
ax1.set_ylabel('Energy Resolution (σ/E)')
ax1.set_title('Energy Resolution Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0.4, 5.2)

# Plot 2: Linearity Comparison
ax2.plot(bgo_energies, bgo_linearity, 'bo-', label='BGO', linewidth=2, markersize=8)
ax2.plot(pbwo4_energies, pbwo4_linearity, 'rs-', label='PbWO4', linewidth=2, markersize=8)
ax2.plot(csi_energies, csi_linearity, 'g^-', label='CsI', linewidth=2, markersize=8)
ax2.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Perfect Linearity')
ax2.set_xlabel('Energy (GeV)')
ax2.set_ylabel('Linearity (E_measured/E_beam)')
ax2.set_title('Linearity Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.95, 0.97)

# Plot 3: Stochastic vs Constant Terms
materials = ['BGO', 'PbWO4', 'CsI']
stochastic_terms = [bgo_stochastic, pbwo4_stochastic, csi_stochastic]
constant_terms = [bgo_constant, pbwo4_constant, csi_constant]

x_pos = np.arange(len(materials))
width = 0.35

ax3.bar(x_pos - width/2, stochastic_terms, width, label='Stochastic Term (%)', alpha=0.8, color='skyblue')
ax3.bar(x_pos + width/2, constant_terms, width, label='Constant Term (%)', alpha=0.8, color='lightcoral')
ax3.set_xlabel('Material')
ax3.set_ylabel('Resolution Term (%)')
ax3.set_title('Stochastic vs Constant Terms')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(materials)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Resolution at 2 GeV (typical energy) with target band
res_2gev = [0.021725, 0.014936, 0.019525]  # BGO, PbWO4, CsI
target_2gev_min = target_min / np.sqrt(2.0)
target_2gev_max = target_max / np.sqrt(2.0)

colors = ['blue', 'red', 'green']
bars = ax4.bar(materials, res_2gev, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax4.axhspan(target_2gev_min, target_2gev_max, alpha=0.2, color='gray', label='Target Range')
ax4.set_ylabel('Energy Resolution (σ/E)')
ax4.set_title('Resolution at 2 GeV vs Target')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, res_2gev):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
             f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('material_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('material_comparison.pdf', bbox_inches='tight')

# Analysis: Check which materials meet target at each energy
print('=== MATERIAL PERFORMANCE ANALYSIS ===')
print(f'Target resolution: {target_min:.1%} - {target_max:.1%} / sqrt(E)')
print()

# Check performance at each energy
for i, energy in enumerate([0.5, 2.0, 5.0]):
    target_min_val = target_min / np.sqrt(energy)
    target_max_val = target_max / np.sqrt(energy)
    
    print(f'At {energy} GeV (target: {target_min_val:.4f} - {target_max_val:.4f}):')
    
    bgo_meets = target_min_val <= bgo_resolutions[i] <= target_max_val
    pbwo4_meets = target_min_val <= pbwo4_resolutions[i] <= target_max_val
    csi_meets = target_min_val <= csi_resolutions[i] <= target_max_val
    
    print(f'  BGO:   {bgo_resolutions[i]:.4f} - {"MEETS" if bgo_meets else "FAILS"} target')
    print(f'  PbWO4: {pbwo4_resolutions[i]:.4f} - {"MEETS" if pbwo4_meets else "FAILS"} target')
    print(f'  CsI:   {csi_resolutions[i]:.4f} - {"MEETS" if csi_meets else "FAILS"} target')
    print()

# Overall assessment
print('=== OVERALL ASSESSMENT ===')

# Count how many energy points each material meets target
bgo_meets_count = sum(1 for i, energy in enumerate([0.5, 2.0, 5.0]) 
                     if target_min/np.sqrt(energy) <= bgo_resolutions[i] <= target_max/np.sqrt(energy))
pbwo4_meets_count = sum(1 for i, energy in enumerate([0.5, 2.0, 5.0]) 
                       if target_min/np.sqrt(energy) <= pbwo4_resolutions[i] <= target_max/np.sqrt(energy))
csi_meets_count = sum(1 for i, energy in enumerate([0.5, 2.0, 5.0]) 
                     if target_min/np.sqrt(energy) <= csi_resolutions[i] <= target_max/np.sqrt(energy))

print(f'BGO meets target at {bgo_meets_count}/3 energy points')
print(f'PbWO4 meets target at {pbwo4_meets_count}/3 energy points')
print(f'CsI meets target at {csi_meets_count}/3 energy points')
print()

# Calculate average linearity
bgo_avg_linearity = np.mean(bgo_linearity)
pbwo4_avg_linearity = np.mean(pbwo4_linearity)
csi_avg_linearity = np.mean(csi_linearity)

print('Average Linearity:')
print(f'BGO:   {bgo_avg_linearity:.4f}')
print(f'PbWO4: {pbwo4_avg_linearity:.4f}')
print(f'CsI:   {csi_avg_linearity:.4f}')
print()

# Determine best material
if pbwo4_meets_count >= max(bgo_meets_count, csi_meets_count):
    if pbwo4_avg_linearity >= max(bgo_avg_linearity, csi_avg_linearity) - 0.001:
        best_material = 'PbWO4'
        best_reason = f'Meets target at {pbwo4_meets_count}/3 energies with best linearity ({pbwo4_avg_linearity:.4f})'
elif csi_meets_count >= bgo_meets_count:
    best_material = 'CsI'
    best_reason = f'Meets target at {csi_meets_count}/3 energies with good resolution scaling'
else:
    best_material = 'BGO'
    best_reason = f'Meets target at {bgo_meets_count}/3 energies with lowest stochastic term'

print(f'RECOMMENDED MATERIAL: {best_material}')
print(f'REASON: {best_reason}')

# Save detailed results
results = {
    'materials_compared': ['BGO', 'PbWO4', 'CsI'],
    'target_resolution': '1-3% / sqrt(E)',
    'bgo_performance': {
        'resolutions': bgo_resolutions,
        'linearity': bgo_linearity,
        'stochastic_term_percent': bgo_stochastic,
        'constant_term_percent': bgo_constant,
        'meets_target_count': bgo_meets_count
    },
    'pbwo4_performance': {
        'resolutions': pbwo4_resolutions,
        'linearity': pbwo4_linearity,
        'stochastic_term_percent': pbwo4_stochastic,
        'constant_term_percent': pbwo4_constant,
        'meets_target_count': pbwo4_meets_count
    },
    'csi_performance': {
        'resolutions': csi_resolutions,
        'linearity': csi_linearity,
        'stochastic_term_percent': csi_stochastic,
        'constant_term_percent': csi_constant,
        'meets_target_count': csi_meets_count
    },
    'best_material': best_material,
    'recommendation_reason': best_reason
}

with open('material_comparison_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Output results for workflow
print(f'RESULT:best_material={best_material}')
print(f'RESULT:bgo_meets_target_count={bgo_meets_count}')
print(f'RESULT:pbwo4_meets_target_count={pbwo4_meets_count}')
print(f'RESULT:csi_meets_target_count={csi_meets_count}')
print(f'RESULT:bgo_avg_linearity={bgo_avg_linearity:.4f}')
print(f'RESULT:pbwo4_avg_linearity={pbwo4_avg_linearity:.4f}')
print(f'RESULT:csi_avg_linearity={csi_avg_linearity:.4f}')
print('RESULT:comparison_plot=material_comparison.png')
print('RESULT:results_json=material_comparison_results.json')
print('RESULT:success=True')