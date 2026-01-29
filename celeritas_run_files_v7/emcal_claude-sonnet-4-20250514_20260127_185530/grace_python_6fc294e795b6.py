import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

# Material performance data from previous step outputs
bgo_results = {
    '0.5gev': {'resolution': 0.026309, 'resolution_err': 0.000263, 'mean_deposit': 481.23, 'linearity': 0.9625},
    '2.0gev': {'resolution': 0.021725, 'resolution_err': 0.000217, 'mean_deposit': 1922.8, 'linearity': 0.9614},
    '5.0gev': {'resolution': 0.017925, 'resolution_err': 0.000179, 'mean_deposit': 4795.16, 'linearity': 0.959}
}

pbwo4_results = {
    '0.5gev': {'resolution': 0.033044, 'resolution_err': 0.00033, 'mean_deposit': 483.91, 'linearity': 0.9678},
    '2.0gev': {'resolution': 0.014936, 'resolution_err': 0.000149, 'mean_deposit': 1934.88, 'linearity': 0.9674},
    '5.0gev': {'resolution': 0.019751, 'resolution_err': 0.000198, 'mean_deposit': 4818.5, 'linearity': 0.9637}
}

csi_results = {
    '0.5gev': {'resolution': 0.031228, 'resolution_err': 0.000312, 'mean_deposit': 477.67, 'linearity': 0.9553},
    '2.0gev': {'resolution': 0.019525, 'resolution_err': 0.000195, 'mean_deposit': 1909.55, 'linearity': 0.9548},
    '5.0gev': {'resolution': 0.016829, 'resolution_err': 0.000168, 'mean_deposit': 4763.59, 'linearity': 0.9527}
}

# Energy points for analysis
energies = [0.5, 2.0, 5.0]

# Extract resolution data for each material
bgo_resolutions = [bgo_results[f'{e}gev']['resolution'] for e in energies]
bgo_errors = [bgo_results[f'{e}gev']['resolution_err'] for e in energies]

pbwo4_resolutions = [pbwo4_results[f'{e}gev']['resolution'] for e in energies]
pbwo4_errors = [pbwo4_results[f'{e}gev']['resolution_err'] for e in energies]

csi_resolutions = [csi_results[f'{e}gev']['resolution'] for e in energies]
csi_errors = [csi_results[f'{e}gev']['resolution_err'] for e in energies]

# Target resolution calculation (1-3% / sqrt(E))
# Calculate target_min and target_max as single scalar values
target_min = 0.01 / np.sqrt(np.max(energies))  # 1% at highest energy
target_max = 0.03 / np.sqrt(np.min(energies))  # 3% at lowest energy

# Create energy array for smooth target curves
energy_smooth = np.linspace(0.5, 5.0, 100)
target_1pct = 0.01 / np.sqrt(energy_smooth)
target_3pct = 0.03 / np.sqrt(energy_smooth)

# Create comparison plot with fixed fill_between
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Resolution vs Energy plot
ax1.fill_between(energy_smooth, target_1pct, target_3pct, alpha=0.2, color='gray', label='Target: 1-3%/√E')
ax1.errorbar(energies, bgo_resolutions, yerr=bgo_errors, marker='o', label='BGO', linewidth=2, markersize=8)
ax1.errorbar(energies, pbwo4_resolutions, yerr=pbwo4_errors, marker='s', label='PbWO4', linewidth=2, markersize=8)
ax1.errorbar(energies, csi_resolutions, yerr=csi_errors, marker='^', label='CsI', linewidth=2, markersize=8)
ax1.set_xlabel('Energy (GeV)')
ax1.set_ylabel('Energy Resolution (σ/E)')
ax1.set_title('Energy Resolution Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 0.04)

# Linearity comparison
bgo_linearity = [bgo_results[f'{e}gev']['linearity'] for e in energies]
pbwo4_linearity = [pbwo4_results[f'{e}gev']['linearity'] for e in energies]
csi_linearity = [csi_results[f'{e}gev']['linearity'] for e in energies]

ax2.plot(energies, bgo_linearity, marker='o', label='BGO', linewidth=2, markersize=8)
ax2.plot(energies, pbwo4_linearity, marker='s', label='PbWO4', linewidth=2, markersize=8)
ax2.plot(energies, csi_linearity, marker='^', label='CsI', linewidth=2, markersize=8)
ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect Linearity')
ax2.set_xlabel('Energy (GeV)')
ax2.set_ylabel('Linearity (E_measured/E_beam)')
ax2.set_title('Linearity Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.94, 0.98)

plt.tight_layout()
plt.savefig('materials_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('materials_comparison.pdf', bbox_inches='tight')
plt.close()

# Performance summary at 2 GeV (typical energy for comparison)
print('\n=== MATERIAL COMPARISON SUMMARY (2 GeV) ===')
print(f'BGO:   Resolution = {bgo_results["2.0gev"]["resolution"]:.4f} ± {bgo_results["2.0gev"]["resolution_err"]:.4f}')
print(f'PbWO4: Resolution = {pbwo4_results["2.0gev"]["resolution"]:.4f} ± {pbwo4_results["2.0gev"]["resolution_err"]:.4f}')
print(f'CsI:   Resolution = {csi_results["2.0gev"]["resolution"]:.4f} ± {csi_results["2.0gev"]["resolution_err"]:.4f}')

# Determine optimal material based on resolution at 2 GeV
resolutions_2gev = {
    'BGO': bgo_results['2.0gev']['resolution'],
    'PbWO4': pbwo4_results['2.0gev']['resolution'],
    'CsI': csi_results['2.0gev']['resolution']
}

optimal_material = min(resolutions_2gev.keys(), key=lambda k: resolutions_2gev[k])
optimal_resolution = resolutions_2gev[optimal_material]

print(f'\nOPTIMAL MATERIAL: {optimal_material} with resolution = {optimal_resolution:.4f}')

# Check target compliance
target_2gev = 0.02 / np.sqrt(2.0)  # 2%/sqrt(2GeV) = 1.41%
print(f'Target at 2 GeV: {target_2gev:.4f}')

for material, resolution in resolutions_2gev.items():
    meets_target = 'YES' if resolution <= target_2gev else 'NO'
    print(f'{material} meets target: {meets_target} ({resolution:.4f} vs {target_2gev:.4f})')

# Save results
comparison_results = {
    'optimal_material': optimal_material,
    'optimal_resolution_2gev': optimal_resolution,
    'target_resolution_2gev': target_2gev,
    'materials_ranking': sorted(resolutions_2gev.items(), key=lambda x: x[1]),
    'target_min_scalar': target_min,
    'target_max_scalar': target_max,
    'comparison_plot': 'materials_comparison.png'
}

with open('materials_comparison_results.json', 'w') as f:
    json.dump(comparison_results, f, indent=2)

print(f'\nRESULT:optimal_material={optimal_material}')
print(f'RESULT:optimal_resolution={optimal_resolution:.4f}')
print(f'RESULT:target_resolution_2gev={target_2gev:.4f}')
print(f'RESULT:comparison_plot=materials_comparison.png')
print(f'RESULT:results_json=materials_comparison_results.json')
print('RESULT:success=True')