import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Extract baseline performance metrics from previous step outputs
baseline_light_yield = 2.12  # pe/keV from analyze_baseline_performance
baseline_energy_resolution = 0.0308
containment_90_mm = 50

# Extract geometry parameters from previous steps
vessel_diameter_m = 0.5  # from generate_baseline_geometry
vessel_height_m = 0.5
current_pmt_count = 30
pmt_diameter_m = 0.2
max_pmt_count = 50  # constraint from step description

# Calculate vessel surface area for PMT placement
radius = vessel_diameter_m / 2
surface_area_barrel = 2 * np.pi * radius * vessel_height_m
surface_area_ends = 2 * np.pi * radius**2
total_surface_area = surface_area_barrel + surface_area_ends

print(f'Vessel geometry: diameter={vessel_diameter_m}m, height={vessel_height_m}m')
print(f'Total surface area: {total_surface_area:.3f} m²')
print(f'Current PMT count: {current_pmt_count}, Max allowed: {max_pmt_count}')

# PMT coverage analysis
pmt_area = np.pi * (pmt_diameter_m/2)**2
current_coverage = (current_pmt_count * pmt_area) / total_surface_area
max_coverage = (max_pmt_count * pmt_area) / total_surface_area

print(f'Current PMT coverage: {current_coverage:.1%}')
print(f'Maximum possible coverage: {max_coverage:.1%}')

# Optimize PMT placement strategy
# Strategy 1: Increase PMT count to maximum
optimal_pmt_count = max_pmt_count

# Strategy 2: Optimize placement pattern
# For cylindrical geometry, distribute PMTs to maximize solid angle coverage
# Place more PMTs on barrel (where most light collection occurs) vs ends

# Calculate optimal distribution between barrel and end caps
# Barrel typically has better light collection efficiency
barrel_fraction = 0.7  # 70% of PMTs on barrel
end_fraction = 0.3     # 30% on end caps

barrel_pmts = int(optimal_pmt_count * barrel_fraction)
end_pmts = optimal_pmt_count - barrel_pmts

print(f'\nOptimal PMT distribution:')
print(f'Barrel PMTs: {barrel_pmts}')
print(f'End cap PMTs: {end_pmts}')
print(f'Total PMTs: {barrel_pmts + end_pmts}')

# Calculate expected light collection improvement
# Light collection scales approximately with PMT count and coverage
pmt_count_improvement = optimal_pmt_count / current_pmt_count
coverage_improvement = max_coverage / current_coverage

# Conservative estimate: light yield improvement scales with sqrt of PMT count
# (due to geometric factors and light transport)
light_yield_improvement = np.sqrt(pmt_count_improvement)
expected_light_yield = baseline_light_yield * light_yield_improvement

# Energy resolution improvement (scales as 1/sqrt(light_yield))
resolution_improvement = np.sqrt(light_yield_improvement)
expected_resolution = baseline_energy_resolution / resolution_improvement

print(f'\nExpected performance improvements:')
print(f'PMT count increase: {pmt_count_improvement:.2f}x')
print(f'Coverage increase: {coverage_improvement:.2f}x')
print(f'Light yield improvement: {light_yield_improvement:.2f}x')
print(f'Expected light yield: {expected_light_yield:.2f} pe/keV')
print(f'Expected energy resolution: {expected_resolution:.4f} (vs {baseline_energy_resolution:.4f})')
print(f'Resolution improvement: {baseline_energy_resolution/expected_resolution:.2f}x')

# Generate PMT placement coordinates
# Barrel PMTs: distributed uniformly in phi and z
barrel_phi = np.linspace(0, 2*np.pi, barrel_pmts, endpoint=False)
barrel_z = np.random.uniform(-vessel_height_m/2, vessel_height_m/2, barrel_pmts)
barrel_x = radius * np.cos(barrel_phi)
barrel_y = radius * np.sin(barrel_phi)

# End cap PMTs: distributed radially on top and bottom
end_pmts_per_cap = end_pmts // 2
end_r = np.random.uniform(0, radius*0.9, end_pmts_per_cap)  # Avoid edge
end_phi = np.random.uniform(0, 2*np.pi, end_pmts_per_cap)

# Top cap PMTs
top_x = end_r * np.cos(end_phi)
top_y = end_r * np.sin(end_phi)
top_z = np.full(end_pmts_per_cap, vessel_height_m/2)

# Bottom cap PMTs
bottom_x = end_r * np.cos(end_phi + np.pi)  # Offset for better coverage
bottom_y = end_r * np.sin(end_phi + np.pi)
bottom_z = np.full(end_pmts_per_cap, -vessel_height_m/2)

# Combine all PMT positions
all_x = np.concatenate([barrel_x, top_x, bottom_x])
all_y = np.concatenate([barrel_y, top_y, bottom_z])
all_z = np.concatenate([barrel_z, top_z, bottom_z])

# Create visualization
fig = plt.figure(figsize=(15, 5))

# PMT placement 3D view
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(barrel_x, barrel_y, barrel_z, c='blue', s=50, label='Barrel PMTs', alpha=0.7)
ax1.scatter(top_x, top_y, top_z, c='red', s=50, label='Top PMTs', alpha=0.7)
ax1.scatter(bottom_x, bottom_y, bottom_z, c='green', s=50, label='Bottom PMTs', alpha=0.7)
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Z (m)')
ax1.set_title(f'Optimized PMT Placement\n({optimal_pmt_count} PMTs)')
ax1.legend()

# Coverage comparison
ax2 = fig.add_subplot(132)
configs = ['Baseline', 'Optimized']
coverages = [current_coverage*100, max_coverage*100]
pmt_counts = [current_pmt_count, optimal_pmt_count]
ax2.bar(configs, coverages, color=['blue', 'green'], alpha=0.7)
ax2.set_ylabel('PMT Coverage (%)')
ax2.set_title('Coverage Comparison')
for i, (cov, count) in enumerate(zip(coverages, pmt_counts)):
    ax2.text(i, cov + 1, f'{count} PMTs\n{cov:.1f}%', ha='center')

# Performance improvement
ax3 = fig.add_subplot(133)
metrics = ['Light Yield\n(pe/keV)', 'Energy Resolution\n(σ/E)']
baseline_vals = [baseline_light_yield, baseline_energy_resolution]
optimized_vals = [expected_light_yield, expected_resolution]
x = np.arange(len(metrics))
width = 0.35
ax3.bar(x - width/2, baseline_vals, width, label='Baseline', color='blue', alpha=0.7)
ax3.bar(x + width/2, optimized_vals, width, label='Optimized', color='green', alpha=0.7)
ax3.set_ylabel('Value')
ax3.set_title('Performance Comparison')
ax3.set_xticks(x)
ax3.set_xticklabels(metrics)
ax3.legend()

plt.tight_layout()
plt.savefig('optimized_pmt_configuration.png', dpi=150, bbox_inches='tight')
plt.savefig('optimized_pmt_configuration.pdf', bbox_inches='tight')
plt.show()

# Save optimization results
optimization_results = {
    'optimal_pmt_count': optimal_pmt_count,
    'barrel_pmts': barrel_pmts,
    'end_cap_pmts': end_pmts,
    'coverage_improvement': coverage_improvement,
    'light_yield_improvement': light_yield_improvement,
    'expected_light_yield_pe_kev': expected_light_yield,
    'expected_energy_resolution': expected_resolution,
    'resolution_improvement_factor': baseline_energy_resolution/expected_resolution,
    'pmt_positions': {
        'x': all_x.tolist(),
        'y': all_y.tolist(),
        'z': all_z.tolist()
    }
}

with open('pmt_optimization_results.json', 'w') as f:
    json.dump(optimization_results, f, indent=2)

# Return key results
print(f'RESULT:optimal_pmt_count={optimal_pmt_count}')
print(f'RESULT:coverage_improvement={coverage_improvement:.3f}')
print(f'RESULT:light_yield_improvement={light_yield_improvement:.3f}')
print(f'RESULT:expected_light_yield={expected_light_yield:.2f}')
print(f'RESULT:expected_resolution={expected_resolution:.4f}')
print(f'RESULT:resolution_improvement={baseline_energy_resolution/expected_resolution:.2f}')
print('RESULT:optimization_plot=optimized_pmt_configuration.png')
print('RESULT:results_file=pmt_optimization_results.json')

print('\n=== PMT OPTIMIZATION COMPLETE ===')
print(f'Optimized configuration uses {optimal_pmt_count} PMTs (within {max_pmt_count} limit)')
print(f'Expected light collection efficiency improvement: {light_yield_improvement:.2f}x')
print(f'Expected energy resolution improvement: {baseline_energy_resolution/expected_resolution:.2f}x')