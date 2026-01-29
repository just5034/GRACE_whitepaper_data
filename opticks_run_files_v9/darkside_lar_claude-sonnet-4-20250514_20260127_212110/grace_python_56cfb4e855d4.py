import matplotlib
matplotlib.use('Agg')
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Base geometry parameters from previous step
base_geometry_file = '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/darkside_lar_claude-sonnet-4-20250514_20260127_212110/darkside_detector.gdml'
vessel_diameter = 3.6  # meters
vessel_height = 3.6    # meters
vessel_surface_area = 57.7268  # m²
pmt_diameter = 0.2     # meters
pmt_area = np.pi * (pmt_diameter/2)**2  # 0.0314 m²

# PMT configurations to test
pmt_configs = [50, 75, 100]
optimization_results = []

print('Starting PMT coverage optimization...')

# Calculate coverage and efficiency for each configuration
for pmt_count in pmt_configs:
    # Calculate coverage fraction
    total_pmt_area = pmt_count * pmt_area
    coverage_fraction = total_pmt_area / vessel_surface_area
    
    # Light collection efficiency model (simplified)
    # Based on solid angle coverage and photon detection probability
    # Efficiency increases with coverage but with diminishing returns
    base_efficiency = coverage_fraction * 0.8  # 80% quantum efficiency
    geometric_factor = 1.0 - np.exp(-coverage_fraction * 2.0)  # geometric acceptance
    light_collection_efficiency = base_efficiency * geometric_factor
    
    # Cost-benefit analysis (efficiency per PMT)
    efficiency_per_pmt = light_collection_efficiency / pmt_count
    
    config_result = {
        'pmt_count': pmt_count,
        'coverage_fraction': round(coverage_fraction, 4),
        'light_collection_efficiency': round(light_collection_efficiency, 4),
        'efficiency_per_pmt': round(efficiency_per_pmt, 6),
        'total_pmt_area_m2': round(total_pmt_area, 3)
    }
    
    optimization_results.append(config_result)
    print(f'PMT Config {pmt_count}: Coverage={coverage_fraction:.3f}, Efficiency={light_collection_efficiency:.3f}')

# Find optimal configuration based on light collection efficiency
optimal_config = max(optimization_results, key=lambda x: x['light_collection_efficiency'])
print(f'\nOptimal PMT configuration: {optimal_config["pmt_count"]} PMTs')
print(f'Light collection efficiency: {optimal_config["light_collection_efficiency"]:.4f}')

# Create comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Light collection efficiency vs PMT count
pmt_counts = [r['pmt_count'] for r in optimization_results]
efficiencies = [r['light_collection_efficiency'] for r in optimization_results]
ax1.plot(pmt_counts, efficiencies, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of PMTs')
ax1.set_ylabel('Light Collection Efficiency')
ax1.set_title('Light Collection Efficiency vs PMT Count')
ax1.grid(True, alpha=0.3)
ax1.axvline(optimal_config['pmt_count'], color='r', linestyle='--', alpha=0.7, label='Optimal')
ax1.legend()

# Plot 2: Coverage fraction vs PMT count
coverages = [r['coverage_fraction'] for r in optimization_results]
ax2.plot(pmt_counts, coverages, 'go-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of PMTs')
ax2.set_ylabel('Coverage Fraction')
ax2.set_title('PMT Coverage vs Count')
ax2.grid(True, alpha=0.3)
ax2.axvline(optimal_config['pmt_count'], color='r', linestyle='--', alpha=0.7, label='Optimal')
ax2.legend()

plt.tight_layout()
plt.savefig('pmt_optimization_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('pmt_optimization_analysis.pdf', bbox_inches='tight')
print('\nSaved optimization plots to pmt_optimization_analysis.png')

# Prepare final results with proper JSON structure
final_results = {
    'optimization_summary': {
        'base_geometry': base_geometry_file,
        'vessel_parameters': {
            'diameter_m': vessel_diameter,
            'height_m': vessel_height,
            'surface_area_m2': vessel_surface_area
        },
        'pmt_parameters': {
            'diameter_m': pmt_diameter,
            'area_m2': round(pmt_area, 6)
        }
    },
    'configurations_tested': optimization_results,
    'optimal_configuration': {
        'pmt_count': optimal_config['pmt_count'],
        'coverage_fraction': optimal_config['coverage_fraction'],
        'light_collection_efficiency': optimal_config['light_collection_efficiency'],
        'efficiency_per_pmt': optimal_config['efficiency_per_pmt']
    },
    'analysis_plots': {
        'optimization_plot': 'pmt_optimization_analysis.png',
        'optimization_plot_pdf': 'pmt_optimization_analysis.pdf'
    },
    'success': True
}

# Write results to output file
output_file = 'pmt_optimization_results.json'
with open(output_file, 'w') as f:
    json.dump(final_results, f, indent=2)

print(f'\nOptimization results written to {output_file}')
print(f'RESULT:optimal_pmt_count={optimal_config["pmt_count"]}')
print(f'RESULT:optimal_efficiency={optimal_config["light_collection_efficiency"]:.4f}')
print(f'RESULT:optimal_coverage={optimal_config["coverage_fraction"]:.4f}')
print(f'RESULT:output_file={output_file}')
print('RESULT:success=True')