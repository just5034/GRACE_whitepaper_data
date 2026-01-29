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
vessel_surface_area = 57.7268  # m^2
pmt_diameter = 0.2     # meters
pmt_area = np.pi * (pmt_diameter/2)**2  # 0.0314 m^2

# PMT configurations to test
pmt_configurations = [50, 75, 100]

# Calculate coverage for each configuration
results = []
for pmt_count in pmt_configurations:
    total_pmt_area = pmt_count * pmt_area
    coverage_fraction = total_pmt_area / vessel_surface_area
    
    # Light collection efficiency estimation
    # Based on solid angle coverage - more PMTs = better light collection
    # Assume geometric efficiency scales with coverage up to saturation
    geometric_efficiency = min(coverage_fraction * 0.8, 0.6)  # Cap at 60% max
    
    # Account for photocathode quantum efficiency (~25% for typical PMTs)
    quantum_efficiency = 0.25
    
    # Overall light collection efficiency
    light_collection_efficiency = geometric_efficiency * quantum_efficiency
    
    results.append({
        'pmt_count': pmt_count,
        'coverage_fraction': coverage_fraction,
        'geometric_efficiency': geometric_efficiency,
        'light_collection_efficiency': light_collection_efficiency
    })
    
    print(f"PMT Config {pmt_count}: Coverage={coverage_fraction:.4f}, Light Collection Eff={light_collection_efficiency:.4f}")

# Convert to DataFrame for analysis
df_results = pd.DataFrame(results)

# Find optimal configuration
optimal_idx = df_results['light_collection_efficiency'].idxmax()
optimal_config = df_results.iloc[optimal_idx]

print(f"\nOptimal PMT Configuration:")
print(f"PMT Count: {optimal_config['pmt_count']}")
print(f"Coverage: {optimal_config['coverage_fraction']:.4f} ({optimal_config['coverage_fraction']*100:.1f}%)")
print(f"Light Collection Efficiency: {optimal_config['light_collection_efficiency']:.4f}")

# Create optimization plot
plt.figure(figsize=(12, 5))

# Plot 1: Coverage vs PMT count
plt.subplot(1, 2, 1)
plt.plot(df_results['pmt_count'], df_results['coverage_fraction']*100, 'bo-', linewidth=2, markersize=8)
plt.xlabel('PMT Count')
plt.ylabel('Surface Coverage (%)')
plt.title('PMT Coverage vs Count')
plt.grid(True, alpha=0.3)
plt.axvline(optimal_config['pmt_count'], color='r', linestyle='--', alpha=0.7, label='Optimal')
plt.legend()

# Plot 2: Light collection efficiency vs PMT count
plt.subplot(1, 2, 2)
plt.plot(df_results['pmt_count'], df_results['light_collection_efficiency']*100, 'go-', linewidth=2, markersize=8)
plt.xlabel('PMT Count')
plt.ylabel('Light Collection Efficiency (%)')
plt.title('Light Collection Efficiency vs PMT Count')
plt.grid(True, alpha=0.3)
plt.axvline(optimal_config['pmt_count'], color='r', linestyle='--', alpha=0.7, label='Optimal')
plt.legend()

plt.tight_layout()
plt.savefig('pmt_optimization.png', dpi=150, bbox_inches='tight')
plt.savefig('pmt_optimization.pdf', bbox_inches='tight')
print("\nPlot saved: pmt_optimization.png")

# Save detailed results
with open('pmt_optimization_results.json', 'w') as f:
    json.dump({
        'configurations': results,
        'optimal_configuration': {
            'pmt_count': int(optimal_config['pmt_count']),
            'coverage_fraction': float(optimal_config['coverage_fraction']),
            'light_collection_efficiency': float(optimal_config['light_collection_efficiency'])
        },
        'vessel_parameters': {
            'diameter_m': vessel_diameter,
            'height_m': vessel_height,
            'surface_area_m2': vessel_surface_area
        }
    }, indent=2)

# Output results for downstream steps
print(f"RESULT:optimal_pmt_count={int(optimal_config['pmt_count'])}")
print(f"RESULT:optimal_coverage={optimal_config['coverage_fraction']:.4f}")
print(f"RESULT:optimal_light_collection_efficiency={optimal_config['light_collection_efficiency']:.4f}")
print(f"RESULT:optimization_plot=pmt_optimization.png")
print(f"RESULT:results_file=pmt_optimization_results.json")
print("RESULT:success=True")