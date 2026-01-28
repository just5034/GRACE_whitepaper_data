import matplotlib
matplotlib.use('Agg')
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load barrel-optimized simulation data
barrel_hits_file = 'barrel_optimized_electron_hits.root'
barrel_parquet_file = 'barrel_optimized_electron_hits_data.parquet'

print('Loading barrel-optimized detector simulation data...')

# Load events data for efficiency calculation
with uproot.open(barrel_hits_file) as f:
    events = f['events'].arrays(library='pd')
    print(f'Loaded {len(events)} events from barrel-optimized detector')

# Calculate light collection efficiency
total_events = len(events)
detected_events = len(events[events['totalEdep'] > 0])
light_collection_efficiency = detected_events / total_events if total_events > 0 else 0
light_collection_efficiency_error = np.sqrt(light_collection_efficiency * (1 - light_collection_efficiency) / total_events) if total_events > 0 else 0

print(f'Light collection efficiency: {light_collection_efficiency:.4f} ± {light_collection_efficiency_error:.4f}')

# Load hit-level data for spatial uniformity (sample first 100k hits to avoid timeout)
print('Analyzing spatial uniformity...')
with uproot.open(barrel_hits_file) as f:
    hits = f['hits'].arrays(['x', 'y', 'z', 'edep'], library='np', entry_stop=100000)

# Calculate spatial uniformity using coefficient of variation of energy deposits
# Create spatial bins and calculate energy deposit distribution
r = np.sqrt(hits['x']**2 + hits['y']**2)  # Radial distance
z = hits['z']

# Create 2D spatial bins (r-z grid for cylindrical detector)
r_bins = np.linspace(0, np.max(r), 20)
z_bins = np.linspace(np.min(z), np.max(z), 20)

# Calculate energy deposit in each spatial bin
hist_2d, r_edges, z_edges = np.histogram2d(r, z, bins=[r_bins, z_bins], weights=hits['edep'])

# Calculate spatial uniformity as coefficient of variation
nonzero_bins = hist_2d[hist_2d > 0]
if len(nonzero_bins) > 1:
    spatial_uniformity = np.std(nonzero_bins) / np.mean(nonzero_bins)
else:
    spatial_uniformity = 0.0

print(f'Spatial uniformity (CV): {spatial_uniformity:.4f}')

# Generate photon hit distribution plot
plt.figure(figsize=(12, 5))

# Plot 1: Energy distribution
plt.subplot(1, 2, 1)
plt.hist(events['totalEdep'], bins=50, histtype='step', linewidth=2, alpha=0.8, label='Barrel Optimized')
plt.xlabel('Total Energy Deposit (MeV)')
plt.ylabel('Events')
plt.title('Energy Distribution - Barrel Optimized')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Spatial hit distribution
plt.subplot(1, 2, 2)
plt.scatter(hits['x'][:10000], hits['y'][:10000], c=hits['edep'][:10000], s=1, alpha=0.6, cmap='viridis')
plt.colorbar(label='Energy Deposit (MeV)')
plt.xlabel('X Position (mm)')
plt.ylabel('Y Position (mm)')
plt.title('Spatial Hit Distribution (10k sample)')
plt.axis('equal')

plt.tight_layout()
plt.savefig('barrel_performance_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('barrel_performance_analysis.pdf', bbox_inches='tight')
plt.close()

# Compare with baseline results (from previous step outputs)
baseline_efficiency = 1.0
baseline_uniformity = 0.7841

# Calculate changes
efficiency_change = light_collection_efficiency - baseline_efficiency
uniformity_change = spatial_uniformity - baseline_uniformity

print('\n=== COMPARISON WITH BASELINE ===')
print(f'Baseline efficiency: {baseline_efficiency:.4f}')
print(f'Barrel efficiency: {light_collection_efficiency:.4f}')
print(f'Efficiency change: {efficiency_change:.4f}')
print(f'Baseline uniformity: {baseline_uniformity:.4f}')
print(f'Barrel uniformity: {spatial_uniformity:.4f}')
print(f'Uniformity change: {uniformity_change:.4f}')

# Generate comparison plot
plt.figure(figsize=(10, 6))

# Efficiency comparison
plt.subplot(1, 2, 1)
configs = ['Baseline', 'Barrel Optimized']
efficiencies = [baseline_efficiency, light_collection_efficiency]
errors = [0.0, light_collection_efficiency_error]
bars1 = plt.bar(configs, efficiencies, yerr=errors, capsize=5, color=['blue', 'green'], alpha=0.7)
plt.ylabel('Light Collection Efficiency')
plt.title('Efficiency Comparison')
plt.ylim(0, 1.1)
for i, v in enumerate(efficiencies):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')

# Uniformity comparison
plt.subplot(1, 2, 2)
uniformities = [baseline_uniformity, spatial_uniformity]
bars2 = plt.bar(configs, uniformities, color=['blue', 'green'], alpha=0.7)
plt.ylabel('Spatial Uniformity (CV)')
plt.title('Uniformity Comparison')
for i, v in enumerate(uniformities):
    plt.text(i, v + 0.05, f'{v:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('barrel_performance_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('barrel_performance_comparison.pdf', bbox_inches='tight')
plt.close()

# Data validation
data_validation_passed = (total_events > 0 and 
                         light_collection_efficiency >= 0 and 
                         light_collection_efficiency <= 1 and
                         spatial_uniformity >= 0)

print(f'\nData validation passed: {data_validation_passed}')

# Return results for downstream workflow steps
print('\n=== RESULTS ===')
print(f'RESULT:light_collection_efficiency={light_collection_efficiency:.6f}')
print(f'RESULT:light_collection_efficiency_error={light_collection_efficiency_error:.6f}')
print(f'RESULT:spatial_uniformity={spatial_uniformity:.4f}')
print(f'RESULT:total_events={total_events}')
print(f'RESULT:detected_events={detected_events}')
print(f'RESULT:efficiency_change={efficiency_change:.6f}')
print(f'RESULT:uniformity_change={uniformity_change:.4f}')
print('RESULT:analysis_plots=barrel_performance_analysis.png')
print('RESULT:comparison_plots=barrel_performance_comparison.png')
print(f'RESULT:data_validation_passed={str(data_validation_passed).lower()}')

print('\nBarrel-optimized detector performance analysis completed successfully!')