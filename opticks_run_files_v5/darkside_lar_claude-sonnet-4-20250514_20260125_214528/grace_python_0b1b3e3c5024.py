import matplotlib
matplotlib.use('Agg')
import uproot
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# File paths from previous steps
optimized_file = 'optimized_geometry_electron_hits.root'
baseline_file = 'baseline_geometry_electron_hits.root'

print('Starting position reconstruction analysis...')

# Function to analyze position reconstruction for a detector
def analyze_position_reconstruction(filename, detector_name):
    print(f'Analyzing {detector_name} detector: {filename}')
    
    with uproot.open(filename) as f:
        # Load events data for energy information
        events = f['events'].arrays(library='pd')
        print(f'{detector_name}: {len(events)} events')
        
        # Sample hits data to avoid memory issues (first 100k hits)
        hits = f['hits'].arrays(['x', 'y', 'z', 'edep', 'eventID'], library='np', entry_stop=100000)
        print(f'{detector_name}: Analyzing {len(hits["x"])} hits (sampled)')
    
    # Convert to cylindrical coordinates for analysis
    x, y, z = hits['x'], hits['y'], hits['z']
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    
    # Detector geometry (from previous steps)
    detector_radius = 250.0  # mm (0.5m diameter = 250mm radius)
    detector_height = 500.0  # mm (0.5m height)
    
    # Calculate position reconstruction metrics
    
    # 1. Radial uniformity - energy deposits vs radius
    r_bins = np.linspace(0, detector_radius, 20)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    r_hist, _ = np.histogram(r, bins=r_bins, weights=hits['edep'])
    r_counts, _ = np.histogram(r, bins=r_bins)
    
    # Normalize by bin area (2πr * dr) for uniform density
    bin_areas = 2 * np.pi * r_centers * (r_bins[1] - r_bins[0])
    r_density = np.divide(r_hist, bin_areas, out=np.zeros_like(r_hist), where=bin_areas!=0)
    
    # 2. Azimuthal uniformity
    phi_bins = np.linspace(-np.pi, np.pi, 24)  # 15-degree bins
    phi_centers = (phi_bins[:-1] + phi_bins[1:]) / 2
    phi_hist, _ = np.histogram(phi, bins=phi_bins, weights=hits['edep'])
    
    # 3. Z uniformity
    z_bins = np.linspace(-detector_height/2, detector_height/2, 20)
    z_centers = (z_bins[:-1] + z_bins[1:]) / 2
    z_hist, _ = np.histogram(z, bins=z_bins, weights=hits['edep'])
    
    # Calculate uniformity metrics (coefficient of variation)
    r_uniformity = np.std(r_density[r_density > 0]) / np.mean(r_density[r_density > 0]) if np.sum(r_density > 0) > 0 else 1.0
    phi_uniformity = np.std(phi_hist) / np.mean(phi_hist) if np.mean(phi_hist) > 0 else 1.0
    z_uniformity = np.std(z_hist) / np.mean(z_hist) if np.mean(z_hist) > 0 else 1.0
    
    # Position reconstruction accuracy (RMS spread)
    # For electrons, expect tight clustering around injection point
    rms_r = np.sqrt(np.average(r**2, weights=hits['edep']))
    rms_z = np.sqrt(np.average(z**2, weights=hits['edep']))
    
    # Calculate containment radii
    total_edep = np.sum(hits['edep'])
    r_sorted_indices = np.argsort(r)
    cumulative_edep = np.cumsum(hits['edep'][r_sorted_indices])
    containment_50_idx = np.where(cumulative_edep >= 0.5 * total_edep)[0]
    containment_90_idx = np.where(cumulative_edep >= 0.9 * total_edep)[0]
    
    r50 = r[r_sorted_indices[containment_50_idx[0]]] if len(containment_50_idx) > 0 else detector_radius
    r90 = r[r_sorted_indices[containment_90_idx[0]]] if len(containment_90_idx) > 0 else detector_radius
    
    return {
        'r_centers': r_centers,
        'r_density': r_density,
        'phi_centers': phi_centers,
        'phi_hist': phi_hist,
        'z_centers': z_centers,
        'z_hist': z_hist,
        'r_uniformity': r_uniformity,
        'phi_uniformity': phi_uniformity,
        'z_uniformity': z_uniformity,
        'rms_r': rms_r,
        'rms_z': rms_z,
        'r50': r50,
        'r90': r90,
        'total_hits': len(hits['x']),
        'total_edep': total_edep
    }

# Analyze both detectors
baseline_results = analyze_position_reconstruction(baseline_file, 'Baseline')
optimized_results = analyze_position_reconstruction(optimized_file, 'Optimized')

# Create comprehensive position reconstruction plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Position Reconstruction Analysis: Baseline vs Optimized', fontsize=16)

# Radial uniformity
axes[0,0].plot(baseline_results['r_centers'], baseline_results['r_density'], 'b-', label='Baseline (30 PMTs)', linewidth=2)
axes[0,0].plot(optimized_results['r_centers'], optimized_results['r_density'], 'r-', label='Optimized (50 PMTs)', linewidth=2)
axes[0,0].set_xlabel('Radius (mm)')
axes[0,0].set_ylabel('Energy Density (MeV/mm²)')
axes[0,0].set_title('Radial Uniformity')
axes[0,0].legend()
axes[0,0].grid(true, alpha=0.3)

# Azimuthal uniformity
axes[0,1].plot(baseline_results['phi_centers'] * 180/np.pi, baseline_results['phi_hist'], 'b-', label='Baseline', linewidth=2)
axes[0,1].plot(optimized_results['phi_centers'] * 180/np.pi, optimized_results['phi_hist'], 'r-', label='Optimized', linewidth=2)
axes[0,1].set_xlabel('Azimuthal Angle (degrees)')
axes[0,1].set_ylabel('Energy Deposit (MeV)')
axes[0,1].set_title('Azimuthal Uniformity')
axes[0,1].legend()
axes[0,1].grid(true, alpha=0.3)

# Z uniformity
axes[0,2].plot(baseline_results['z_centers'], baseline_results['z_hist'], 'b-', label='Baseline', linewidth=2)
axes[0,2].plot(optimized_results['z_centers'], optimized_results['z_hist'], 'r-', label='Optimized', linewidth=2)
axes[0,2].set_xlabel('Z Position (mm)')
axes[0,2].set_ylabel('Energy Deposit (MeV)')
axes[0,2].set_title('Longitudinal Uniformity')
axes[0,2].legend()
axes[0,2].grid(true, alpha=0.3)

# Uniformity comparison bar chart
uniformity_metrics = ['Radial', 'Azimuthal', 'Longitudinal']
baseline_uniformities = [baseline_results['r_uniformity'], baseline_results['phi_uniformity'], baseline_results['z_uniformity']]
optimized_uniformities = [optimized_results['r_uniformity'], optimized_results['phi_uniformity'], optimized_results['z_uniformity']]

x_pos = np.arange(len(uniformity_metrics))
width = 0.35

axes[1,0].bar(x_pos - width/2, baseline_uniformities, width, label='Baseline', color='blue', alpha=0.7)
axes[1,0].bar(x_pos + width/2, optimized_uniformities, width, label='Optimized', color='red', alpha=0.7)
axes[1,0].set_xlabel('Uniformity Type')
axes[1,0].set_ylabel('Coefficient of Variation')
axes[1,0].set_title('Uniformity Comparison (lower = better)')
axes[1,0].set_xticks(x_pos)
axes[1,0].set_xticklabels(uniformity_metrics)
axes[1,0].legend()
axes[1,0].grid(true, alpha=0.3)

# Position resolution comparison
resolution_metrics = ['RMS Radial', 'RMS Z', 'R50', 'R90']
baseline_resolutions = [baseline_results['rms_r'], baseline_results['rms_z'], baseline_results['r50'], baseline_results['r90']]
optimized_resolutions = [optimized_results['rms_r'], optimized_results['rms_z'], optimized_results['r50'], optimized_results['r90']]

x_pos = np.arange(len(resolution_metrics))
axes[1,1].bar(x_pos - width/2, baseline_resolutions, width, label='Baseline', color='blue', alpha=0.7)
axes[1,1].bar(x_pos + width/2, optimized_resolutions, width, label='Optimized', color='red', alpha=0.7)
axes[1,1].set_xlabel('Resolution Metric')
axes[1,1].set_ylabel('Distance (mm)')
axes[1,1].set_title('Position Resolution (lower = better)')
axes[1,1].set_xticks(x_pos)
axes[1,1].set_xticklabels(resolution_metrics, rotation=45)
axes[1,1].legend()
axes[1,1].grid(true, alpha=0.3)

# Overall improvement summary
improvements = {
    'Radial Uniformity': (baseline_results['r_uniformity'] - optimized_results['r_uniformity']) / baseline_results['r_uniformity'] * 100,
    'Azimuthal Uniformity': (baseline_results['phi_uniformity'] - optimized_results['phi_uniformity']) / baseline_results['phi_uniformity'] * 100,
    'Z Uniformity': (baseline_results['z_uniformity'] - optimized_results['z_uniformity']) / baseline_results['z_uniformity'] * 100,
    'RMS Radial': (baseline_results['rms_r'] - optimized_results['rms_r']) / baseline_results['rms_r'] * 100,
    'RMS Z': (baseline_results['rms_z'] - optimized_results['rms_z']) / baseline_results['rms_z'] * 100
}

improvement_names = list(improvements.keys())
improvement_values = list(improvements.values())

colors = ['green' if x > 0 else 'red' for x in improvement_values]
axes[1,2].barh(improvement_names, improvement_values, color=colors, alpha=0.7)
axes[1,2].set_xlabel('Improvement (%)')
axes[1,2].set_title('Position Reconstruction Improvements')
axes[1,2].axvline(0, color='black', linestyle='-', alpha=0.5)
axes[1,2].grid(true, alpha=0.3)

plt.tight_layout()
plt.savefig('position_reconstruction_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('position_reconstruction_analysis.pdf', bbox_inches='tight')
plt.show()

# Calculate overall position reconstruction accuracy
baseline_overall_accuracy = np.sqrt(baseline_results['rms_r']**2 + baseline_results['rms_z']**2)
optimized_overall_accuracy = np.sqrt(optimized_results['rms_r']**2 + optimized_results['rms_z']**2)
accuracy_improvement = (baseline_overall_accuracy - optimized_overall_accuracy) / baseline_overall_accuracy * 100

# Calculate average uniformity improvement
avg_uniformity_improvement = np.mean([improvements['Radial Uniformity'], improvements['Azimuthal Uniformity'], improvements['Z Uniformity']])

# Print results
print('\n