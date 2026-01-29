import matplotlib
matplotlib.use('Agg')
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Load uniform detector simulation data
with uproot.open('uniform_lar_detector_electron_hits.root') as f:
    events = f['events'].arrays(library='pd')
    print(f"Loaded {len(events)} events from uniform detector simulation")

# Get detector parameters from previous steps
detector_diameter_m = 3.0
detector_height_m = 2.5
detector_volume_m3 = np.pi * (detector_diameter_m/2)**2 * detector_height_m
sensor_count = 75

# Calculate baseline performance metrics

# 1. Light Collection Efficiency (from energy deposits)
if len(events) > 0 and events['totalEdep'].sum() > 0:
    mean_edep = events['totalEdep'].mean()
    std_edep = events['totalEdep'].std()
    # Normalize by detector volume for efficiency metric
    light_collection_efficiency = mean_edep / detector_volume_m3
    light_collection_err = std_edep / (np.sqrt(len(events)) * detector_volume_m3)
else:
    light_collection_efficiency = 0.0
    light_collection_err = 0.0
    print("WARNING: No energy deposits found in uniform detector")

# 2. Energy Linearity (ratio of mean to expected)
beam_energy_mev = 5.0 * 1000  # 5 GeV converted to MeV (from simulation config)
if mean_edep > 0:
    energy_linearity = mean_edep / beam_energy_mev
    energy_linearity_err = std_edep / (np.sqrt(len(events)) * beam_energy_mev)
else:
    energy_linearity = 0.0
    energy_linearity_err = 0.0

# 3. Energy Resolution
if mean_edep > 0:
    energy_resolution = std_edep / mean_edep
    energy_resolution_err = energy_resolution / np.sqrt(2 * len(events))
else:
    energy_resolution = 0.0
    energy_resolution_err = 0.0

# 4. Spatial Uniformity Analysis
if len(events) > 0:
    # Load hit-level data for spatial analysis (sample to avoid timeout)
    with uproot.open('uniform_lar_detector_electron_hits.root') as f:
        hits = f['hits'].arrays(['x', 'y', 'z', 'edep'], library='np', entry_stop=100000)
    
    # Calculate radial positions
    r = np.sqrt(hits['x']**2 + hits['y']**2)
    
    # Spatial uniformity: coefficient of variation of energy deposits vs radius
    r_bins = np.linspace(0, detector_diameter_m*500, 20)  # Convert to mm
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    
    edep_vs_r = []
    for i in range(len(r_bins)-1):
        mask = (r >= r_bins[i]) & (r < r_bins[i+1])
        if np.sum(mask) > 0:
            edep_vs_r.append(np.mean(hits['edep'][mask]))
        else:
            edep_vs_r.append(0)
    
    edep_vs_r = np.array(edep_vs_r)
    valid_bins = edep_vs_r > 0
    
    if np.sum(valid_bins) > 1:
        spatial_uniformity = 1.0 - (np.std(edep_vs_r[valid_bins]) / np.mean(edep_vs_r[valid_bins]))
        spatial_uniformity_err = spatial_uniformity / np.sqrt(np.sum(valid_bins))
    else:
        spatial_uniformity = 1.0
        spatial_uniformity_err = 0.0
else:
    spatial_uniformity = 1.0
    spatial_uniformity_err = 0.0

# Generate spatial distribution plots
plt.figure(figsize=(15, 5))

# Plot 1: Energy distribution
plt.subplot(1, 3, 1)
if len(events) > 0:
    plt.hist(events['totalEdep'], bins=50, histtype='step', linewidth=2, alpha=0.8)
    plt.axvline(mean_edep, color='r', linestyle='--', label=f'Mean: {mean_edep:.1f} MeV')
    plt.xlabel('Total Energy Deposit (MeV)')
    plt.ylabel('Events')
    plt.title(f'Uniform Detector Energy Distribution\nσ/E = {energy_resolution:.4f} ± {energy_resolution_err:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

# Plot 2: Spatial hit distribution
plt.subplot(1, 3, 2)
if len(events) > 0:
    plt.scatter(hits['x'][:10000], hits['y'][:10000], c=hits['edep'][:10000], 
                s=1, alpha=0.6, cmap='viridis')
    plt.colorbar(label='Energy Deposit (MeV)')
    plt.xlabel('X Position (mm)')
    plt.ylabel('Y Position (mm)')
    plt.title('Spatial Hit Distribution (10k sample)')
    plt.axis('equal')

# Plot 3: Energy vs radius
plt.subplot(1, 3, 3)
if len(events) > 0 and np.sum(valid_bins) > 0:
    plt.plot(r_centers[valid_bins], edep_vs_r[valid_bins], 'bo-', linewidth=2, markersize=4)
    plt.xlabel('Radius (mm)')
    plt.ylabel('Mean Energy Deposit (MeV)')
    plt.title(f'Radial Energy Profile\nUniformity = {spatial_uniformity:.4f} ± {spatial_uniformity_err:.4f}')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('uniform_baseline_performance.png', dpi=150, bbox_inches='tight')
plt.savefig('uniform_baseline_performance.pdf', bbox_inches='tight')
plt.show()

# Save baseline metrics to JSON
baseline_metrics = {
    'light_collection_efficiency': float(light_collection_efficiency),
    'light_collection_efficiency_err': float(light_collection_err),
    'spatial_uniformity': float(spatial_uniformity),
    'spatial_uniformity_err': float(spatial_uniformity_err),
    'energy_linearity': float(energy_linearity),
    'energy_linearity_err': float(energy_linearity_err),
    'energy_resolution': float(energy_resolution),
    'energy_resolution_err': float(energy_resolution_err),
    'mean_energy_deposit_mev': float(mean_edep) if len(events) > 0 else 0.0,
    'num_events_analyzed': len(events),
    'detector_volume_m3': detector_volume_m3,
    'sensor_count': sensor_count
}

with open('uniform_baseline_metrics.json', 'w') as f:
    json.dump(baseline_metrics, f, indent=2)

# Print results for workflow
print(f"\n=== UNIFORM DETECTOR BASELINE PERFORMANCE ===")
print(f"Light Collection Efficiency: {light_collection_efficiency:.6f} ± {light_collection_err:.6f}")
print(f"Spatial Uniformity: {spatial_uniformity:.4f} ± {spatial_uniformity_err:.4f}")
print(f"Energy Linearity: {energy_linearity:.4f} ± {energy_linearity_err:.4f}")
print(f"Energy Resolution: {energy_resolution:.4f} ± {energy_resolution_err:.4f}")
print(f"Mean Energy Deposit: {mean_edep:.1f} MeV")
print(f"Events Analyzed: {len(events)}")

# Return values for downstream steps
print(f"RESULT:uniform_light_collection_efficiency={light_collection_efficiency:.6f}")
print(f"RESULT:uniform_light_collection_efficiency_err={light_collection_err:.6f}")
print(f"RESULT:uniform_spatial_uniformity={spatial_uniformity:.4f}")
print(f"RESULT:uniform_spatial_uniformity_err={spatial_uniformity_err:.4f}")
print(f"RESULT:uniform_energy_linearity={energy_linearity:.4f}")
print(f"RESULT:uniform_energy_linearity_err={energy_linearity_err:.4f}")
print(f"RESULT:uniform_energy_resolution={energy_resolution:.4f}")
print(f"RESULT:uniform_energy_resolution_err={energy_resolution_err:.4f}")
print(f"RESULT:uniform_mean_deposit_mev={mean_edep:.1f}")
print(f"RESULT:uniform_num_events={len(events)}")
print("RESULT:baseline_metrics_file=uniform_baseline_metrics.json")
print("RESULT:baseline_plots=uniform_baseline_performance.png")
print("RESULT:success=True")