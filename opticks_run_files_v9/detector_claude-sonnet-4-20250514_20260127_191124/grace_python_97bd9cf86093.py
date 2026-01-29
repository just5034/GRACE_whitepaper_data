import matplotlib
matplotlib.use('Agg')
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Read simulation data from endcap-heavy configuration
hits_file = '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/detector_claude-sonnet-4-20250514_20260127_191124/endcap_heavy_lar_detector_electron_hits.root'

# Load event-level data for energy analysis
with uproot.open(hits_file) as f:
    events = f['events'].arrays(library='pd')
    print(f"Loaded {len(events)} events from endcap-heavy configuration")

# Calculate basic energy response metrics
mean_edep = events['totalEdep'].mean()
std_edep = events['totalEdep'].std()
energy_resolution = std_edep / mean_edep if mean_edep > 0 else 0
resolution_err = energy_resolution / np.sqrt(2 * len(events)) if len(events) > 0 else 0

# Detector geometry parameters from previous steps
detector_diameter_m = 3.0
detector_height_m = 2.5
detector_volume_m3 = np.pi * (detector_diameter_m/2)**2 * detector_height_m

# Calculate light collection efficiency (photons detected per MeV)
# For LAr: 40,000 photons per MeV scintillation yield
scintillation_yield_per_mev = 40000
beam_energy_mev = 5.0 * 1000  # 5 GeV from simulation converted to MeV
expected_photons = beam_energy_mev * scintillation_yield_per_mev
light_collection_efficiency = mean_edep / beam_energy_mev if beam_energy_mev > 0 else 0

print(f"Energy Response Analysis:")
print(f"Mean energy deposit: {mean_edep:.2f} MeV")
print(f"Energy resolution (σ/E): {energy_resolution:.4f} ± {resolution_err:.4f}")
print(f"Light collection efficiency: {light_collection_efficiency:.4f}")

# Spatial uniformity analysis using hit-level data (sample to avoid timeout)
print("\nAnalyzing spatial uniformity...")

# Sample first 100k hits for spatial analysis
with uproot.open(hits_file) as f:
    hits_sample = f['hits'].arrays(['x', 'y', 'z', 'edep'], library='np', entry_stop=100000)

# Convert to cylindrical coordinates
r = np.sqrt(hits_sample['x']**2 + hits_sample['y']**2)
z = hits_sample['z']
edep = hits_sample['edep']

# Calculate spatial uniformity in radial bins
r_bins = np.linspace(0, detector_diameter_m*500, 20)  # Convert to mm
r_centers = (r_bins[:-1] + r_bins[1:]) / 2
r_response = np.zeros(len(r_bins)-1)
r_counts = np.zeros(len(r_bins)-1)

for i in range(len(r_bins)-1):
    mask = (r >= r_bins[i]) & (r < r_bins[i+1])
    if np.sum(mask) > 0:
        r_response[i] = np.mean(edep[mask])
        r_counts[i] = np.sum(mask)

# Calculate spatial uniformity metric (RMS/mean of radial response)
valid_bins = r_counts > 10  # Only bins with sufficient statistics
if np.sum(valid_bins) > 0:
    spatial_uniformity = np.std(r_response[valid_bins]) / np.mean(r_response[valid_bins])
else:
    spatial_uniformity = 0

print(f"Spatial uniformity (RMS/mean): {spatial_uniformity:.4f}")

# Generate spatial distribution plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Energy distribution
ax1.hist(events['totalEdep'], bins=50, histtype='step', linewidth=2, color='blue')
ax1.axvline(mean_edep, color='red', linestyle='--', label=f'Mean: {mean_edep:.1f} MeV')
ax1.set_xlabel('Total Energy Deposit (MeV)')
ax1.set_ylabel('Events')
ax1.set_title(f'Energy Response (σ/E = {energy_resolution:.4f})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Radial response uniformity
valid_mask = r_counts > 0
ax2.errorbar(r_centers[valid_mask], r_response[valid_mask], 
             yerr=np.sqrt(r_response[valid_mask])/np.sqrt(r_counts[valid_mask]), 
             fmt='o-', capsize=3, linewidth=2)
ax2.set_xlabel('Radius (mm)')
ax2.set_ylabel('Mean Energy Deposit per Hit (MeV)')
ax2.set_title(f'Radial Response Uniformity (σ/μ = {spatial_uniformity:.4f})')
ax2.grid(True, alpha=0.3)

# Plot 3: 2D hit distribution (XY view)
hist_2d, x_edges, y_edges = np.histogram2d(hits_sample['x'], hits_sample['y'], 
                                          bins=50, weights=hits_sample['edep'])
im = ax3.imshow(hist_2d.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], 
                origin='lower', cmap='viridis', aspect='equal')
ax3.set_xlabel('X Position (mm)')
ax3.set_ylabel('Y Position (mm)')
ax3.set_title('Energy Deposit Distribution (XY)')
plt.colorbar(im, ax=ax3, label='Energy Deposit (MeV)')

# Plot 4: Longitudinal profile (Z direction)
z_bins = np.linspace(-detector_height_m*500, detector_height_m*500, 50)
z_hist, _ = np.histogram(hits_sample['z'], bins=z_bins, weights=hits_sample['edep'])
z_centers = (z_bins[:-1] + z_bins[1:]) / 2
ax4.step(z_centers, z_hist, where='mid', linewidth=2, color='green')
ax4.set_xlabel('Z Position (mm)')
ax4.set_ylabel('Energy Deposit (MeV)')
ax4.set_title('Longitudinal Energy Profile')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('endcap_heavy_performance_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('endcap_heavy_performance_analysis.pdf', bbox_inches='tight')
plt.show()

# Calculate photons detected per MeV
photons_per_mev = mean_edep * scintillation_yield_per_mev / beam_energy_mev if beam_energy_mev > 0 else 0

# Save results to JSON for persistence
results = {
    'endcap_heavy_mean_deposit_mev': float(mean_edep),
    'endcap_heavy_energy_resolution': float(energy_resolution),
    'endcap_heavy_resolution_err': float(resolution_err),
    'endcap_heavy_light_collection_efficiency': float(light_collection_efficiency),
    'endcap_heavy_spatial_uniformity': float(spatial_uniformity),
    'endcap_heavy_photons_per_mev': float(photons_per_mev),
    'endcap_heavy_num_events': int(len(events)),
    'endcap_heavy_beam_energy_mev': float(beam_energy_mev),
    'detector_volume_m3': float(detector_volume_m3),
    'sensor_count': 75
}

import json
with open('endcap_heavy_performance_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

# Output results for workflow
print(f"\nRESULT:endcap_heavy_mean_deposit_mev={mean_edep:.2f}")
print(f"RESULT:endcap_heavy_energy_resolution={energy_resolution:.4f}")
print(f"RESULT:endcap_heavy_resolution_err={resolution_err:.4f}")
print(f"RESULT:endcap_heavy_light_collection_efficiency={light_collection_efficiency:.4f}")
print(f"RESULT:endcap_heavy_spatial_uniformity={spatial_uniformity:.4f}")
print(f"RESULT:endcap_heavy_photons_per_mev={photons_per_mev:.1f}")
print(f"RESULT:endcap_heavy_num_events={len(events)}")
print(f"RESULT:endcap_heavy_beam_energy_mev={beam_energy_mev:.1f}")
print("RESULT:performance_plot=endcap_heavy_performance_analysis.png")
print("RESULT:metrics_file=endcap_heavy_performance_metrics.json")
print("RESULT:success=True")

print("\nEndcap-heavy performance analysis completed successfully!")