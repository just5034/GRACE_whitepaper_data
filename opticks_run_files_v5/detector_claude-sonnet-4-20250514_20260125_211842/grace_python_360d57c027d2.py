import matplotlib
matplotlib.use('Agg')
import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load baseline simulation data
with uproot.open('/u/jhill5/grace/work/benchmarks/opticks_20260125_211816/detector_claude-sonnet-4-20250514_20260125_211842/baseline_uniform_electron_hits.root') as f:
    # Load events data for efficiency calculations
    events = f['events'].arrays(library='pd')
    
    # Sample hits data to avoid memory issues (first 100k hits)
    hits = f['hits'].arrays(['x', 'y', 'z', 'edep', 'eventID'], library='pd', entry_stop=100000)

# Calculate light collection efficiency
# For optical detectors, this is ratio of detected photons to generated photons
total_events = len(events)
detected_events = len(events[events['nHits'] > 0]) if 'nHits' in events.columns else len(events[events['totalEdep'] > 0])
light_collection_efficiency = detected_events / total_events if total_events > 0 else 0
efficiency_err = np.sqrt(light_collection_efficiency * (1 - light_collection_efficiency) / total_events) if total_events > 0 else 0

print(f'Total events simulated: {total_events}')
print(f'Events with detected hits: {detected_events}')
print(f'Light collection efficiency: {light_collection_efficiency:.4f} ± {efficiency_err:.4f}')

# Calculate spatial uniformity from hit positions
# Analyze radial distribution of hits for cylindrical detector
r_hits = np.sqrt(hits['x']**2 + hits['y']**2)
z_hits = hits['z']

# Radial uniformity analysis
r_bins = np.linspace(0, 500, 20)  # 0 to 500mm in 20 bins
r_hist, _ = np.histogram(r_hits, bins=r_bins, weights=hits['edep'])
r_centers = (r_bins[:-1] + r_bins[1:]) / 2

# Calculate spatial uniformity as coefficient of variation
r_uniformity = np.std(r_hist) / np.mean(r_hist) if np.mean(r_hist) > 0 else 0

# Z uniformity analysis
z_bins = np.linspace(-500, 500, 20)  # -500 to 500mm in 20 bins
z_hist, _ = np.histogram(z_hits, bins=z_bins, weights=hits['edep'])
z_centers = (z_bins[:-1] + z_bins[1:]) / 2
z_uniformity = np.std(z_hist) / np.mean(z_hist) if np.mean(z_hist) > 0 else 0

# Overall spatial uniformity (RMS of both)
spatial_uniformity = np.sqrt(r_uniformity**2 + z_uniformity**2)

print(f'Radial uniformity (CV): {r_uniformity:.4f}')
print(f'Z uniformity (CV): {z_uniformity:.4f}')
print(f'Overall spatial uniformity: {spatial_uniformity:.4f}')

# Plot raw photon hit distributions
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# 1. Energy deposit distribution
if 'totalEdep' in events.columns:
    ax1.hist(events['totalEdep'], bins=50, histtype='step', linewidth=2, color='blue')
    ax1.set_xlabel('Total Energy Deposit (MeV)')
    ax1.set_ylabel('Events')
    ax1.set_title('Raw Energy Distribution')
    ax1.grid(True, alpha=0.3)

# 2. Radial hit distribution
ax2.hist(r_hits, bins=30, histtype='step', linewidth=2, color='red', weights=hits['edep'])
ax2.set_xlabel('Radial Position (mm)')
ax2.set_ylabel('Energy-weighted Hits')
ax2.set_title('Radial Hit Distribution')
ax2.grid(True, alpha=0.3)

# 3. Z hit distribution
ax3.hist(z_hits, bins=30, histtype='step', linewidth=2, color='green', weights=hits['edep'])
ax3.set_xlabel('Z Position (mm)')
ax3.set_ylabel('Energy-weighted Hits')
ax3.set_title('Longitudinal Hit Distribution')
ax3.grid(True, alpha=0.3)

# 4. 2D hit map (X-Y projection)
ax4.scatter(hits['x'], hits['y'], c=hits['edep'], s=1, alpha=0.6, cmap='viridis')
ax4.set_xlabel('X Position (mm)')
ax4.set_ylabel('Y Position (mm)')
ax4.set_title('Hit Map (X-Y Projection)')
ax4.set_aspect('equal')

plt.tight_layout()
plt.savefig('baseline_photon_distributions.png', dpi=150, bbox_inches='tight')
plt.savefig('baseline_photon_distributions.pdf', bbox_inches='tight')
plt.show()

# Create spatial uniformity plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Radial uniformity
ax1.step(r_centers, r_hist, where='mid', linewidth=2, color='red')
ax1.set_xlabel('Radius (mm)')
ax1.set_ylabel('Energy-weighted Hits')
ax1.set_title(f'Radial Uniformity (CV = {r_uniformity:.3f})')
ax1.grid(True, alpha=0.3)

# Z uniformity
ax2.step(z_centers, z_hist, where='mid', linewidth=2, color='green')
ax2.set_xlabel('Z Position (mm)')
ax2.set_ylabel('Energy-weighted Hits')
ax2.set_title(f'Z Uniformity (CV = {z_uniformity:.3f})')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('baseline_spatial_uniformity.png', dpi=150, bbox_inches='tight')
plt.savefig('baseline_spatial_uniformity.pdf', bbox_inches='tight')
plt.show()

# Output results for downstream workflow steps
print(f'RESULT:light_collection_efficiency={light_collection_efficiency:.4f}')
print(f'RESULT:light_collection_efficiency_err={efficiency_err:.4f}')
print(f'RESULT:spatial_uniformity={spatial_uniformity:.4f}')
print(f'RESULT:radial_uniformity={r_uniformity:.4f}')
print(f'RESULT:z_uniformity={z_uniformity:.4f}')
print('RESULT:photon_distributions_plot=baseline_photon_distributions.png')
print('RESULT:spatial_uniformity_plot=baseline_spatial_uniformity.png')

# Save detailed results to JSON for persistence
results = {
    'light_collection_efficiency': float(light_collection_efficiency),
    'light_collection_efficiency_err': float(efficiency_err),
    'spatial_uniformity': float(spatial_uniformity),
    'radial_uniformity': float(r_uniformity),
    'z_uniformity': float(z_uniformity),
    'total_events': int(total_events),
    'detected_events': int(detected_events),
    'analysis_type': 'optical_efficiency',
    'detector_config': 'baseline_uniform'
}

import json
with open('baseline_performance_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Analysis complete. Results saved to baseline_performance_metrics.json')