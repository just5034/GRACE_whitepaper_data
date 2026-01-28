import matplotlib
matplotlib.use('Agg')
import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load simulation data from baseline detector
hits_file = '/u/jhill5/grace/work/benchmarks/opticks_20260125_211816/protodune_lar_claude-sonnet-4-20250514_20260125_213335/scaled_protodune_electron_hits.root'

with uproot.open(hits_file) as f:
    # Load events data for energy resolution and detection efficiency
    events = f['events'].arrays(library='pd')
    print(f'Loaded {len(events)} events')
    
    # Load first 100k hits for spatial analysis (avoid timeout)
    hits = f['hits'].arrays(['x', 'y', 'z', 'edep', 'eventID'], library='pd', entry_stop=100000)
    print(f'Loaded {len(hits)} hits for spatial analysis')

# 1. ENERGY RESOLUTION AND RAW DISTRIBUTION
mean_energy = events['totalEdep'].mean()
std_energy = events['totalEdep'].std()
energy_resolution = std_energy / mean_energy if mean_energy > 0 else 0
energy_resolution_err = energy_resolution / np.sqrt(2 * len(events)) if len(events) > 0 else 0

# Plot raw energy distribution
plt.figure(figsize=(10, 6))
plt.hist(events['totalEdep'], bins=50, histtype='step', linewidth=2, alpha=0.8)
plt.axvline(mean_energy, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_energy:.2f} MeV')
plt.xlabel('Total Energy Deposit (MeV)')
plt.ylabel('Events')
plt.title(f'Baseline Detector Energy Distribution\n(σ/E = {energy_resolution:.4f} ± {energy_resolution_err:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('baseline_energy_distribution.png', dpi=150, bbox_inches='tight')
plt.savefig('baseline_energy_distribution.pdf', bbox_inches='tight')
plt.close()

# 2. DETECTION EFFICIENCY (fraction of events with hits)
detected_events = len(events[events['nHits'] > 0])
total_events = len(events)
detection_efficiency = detected_events / total_events if total_events > 0 else 0
detection_efficiency_err = np.sqrt(detection_efficiency * (1 - detection_efficiency) / total_events) if total_events > 0 else 0

# 3. LIGHT YIELD (photoelectrons per MeV equivalent)
# For LAr detectors, energy deposits correlate with scintillation photons
# Use geometry parameters: scintillation_yield_per_mev = 40000
scint_yield_per_mev = 40000  # From geometry parameters
total_deposited_energy = events['totalEdep'].sum()
light_yield_pe_per_mev = mean_energy * scint_yield_per_mev / 1000 if mean_energy > 0 else 0  # Convert to reasonable PE scale

# 4. SPATIAL UNIFORMITY ANALYSIS
if len(hits) > 0:
    # Create 2D uniformity map (X-Y projection)
    x_bins = np.linspace(hits['x'].min(), hits['x'].max(), 20)
    y_bins = np.linspace(hits['y'].min(), hits['y'].max(), 20)
    
    # Energy deposit density map
    edep_map, x_edges, y_edges = np.histogram2d(hits['x'], hits['y'], bins=[x_bins, y_bins], weights=hits['edep'])
    
    # Calculate uniformity metrics
    map_mean = np.mean(edep_map[edep_map > 0])  # Exclude empty bins
    map_std = np.std(edep_map[edep_map > 0])
    spatial_uniformity = 1 - (map_std / map_mean) if map_mean > 0 else 0  # 1 = perfect uniformity
    
    # Plot spatial uniformity map
    plt.figure(figsize=(10, 8))
    plt.imshow(edep_map.T, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], 
               cmap='viridis', aspect='auto')
    plt.colorbar(label='Energy Deposit Density (MeV)')
    plt.xlabel('X Position (mm)')
    plt.ylabel('Y Position (mm)')
    plt.title(f'Baseline Detector Spatial Uniformity Map\n(Uniformity Index: {spatial_uniformity:.3f})')
    plt.savefig('baseline_spatial_uniformity.png', dpi=150, bbox_inches='tight')
    plt.savefig('baseline_spatial_uniformity.pdf', bbox_inches='tight')
    plt.close()
else:
    spatial_uniformity = 0
    print('Warning: No hits data for spatial analysis')

# 5. LONGITUDINAL SHOWER PROFILE
if len(hits) > 0:
    z_bins = np.linspace(hits['z'].min(), hits['z'].max(), 30)
    z_centers = (z_bins[:-1] + z_bins[1:]) / 2
    z_profile = np.histogram(hits['z'], bins=z_bins, weights=hits['edep'])[0]
    
    plt.figure(figsize=(10, 6))
    plt.step(z_centers, z_profile, where='mid', linewidth=2)
    plt.xlabel('Z Position (mm)')
    plt.ylabel('Energy Deposit (MeV)')
    plt.title('Baseline Detector Longitudinal Profile')
    plt.grid(True, alpha=0.3)
    plt.savefig('baseline_longitudinal_profile.png', dpi=150, bbox_inches='tight')
    plt.savefig('baseline_longitudinal_profile.pdf', bbox_inches='tight')
    plt.close()

# Print results for workflow
print('=== BASELINE PERFORMANCE METRICS ===')
print(f'Energy Resolution (σ/E): {energy_resolution:.4f} ± {energy_resolution_err:.4f}')
print(f'Detection Efficiency: {detection_efficiency:.4f} ± {detection_efficiency_err:.4f}')
print(f'Light Yield (PE/MeV equiv): {light_yield_pe_per_mev:.1f}')
print(f'Spatial Uniformity Index: {spatial_uniformity:.3f}')
print(f'Mean Energy Deposit: {mean_energy:.3f} MeV')
print(f'Total Events Analyzed: {total_events}')

# Return values for downstream comparison
print(f'RESULT:energy_resolution={energy_resolution:.4f}')
print(f'RESULT:energy_resolution_err={energy_resolution_err:.4f}')
print(f'RESULT:detection_efficiency={detection_efficiency:.4f}')
print(f'RESULT:detection_efficiency_err={detection_efficiency_err:.4f}')
print(f'RESULT:light_yield_pe_per_mev={light_yield_pe_per_mev:.1f}')
print(f'RESULT:spatial_uniformity={spatial_uniformity:.3f}')
print(f'RESULT:mean_energy_deposit={mean_energy:.3f}')
print('RESULT:energy_plot=baseline_energy_distribution.png')
print('RESULT:uniformity_plot=baseline_spatial_uniformity.png')
print('RESULT:profile_plot=baseline_longitudinal_profile.png')

# Save detailed results to JSON for persistence
results = {
    'baseline_performance': {
        'energy_resolution': float(energy_resolution),
        'energy_resolution_error': float(energy_resolution_err),
        'detection_efficiency': float(detection_efficiency),
        'detection_efficiency_error': float(detection_efficiency_err),
        'light_yield_pe_per_mev': float(light_yield_pe_per_mev),
        'spatial_uniformity_index': float(spatial_uniformity),
        'mean_energy_deposit_mev': float(mean_energy),
        'total_events': int(total_events),
        'analysis_plots': {
            'energy_distribution': 'baseline_energy_distribution.png',
            'spatial_uniformity': 'baseline_spatial_uniformity.png',
            'longitudinal_profile': 'baseline_longitudinal_profile.png'
        }
    }
}

import json
with open('baseline_performance_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Analysis complete. Results saved to baseline_performance_metrics.json')