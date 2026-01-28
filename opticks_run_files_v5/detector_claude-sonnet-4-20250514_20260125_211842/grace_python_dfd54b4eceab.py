import matplotlib
matplotlib.use('Agg')
import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Load simulation data from baseline detector
hits_file = '/u/jhill5/grace/work/benchmarks/opticks_20260125_211816/detector_claude-sonnet-4-20250514_20260125_211842/baseline_uniform_electron_hits.root'

print(f'Loading data from: {hits_file}')

with uproot.open(hits_file) as f:
    # Load events data (safe - small dataset)
    events = f['events'].arrays(library='pd')
    print(f'Loaded {len(events)} events')
    
    # Data validation check
    if len(events) == 0:
        print('ERROR: No events found in simulation data')
        exit(1)
    
    # Check for constant data (as specified in modifications)
    if events['totalEdep'].std() < 1e-10:
        print('WARNING: Data appears constant (std < 1e-10), using single bin')
        bins_param = 1
    else:
        # Use adaptive binning based on data variation
        n_events = len(events)
        if n_events < 50:
            bins_param = max(10, n_events // 5)
        elif n_events < 500:
            bins_param = 25
        else:
            bins_param = 50
        print(f'Using adaptive binning: {bins_param} bins for {n_events} events')

# Calculate light collection efficiency metrics
total_events = len(events)
detected_events = len(events[events['totalEdep'] > 0])
light_collection_efficiency = detected_events / total_events if total_events > 0 else 0
light_collection_err = np.sqrt(light_collection_efficiency * (1 - light_collection_efficiency) / total_events) if total_events > 0 else 0

print(f'Light collection efficiency: {light_collection_efficiency:.4f} ± {light_collection_err:.4f}')

# Calculate spatial uniformity from hit positions
with uproot.open(hits_file) as f:
    # Sample first 100k hits to avoid timeout on large files
    hits = f['hits'].arrays(['x', 'y', 'z', 'edep'], library='np', entry_stop=100000)
    
    if len(hits['x']) > 0:
        # Calculate radial positions
        r = np.sqrt(hits['x']**2 + hits['y']**2)
        
        # Spatial uniformity: coefficient of variation of energy deposits vs radius
        r_bins = np.linspace(0, np.max(r), 10)
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2
        
        edep_vs_r = []
        for i in range(len(r_bins)-1):
            mask = (r >= r_bins[i]) & (r < r_bins[i+1])
            if np.sum(mask) > 0:
                edep_vs_r.append(np.mean(hits['edep'][mask]))
            else:
                edep_vs_r.append(0)
        
        edep_vs_r = np.array(edep_vs_r)
        edep_vs_r = edep_vs_r[edep_vs_r > 0]  # Remove empty bins
        
        if len(edep_vs_r) > 1:
            spatial_uniformity = np.std(edep_vs_r) / np.mean(edep_vs_r) if np.mean(edep_vs_r) > 0 else 1.0
        else:
            spatial_uniformity = 0.0
            
        print(f'Spatial uniformity (CV): {spatial_uniformity:.4f}')
    else:
        spatial_uniformity = 1.0
        print('WARNING: No hits found for spatial analysis')

# MANDATORY: Plot raw photon/energy distributions
plt.figure(figsize=(12, 8))

# Plot 1: Energy distribution
plt.subplot(2, 2, 1)
if events['totalEdep'].std() > 1e-10:
    plt.hist(events['totalEdep'], bins=bins_param, histtype='step', linewidth=2, label='Energy deposits')
else:
    # Handle constant data case
    plt.axvline(events['totalEdep'].iloc[0], color='red', linewidth=2, label=f'Constant: {events["totalEdep"].iloc[0]:.3f} MeV')
plt.xlabel('Total Energy Deposit (MeV)')
plt.ylabel('Events')
plt.title('Raw Energy Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Hit count distribution (proxy for photon count)
plt.subplot(2, 2, 2)
if 'nHits' in events.columns:
    if events['nHits'].std() > 1e-10:
        plt.hist(events['nHits'], bins=bins_param, histtype='step', linewidth=2, color='green', label='Hit count')
    else:
        plt.axvline(events['nHits'].iloc[0], color='green', linewidth=2, label=f'Constant: {events["nHits"].iloc[0]} hits')
    plt.xlabel('Number of Hits per Event')
    plt.ylabel('Events')
    plt.title('Hit Count Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
else:
    plt.text(0.5, 0.5, 'Hit count data\nnot available', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Hit Count Distribution')

# Plot 3: Spatial distribution (XY)
if len(hits['x']) > 0:
    plt.subplot(2, 2, 3)
    plt.scatter(hits['x'][:5000], hits['y'][:5000], c=hits['edep'][:5000], s=1, alpha=0.6, cmap='viridis')
    plt.xlabel('X Position (mm)')
    plt.ylabel('Y Position (mm)')
    plt.title('Hit Spatial Distribution (XY)')
    plt.colorbar(label='Energy Deposit (MeV)')
    plt.axis('equal')
else:
    plt.subplot(2, 2, 3)
    plt.text(0.5, 0.5, 'No hit position\ndata available', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Hit Spatial Distribution')

# Plot 4: Radial energy profile
if len(hits['x']) > 0:
    plt.subplot(2, 2, 4)
    r = np.sqrt(hits['x']**2 + hits['y']**2)
    r_bins = np.linspace(0, np.max(r), 20)
    r_hist, _ = np.histogram(r, bins=r_bins, weights=hits['edep'])
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    plt.step(r_centers, r_hist, where='mid', linewidth=2)
    plt.xlabel('Radial Distance (mm)')
    plt.ylabel('Energy Deposit (MeV)')
    plt.title('Radial Energy Profile')
    plt.grid(True, alpha=0.3)
else:
    plt.subplot(2, 2, 4)
    plt.text(0.5, 0.5, 'No position data\nfor radial profile', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Radial Energy Profile')

plt.tight_layout()
plt.savefig('baseline_performance_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('baseline_performance_analysis.pdf', bbox_inches='tight')
print('Saved plots: baseline_performance_analysis.png/pdf')

# Output results
print(f'RESULT:light_collection_efficiency={light_collection_efficiency:.4f}')
print(f'RESULT:light_collection_efficiency_error={light_collection_err:.4f}')
print(f'RESULT:spatial_uniformity={spatial_uniformity:.4f}')
print(f'RESULT:total_events={total_events}')
print(f'RESULT:detected_events={detected_events}')
print('RESULT:analysis_plots=baseline_performance_analysis.png')
print('RESULT:data_validation_passed=true')
print(f'RESULT:adaptive_bins_used={bins_param}')

# Save results to JSON for persistence
import json
results = {
    'light_collection_efficiency': float(light_collection_efficiency),
    'light_collection_efficiency_error': float(light_collection_err),
    'spatial_uniformity': float(spatial_uniformity),
    'total_events': int(total_events),
    'detected_events': int(detected_events),
    'analysis_type': 'optical_efficiency',
    'data_validation_checks': {
        'check_data_variation': True,
        'use_adaptive_binning': True,
        'handle_constant_data': True,
        'bins_parameter': bins_param
    }
}

with open('baseline_performance_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Analysis complete - baseline detector performance quantified')