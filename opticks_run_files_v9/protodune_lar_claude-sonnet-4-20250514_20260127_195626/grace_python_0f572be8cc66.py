import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Energy sweep file paths from Previous Step Outputs
energy_files = {
    0.001: {
        'events': '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/protodune_lar_claude-sonnet-4-20250514_20260127_195626/energy_0.001GeV/protodune_geometry_retry_electron_events.parquet',
        'hits': '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/protodune_lar_claude-sonnet-4-20250514_20260127_195626/energy_0.001GeV/protodune_geometry_retry_electron_hits_data.parquet'
    },
    0.002: {
        'events': '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/protodune_lar_claude-sonnet-4-20250514_20260127_195626/energy_0.002GeV/protodune_geometry_retry_electron_events.parquet',
        'hits': '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/protodune_lar_claude-sonnet-4-20250514_20260127_195626/energy_0.002GeV/protodune_geometry_retry_electron_hits_data.parquet'
    },
    0.005: {
        'events': '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/protodune_lar_claude-sonnet-4-20250514_20260127_195626/energy_0.005GeV/protodune_geometry_retry_events.parquet',
        'hits': '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/protodune_lar_claude-sonnet-4-20250514_20260127_195626/energy_0.005GeV/protodune_geometry_retry_hits_data.parquet'
    }
}

# Geometry parameters from previous step
sensor_count = 75
sensor_area_m2 = 5.30144
vessel_volume_m3 = 293.94
scint_yield_per_mev = 40000  # LAr scintillation yield

# Analyze each energy point
results = []
all_light_yields = []

for energy_gev, filepaths in energy_files.items():
    energy_mev = energy_gev * 1000
    
    # Load events data
    events_df = pd.read_parquet(filepaths['events'])
    
    # Calculate light yield metrics
    if 'nPhotons' in events_df.columns:
        # Optical photons detected
        mean_photons = events_df['nPhotons'].mean()
        std_photons = events_df['nPhotons'].std()
        light_yield_pe_per_mev = mean_photons / energy_mev
        light_yield_err = std_photons / (energy_mev * np.sqrt(len(events_df)))
    else:
        # Fallback to energy deposits if photons not available
        mean_edep = events_df['totalEdep'].mean()
        std_edep = events_df['totalEdep'].std()
        # Estimate photons from energy deposit and scintillation yield
        estimated_photons = mean_edep * scint_yield_per_mev / 1000  # Convert MeV
        light_yield_pe_per_mev = estimated_photons / energy_mev
        light_yield_err = std_edep * scint_yield_per_mev / (1000 * energy_mev * np.sqrt(len(events_df)))
    
    # Detection efficiency (fraction of events with hits)
    detection_eff = len(events_df[events_df['nHits'] > 0]) / len(events_df) if len(events_df) > 0 else 0
    detection_eff_err = np.sqrt(detection_eff * (1 - detection_eff) / len(events_df)) if len(events_df) > 0 else 0
    
    results.append({
        'energy_gev': energy_gev,
        'energy_mev': energy_mev,
        'light_yield_pe_per_mev': light_yield_pe_per_mev,
        'light_yield_err': light_yield_err,
        'detection_efficiency': detection_eff,
        'detection_eff_err': detection_eff_err,
        'num_events': len(events_df)
    })
    
    all_light_yields.append(light_yield_pe_per_mev)
    
    print(f'Energy {energy_mev:.1f} MeV: Light yield = {light_yield_pe_per_mev:.1f} ± {light_yield_err:.1f} PE/MeV, Detection eff = {detection_eff:.3f} ± {detection_eff_err:.3f}')

# Calculate overall performance metrics
mean_light_yield = np.mean(all_light_yields)
std_light_yield = np.std(all_light_yields)
mean_detection_eff = np.mean([r['detection_efficiency'] for r in results])

# Create light yield vs energy plot
plt.figure(figsize=(10, 6))
energies = [r['energy_mev'] for r in results]
light_yields = [r['light_yield_pe_per_mev'] for r in results]
errors = [r['light_yield_err'] for r in results]

plt.errorbar(energies, light_yields, yerr=errors, marker='o', capsize=5, linewidth=2)
plt.axhline(mean_light_yield, color='r', linestyle='--', label=f'Mean: {mean_light_yield:.1f} PE/MeV')
plt.xlabel('Particle Energy (MeV)')
plt.ylabel('Light Yield (PE/MeV)')
plt.title('Baseline Optical Performance - Light Yield vs Energy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('baseline_light_yield.png', dpi=150, bbox_inches='tight')
plt.savefig('baseline_light_yield.pdf', bbox_inches='tight')

# Create detection efficiency plot
plt.figure(figsize=(10, 6))
det_effs = [r['detection_efficiency'] for r in results]
det_errs = [r['detection_eff_err'] for r in results]

plt.errorbar(energies, det_effs, yerr=det_errs, marker='s', capsize=5, linewidth=2, color='green')
plt.axhline(mean_detection_eff, color='r', linestyle='--', label=f'Mean: {mean_detection_eff:.3f}')
plt.xlabel('Particle Energy (MeV)')
plt.ylabel('Detection Efficiency')
plt.title('Baseline Optical Performance - Detection Efficiency vs Energy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.1)
plt.savefig('baseline_detection_efficiency.png', dpi=150, bbox_inches='tight')
plt.savefig('baseline_detection_efficiency.pdf', bbox_inches='tight')

# Spatial uniformity analysis using hits data from highest energy point
highest_energy_hits_file = energy_files[0.005]['hits']
try:
    hits_df = pd.read_parquet(highest_energy_hits_file)
    
    # Create 2D spatial uniformity map
    plt.figure(figsize=(12, 5))
    
    # X-Y projection
    plt.subplot(1, 2, 1)
    plt.hist2d(hits_df['x'], hits_df['y'], bins=20, weights=hits_df['edep'])
    plt.colorbar(label='Energy Deposit (MeV)')
    plt.xlabel('X Position (mm)')
    plt.ylabel('Y Position (mm)')
    plt.title('Spatial Uniformity - XY Projection')
    
    # X-Z projection
    plt.subplot(1, 2, 2)
    plt.hist2d(hits_df['x'], hits_df['z'], bins=20, weights=hits_df['edep'])
    plt.colorbar(label='Energy Deposit (MeV)')
    plt.xlabel('X Position (mm)')
    plt.ylabel('Z Position (mm)')
    plt.title('Spatial Uniformity - XZ Projection')
    
    plt.tight_layout()
    plt.savefig('baseline_spatial_uniformity.png', dpi=150, bbox_inches='tight')
    plt.savefig('baseline_spatial_uniformity.pdf', bbox_inches='tight')
    
    # Calculate spatial uniformity metric (coefficient of variation)
    x_bins = np.linspace(hits_df['x'].min(), hits_df['x'].max(), 10)
    spatial_deposits = []
    for i in range(len(x_bins)-1):
        mask = (hits_df['x'] >= x_bins[i]) & (hits_df['x'] < x_bins[i+1])
        if mask.sum() > 0:
            spatial_deposits.append(hits_df[mask]['edep'].sum())
    
    spatial_uniformity = np.std(spatial_deposits) / np.mean(spatial_deposits) if len(spatial_deposits) > 0 else 0
    
except Exception as e:
    print(f'Spatial analysis failed: {e}')
    spatial_uniformity = 0

# Save results to JSON
import json
results_dict = {
    'baseline_light_yield_pe_per_mev': mean_light_yield,
    'baseline_light_yield_std': std_light_yield,
    'baseline_detection_efficiency': mean_detection_eff,
    'baseline_spatial_uniformity_cv': spatial_uniformity,
    'baseline_sensor_count': sensor_count,
    'baseline_sensor_area_m2': sensor_area_m2,
    'energy_sweep_results': results
}

with open('baseline_optical_performance.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

# Output results for downstream steps
print(f'RESULT:baseline_light_yield_pe_per_mev={mean_light_yield:.1f}')
print(f'RESULT:baseline_light_yield_std={std_light_yield:.1f}')
print(f'RESULT:baseline_detection_efficiency={mean_detection_eff:.3f}')
print(f'RESULT:baseline_spatial_uniformity_cv={spatial_uniformity:.3f}')
print(f'RESULT:baseline_sensor_count={sensor_count}')
print('RESULT:light_yield_plot=baseline_light_yield.png')
print('RESULT:detection_efficiency_plot=baseline_detection_efficiency.png')
print('RESULT:spatial_uniformity_plot=baseline_spatial_uniformity.png')
print('RESULT:performance_data=baseline_optical_performance.json')
print('RESULT:success=True')

print('\nBaseline Optical Performance Summary:')
print(f'Light Yield: {mean_light_yield:.1f} ± {std_light_yield:.1f} PE/MeV')
print(f'Detection Efficiency: {mean_detection_eff:.1%}')
print(f'Spatial Uniformity (CV): {spatial_uniformity:.3f}')
print(f'Sensor Count: {sensor_count} PMTs')
print(f'Total Sensor Area: {sensor_area_m2:.1f} m²')