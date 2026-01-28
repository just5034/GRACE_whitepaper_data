import matplotlib
matplotlib.use('Agg')
import uproot
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# File path from previous step
hits_file = '/u/jhill5/grace/work/benchmarks/celeritas_20260125_192529/emcal_claude-sonnet-4-20250514_20260125_192551/baseline_calorimeter_electron_hits.root'

print(f'Analyzing baseline performance from: {hits_file}')

# Check if file exists
if not Path(hits_file).exists():
    print(f'ERROR: File not found: {hits_file}')
    exit(1)

with uproot.open(hits_file) as f:
    print('Available trees:', f.keys())
    
    # Load events data for energy resolution
    events = f['events'].arrays(library='pd')
    print(f'Loaded {len(events)} events')
    print(f'Event columns: {list(events.columns)}')
    
    # 1. ENERGY RESOLUTION ANALYSIS
    mean_E = events['totalEdep'].mean()
    std_E = events['totalEdep'].std()
    resolution = std_E / mean_E if mean_E > 0 else 0
    resolution_err = resolution / np.sqrt(2 * len(events)) if len(events) > 1 else 0
    
    print(f'Mean energy: {mean_E:.3f} MeV')
    print(f'Energy std: {std_E:.3f} MeV')
    print(f'Energy resolution (σ/E): {resolution:.4f} ± {resolution_err:.4f}')
    
    # Plot energy distribution
    plt.figure(figsize=(10, 6))
    plt.hist(events['totalEdep'], bins=50, histtype='step', linewidth=2, alpha=0.8)
    plt.axvline(mean_E, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_E:.2f} MeV')
    plt.axvline(mean_E + std_E, color='r', linestyle=':', alpha=0.7, label=f'±1σ: {std_E:.2f} MeV')
    plt.axvline(mean_E - std_E, color='r', linestyle=':', alpha=0.7)
    plt.xlabel('Total Energy Deposit (MeV)')
    plt.ylabel('Events')
    plt.title(f'Baseline Energy Distribution\nResolution σ/E = {resolution:.4f} ± {resolution_err:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('baseline_energy_distribution.png', dpi=150, bbox_inches='tight')
    plt.savefig('baseline_energy_distribution.pdf', bbox_inches='tight')
    plt.close()
    
    # 2. LINEARITY ANALYSIS (single energy point - report as baseline)
    # For single energy, linearity is the ratio of reconstructed to true energy
    # Assuming 1 GeV true energy (default from simulation)
    true_energy_mev = 1000.0  # 1 GeV = 1000 MeV
    linearity = mean_E / true_energy_mev
    linearity_err = std_E / (true_energy_mev * np.sqrt(len(events)))
    
    print(f'Linearity (reco/true): {linearity:.4f} ± {linearity_err:.4f}')
    
    # 3. SHOWER CONTAINMENT ANALYSIS
    print('Analyzing shower containment...')
    
    # Sample hits to avoid timeout on large files
    max_hits = 500000
    hits = f['hits'].arrays(['x', 'y', 'z', 'edep'], library='np', entry_stop=max_hits)
    print(f'Loaded {len(hits["x"])} hits (max {max_hits})')
    
    # Calculate radial distance from beam axis (x=0, y=0)
    r = np.sqrt(hits['x']**2 + hits['y']**2)
    total_edep = np.sum(hits['edep'])
    
    # Containment vs radius
    radius_bins = np.linspace(0, 200, 50)  # 0 to 200 mm
    contained_energy = np.array([np.sum(hits['edep'][r <= r_cut]) for r_cut in radius_bins])
    containment_fractions = contained_energy / total_edep if total_edep > 0 else contained_energy
    
    # Find 90% and 95% containment radii
    idx_90 = np.where(containment_fractions >= 0.9)[0]
    idx_95 = np.where(containment_fractions >= 0.95)[0]
    r90 = radius_bins[idx_90[0]] if len(idx_90) > 0 else radius_bins[-1]
    r95 = radius_bins[idx_95[0]] if len(idx_95) > 0 else radius_bins[-1]
    
    print(f'90% containment radius: {r90:.1f} mm')
    print(f'95% containment radius: {r95:.1f} mm')
    
    # Plot containment
    plt.figure(figsize=(10, 6))
    plt.plot(radius_bins, containment_fractions * 100, linewidth=2, label='Radial containment')
    plt.axhline(90, color='r', linestyle='--', alpha=0.7, label=f'90% at r={r90:.1f} mm')
    plt.axhline(95, color='orange', linestyle='--', alpha=0.7, label=f'95% at r={r95:.1f} mm')
    plt.axvline(r90, color='r', linestyle=':', alpha=0.5)
    plt.axvline(r95, color='orange', linestyle=':', alpha=0.5)
    plt.xlabel('Radius (mm)')
    plt.ylabel('Energy Containment (%)')
    plt.title('Baseline Shower Containment')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 150)
    plt.ylim(0, 100)
    plt.savefig('baseline_containment.png', dpi=150, bbox_inches='tight')
    plt.savefig('baseline_containment.pdf', bbox_inches='tight')
    plt.close()
    
    # 4. LONGITUDINAL SHOWER PROFILE
    z_bins = np.linspace(hits['z'].min(), hits['z'].max(), 50)
    z_centers = (z_bins[:-1] + z_bins[1:]) / 2
    z_hist, _ = np.histogram(hits['z'], bins=z_bins, weights=hits['edep'])
    
    plt.figure(figsize=(10, 6))
    plt.step(z_centers, z_hist, where='mid', linewidth=2)
    plt.xlabel('Z Position (mm)')
    plt.ylabel('Energy Deposit (MeV)')
    plt.title('Baseline Longitudinal Shower Profile')
    plt.grid(True, alpha=0.3)
    plt.savefig('baseline_shower_profile.png', dpi=150, bbox_inches='tight')
    plt.savefig('baseline_shower_profile.pdf', bbox_inches='tight')
    plt.close()

# Save results to JSON
results = {
    'energy_resolution': float(resolution),
    'energy_resolution_error': float(resolution_err),
    'mean_energy_mev': float(mean_E),
    'energy_std_mev': float(std_E),
    'linearity': float(linearity),
    'linearity_error': float(linearity_err),
    'containment_90_radius_mm': float(r90),
    'containment_95_radius_mm': float(r95),
    'num_events': int(len(events)),
    'num_hits_analyzed': int(len(hits['x']))
}

with open('baseline_performance_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\n=== BASELINE PERFORMANCE SUMMARY ===')
print(f'Energy Resolution: {resolution:.4f} ± {resolution_err:.4f}')
print(f'Linearity: {linearity:.4f} ± {linearity_err:.4f}')
print(f'90% Containment: {r90:.1f} mm')
print(f'95% Containment: {r95:.1f} mm')

# Output results for downstream steps
print(f'RESULT:energy_resolution={resolution:.6f}')
print(f'RESULT:energy_resolution_error={resolution_err:.6f}')
print(f'RESULT:linearity={linearity:.6f}')
print(f'RESULT:linearity_error={linearity_err:.6f}')
print(f'RESULT:containment_90_radius={r90:.2f}')
print(f'RESULT:containment_95_radius={r95:.2f}')
print('RESULT:energy_plot=baseline_energy_distribution.png')
print('RESULT:containment_plot=baseline_containment.png')
print('RESULT:shower_profile_plot=baseline_shower_profile.png')
print('RESULT:metrics_file=baseline_performance_metrics.json')

print('\nBaseline performance analysis completed successfully!')