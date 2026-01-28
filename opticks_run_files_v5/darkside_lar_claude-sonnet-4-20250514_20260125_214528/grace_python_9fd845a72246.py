import matplotlib
matplotlib.use('Agg')
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Load simulation data from baseline detector
hits_file = 'baseline_geometry_electron_hits.root'
events_parquet = 'baseline_geometry_electron_events.parquet'
hits_parquet = 'baseline_geometry_electron_hits_data.parquet'

print('Loading simulation data...')

# Load events data (energy deposits and photon counts)
if Path(events_parquet).exists():
    events = pd.read_parquet(events_parquet)
    print(f'Loaded {len(events)} events from parquet')
else:
    with uproot.open(hits_file) as f:
        events = f['events'].arrays(library='pd')
    print(f'Loaded {len(events)} events from ROOT')

print(f'Event columns: {list(events.columns)}')
print(f'Events summary: mean_E={events["totalEdep"].mean():.3f} MeV, std_E={events["totalEdep"].std():.3f} MeV')

# 1. ENERGY RESOLUTION ANALYSIS
mean_E = events['totalEdep'].mean()
std_E = events['totalEdep'].std()
energy_resolution = std_E / mean_E if mean_E > 0 else 0
energy_resolution_err = energy_resolution / np.sqrt(2 * len(events)) if len(events) > 1 else 0

print(f'Energy Resolution: {energy_resolution:.4f} ± {energy_resolution_err:.4f}')

# Plot energy distribution
plt.figure(figsize=(10, 6))
plt.hist(events['totalEdep'], bins=50, histtype='step', linewidth=2, alpha=0.8)
plt.axvline(mean_E, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_E:.3f} MeV')
plt.xlabel('Total Energy Deposit (MeV)')
plt.ylabel('Events')
plt.title(f'Energy Distribution (σ/E = {energy_resolution:.4f} ± {energy_resolution_err:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('baseline_energy_distribution.png', dpi=150, bbox_inches='tight')
plt.savefig('baseline_energy_distribution.pdf', bbox_inches='tight')
plt.close()

# 2. LIGHT YIELD ANALYSIS (if optical photons available)
light_yield_pe_kev = 0
if 'nPhotons' in events.columns:
    # Calculate light yield in PE/keV
    photons_per_mev = events['nPhotons'].mean() / mean_E if mean_E > 0 else 0
    light_yield_pe_kev = photons_per_mev / 1000  # Convert MeV to keV
    
    print(f'Light Yield: {light_yield_pe_kev:.2f} PE/keV')
    
    # Plot photon distribution
    plt.figure(figsize=(10, 6))
    plt.hist(events['nPhotons'], bins=50, histtype='step', linewidth=2, alpha=0.8)
    plt.xlabel('Number of Photons')
    plt.ylabel('Events')
    plt.title(f'Optical Photon Distribution (Mean: {events["nPhotons"].mean():.0f} photons)')
    plt.grid(True, alpha=0.3)
    plt.savefig('baseline_photon_distribution.png', dpi=150, bbox_inches='tight')
    plt.savefig('baseline_photon_distribution.pdf', bbox_inches='tight')
    plt.close()
else:
    print('No optical photon data found - detector may not have optical physics enabled')

# 3. SIGNAL CHARACTERISTICS - PMT Response Analysis
# Load hit-level data for spatial analysis (sample to avoid timeout)
print('Analyzing spatial hit distribution...')

if Path(hits_parquet).exists():
    # Use parquet for faster loading
    hits = pd.read_parquet(hits_parquet)
    if len(hits) > 100000:
        hits = hits.sample(n=100000, random_state=42)  # Sample for performance
    print(f'Loaded {len(hits)} hits from parquet')
else:
    # Use ROOT with sampling
    with uproot.open(hits_file) as f:
        hits = f['hits'].arrays(['x', 'y', 'z', 'edep'], library='pd', entry_stop=100000)
    print(f'Loaded {len(hits)} hits from ROOT')

# Calculate radial distribution (cylindrical detector)
r = np.sqrt(hits['x']**2 + hits['y']**2)

# Radial energy deposit profile
radius_bins = np.linspace(0, r.max(), 30)
r_centers = (radius_bins[:-1] + radius_bins[1:]) / 2
r_hist, _ = np.histogram(r, bins=radius_bins, weights=hits['edep'])

plt.figure(figsize=(10, 6))
plt.step(r_centers, r_hist, where='mid', linewidth=2)
plt.xlabel('Radius (mm)')
plt.ylabel('Energy Deposit (MeV)')
plt.title('Radial Energy Deposit Profile')
plt.grid(True, alpha=0.3)
plt.savefig('baseline_radial_profile.png', dpi=150, bbox_inches='tight')
plt.savefig('baseline_radial_profile.pdf', bbox_inches='tight')
plt.close()

# 4. DETECTOR PERFORMANCE METRICS
total_edep = hits['edep'].sum()
containment_radii = [50, 100, 150, 200]  # mm
containment_fractions = []

for r_cut in containment_radii:
    contained_edep = hits[r <= r_cut]['edep'].sum()
    fraction = contained_edep / total_edep if total_edep > 0 else 0
    containment_fractions.append(fraction)

# Find 90% containment radius
containment_90 = 0
for i, frac in enumerate(containment_fractions):
    if frac >= 0.9:
        containment_90 = containment_radii[i]
        break
if containment_90 == 0:
    containment_90 = containment_radii[-1]  # Use max if 90% not reached

# Summary metrics
metrics = {
    'energy_resolution': float(energy_resolution),
    'energy_resolution_error': float(energy_resolution_err),
    'mean_energy_deposit_mev': float(mean_E),
    'light_yield_pe_per_kev': float(light_yield_pe_kev),
    'containment_90_radius_mm': float(containment_90),
    'total_events': int(len(events)),
    'total_hits': int(len(hits))
}

# Save metrics to JSON
with open('baseline_performance_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Print results for workflow
print('\n=== BASELINE DETECTOR PERFORMANCE ===')
print(f'Energy Resolution: {energy_resolution:.4f} ± {energy_resolution_err:.4f}')
print(f'Mean Energy Deposit: {mean_E:.3f} MeV')
if light_yield_pe_kev > 0:
    print(f'Light Yield: {light_yield_pe_kev:.2f} PE/keV')
print(f'90% Containment Radius: {containment_90:.1f} mm')
print(f'Total Events Analyzed: {len(events)}')

# Return values for downstream steps
print(f'RESULT:energy_resolution={energy_resolution:.4f}')
print(f'RESULT:energy_resolution_error={energy_resolution_err:.4f}')
print(f'RESULT:mean_energy_mev={mean_E:.3f}')
print(f'RESULT:light_yield_pe_kev={light_yield_pe_kev:.2f}')
print(f'RESULT:containment_90_mm={containment_90:.1f}')
print('RESULT:energy_plot=baseline_energy_distribution.png')
print('RESULT:radial_plot=baseline_radial_profile.png')
print('RESULT:metrics_file=baseline_performance_metrics.json')

print('\nBaseline performance analysis completed successfully!')