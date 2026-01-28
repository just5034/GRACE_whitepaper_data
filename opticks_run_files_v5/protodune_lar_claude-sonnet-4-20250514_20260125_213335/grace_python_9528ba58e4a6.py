import matplotlib
matplotlib.use('Agg')
import uproot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Set up publication-quality plotting
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (10, 8)
})

# Load simulation data
hits_file = '/u/jhill5/grace/work/benchmarks/opticks_20260125_211816/protodune_lar_claude-sonnet-4-20250514_20260125_213335/scaled_protodune_electron_hits.root'

# Check if files exist
if not Path(hits_file).exists():
    print(f'ERROR: Hits file not found: {hits_file}')
    exit(1)

# Load data from ROOT file
with uproot.open(hits_file) as f:
    # Load events data (safe - typically small)
    events = f['events'].arrays(library='pd')
    print(f'Loaded {len(events)} events')
    
    # Sample hits data to avoid memory issues (first 100k hits)
    hits = f['hits'].arrays(['x', 'y', 'z', 'edep', 'time'], library='pd', entry_stop=100000)
    print(f'Loaded {len(hits)} hits (sampled)')

# Use analysis results from previous step
energy_resolution = 3.1729
energy_resolution_err = 0.3173
detection_efficiency = 0.1
mean_energy_deposit = 0.017

# 1. Energy Distribution Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Raw energy distribution
ax1.hist(events['totalEdep'], bins=30, histtype='step', linewidth=2, color='blue', alpha=0.8)
mean_E = events['totalEdep'].mean()
std_E = events['totalEdep'].std()
ax1.axvline(mean_E, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_E:.3f} MeV')
ax1.set_xlabel('Total Energy Deposit (MeV)')
ax1.set_ylabel('Number of Events')
ax1.set_title(f'Baseline Energy Distribution\n(σ/E = {energy_resolution:.2f} ± {energy_resolution_err:.2f}%)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Hit energy distribution
ax2.hist(hits['edep'], bins=50, histtype='step', linewidth=2, color='green', alpha=0.8)
ax2.set_xlabel('Hit Energy Deposit (MeV)')
ax2.set_ylabel('Number of Hits')
ax2.set_title('Individual Hit Energy Distribution')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('baseline_energy_distributions.png', dpi=150, bbox_inches='tight')
plt.savefig('baseline_energy_distributions.pdf', bbox_inches='tight')
plt.close()

# 2. Spatial Hit Distribution Maps
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# XY projection (transverse)
h1 = ax1.hist2d(hits['x'], hits['y'], bins=40, weights=hits['edep'], cmap='viridis')
ax1.set_xlabel('X Position (mm)')
ax1.set_ylabel('Y Position (mm)')
ax1.set_title('Energy Deposit Map (XY Projection)')
ax1.set_aspect('equal')
plt.colorbar(h1[3], ax=ax1, label='Energy Deposit (MeV)')

# XZ projection (longitudinal)
h2 = ax2.hist2d(hits['x'], hits['z'], bins=40, weights=hits['edep'], cmap='plasma')
ax2.set_xlabel('X Position (mm)')
ax2.set_ylabel('Z Position (mm)')
ax2.set_title('Energy Deposit Map (XZ Projection)')
plt.colorbar(h2[3], ax=ax2, label='Energy Deposit (MeV)')

# YZ projection
h3 = ax3.hist2d(hits['y'], hits['z'], bins=40, weights=hits['edep'], cmap='inferno')
ax3.set_xlabel('Y Position (mm)')
ax3.set_ylabel('Z Position (mm)')
ax3.set_title('Energy Deposit Map (YZ Projection)')
plt.colorbar(h3[3], ax=ax3, label='Energy Deposit (MeV)')

# Hit count distribution (XY)
h4 = ax4.hist2d(hits['x'], hits['y'], bins=40, cmap='Blues')
ax4.set_xlabel('X Position (mm)')
ax4.set_ylabel('Y Position (mm)')
ax4.set_title('Hit Count Map (XY Projection)')
ax4.set_aspect('equal')
plt.colorbar(h4[3], ax=ax4, label='Number of Hits')

plt.tight_layout()
plt.savefig('baseline_spatial_maps.png', dpi=150, bbox_inches='tight')
plt.savefig('baseline_spatial_maps.pdf', bbox_inches='tight')
plt.close()

# 3. Light Collection Uniformity Analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Radial uniformity
r = np.sqrt(hits['x']**2 + hits['y']**2)
r_bins = np.linspace(0, r.max(), 20)
r_centers = (r_bins[:-1] + r_bins[1:]) / 2
r_edep = np.array([hits['edep'][((r >= r_bins[i]) & (r < r_bins[i+1]))].sum() for i in range(len(r_bins)-1)])
r_counts = np.array([len(hits['edep'][((r >= r_bins[i]) & (r < r_bins[i+1]))]) for i in range(len(r_bins)-1)])
r_avg_edep = np.divide(r_edep, r_counts, out=np.zeros_like(r_edep), where=r_counts!=0)

ax1.plot(r_centers, r_avg_edep, 'o-', linewidth=2, markersize=6, color='blue')
ax1.set_xlabel('Radial Distance (mm)')
ax1.set_ylabel('Average Energy per Hit (MeV)')
ax1.set_title('Radial Light Collection Uniformity')
ax1.grid(True, alpha=0.3)

# Calculate uniformity metric
if len(r_avg_edep[r_avg_edep > 0]) > 0:
    uniformity = r_avg_edep[r_avg_edep > 0].std() / r_avg_edep[r_avg_edep > 0].mean()
else:
    uniformity = 0.0

# Z uniformity
z_bins = np.linspace(hits['z'].min(), hits['z'].max(), 20)
z_centers = (z_bins[:-1] + z_bins[1:]) / 2
z_edep = np.array([hits['edep'][((hits['z'] >= z_bins[i]) & (hits['z'] < z_bins[i+1]))].sum() for i in range(len(z_bins)-1)])
z_counts = np.array([len(hits['edep'][((hits['z'] >= z_bins[i]) & (hits['z'] < z_bins[i+1]))]) for i in range(len(z_bins)-1)])
z_avg_edep = np.divide(z_edep, z_counts, out=np.zeros_like(z_edep), where=z_counts!=0)

ax2.plot(z_centers, z_avg_edep, 's-', linewidth=2, markersize=6, color='red')
ax2.set_xlabel('Z Position (mm)')
ax2.set_ylabel('Average Energy per Hit (MeV)')
ax2.set_title('Longitudinal Light Collection Profile')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('baseline_uniformity_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('baseline_uniformity_analysis.pdf', bbox_inches='tight')
plt.close()

# 4. Summary Performance Plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Energy resolution vs event number (stability)
event_batches = np.array_split(events['totalEdep'], 5)
batch_resolutions = [batch.std() / batch.mean() if batch.mean() > 0 else 0 for batch in event_batches]
batch_numbers = np.arange(1, len(batch_resolutions) + 1)

ax1.plot(batch_numbers, np.array(batch_resolutions) * 100, 'o-', linewidth=2, markersize=8)
ax1.axhline(energy_resolution, color='red', linestyle='--', label=f'Overall: {energy_resolution:.1f}%')
ax1.set_xlabel('Event Batch')
ax1.set_ylabel('Energy Resolution (%)')
ax1.set_title('Energy Resolution Stability')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Hit multiplicity
ax2.hist(events['nHits'], bins=20, histtype='step', linewidth=2, color='purple')
ax2.set_xlabel('Number of Hits per Event')
ax2.set_ylabel('Number of Events')
ax2.set_title(f'Hit Multiplicity\n(Mean: {events["nHits"].mean():.1f} hits/event)')
ax2.grid(True, alpha=0.3)

# Time distribution
ax3.hist(hits['time'], bins=30, histtype='step', linewidth=2, color='orange')
ax3.set_xlabel('Hit Time (ns)')
ax3.set_ylabel('Number of Hits')
ax3.set_title('Hit Time Distribution')
ax3.grid(True, alpha=0.3)

# Performance summary
metrics = ['Energy\nResolution\n(%)', 'Detection\nEfficiency\n(%)', 'Mean Energy\nDeposit\n(MeV)', 'Spatial\nUniformity']
values = [energy_resolution, detection_efficiency * 100, mean_energy_deposit * 1000, uniformity]
colors = ['blue', 'green', 'red', 'purple']

bars = ax4.bar(range(len(metrics)), values, color=colors, alpha=0.7)
ax4.set_xticks(range(len(metrics)))
ax4.set_xticklabels(metrics, rotation=45, ha='right')
ax4.set_ylabel('Value')
ax4.set_title('Baseline Detector Performance Summary')
ax4.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{value:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('baseline_performance_summary.png', dpi=150, bbox_inches='tight')
plt.savefig('baseline_performance_summary.pdf', bbox_inches='tight')
plt.close()

print('RESULT:energy_distribution_plot=baseline_energy_distributions.png')
print('RESULT:spatial_maps_plot=baseline_spatial_maps.png')
print('RESULT:uniformity_plot=baseline_uniformity_analysis.png')
print('RESULT:summary_plot=baseline_performance_summary.png')
print(f'RESULT:spatial_uniformity_calculated={uniformity:.4f}')
print(f'RESULT:mean_hits_per_event={events["nHits"].mean():.1f}')
print(f'RESULT:total_events_analyzed={len(events)}')
print(f'RESULT:total_hits_analyzed={len(hits)}')
print('SUCCESS: Generated publication-quality baseline detector plots with proper axes and units')