import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Load optimized detector hits data from validate_optimized_design step
hits_file = '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/darkside_lar_claude-sonnet-4-20250514_20260127_212110/energy_0.005GeV/darkside_optimized_hits_data.parquet'
df_hits = pd.read_parquet(hits_file)

# PMT geometry parameters from generate_optimized_geometry
pmt_count = 100
vessel_diameter = 3.6  # meters
vessel_height = 3.6    # meters
pmt_diameter = 0.2     # meters

print(f'Loaded {len(df_hits)} hits from {pmt_count} PMTs')
print(f'Detector: {vessel_diameter}m diameter x {vessel_height}m height cylinder')

# Group hits by event and PMT to get hit patterns
event_hits = df_hits.groupby(['eventID', 'pmtID']).agg({
    'x': 'first',
    'y': 'first', 
    'z': 'first',
    'edep': 'sum',
    'time': 'mean'
}).reset_index()

# Calculate event-level PMT hit patterns
event_summary = event_hits.groupby('eventID').agg({
    'pmtID': 'count',  # number of PMTs hit
    'edep': 'sum',     # total light collected
    'x': lambda x: np.sqrt(np.mean(x**2)),  # RMS x position
    'y': lambda x: np.sqrt(np.mean(x**2)),  # RMS y position
    'z': 'mean'        # mean z position
}).rename(columns={'pmtID': 'nPMTsHit', 'edep': 'totalLight'})

print(f'Analyzed {len(event_summary)} events')
print(f'Average PMTs hit per event: {event_summary["nPMTsHit"].mean():.1f}')

# Position reconstruction using center-of-light method
position_data = []
for event_id in event_summary.index[:1000]:  # Sample first 1000 events
    event_pmts = event_hits[event_hits['eventID'] == event_id]
    if len(event_pmts) < 3:  # Need minimum 3 PMTs for position reconstruction
        continue
    
    # Weight PMT positions by light collected
    weights = event_pmts['edep'].values
    if np.sum(weights) == 0:
        continue
        
    # Calculate center of light
    x_recon = np.average(event_pmts['x'], weights=weights)
    y_recon = np.average(event_pmts['y'], weights=weights)
    z_recon = np.average(event_pmts['z'], weights=weights)
    
    # Calculate radial position
    r_recon = np.sqrt(x_recon**2 + y_recon**2)
    
    position_data.append({
        'eventID': event_id,
        'x_recon': x_recon,
        'y_recon': y_recon,
        'z_recon': z_recon,
        'r_recon': r_recon,
        'nPMTs': len(event_pmts),
        'totalLight': np.sum(weights)
    })

pos_df = pd.DataFrame(position_data)
print(f'Position reconstruction successful for {len(pos_df)} events')

# Calculate spatial resolution metrics
x_resolution = pos_df['x_recon'].std()
y_resolution = pos_df['y_recon'].std()
z_resolution = pos_df['z_recon'].std()
r_resolution = pos_df['r_recon'].std()

# Position reconstruction bias (should be near zero for centered events)
x_bias = pos_df['x_recon'].mean()
y_bias = pos_df['y_recon'].mean()
z_bias = pos_df['z_recon'].mean()

# Spatial uniformity - check position dependence
radial_bins = np.linspace(0, vessel_diameter/2, 10)
radial_centers = (radial_bins[:-1] + radial_bins[1:]) / 2
radial_resolution = []

for i in range(len(radial_bins)-1):
    mask = (pos_df['r_recon'] >= radial_bins[i]) & (pos_df['r_recon'] < radial_bins[i+1])
    if np.sum(mask) > 10:  # Need sufficient statistics
        r_res = pos_df[mask]['r_recon'].std()
        radial_resolution.append(r_res)
    else:
        radial_resolution.append(np.nan)

# Create comprehensive position reconstruction plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: XY position reconstruction
ax1.scatter(pos_df['x_recon'], pos_df['y_recon'], alpha=0.6, s=20)
circle = plt.Circle((0, 0), vessel_diameter/2, fill=False, color='red', linestyle='--', label='Vessel boundary')
ax1.add_patch(circle)
ax1.set_xlabel('Reconstructed X (m)')
ax1.set_ylabel('Reconstructed Y (m)')
ax1.set_title(f'XY Position Reconstruction\n(σ_x = {x_resolution:.3f} m, σ_y = {y_resolution:.3f} m)')
ax1.set_aspect('equal')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Z position distribution
ax2.hist(pos_df['z_recon'], bins=30, alpha=0.7, edgecolor='black')
ax2.axvline(z_bias, color='red', linestyle='--', label=f'Mean: {z_bias:.3f} m')
ax2.set_xlabel('Reconstructed Z (m)')
ax2.set_ylabel('Events')
ax2.set_title(f'Z Position Reconstruction\n(σ_z = {z_resolution:.3f} m)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Radial resolution vs radius
valid_mask = ~np.isnan(radial_resolution)
ax3.plot(radial_centers[valid_mask], np.array(radial_resolution)[valid_mask], 'bo-', linewidth=2, markersize=6)
ax3.set_xlabel('Radial Position (m)')
ax3.set_ylabel('Radial Resolution (m)')
ax3.set_title('Spatial Uniformity\n(Resolution vs Radius)')
ax3.grid(True, alpha=0.3)

# Plot 4: PMT multiplicity vs light yield
ax4.scatter(pos_df['nPMTs'], pos_df['totalLight'], alpha=0.6, s=20)
ax4.set_xlabel('Number of PMTs Hit')
ax4.set_ylabel('Total Light Collected')
ax4.set_title('PMT Hit Pattern Analysis')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('position_reconstruction_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('position_reconstruction_analysis.pdf', bbox_inches='tight')
plt.show()

# Calculate overall position resolution
overall_resolution = np.sqrt(x_resolution**2 + y_resolution**2 + z_resolution**2)
radial_resolution_avg = np.nanmean(radial_resolution)

# Assess spatial uniformity
uniformity_metric = np.nanstd(radial_resolution) / np.nanmean(radial_resolution) if np.nanmean(radial_resolution) > 0 else 0

# Calculate fiducial volume efficiency
fiducial_radius = vessel_diameter/2 * 0.8  # 80% of vessel radius
fiducial_events = np.sum(pos_df['r_recon'] < fiducial_radius)
fiducial_efficiency = fiducial_events / len(pos_df)

print('\n=== POSITION RECONSTRUCTION RESULTS ===')
print(f'Overall 3D position resolution: {overall_resolution:.3f} m')
print(f'X resolution: {x_resolution:.3f} m')
print(f'Y resolution: {y_resolution:.3f} m')
print(f'Z resolution: {z_resolution:.3f} m')
print(f'Radial resolution (average): {radial_resolution_avg:.3f} m')
print(f'Position bias - X: {x_bias:.3f} m, Y: {y_bias:.3f} m, Z: {z_bias:.3f} m')
print(f'Spatial uniformity metric: {uniformity_metric:.3f} (lower is better)')
print(f'Fiducial volume efficiency (80% radius): {fiducial_efficiency:.3f}')

# Save results
results = {
    'overall_position_resolution_m': float(overall_resolution),
    'x_resolution_m': float(x_resolution),
    'y_resolution_m': float(y_resolution),
    'z_resolution_m': float(z_resolution),
    'radial_resolution_avg_m': float(radial_resolution_avg),
    'position_bias_x_m': float(x_bias),
    'position_bias_y_m': float(y_bias),
    'position_bias_z_m': float(z_bias),
    'spatial_uniformity_metric': float(uniformity_metric),
    'fiducial_efficiency': float(fiducial_efficiency),
    'events_analyzed': len(pos_df),
    'avg_pmts_per_event': float(event_summary['nPMTsHit'].mean()),
    'pmt_count': pmt_count,
    'vessel_diameter_m': vessel_diameter
}

with open('position_reconstruction_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Output results for workflow
print(f'RESULT:position_resolution_3d={overall_resolution:.4f}')
print(f'RESULT:position_resolution_x={x_resolution:.4f}')
print(f'RESULT:position_resolution_y={y_resolution:.4f}')
print(f'RESULT:position_resolution_z={z_resolution:.4f}')
print(f'RESULT:spatial_uniformity={uniformity_metric:.4f}')
print(f'RESULT:fiducial_efficiency={fiducial_efficiency:.4f}')
print(f'RESULT:position_bias_magnitude={np.sqrt(x_bias**2 + y_bias**2 + z_bias**2):.4f}')
print('RESULT:position_analysis_plot=position_reconstruction_analysis.png')
print('RESULT:analysis_results_file=position_reconstruction_results.json')
print('RESULT:success=True')