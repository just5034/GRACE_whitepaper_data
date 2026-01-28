import matplotlib
matplotlib.use('Agg')
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Load optimized detector data
print('Loading optimized detector simulation data...')
with uproot.open('optimized_geometry_electron_hits.root') as f:
    opt_events = f['events'].arrays(library='pd')
    # Sample hits to avoid timeout on large files
    opt_hits = f['hits'].arrays(['x', 'y', 'z', 'edep', 'eventID'], library='pd', entry_stop=100000)

# Load baseline detector data
print('Loading baseline detector simulation data...')
with uproot.open('baseline_geometry_electron_hits.root') as f:
    base_events = f['events'].arrays(library='pd')
    # Sample hits to avoid timeout on large files
    base_hits = f['hits'].arrays(['x', 'y', 'z', 'edep', 'eventID'], library='pd', entry_stop=100000)

print(f'Loaded {len(opt_events)} optimized events, {len(opt_hits)} optimized hits')
print(f'Loaded {len(base_events)} baseline events, {len(base_hits)} baseline hits')

# Calculate position reconstruction accuracy
def calculate_position_reconstruction(hits_df, events_df):
    """Calculate position reconstruction metrics"""
    # Group hits by event and calculate energy-weighted centroids
    event_positions = []
    
    for event_id in events_df['eventID']:
        event_hits = hits_df[hits_df['eventID'] == event_id]
        if len(event_hits) > 0:
            total_edep = event_hits['edep'].sum()
            if total_edep > 0:
                x_centroid = (event_hits['x'] * event_hits['edep']).sum() / total_edep
                y_centroid = (event_hits['y'] * event_hits['edep']).sum() / total_edep
                z_centroid = (event_hits['z'] * event_hits['edep']).sum() / total_edep
                
                # Calculate RMS spread around centroid
                x_rms = np.sqrt(((event_hits['x'] - x_centroid)**2 * event_hits['edep']).sum() / total_edep)
                y_rms = np.sqrt(((event_hits['y'] - y_centroid)**2 * event_hits['edep']).sum() / total_edep)
                z_rms = np.sqrt(((event_hits['z'] - z_centroid)**2 * event_hits['edep']).sum() / total_edep)
                
                event_positions.append({
                    'eventID': event_id,
                    'x_reco': x_centroid,
                    'y_reco': y_centroid, 
                    'z_reco': z_centroid,
                    'x_rms': x_rms,
                    'y_rms': y_rms,
                    'z_rms': z_rms,
                    'total_edep': total_edep
                })
    
    return pd.DataFrame(event_positions)

# Calculate position reconstruction for both detectors
print('Calculating position reconstruction for optimized detector...')
opt_positions = calculate_position_reconstruction(opt_hits, opt_events)

print('Calculating position reconstruction for baseline detector...')
base_positions = calculate_position_reconstruction(base_hits, base_events)

# Calculate position resolution metrics
opt_pos_res_x = opt_positions['x_rms'].mean()
opt_pos_res_y = opt_positions['y_rms'].mean() 
opt_pos_res_z = opt_positions['z_rms'].mean()
opt_pos_res_3d = np.sqrt(opt_pos_res_x**2 + opt_pos_res_y**2 + opt_pos_res_z**2)

base_pos_res_x = base_positions['x_rms'].mean()
base_pos_res_y = base_positions['y_rms'].mean()
base_pos_res_z = base_positions['z_rms'].mean() 
base_pos_res_3d = np.sqrt(base_pos_res_x**2 + base_pos_res_y**2 + base_pos_res_z**2)

print(f'Optimized position resolution: X={opt_pos_res_x:.2f}mm, Y={opt_pos_res_y:.2f}mm, Z={opt_pos_res_z:.2f}mm')
print(f'Baseline position resolution: X={base_pos_res_x:.2f}mm, Y={base_pos_res_y:.2f}mm, Z={base_pos_res_z:.2f}mm')

# Calculate improvement
pos_improvement = (base_pos_res_3d - opt_pos_res_3d) / base_pos_res_3d * 100
print(f'Position reconstruction improvement: {pos_improvement:.1f}%')

# Generate uniformity maps
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Position Reconstruction Uniformity Maps', fontsize=16)

# Baseline detector uniformity maps
axes[0,0].hist2d(base_positions['x_reco'], base_positions['y_reco'], bins=20, cmap='viridis')
axes[0,0].set_title('Baseline: XY Position Distribution')
axes[0,0].set_xlabel('X (mm)')
axes[0,0].set_ylabel('Y (mm)')

axes[0,1].hist2d(base_positions['x_reco'], base_positions['z_reco'], bins=20, cmap='viridis')
axes[0,1].set_title('Baseline: XZ Position Distribution')
axes[0,1].set_xlabel('X (mm)')
axes[0,1].set_ylabel('Z (mm)')

axes[0,2].hist2d(base_positions['x_reco'], base_positions['x_rms'], bins=20, cmap='plasma')
axes[0,2].set_title('Baseline: Position vs Resolution')
axes[0,2].set_xlabel('X Position (mm)')
axes[0,2].set_ylabel('X Resolution (mm)')

# Optimized detector uniformity maps
axes[1,0].hist2d(opt_positions['x_reco'], opt_positions['y_reco'], bins=20, cmap='viridis')
axes[1,0].set_title('Optimized: XY Position Distribution')
axes[1,0].set_xlabel('X (mm)')
axes[1,0].set_ylabel('Y (mm)')

axes[1,1].hist2d(opt_positions['x_reco'], opt_positions['z_reco'], bins=20, cmap='viridis')
axes[1,1].set_title('Optimized: XZ Position Distribution')
axes[1,1].set_xlabel('X (mm)')
axes[1,1].set_ylabel('Z (mm)')

axes[1,2].hist2d(opt_positions['x_reco'], opt_positions['x_rms'], bins=20, cmap='plasma')
axes[1,2].set_title('Optimized: Position vs Resolution')
axes[1,2].set_xlabel('X Position (mm)')
axes[1,2].set_ylabel('X Resolution (mm)')

plt.tight_layout()
plt.savefig('position_reconstruction_uniformity_maps.png', dpi=150, bbox_inches='tight')
plt.savefig('position_reconstruction_uniformity_maps.pdf', bbox_inches='tight')
plt.close()

# Generate position resolution comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Resolution comparison bar chart
resolutions = ['X', 'Y', 'Z', '3D']
base_res = [base_pos_res_x, base_pos_res_y, base_pos_res_z, base_pos_res_3d]
opt_res = [opt_pos_res_x, opt_pos_res_y, opt_pos_res_z, opt_pos_res_3d]

x_pos = np.arange(len(resolutions))
width = 0.35

ax1.bar(x_pos - width/2, base_res, width, label='Baseline', alpha=0.7, color='blue')
ax1.bar(x_pos + width/2, opt_res, width, label='Optimized', alpha=0.7, color='green')
ax1.set_xlabel('Coordinate')
ax1.set_ylabel('Position Resolution (mm)')
ax1.set_title('Position Resolution Comparison')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(resolutions)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Resolution vs energy deposit
ax2.scatter(base_positions['total_edep'], base_positions['x_rms'], alpha=0.5, label='Baseline', color='blue')
ax2.scatter(opt_positions['total_edep'], opt_positions['x_rms'], alpha=0.5, label='Optimized', color='green')
ax2.set_xlabel('Total Energy Deposit (MeV)')
ax2.set_ylabel('X Position Resolution (mm)')
ax2.set_title('Resolution vs Energy Deposit')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('position_reconstruction_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('position_reconstruction_comparison.pdf', bbox_inches='tight')
plt.close()

# Save results to JSON
results = {
    'baseline_position_resolution': {
        'x_mm': float(base_pos_res_x),
        'y_mm': float(base_pos_res_y),
        'z_mm': float(base_pos_res_z),
        '3d_mm': float(base_pos_res_3d)
    },
    'optimized_position_resolution': {
        'x_mm': float(opt_pos_res_x),
        'y_mm': float(opt_pos_res_y), 
        'z_mm': float(opt_pos_res_z),
        '3d_mm': float(opt_pos_res_3d)
    },
    'improvement_percent': float(pos_improvement),
    'num_events_analyzed': {
        'baseline': len(base_positions),
        'optimized': len(opt_positions)
    }
}

with open('position_reconstruction_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('RESULT:baseline_position_resolution_3d=' + str(base_pos_res_3d))
print('RESULT:optimized_position_resolution_3d=' + str(opt_pos_res_3d))
print('RESULT:position_improvement_percent=' + str(pos_improvement))
print('RESULT:uniformity_maps=position_reconstruction_uniformity_maps.png')
print('RESULT:comparison_plot=position_reconstruction_comparison.png')
print('RESULT:results_file=position_reconstruction_results.json')
print('Position reconstruction analysis completed successfully!')