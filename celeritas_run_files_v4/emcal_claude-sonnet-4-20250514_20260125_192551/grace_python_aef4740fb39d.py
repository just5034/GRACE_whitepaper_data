import matplotlib
matplotlib.use('Agg')
import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json

# Configuration from optimal design (Accordion CsI)
optimal_config = 'Accordion (CsI)'
hits_file = '/u/jhill5/grace/work/benchmarks/celeritas_20260125_192529/emcal_claude-sonnet-4-20250514_20260125_192551/accordion_csi_calorimeter_electron_hits.root'

print(f'Generating final performance plots for optimal configuration: {optimal_config}')
print(f'Using data file: {hits_file}')

# Create figure with subplots for comprehensive final plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'Final Performance Summary: {optimal_config}', fontsize=16, fontweight='bold')

# Load and process data with array conversions
with uproot.open(hits_file) as f:
    # Events data for resolution and linearity
    events = f['events'].arrays(library='pd')
    
    # Convert to numpy arrays as specified in modifications
    total_edep = np.array(events['totalEdep'])
    event_ids = np.array(events['eventID'])
    
    print(f'Loaded {len(events)} events')
    
    # 1. Final Resolution Curve
    mean_E = np.mean(total_edep)
    std_E = np.std(total_edep)
    resolution = std_E / mean_E if mean_E > 0 else 0
    resolution_err = resolution / np.sqrt(2 * len(total_edep)) if len(total_edep) > 0 else 0
    
    ax1.hist(total_edep, bins=50, histtype='step', linewidth=2, color='blue', alpha=0.8)
    ax1.axvline(mean_E, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_E:.1f} MeV')
    ax1.fill_between([mean_E-std_E, mean_E+std_E], 0, ax1.get_ylim()[1], alpha=0.2, color='red', label=f'±1σ: {std_E:.1f} MeV')
    ax1.set_xlabel('Total Energy Deposit (MeV)', fontsize=12)
    ax1.set_ylabel('Events', fontsize=12)
    ax1.set_title(f'Energy Resolution: σ/E = {resolution:.4f} ± {resolution_err:.4f}', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Energy Linearity (using known 100 GeV input)
    true_energy = 100000  # 100 GeV in MeV
    linearity = mean_E / true_energy if true_energy > 0 else 0
    linearity_err = std_E / (true_energy * np.sqrt(len(total_edep))) if true_energy > 0 and len(total_edep) > 0 else 0
    
    ax2.scatter([true_energy], [mean_E], s=100, color='blue', zorder=5)
    ax2.errorbar([true_energy], [mean_E], yerr=[std_E], fmt='o', color='blue', capsize=5, capthick=2)
    ax2.plot([0, true_energy*1.1], [0, true_energy*1.1], 'k--', alpha=0.5, label='Perfect Linearity')
    ax2.set_xlabel('True Energy (MeV)', fontsize=12)
    ax2.set_ylabel('Reconstructed Energy (MeV)', fontsize=12)
    ax2.set_title(f'Energy Linearity: {linearity:.4f} ± {linearity_err:.4f}', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Shower Profiles (longitudinal and lateral)
    # Process hits data in chunks with array conversions
    z_bins = np.linspace(1400, 1800, 50)  # Accordion geometry range
    z_centers = (z_bins[:-1] + z_bins[1:]) / 2
    z_hist = np.zeros(len(z_bins)-1)
    
    r_bins = np.linspace(0, 300, 50)  # mm
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    r_hist = np.zeros(len(r_bins)-1)
    
    total_hits_processed = 0
    for batch in f['hits'].iterate(['x', 'y', 'z', 'edep'], step_size='50 MB'):
        # Convert awkward arrays to numpy as specified
        x_arr = np.array(batch['x'])
        y_arr = np.array(batch['y'])
        z_arr = np.array(batch['z'])
        edep_arr = np.array(batch['edep'])  # weights conversion
        
        # Fix histogram accumulation with numpy arrays
        z_hist += np.histogram(z_arr, bins=z_bins, weights=edep_arr)[0]
        
        r_arr = np.sqrt(x_arr**2 + y_arr**2)
        r_hist += np.histogram(r_arr, bins=r_bins, weights=edep_arr)[0]
        
        total_hits_processed += len(x_arr)
    
    print(f'Processed {total_hits_processed} hits for shower profiles')
    
    # Normalize shower profiles
    z_hist = z_hist / np.sum(z_hist) if np.sum(z_hist) > 0 else z_hist
    r_hist = r_hist / np.sum(r_hist) if np.sum(r_hist) > 0 else r_hist
    
    ax3.step(z_centers, z_hist, where='mid', linewidth=2, color='green', label='Longitudinal Profile')
    ax3.set_xlabel('Z Position (mm)', fontsize=12)
    ax3.set_ylabel('Normalized Energy Fraction', fontsize=12)
    ax3.set_title('Longitudinal Shower Profile', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Containment Efficiency
    # Calculate containment from processed data
    total_energy_in_r = np.cumsum(r_hist) * np.sum(r_hist) if np.sum(r_hist) > 0 else np.zeros_like(r_hist)
    containment_fractions = total_energy_in_r / total_energy_in_r[-1] if total_energy_in_r[-1] > 0 else np.zeros_like(total_energy_in_r)
    
    ax4.plot(r_centers, containment_fractions * 100, linewidth=2, color='purple', label='Radial Containment')
    ax4.axhline(90, color='red', linestyle='--', linewidth=2, label='90% Containment')
    ax4.axhline(95, color='orange', linestyle='--', linewidth=2, label='95% Containment')
    ax4.set_xlabel('Radius (mm)', fontsize=12)
    ax4.set_ylabel('Containment (%)', fontsize=12)
    ax4.set_title('Radial Containment Efficiency', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 105)
    
    # Find containment radii
    idx_90 = np.where(containment_fractions >= 0.9)[0]
    idx_95 = np.where(containment_fractions >= 0.95)[0]
    r90 = r_centers[idx_90[0]] if len(idx_90) > 0 else r_centers[-1]
    r95 = r_centers[idx_95[0]] if len(idx_95) > 0 else r_centers[-1]

plt.tight_layout()
plt.savefig('final_performance_comprehensive.png', dpi=300, bbox_inches='tight')
plt.savefig('final_performance_comprehensive.pdf', bbox_inches='tight')
print('RESULT:final_performance_plot=final_performance_comprehensive.png')

# Generate individual high-quality plots
# Resolution curve
plt.figure(figsize=(10, 8))
plt.hist(total_edep, bins=60, histtype='step', linewidth=3, color='navy', alpha=0.8, label='Energy Distribution')
plt.axvline(mean_E, color='crimson', linestyle='--', linewidth=3, label=f'Mean: {mean_E:.1f} MeV')
plt.fill_between([mean_E-std_E, mean_E+std_E], 0, plt.gca().get_ylim()[1], alpha=0.3, color='crimson', label=f'±1σ: {std_E:.1f} MeV')
plt.xlabel('Total Energy Deposit (MeV)', fontsize=14, fontweight='bold')
plt.ylabel('Events', fontsize=14, fontweight='bold')
plt.title(f'Final Energy Resolution: {optimal_config}\nσ/E = {resolution:.4f} ± {resolution_err:.4f}', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('final_resolution_curve.png', dpi=300, bbox_inches='tight')
plt.savefig('final_resolution_curve.pdf', bbox_inches='tight')
print('RESULT:resolution_curve_plot=final_resolution_curve.png')

# Save comprehensive metrics
final_metrics = {
    'optimal_configuration': optimal_config,
    'energy_resolution': float(resolution),
    'energy_resolution_error': float(resolution_err),
    'energy_linearity': float(linearity),
    'energy_linearity_error': float(linearity_err),
    'containment_90_radius_mm': float(r90),
    'containment_95_radius_mm': float(r95),
    'mean_energy_deposit_mev': float(mean_E),
    'energy_deposit_std_mev': float(std_E),
    'total_events_analyzed': int(len(total_edep)),
    'total_hits_processed': int(total_hits_processed)
}

with open('final_performance_metrics.json', 'w') as f:
    json.dump(final_metrics, f, indent=2)

print('\n=== FINAL PERFORMANCE SUMMARY ===')
print(f'Optimal Configuration: {optimal_config}')
print(f'Energy Resolution: {resolution:.4f} ± {resolution_err:.4f}')
print(f'Energy Linearity: {linearity:.4f} ± {linearity_err:.4f}')
print(f'90% Containment Radius: {r90:.1f} mm')
print(f'95% Containment Radius: {r95:.1f} mm')
print('\nFinal performance plots generated successfully!')

print('RESULT:success=True')
print(f'RESULT:energy_resolution={resolution:.6f}')
print(f'RESULT:linearity={linearity:.6f}')
print(f'RESULT:containment_90_radius={r90:.1f}')
print(f'RESULT:containment_95_radius={r95:.1f}')
print('RESULT:metrics_file=final_performance_metrics.json')
print('RESULT:plots_created=4')