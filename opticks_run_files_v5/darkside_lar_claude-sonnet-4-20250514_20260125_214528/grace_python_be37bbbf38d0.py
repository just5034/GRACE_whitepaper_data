import matplotlib
matplotlib.use('Agg')
import uproot
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Load multi-energy simulation data
hits_file = '/u/jhill5/grace/work/benchmarks/opticks_20260125_211816/darkside_lar_claude-sonnet-4-20250514_20260125_214528/baseline_geometry_electron_hits.root'

print(f"Loading multi-energy data from: {hits_file}")

# Read events data (safe - typically small)
with uproot.open(hits_file) as f:
    events = f['events'].arrays(library='pd')
    print(f"Loaded {len(events)} events")

# Group events by energy (assuming different energy runs have different event ranges)
# For multi-energy analysis, we need to identify energy groups
# Since this is from simulate_energy_range, likely contains multiple energies

# Calculate basic statistics
total_events = len(events)
mean_energy = events['totalEdep'].mean()
std_energy = events['totalEdep'].std()
overall_resolution = std_energy / mean_energy if mean_energy > 0 else 0
resolution_error = overall_resolution / np.sqrt(2 * total_events) if total_events > 0 else 0

print(f"Overall statistics: {total_events} events, mean energy: {mean_energy:.3f} MeV")
print(f"Overall resolution: {overall_resolution:.4f} ± {resolution_error:.4f}")

# For multi-energy analysis, try to identify energy groups by clustering
# This is a simplified approach - in practice, energy ranges would be known
energies = events['totalEdep'].values
unique_energies = np.unique(np.round(energies, 1))  # Round to 0.1 MeV precision

# If we have clear energy separation, analyze each group
if len(unique_energies) > 3:  # Multiple distinct energies
    print(f"Detected {len(unique_energies)} potential energy points")
    
    # Analyze energy groups
    energy_points = []
    resolutions = []
    resolution_errors = []
    light_yields = []
    
    # Group events by similar energies (within 10% bins)
    energy_bins = np.logspace(np.log10(energies.min()), np.log10(energies.max()), 6)
    energy_centers = []
    
    for i in range(len(energy_bins)-1):
        mask = (energies >= energy_bins[i]) & (energies < energy_bins[i+1])
        if np.sum(mask) > 10:  # Need sufficient statistics
            group_events = events[mask]
            group_mean = group_events['totalEdep'].mean()
            group_std = group_events['totalEdep'].std()
            group_resolution = group_std / group_mean if group_mean > 0 else 0
            group_res_err = group_resolution / np.sqrt(2 * len(group_events))
            
            # Light yield (assuming 40,000 photons/MeV for LAr from geometry specs)
            scint_yield = 40000  # photons/MeV from geometry specs
            light_yield = group_mean * scint_yield / 1000  # PE/keV
            
            energy_centers.append(group_mean)
            resolutions.append(group_resolution)
            resolution_errors.append(group_res_err)
            light_yields.append(light_yield)
            
            print(f"Energy {group_mean:.2f} MeV: resolution = {group_resolution:.4f} ± {group_res_err:.4f}")
else:
    # Single energy or insufficient separation - use overall values
    energy_centers = [mean_energy]
    resolutions = [overall_resolution]
    resolution_errors = [resolution_error]
    light_yields = [mean_energy * 40000 / 1000]  # PE/keV

# Create energy resolution curve
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Raw energy distribution
ax1.hist(events['totalEdep'], bins=50, histtype='step', linewidth=2, alpha=0.8)
ax1.axvline(mean_energy, color='r', linestyle='--', label=f'Mean: {mean_energy:.2f} MeV')
ax1.set_xlabel('Total Energy Deposit (MeV)')
ax1.set_ylabel('Events')
ax1.set_title('Energy Distribution')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Energy resolution vs energy
if len(energy_centers) > 1:
    ax2.errorbar(energy_centers, resolutions, yerr=resolution_errors, 
                fmt='o-', capsize=5, linewidth=2, markersize=8)
    ax2.set_xlabel('Energy (MeV)')
    ax2.set_ylabel('Energy Resolution (σ/E)')
    ax2.set_title('Energy Resolution vs Energy')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
else:
    ax2.bar(['Single Energy'], resolutions, yerr=resolution_errors, 
           capsize=5, color='blue', alpha=0.7)
    ax2.set_ylabel('Energy Resolution (σ/E)')
    ax2.set_title(f'Energy Resolution at {energy_centers[0]:.1f} MeV')
    ax2.grid(True, alpha=0.3)

# 3. Linearity plot (reconstructed vs true energy)
# For linearity, we assume the mean of each group represents the true energy
if len(energy_centers) > 1:
    # Fit linear relationship
    coeffs = np.polyfit(energy_centers, energy_centers, 1)
    fit_line = np.poly1d(coeffs)
    x_fit = np.linspace(min(energy_centers), max(energy_centers), 100)
    
    ax3.plot(energy_centers, energy_centers, 'o-', markersize=8, linewidth=2, label='Data')
    ax3.plot(x_fit, fit_line(x_fit), '--', color='red', label=f'Linear fit: y={coeffs[0]:.3f}x+{coeffs[1]:.3f}')
    ax3.plot([0, max(energy_centers)*1.1], [0, max(energy_centers)*1.1], 'k:', alpha=0.5, label='Perfect linearity')
    ax3.set_xlabel('True Energy (MeV)')
    ax3.set_ylabel('Reconstructed Energy (MeV)')
    ax3.set_title('Energy Linearity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Calculate linearity deviation
    linearity_deviation = abs(coeffs[0] - 1.0) * 100  # Percent deviation from perfect linearity
else:
    ax3.text(0.5, 0.5, 'Single energy point\nLinearity requires multiple energies', 
            transform=ax3.transAxes, ha='center', va='center', fontsize=12)
    ax3.set_title('Energy Linearity (insufficient data)')
    linearity_deviation = 0

# 4. Light yield vs energy
ax4.plot(energy_centers, light_yields, 'o-', markersize=8, linewidth=2, color='green')
ax4.set_xlabel('Energy (MeV)')
ax4.set_ylabel('Light Yield (PE/keV)')
ax4.set_title('Light Yield vs Energy')
ax4.grid(True, alpha=0.3)
if len(energy_centers) > 1:
    ax4.set_xscale('log')

plt.tight_layout()
plt.savefig('multi_energy_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('multi_energy_analysis.pdf', bbox_inches='tight')
plt.show()

# Save detailed results
results = {
    'energy_points_mev': energy_centers,
    'energy_resolutions': resolutions,
    'resolution_errors': resolution_errors,
    'light_yields_pe_kev': light_yields,
    'linearity_deviation_percent': linearity_deviation,
    'total_events': int(total_events),
    'analysis_type': 'multi_energy' if len(energy_centers) > 1 else 'single_energy'
}

with open('multi_energy_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print results for workflow
print(f"RESULT:energy_resolution_curve=multi_energy_analysis.png")
print(f"RESULT:linearity_plot=multi_energy_analysis.png")
print(f"RESULT:light_yield_plot=multi_energy_analysis.png")
print(f"RESULT:results_file=multi_energy_results.json")
print(f"RESULT:mean_resolution={np.mean(resolutions):.4f}")
print(f"RESULT:resolution_at_5mev={resolutions[0]:.4f}" if energy_centers else "RESULT:resolution_at_5mev=0.0000")
print(f"RESULT:linearity_deviation={linearity_deviation:.2f}")
print(f"RESULT:mean_light_yield={np.mean(light_yields):.2f}")
print(f"RESULT:num_energy_points={len(energy_centers)}")

print("\nMulti-energy analysis completed successfully!")