import matplotlib
matplotlib.use('Agg')
import uproot
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Use analysis results from previous step
energy_resolution = 0.034363
energy_resolution_error = 0.000768
linearity = 0.467007
linearity_error = 0.000507
containment_90_radius = 40.82
containment_95_radius = 53.06

# Read simulation data for detailed plots
hits_file = 'baseline_calorimeter_electron_hits.root'

print('Creating publication-quality baseline performance plots...')

# Set publication style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

# 1. Energy Distribution Plot
with uproot.open(hits_file) as f:
    events = f['events'].arrays(library='pd')

fig, ax = plt.subplots(figsize=(10, 8))
mean_E = events['totalEdep'].mean()
std_E = events['totalEdep'].std()
n_events = len(events)

# Create histogram with error bars
counts, bins, patches = ax.hist(events['totalEdep'], bins=50, histtype='step', 
                               linewidth=2, color='blue', label='Energy deposits')
# Add error bars (Poisson statistics)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_errors = np.sqrt(counts)
ax.errorbar(bin_centers, counts, yerr=bin_errors, fmt='none', 
           color='blue', alpha=0.5, capsize=2)

# Add statistics
ax.axvline(mean_E, color='red', linestyle='--', linewidth=2, 
          label=f'Mean: {mean_E:.2f} MeV')
ax.axvline(mean_E + std_E, color='orange', linestyle=':', alpha=0.7, 
          label=f'±1σ: {std_E:.2f} MeV')
ax.axvline(mean_E - std_E, color='orange', linestyle=':', alpha=0.7)

ax.set_xlabel('Total Energy Deposit (MeV)')
ax.set_ylabel('Events')
ax.set_title(f'Baseline Energy Distribution\nResolution σ/E = {energy_resolution:.4f} ± {energy_resolution_error:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('baseline_energy_distribution_publication.png', dpi=300, bbox_inches='tight')
plt.savefig('baseline_energy_distribution_publication.pdf', bbox_inches='tight')
plt.close()

# 2. Resolution vs Energy Curve (using single energy point with error)
fig, ax = plt.subplots(figsize=(10, 8))
# For single energy simulation, show the measured resolution with error bars
particle_energy = 1000  # 1 GeV in MeV from simulation
ax.errorbar([particle_energy], [energy_resolution * 100], 
           yerr=[energy_resolution_error * 100], 
           fmt='o', markersize=8, capsize=5, color='blue', 
           label='Measured resolution')

# Add theoretical curve for comparison (1/√E scaling)
energy_range = np.linspace(100, 2000, 100)
theoretical_res = energy_resolution * 100 * np.sqrt(particle_energy / energy_range)
ax.plot(energy_range, theoretical_res, '--', color='red', alpha=0.7, 
       label='1/√E scaling')

ax.set_xlabel('Particle Energy (MeV)')
ax.set_ylabel('Energy Resolution σ/E (%)')
ax.set_title('Baseline Energy Resolution vs Energy')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 2000)
ax.set_ylim(0, 10)
plt.tight_layout()
plt.savefig('baseline_resolution_curve_publication.png', dpi=300, bbox_inches='tight')
plt.savefig('baseline_resolution_curve_publication.pdf', bbox_inches='tight')
plt.close()

# 3. Shower Profile with error bars (sample first 500k hits to avoid timeout)
print('Creating shower profile plot...')
z_bins = np.linspace(-50, 350, 50)  # mm, covering detector depth
z_centers = (z_bins[:-1] + z_bins[1:]) / 2
z_hist = np.zeros(len(z_bins)-1)

# Sample hits to avoid memory issues
with uproot.open(hits_file) as f:
    hits = f['hits'].arrays(['z', 'edep'], library='np', entry_stop=500000)
    z_hist, _ = np.histogram(hits['z'], bins=z_bins, weights=hits['edep'])

# Calculate errors (assume Poisson statistics for energy deposits)
z_errors = np.sqrt(z_hist)

fig, ax = plt.subplots(figsize=(10, 8))
ax.step(z_centers, z_hist, where='mid', linewidth=2, color='blue', 
       label='Longitudinal profile')
ax.errorbar(z_centers, z_hist, yerr=z_errors, fmt='none', 
           color='blue', alpha=0.5, capsize=2)

ax.set_xlabel('Z Position (mm)')
ax.set_ylabel('Energy Deposit (MeV)')
ax.set_title('Baseline Longitudinal Shower Profile')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('baseline_shower_profile_publication.png', dpi=300, bbox_inches='tight')
plt.savefig('baseline_shower_profile_publication.pdf', bbox_inches='tight')
plt.close()

# 4. Summary metrics plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Energy resolution
ax1.bar(['Baseline'], [energy_resolution * 100], 
       yerr=[energy_resolution_error * 100], 
       capsize=5, color='blue', alpha=0.7)
ax1.set_ylabel('Energy Resolution (%)')
ax1.set_title('Energy Resolution σ/E')
ax1.grid(True, alpha=0.3)

# Linearity
ax2.bar(['Baseline'], [linearity], 
       yerr=[linearity_error], 
       capsize=5, color='green', alpha=0.7)
ax2.set_ylabel('Linearity (Measured/True)')
ax2.set_title('Energy Linearity')
ax2.grid(True, alpha=0.3)

# Containment radii
containment_radii = [containment_90_radius, containment_95_radius]
containment_labels = ['90%', '95%']
ax3.bar(containment_labels, containment_radii, 
       color=['orange', 'red'], alpha=0.7)
ax3.set_ylabel('Containment Radius (mm)')
ax3.set_title('Radial Containment')
ax3.grid(True, alpha=0.3)

# Event statistics
ax4.text(0.1, 0.8, f'Total Events: {n_events}', transform=ax4.transAxes, fontsize=14)
ax4.text(0.1, 0.6, f'Mean Energy: {mean_E:.2f} ± {std_E/np.sqrt(n_events):.2f} MeV', 
        transform=ax4.transAxes, fontsize=14)
ax4.text(0.1, 0.4, f'RMS: {std_E:.2f} MeV', transform=ax4.transAxes, fontsize=14)
ax4.text(0.1, 0.2, f'Resolution: {energy_resolution:.4f} ± {energy_resolution_error:.4f}', 
        transform=ax4.transAxes, fontsize=14)
ax4.set_title('Statistics Summary')
ax4.axis('off')

plt.suptitle('Baseline Calorimeter Performance Summary', fontsize=18)
plt.tight_layout()
plt.savefig('baseline_performance_summary.png', dpi=300, bbox_inches='tight')
plt.savefig('baseline_performance_summary.pdf', bbox_inches='tight')
plt.close()

# Save results
results = {
    'energy_distribution_plot': 'baseline_energy_distribution_publication.png',
    'resolution_curve_plot': 'baseline_resolution_curve_publication.png', 
    'shower_profile_plot': 'baseline_shower_profile_publication.png',
    'summary_plot': 'baseline_performance_summary.png',
    'energy_resolution': energy_resolution,
    'energy_resolution_error': energy_resolution_error,
    'linearity': linearity,
    'containment_90_radius': containment_90_radius,
    'containment_95_radius': containment_95_radius,
    'num_events': int(n_events),
    'mean_energy': float(mean_E),
    'rms_energy': float(std_E)
}

with open('baseline_plots_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Publication-quality plots created successfully!')
print(f'RESULT:energy_distribution_plot=baseline_energy_distribution_publication.png')
print(f'RESULT:resolution_curve_plot=baseline_resolution_curve_publication.png')
print(f'RESULT:shower_profile_plot=baseline_shower_profile_publication.png')
print(f'RESULT:summary_plot=baseline_performance_summary.png')
print(f'RESULT:plots_created=4')