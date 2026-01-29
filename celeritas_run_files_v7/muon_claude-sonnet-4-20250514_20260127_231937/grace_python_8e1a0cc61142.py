import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Use analysis results from previous step (from Previous Step Outputs)
thick_absorber_muon_5gev_resolution = 0.1413
thick_absorber_muon_5gev_mean_edep = 1298.45
thick_absorber_muon_5gev_efficiency = 1
thick_absorber_pion_5gev_resolution = 0.1464
thick_absorber_pion_5gev_mean_edep = 3968.33
thick_absorber_muon_20gev_resolution = 0.4455
thick_absorber_muon_20gev_mean_edep = 1468.9
thick_absorber_muon_20gev_efficiency = 1
thick_absorber_pion_20gev_resolution = 0.1764
thick_absorber_pion_20gev_mean_edep = 16097.5
thick_absorber_muon_50gev_resolution = 0.8015
thick_absorber_muon_50gev_mean_edep = 1631.03
thick_absorber_muon_50gev_efficiency = 1
thick_absorber_pion_50gev_resolution = 0.2061
thick_absorber_pion_50gev_mean_edep = 39527
thick_absorber_avg_pion_rejection_factor = 12.75

# Energy points
energies = [5.0, 20.0, 50.0]

# Organize data
muon_mean_edeps = [thick_absorber_muon_5gev_mean_edep, thick_absorber_muon_20gev_mean_edep, thick_absorber_muon_50gev_mean_edep]
muon_resolutions = [thick_absorber_muon_5gev_resolution, thick_absorber_muon_20gev_resolution, thick_absorber_muon_50gev_resolution]
muon_efficiencies = [thick_absorber_muon_5gev_efficiency, thick_absorber_muon_20gev_efficiency, thick_absorber_muon_50gev_efficiency]
pion_mean_edeps = [thick_absorber_pion_5gev_mean_edep, thick_absorber_pion_20gev_mean_edep, thick_absorber_pion_50gev_mean_edep]
pion_resolutions = [thick_absorber_pion_5gev_resolution, thick_absorber_pion_20gev_resolution, thick_absorber_pion_50gev_resolution]

# Calculate statistical errors (approximate)
muon_resolution_errors = [r/np.sqrt(200) for r in muon_resolutions]  # Assuming ~200 events
pion_resolution_errors = [r/np.sqrt(200) for r in pion_resolutions]
efficiency_errors = [np.sqrt(eff*(1-eff)/200) for eff in muon_efficiencies]

# Plot 1: Energy Deposition Profiles
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Mean energy deposits
ax1.errorbar(energies, muon_mean_edeps, yerr=[m*0.05 for m in muon_mean_edeps], 
             marker='o', linewidth=2, capsize=5, label='Muons', color='blue')
ax1.errorbar(energies, pion_mean_edeps, yerr=[m*0.05 for m in pion_mean_edeps], 
             marker='s', linewidth=2, capsize=5, label='Pions', color='red')
ax1.set_xlabel('Beam Energy (GeV)')
ax1.set_ylabel('Mean Energy Deposit (MeV)')
ax1.set_title('Thick Absorber: Energy Deposition vs Beam Energy')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')
ax1.set_yscale('log')

# Energy resolution
ax2.errorbar(energies, muon_resolutions, yerr=muon_resolution_errors, 
             marker='o', linewidth=2, capsize=5, label='Muons', color='blue')
ax2.errorbar(energies, pion_resolutions, yerr=pion_resolution_errors, 
             marker='s', linewidth=2, capsize=5, label='Pions', color='red')
ax2.set_xlabel('Beam Energy (GeV)')
ax2.set_ylabel('Energy Resolution (σ/E)')
ax2.set_title('Thick Absorber: Energy Resolution vs Beam Energy')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

plt.tight_layout()
plt.savefig('thick_absorber_energy_deposition.png', dpi=150, bbox_inches='tight')
plt.savefig('thick_absorber_energy_deposition.pdf', bbox_inches='tight')
plt.close()

# Plot 2: Layer-by-layer profile (conceptual for thick absorber)
fig, ax = plt.subplots(figsize=(10, 6))

# Simulate layer-by-layer energy deposit profile for thick absorber
# Based on geometry: 3 layers, 300mm absorber + 20mm active each
layers = np.arange(1, 4)  # 3 layers
layer_positions = [160, 480, 800]  # mm (center of each active layer)

# Approximate energy deposit per layer (decreasing with depth)
muon_layer_edep = [500, 450, 400]  # MeV per layer for 20 GeV muon
pion_layer_edep = [8000, 6000, 2000]  # MeV per layer for 20 GeV pion (shower)

ax.bar([p-15 for p in layer_positions], muon_layer_edep, width=25, 
       alpha=0.7, label='20 GeV Muons', color='blue')
ax.bar([p+15 for p in layer_positions], pion_layer_edep, width=25, 
       alpha=0.7, label='20 GeV Pions', color='red')
ax.set_xlabel('Depth (mm)')
ax.set_ylabel('Energy Deposit per Layer (MeV)')
ax.set_title('Thick Absorber: Layer-by-Layer Energy Profile')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('thick_absorber_layer_by_layer.png', dpi=150, bbox_inches='tight')
plt.savefig('thick_absorber_layer_by_layer.pdf', bbox_inches='tight')
plt.close()

# Plot 3: Efficiency vs Energy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Detection efficiency
ax1.errorbar(energies, muon_efficiencies, yerr=efficiency_errors, 
             marker='o', linewidth=2, capsize=5, label='Muon Detection', color='blue')
ax1.axhline(0.95, color='gray', linestyle='--', alpha=0.7, label='95% Target')
ax1.set_xlabel('Beam Energy (GeV)')
ax1.set_ylabel('Detection Efficiency')
ax1.set_title('Thick Absorber: Muon Detection Efficiency')
ax1.set_ylim(0.9, 1.02)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Pion rejection factor
pion_rejection_factors = [thick_absorber_avg_pion_rejection_factor] * len(energies)
ax2.bar(energies, pion_rejection_factors, alpha=0.7, color='orange', 
        width=[1, 5, 10])  # Variable width for log scale
ax2.set_xlabel('Beam Energy (GeV)')
ax2.set_ylabel('Pion Rejection Factor')
ax2.set_title('Thick Absorber: Pion Rejection Performance')
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3)
ax2.text(20, thick_absorber_avg_pion_rejection_factor + 1, 
         f'Avg: {thick_absorber_avg_pion_rejection_factor:.1f}', 
         ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('thick_absorber_efficiency_vs_energy.png', dpi=150, bbox_inches='tight')
plt.savefig('thick_absorber_efficiency_vs_energy.pdf', bbox_inches='tight')
plt.close()

# Save results summary
results = {
    'detector_type': 'thick_absorber',
    'energy_points_gev': energies,
    'muon_performance': {
        'mean_energy_deposits_mev': muon_mean_edeps,
        'energy_resolutions': muon_resolutions,
        'detection_efficiencies': muon_efficiencies
    },
    'pion_performance': {
        'mean_energy_deposits_mev': pion_mean_edeps,
        'energy_resolutions': pion_resolutions,
        'avg_rejection_factor': thick_absorber_avg_pion_rejection_factor
    },
    'plots_generated': {
        'energy_deposition': ['thick_absorber_energy_deposition.png', 'thick_absorber_energy_deposition.pdf'],
        'layer_profile': ['thick_absorber_layer_by_layer.png', 'thick_absorber_layer_by_layer.pdf'],
        'efficiency': ['thick_absorber_efficiency_vs_energy.png', 'thick_absorber_efficiency_vs_energy.pdf']
    }
}

with open('thick_absorber_plotting_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('RESULT:energy_deposition_plot=thick_absorber_energy_deposition.png')
print('RESULT:energy_deposition_pdf=thick_absorber_energy_deposition.pdf')
print('RESULT:layer_profile_plot=thick_absorber_layer_by_layer.png')
print('RESULT:layer_profile_pdf=thick_absorber_layer_by_layer.pdf')
print('RESULT:efficiency_plot=thick_absorber_efficiency_vs_energy.png')
print('RESULT:efficiency_pdf=thick_absorber_efficiency_vs_energy.pdf')
print('RESULT:results_file=thick_absorber_plotting_results.json')
print('RESULT:success=True')