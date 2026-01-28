import matplotlib
matplotlib.use('Agg')
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# File paths from simulation outputs
muon_files = [
    'baseline_planar_muon_spectrometer_mum_hits.root',
    'baseline_planar_muon_spectrometer_mum_hits.root',
    'baseline_planar_muon_spectrometer_mum_hits.root',
    'baseline_planar_muon_spectrometer_mum_hits.root'
]
pion_files = [
    'baseline_planar_muon_spectrometer_pim_hits.root',
    'baseline_planar_muon_spectrometer_pim_hits.root', 
    'baseline_planar_muon_spectrometer_pim_hits.root',
    'baseline_planar_muon_spectrometer_pim_hits.root'
]
energies = [5, 20, 50, 100]

# Initialize results storage
results = {
    'energies': energies,
    'muon_efficiency': [],
    'muon_efficiency_err': [],
    'pion_rejection': [],
    'pion_rejection_err': [],
    'muon_mean_edep': [],
    'pion_mean_edep': [],
    'energy_resolution_muon': [],
    'energy_resolution_pion': []
}

# Detector geometry parameters from previous steps
num_layers = 4
total_depth = 0.88  # meters
last_layer_z = 880  # mm (0.88m converted to mm)

print('Analyzing baseline detector performance...')

# Process each energy point
for i, energy in enumerate(energies):
    print(f'Processing {energy} GeV data...')
    
    # Load muon data
    try:
        with uproot.open(muon_files[i]) as f:
            muon_events = f['events'].arrays(library='pd')
            muon_hits = f['hits'].arrays(['z', 'edep', 'eventID'], library='pd')
    except:
        print(f'Warning: Could not load muon file for {energy} GeV')
        continue
        
    # Load pion data  
    try:
        with uproot.open(pion_files[i]) as f:
            pion_events = f['events'].arrays(library='pd')
            pion_hits = f['hits'].arrays(['z', 'edep', 'eventID'], library='pd')
    except:
        print(f'Warning: Could not load pion file for {energy} GeV')
        continue
    
    # Calculate muon efficiency (fraction with hits detected)
    total_muons = len(muon_events)
    detected_muons = len(muon_events[muon_events['nHits'] > 0])
    muon_eff = detected_muons / total_muons if total_muons > 0 else 0
    muon_eff_err = np.sqrt(muon_eff * (1 - muon_eff) / total_muons) if total_muons > 0 else 0
    
    # Calculate pion rejection (fraction that don't penetrate fully)
    total_pions = len(pion_events)
    # Find maximum z position for each pion event
    pion_max_z = pion_hits.groupby('eventID')['z'].max()
    penetrating_pions = len(pion_max_z[pion_max_z > (last_layer_z - 50)])  # Within 50mm of end
    pion_rej = 1.0 - (penetrating_pions / total_pions) if total_pions > 0 else 0
    pion_rej_err = np.sqrt(pion_rej * (1 - pion_rej) / total_pions) if total_pions > 0 else 0
    
    # Calculate mean energy deposits
    muon_mean = muon_events['totalEdep'].mean()
    pion_mean = pion_events['totalEdep'].mean()
    
    # Calculate energy resolution
    muon_res = muon_events['totalEdep'].std() / muon_mean if muon_mean > 0 else 0
    pion_res = pion_events['totalEdep'].std() / pion_mean if pion_mean > 0 else 0
    
    # Store results
    results['muon_efficiency'].append(muon_eff)
    results['muon_efficiency_err'].append(muon_eff_err)
    results['pion_rejection'].append(pion_rej)
    results['pion_rejection_err'].append(pion_rej_err)
    results['muon_mean_edep'].append(muon_mean)
    results['pion_mean_edep'].append(pion_mean)
    results['energy_resolution_muon'].append(muon_res)
    results['energy_resolution_pion'].append(pion_res)
    
    print(f'{energy} GeV: Muon eff={muon_eff:.3f}±{muon_eff_err:.3f}, Pion rej={pion_rej:.3f}±{pion_rej_err:.3f}')

# Generate performance plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Muon efficiency vs energy
ax1.errorbar(energies, results['muon_efficiency'], yerr=results['muon_efficiency_err'], 
             marker='o', capsize=5, linewidth=2)
ax1.set_xlabel('Energy (GeV)')
ax1.set_ylabel('Muon Efficiency')
ax1.set_title('Muon Detection Efficiency')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.1)

# Pion rejection vs energy
ax2.errorbar(energies, results['pion_rejection'], yerr=results['pion_rejection_err'],
             marker='s', color='red', capsize=5, linewidth=2)
ax2.set_xlabel('Energy (GeV)')
ax2.set_ylabel('Pion Rejection Fraction')
ax2.set_title('Pion Rejection Performance')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.1)

# Energy deposits comparison
ax3.plot(energies, results['muon_mean_edep'], 'o-', label='Muons', linewidth=2)
ax3.plot(energies, results['pion_mean_edep'], 's-', label='Pions', linewidth=2)
ax3.set_xlabel('Energy (GeV)')
ax3.set_ylabel('Mean Energy Deposit (MeV)')
ax3.set_title('Energy Deposit Comparison')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')

# Energy resolution
ax4.plot(energies, results['energy_resolution_muon'], 'o-', label='Muons', linewidth=2)
ax4.plot(energies, results['energy_resolution_pion'], 's-', label='Pions', linewidth=2)
ax4.set_xlabel('Energy (GeV)')
ax4.set_ylabel('Energy Resolution (σ/E)')
ax4.set_title('Energy Resolution')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('baseline_performance_summary.png', dpi=150, bbox_inches='tight')
plt.savefig('baseline_performance_summary.pdf', bbox_inches='tight')
plt.show()

# Save detailed results to JSON
with open('baseline_performance_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

# Calculate overall performance metrics
avg_muon_eff = np.mean(results['muon_efficiency'])
avg_pion_rej = np.mean(results['pion_rejection'])
muon_eff_100gev = results['muon_efficiency'][-1] if len(results['muon_efficiency']) > 0 else 0
pion_rej_100gev = results['pion_rejection'][-1] if len(results['pion_rejection']) > 0 else 0

print('\n=== BASELINE PERFORMANCE SUMMARY ===')
print(f'Average muon efficiency: {avg_muon_eff:.3f}')
print(f'Average pion rejection: {avg_pion_rej:.3f}')
print(f'Muon efficiency at 100 GeV: {muon_eff_100gev:.3f}')
print(f'Pion rejection at 100 GeV: {pion_rej_100gev:.3f}')

# Return key metrics for downstream steps
print(f'RESULT:avg_muon_efficiency={avg_muon_eff:.4f}')
print(f'RESULT:avg_pion_rejection={avg_pion_rej:.4f}')
print(f'RESULT:muon_efficiency_100gev={muon_eff_100gev:.4f}')
print(f'RESULT:pion_rejection_100gev={pion_rej_100gev:.4f}')
print('RESULT:performance_plot=baseline_performance_summary.png')
print('RESULT:metrics_file=baseline_performance_metrics.json')

print('\nBaseline performance analysis completed successfully!')