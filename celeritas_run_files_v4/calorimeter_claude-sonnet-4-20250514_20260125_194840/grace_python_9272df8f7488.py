import matplotlib
matplotlib.use('Agg')
import uproot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

# Energy points from simulations
energies_gev = [10, 100, 200]
file_path = 'baseline_calorimeter_pip_hits.root'

# Storage for results
results = {}
resolutions = []
resolution_errors = []
mean_energies = []
mean_energy_errors = []

print('Analyzing baseline calorimeter performance across energies...')

# Analyze each energy point
for i, true_energy in enumerate(energies_gev):
    print(f'\nAnalyzing {true_energy} GeV data...')
    
    try:
        with uproot.open(file_path) as f:
            # Load events tree for energy resolution
            events = f['events'].arrays(library='pd')
            
            if len(events) == 0:
                print(f'Warning: No events found for {true_energy} GeV')
                continue
                
            # Calculate energy resolution metrics
            total_edep = events['totalEdep']  # MeV
            mean_E = total_edep.mean()
            std_E = total_edep.std()
            resolution = std_E / mean_E if mean_E > 0 else 0
            resolution_err = resolution / np.sqrt(2 * len(events)) if len(events) > 1 else 0
            mean_err = std_E / np.sqrt(len(events)) if len(events) > 0 else 0
            
            # Store results
            resolutions.append(resolution)
            resolution_errors.append(resolution_err)
            mean_energies.append(mean_E)
            mean_energy_errors.append(mean_err)
            
            results[f'{true_energy}gev'] = {
                'mean_energy_mev': float(mean_E),
                'std_energy_mev': float(std_E),
                'resolution': float(resolution),
                'resolution_error': float(resolution_err),
                'num_events': len(events)
            }
            
            print(f'Mean energy: {mean_E:.1f} ± {mean_err:.1f} MeV')
            print(f'Resolution (σ/E): {resolution:.4f} ± {resolution_err:.4f}')
            
    except Exception as e:
        print(f'Error analyzing {true_energy} GeV: {e}')
        continue

# Generate energy resolution plot
if len(resolutions) > 0:
    plt.figure(figsize=(10, 6))
    plt.errorbar(energies_gev[:len(resolutions)], resolutions, yerr=resolution_errors, 
                 fmt='bo-', capsize=5, linewidth=2, markersize=8)
    plt.xlabel('True Energy (GeV)')
    plt.ylabel('Energy Resolution (σ/E)')
    plt.title('Baseline Calorimeter Energy Resolution vs Energy')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('baseline_energy_resolution.png', dpi=150, bbox_inches='tight')
    plt.savefig('baseline_energy_resolution.pdf', bbox_inches='tight')
    plt.close()
    print('RESULT:resolution_plot=baseline_energy_resolution.png')

# Generate linearity plot
if len(mean_energies) > 0:
    # Convert to GeV for linearity plot
    mean_energies_gev = np.array(mean_energies) / 1000.0
    mean_errors_gev = np.array(mean_energy_errors) / 1000.0
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(energies_gev[:len(mean_energies)], mean_energies_gev, 
                 yerr=mean_errors_gev, fmt='ro-', capsize=5, linewidth=2, markersize=8, label='Measured')
    
    # Perfect linearity reference
    energy_range = np.array(energies_gev[:len(mean_energies)])
    plt.plot(energy_range, energy_range, 'k--', alpha=0.7, label='Perfect linearity')
    
    plt.xlabel('True Energy (GeV)')
    plt.ylabel('Reconstructed Energy (GeV)')
    plt.title('Baseline Calorimeter Energy Linearity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('baseline_linearity.png', dpi=150, bbox_inches='tight')
    plt.savefig('baseline_linearity.pdf', bbox_inches='tight')
    plt.close()
    print('RESULT:linearity_plot=baseline_linearity.png')
    
    # Calculate linearity deviation
    if len(mean_energies_gev) > 1:
        linearity_deviation = np.std((mean_energies_gev - energy_range) / energy_range) * 100
        print(f'RESULT:linearity_deviation={linearity_deviation:.2f}')

# Analyze shower profile (using hits from last energy for profile)
print('\nAnalyzing longitudinal shower profile...')
try:
    z_bins = np.linspace(0, 1000, 50)  # 0-1m depth in mm
    z_centers = (z_bins[:-1] + z_bins[1:]) / 2
    z_hist = np.zeros(len(z_bins)-1)
    
    # Sample hits to avoid memory issues
    with uproot.open(file_path) as f:
        for batch in f['hits'].iterate(['z', 'edep'], step_size='50 MB'):
            z_hist += np.histogram(batch['z'], bins=z_bins, weights=batch['edep'])[0]
    
    # Normalize profile
    if np.sum(z_hist) > 0:
        z_hist = z_hist / np.sum(z_hist)
        
        plt.figure(figsize=(10, 6))
        plt.step(z_centers, z_hist, where='mid', linewidth=2)
        plt.xlabel('Depth (mm)')
        plt.ylabel('Normalized Energy Deposit')
        plt.title('Baseline Calorimeter Longitudinal Shower Profile')
        plt.grid(True, alpha=0.3)
        plt.savefig('baseline_shower_profile.png', dpi=150, bbox_inches='tight')
        plt.savefig('baseline_shower_profile.pdf', bbox_inches='tight')
        plt.close()
        print('RESULT:shower_profile_plot=baseline_shower_profile.png')
        
        # Calculate shower maximum position
        max_idx = np.argmax(z_hist)
        shower_max_mm = z_centers[max_idx]
        print(f'RESULT:shower_max_depth={shower_max_mm:.1f}')
        
except Exception as e:
    print(f'Error analyzing shower profile: {e}')

# Save detailed results
results['summary'] = {
    'energies_gev': energies_gev[:len(resolutions)],
    'resolutions': resolutions,
    'resolution_errors': resolution_errors,
    'mean_energies_mev': mean_energies,
    'analysis_type': 'energy_resolution_linearity'
}

with open('baseline_performance_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

# Output key metrics
if len(resolutions) > 0:
    print(f'\nRESULT:num_energy_points={len(resolutions)}')
    print(f'RESULT:best_resolution={min(resolutions):.4f}')
    print(f'RESULT:worst_resolution={max(resolutions):.4f}')
    print(f'RESULT:analysis_file=baseline_performance_analysis.json')
    print('RESULT:baseline_analysis_completed=True')
else:
    print('ERROR: No valid energy resolution data extracted')
    print('RESULT:baseline_analysis_completed=False')