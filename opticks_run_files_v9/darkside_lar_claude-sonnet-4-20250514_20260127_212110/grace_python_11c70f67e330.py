import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Read analysis results from previous step
# Use the return values from analyze_scintillation_response step
average_light_yield_pe_per_kev = 0.29
light_yield_std = 0
best_energy_resolution = 0.0258
num_energy_points = 2

# Energy sweep file paths from simulate_low_energy_electrons
energy_files = {
    0.0001: '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/darkside_lar_claude-sonnet-4-20250514_20260127_212110/energy_0.000GeV/darkside_detector_electron_events.parquet',
    0.001: '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/darkside_lar_claude-sonnet-4-20250514_20260127_212110/energy_0.001GeV/darkside_detector_electron_events.parquet',
    0.005: '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/darkside_lar_claude-sonnet-4-20250514_20260127_212110/energy_0.005GeV/darkside_detector_events.parquet'
}

# Initialize results storage
results = []
energies_gev = []
light_yields = []
light_yield_errors = []
resolutions = []
resolution_errors = []

# Process each energy point
for energy_gev, filepath in energy_files.items():
    try:
        # Check if file exists
        if not Path(filepath).exists():
            print(f'Warning: File not found: {filepath}')
            continue
            
        # Read events data
        df = pd.read_parquet(filepath)
        
        if len(df) == 0:
            print(f'Warning: No data in {filepath}')
            continue
            
        # Check data variation (modification requirement)
        if 'nPhotons' in df.columns:
            photon_data = df['nPhotons'].values
        else:
            print(f'Warning: nPhotons column not found in {filepath}')
            continue
            
        # Handle constant data check
        if photon_data.std() <= 1e-10:
            print(f'Warning: Constant photon data at {energy_gev} GeV: {photon_data.mean():.2f}')
            light_yield = photon_data.mean() / (energy_gev * 1000)  # Convert GeV to MeV
            light_yield_err = 0
            resolution = 0
            resolution_err = 0
        else:
            # Calculate light yield (photons per MeV)
            beam_energy_mev = energy_gev * 1000
            light_yield = photon_data.mean() / beam_energy_mev
            light_yield_err = photon_data.std() / (np.sqrt(len(photon_data)) * beam_energy_mev)
            
            # Calculate energy resolution
            if photon_data.mean() > 0:
                resolution = photon_data.std() / photon_data.mean()
                resolution_err = resolution / np.sqrt(2 * len(photon_data))
            else:
                resolution = 0
                resolution_err = 0
        
        # Store results
        energies_gev.append(energy_gev)
        light_yields.append(light_yield)
        light_yield_errors.append(light_yield_err)
        resolutions.append(resolution)
        resolution_errors.append(resolution_err)
        
        print(f'Energy: {energy_gev} GeV, Light Yield: {light_yield:.2f} ± {light_yield_err:.2f} PE/MeV, Resolution: {resolution:.4f} ± {resolution_err:.4f}')
        
    except Exception as e:
        print(f'Error processing {filepath}: {e}')
        continue

# Create energy response plots with error bars
if len(energies_gev) > 0:
    # Convert to numpy arrays for plotting
    energies_mev = np.array(energies_gev) * 1000
    light_yields = np.array(light_yields)
    light_yield_errors = np.array(light_yield_errors)
    resolutions = np.array(resolutions)
    resolution_errors = np.array(resolution_errors)
    
    # Plot 1: Light yield vs energy
    plt.figure(figsize=(10, 6))
    plt.errorbar(energies_mev, light_yields, yerr=light_yield_errors, 
                fmt='o-', capsize=5, linewidth=2, markersize=8)
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Light Yield (PE/MeV)')
    plt.title('Light Yield vs Energy with Statistical Uncertainties')
    plt.grid(True, alpha=0.3)
    plt.savefig('light_yield_curve.png', dpi=150, bbox_inches='tight')
    plt.savefig('light_yield_curve.pdf', bbox_inches='tight')
    plt.close()
    
    # Plot 2: Energy resolution vs energy
    plt.figure(figsize=(10, 6))
    plt.errorbar(energies_mev, resolutions * 100, yerr=resolution_errors * 100,
                fmt='s-', capsize=5, linewidth=2, markersize=8, color='red')
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Energy Resolution (%)')
    plt.title('Energy Resolution vs Energy with Statistical Uncertainties')
    plt.grid(True, alpha=0.3)
    plt.savefig('energy_resolution_vs_energy.png', dpi=150, bbox_inches='tight')
    plt.savefig('energy_resolution_vs_energy.pdf', bbox_inches='tight')
    plt.close()
    
    # Plot 3: Photoelectron distributions for each energy
    fig, axes = plt.subplots(1, min(3, len(energies_gev)), figsize=(15, 5))
    if len(energies_gev) == 1:
        axes = [axes]
    
    for i, (energy_gev, filepath) in enumerate(list(energy_files.items())[:3]):
        if Path(filepath).exists():
            df = pd.read_parquet(filepath)
            if 'nPhotons' in df.columns and len(df) > 0:
                photon_data = df['nPhotons'].values
                
                # Use adaptive binning
                if photon_data.std() > 1e-10:
                    bins = 'auto'  # Safe histogram code modification
                else:
                    bins = 10  # Fallback for constant data
                
                axes[i].hist(photon_data, bins=bins, histtype='step', linewidth=2)
                axes[i].set_xlabel('Number of Photons')
                axes[i].set_ylabel('Events')
                axes[i].set_title(f'{energy_gev*1000:.1f} MeV')
                axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('photoelectron_distributions.png', dpi=150, bbox_inches='tight')
    plt.savefig('photoelectron_distributions.pdf', bbox_inches='tight')
    plt.close()
    
    # Save results to JSON
    results_dict = {
        'energies_mev': energies_mev.tolist(),
        'light_yields_pe_per_mev': light_yields.tolist(),
        'light_yield_errors': light_yield_errors.tolist(),
        'energy_resolutions': resolutions.tolist(),
        'resolution_errors': resolution_errors.tolist(),
        'average_light_yield': float(np.mean(light_yields)),
        'best_resolution': float(np.min(resolutions)),
        'plots_generated': ['light_yield_curve.png', 'energy_resolution_vs_energy.png', 'photoelectron_distributions.png']
    }
    
    with open('energy_response_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print('RESULT:light_yield_plot=light_yield_curve.png')
    print('RESULT:resolution_plot=energy_resolution_vs_energy.png')
    print('RESULT:photoelectron_plot=photoelectron_distributions.png')
    print(f'RESULT:average_light_yield={np.mean(light_yields):.3f}')
    print(f'RESULT:best_resolution={np.min(resolutions):.4f}')
    print('RESULT:plots_with_error_bars=True')
    print('RESULT:success=True')
    
else:
    print('Error: No valid data found for plotting')
    print('RESULT:success=False')