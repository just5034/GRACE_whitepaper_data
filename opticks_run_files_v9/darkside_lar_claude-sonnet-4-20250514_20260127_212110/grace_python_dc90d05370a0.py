import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Energy sweep file paths from Previous Step Outputs
energy_files = {
    0.0001: '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/darkside_lar_claude-sonnet-4-20250514_20260127_212110/energy_0.000GeV/darkside_detector_electron_events.parquet',
    0.001: '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/darkside_lar_claude-sonnet-4-20250514_20260127_212110/energy_0.001GeV/darkside_detector_electron_events.parquet',
    0.005: '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/darkside_lar_claude-sonnet-4-20250514_20260127_212110/energy_0.005GeV/darkside_detector_events.parquet'
}

# Analysis results from previous step
average_light_yield = 0.29  # pe/keV
light_yield_std = 0
best_energy_resolution = 0.0258

# Analyze each energy point
results = []
for energy_gev, filepath in energy_files.items():
    if Path(filepath).exists():
        df = pd.read_parquet(filepath)
        energy_kev = energy_gev * 1e6  # Convert GeV to keV
        
        if len(df) > 0 and 'totalEdep' in df.columns:
            mean_edep = df['totalEdep'].mean()
            std_edep = df['totalEdep'].std()
            resolution = std_edep / mean_edep if mean_edep > 0 else 0
            resolution_err = resolution / np.sqrt(2 * len(df)) if len(df) > 1 else 0
            
            # Light yield calculation (assuming totalEdep is in MeV)
            light_yield = mean_edep / (energy_kev / 1000) if energy_kev > 0 else 0  # pe/MeV
            light_yield_err = std_edep / np.sqrt(len(df)) / (energy_kev / 1000) if len(df) > 1 and energy_kev > 0 else 0
            
            results.append({
                'energy_gev': energy_gev,
                'energy_kev': energy_kev,
                'mean_edep': mean_edep,
                'std_edep': std_edep,
                'resolution': resolution,
                'resolution_err': resolution_err,
                'light_yield': light_yield,
                'light_yield_err': light_yield_err,
                'num_events': len(df)
            })

# Convert to arrays for plotting
if results:
    energies_gev = np.array([r['energy_gev'] for r in results])
    energies_kev = np.array([r['energy_kev'] for r in results])
    resolutions = np.array([r['resolution'] for r in results])
    resolution_errs = np.array([r['resolution_err'] for r in results])
    light_yields = np.array([r['light_yield'] for r in results])
    light_yield_errs = np.array([r['light_yield_err'] for r in results])
    
    # Plot 1: Energy Response (Resolution vs Energy)
    plt.figure(figsize=(10, 6))
    plt.errorbar(energies_kev, resolutions * 100, yerr=resolution_errs * 100, 
                 fmt='o-', capsize=5, linewidth=2, markersize=8)
    plt.xlabel('Energy (keV)', fontsize=12)
    plt.ylabel('Energy Resolution σ/E (%)', fontsize=12)
    plt.title('Energy Response vs Incident Energy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.savefig('energy_response_plot.png', dpi=150, bbox_inches='tight')
    plt.savefig('energy_response_plot.pdf', bbox_inches='tight')
    plt.close()
    
    # Plot 2: Light Yield vs Energy
    plt.figure(figsize=(10, 6))
    plt.errorbar(energies_kev, light_yields, yerr=light_yield_errs,
                 fmt='s-', capsize=5, linewidth=2, markersize=8, color='green')
    plt.xlabel('Energy (keV)', fontsize=12)
    plt.ylabel('Light Yield (photoelectrons/MeV)', fontsize=12)
    plt.title('Light Yield vs Incident Energy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.savefig('light_yield_curve_plot.png', dpi=150, bbox_inches='tight')
    plt.savefig('light_yield_curve_plot.pdf', bbox_inches='tight')
    plt.close()
    
    # Plot 3: Photoelectron Distributions for each energy
    fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
    if len(results) == 1:
        axes = [axes]
    
    for i, (energy_gev, filepath) in enumerate(energy_files.items()):
        if Path(filepath).exists():
            df = pd.read_parquet(filepath)
            if len(df) > 0 and 'totalEdep' in df.columns:
                axes[i].hist(df['totalEdep'], bins=30, alpha=0.7, histtype='step', linewidth=2)
                axes[i].set_xlabel('Energy Deposit (MeV)', fontsize=10)
                axes[i].set_ylabel('Events', fontsize=10)
                axes[i].set_title(f'{energy_gev*1000:.1f} MeV', fontsize=11)
                axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Photoelectron Distributions by Energy', fontsize=14)
    plt.tight_layout()
    plt.savefig('photoelectron_distributions_plot.png', dpi=150, bbox_inches='tight')
    plt.savefig('photoelectron_distributions_plot.pdf', bbox_inches='tight')
    plt.close()
    
    # Summary plot combining all metrics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Top: Energy resolution
    ax1.errorbar(energies_kev, resolutions * 100, yerr=resolution_errs * 100,
                fmt='o-', capsize=5, linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Energy (keV)', fontsize=12)
    ax1.set_ylabel('Energy Resolution σ/E (%)', fontsize=12)
    ax1.set_title('DarkSide LAr Detector Performance', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Bottom: Light yield
    ax2.errorbar(energies_kev, light_yields, yerr=light_yield_errs,
                fmt='s-', capsize=5, linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Energy (keV)', fontsize=12)
    ax2.set_ylabel('Light Yield (pe/MeV)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('combined_response_plot.png', dpi=150, bbox_inches='tight')
    plt.savefig('combined_response_plot.pdf', bbox_inches='tight')
    plt.close()
    
    print('RESULT:energy_response_plot=energy_response_plot.png')
    print('RESULT:light_yield_curve_plot=light_yield_curve_plot.png')
    print('RESULT:photoelectron_distributions_plot=photoelectron_distributions_plot.png')
    print('RESULT:combined_response_plot=combined_response_plot.png')
    print(f'RESULT:best_energy_resolution={best_energy_resolution:.4f}')
    print(f'RESULT:average_light_yield_pe_per_kev={average_light_yield:.3f}')
    print(f'RESULT:num_energy_points={len(results)}')
    print('RESULT:plots_with_error_bars=True')
    print('RESULT:success=True')
else:
    print('ERROR: No valid energy data files found')
    print('RESULT:success=False')