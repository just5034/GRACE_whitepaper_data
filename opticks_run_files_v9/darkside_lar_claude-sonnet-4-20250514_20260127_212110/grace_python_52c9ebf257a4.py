import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Energy sweep file paths from Previous Step Outputs
energy_files = {
    0.1: '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/darkside_lar_claude-sonnet-4-20250514_20260127_212110/energy_0.000GeV/darkside_detector_electron_events.parquet',
    1.0: '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/darkside_lar_claude-sonnet-4-20250514_20260127_212110/energy_0.001GeV/darkside_detector_electron_events.parquet',
    5.0: '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/darkside_lar_claude-sonnet-4-20250514_20260127_212110/energy_0.005GeV/darkside_detector_electron_events.parquet'
}

# Analyze each energy point
results = []
for energy_mev, filepath in energy_files.items():
    if Path(filepath).exists():
        df = pd.read_parquet(filepath)
        
        # Extract optical response metrics
        if 'nPhotons' in df.columns:
            # Light yield analysis
            mean_photons = df['nPhotons'].mean()
            std_photons = df['nPhotons'].std()
            
            # Convert to photoelectrons (assuming ~20% quantum efficiency for LAr)
            qe = 0.20
            mean_pe = mean_photons * qe
            std_pe = std_photons * qe
            
            # Light yield in PE/keV
            light_yield_pe_per_kev = mean_pe / (energy_mev * 1000)
            
            # Energy resolution from photoelectron statistics
            if mean_pe > 0:
                pe_resolution = std_pe / mean_pe
                pe_resolution_err = pe_resolution / np.sqrt(2 * len(df))
            else:
                pe_resolution = 0
                pe_resolution_err = 0
            
            results.append({
                'energy_mev': energy_mev,
                'mean_photons': mean_photons,
                'mean_pe': mean_pe,
                'light_yield_pe_per_kev': light_yield_pe_per_kev,
                'pe_resolution': pe_resolution,
                'pe_resolution_err': pe_resolution_err,
                'num_events': len(df)
            })
            
            # Plot S1 distribution for this energy
            plt.figure(figsize=(10, 6))
            plt.hist(df['nPhotons'] * qe, bins=50, histtype='step', linewidth=2, 
                    label=f'{energy_mev} MeV electrons')
            plt.axvline(mean_pe, color='r', linestyle='--', 
                       label=f'Mean: {mean_pe:.1f} PE')
            plt.xlabel('S1 Signal (Photoelectrons)')
            plt.ylabel('Events')
            plt.title(f'S1 Distribution - {energy_mev} MeV (σ/μ = {pe_resolution:.4f})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f's1_distribution_{energy_mev}mev.png', dpi=150, bbox_inches='tight')
            plt.savefig(f's1_distribution_{energy_mev}mev.pdf', bbox_inches='tight')
            plt.close()
            
        else:
            print(f'Warning: No nPhotons column found in {filepath}')
            # Fallback to energy deposit analysis
            if 'totalEdep' in df.columns:
                mean_edep = df['totalEdep'].mean()
                std_edep = df['totalEdep'].std()
                edep_resolution = std_edep / mean_edep if mean_edep > 0 else 0
                
                results.append({
                    'energy_mev': energy_mev,
                    'mean_edep_mev': mean_edep,
                    'edep_resolution': edep_resolution,
                    'num_events': len(df)
                })

# Summary analysis across energies
if results:
    # Plot light yield vs energy
    energies = [r['energy_mev'] for r in results if 'light_yield_pe_per_kev' in r]
    light_yields = [r['light_yield_pe_per_kev'] for r in results if 'light_yield_pe_per_kev' in r]
    
    if light_yields:
        plt.figure(figsize=(10, 6))
        plt.plot(energies, light_yields, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Electron Energy (MeV)')
        plt.ylabel('Light Yield (PE/keV)')
        plt.title('Light Yield vs Energy')
        plt.grid(True, alpha=0.3)
        plt.savefig('light_yield_vs_energy.png', dpi=150, bbox_inches='tight')
        plt.savefig('light_yield_vs_energy.pdf', bbox_inches='tight')
        plt.close()
        
        # Calculate average light yield
        avg_light_yield = np.mean(light_yields)
        light_yield_std = np.std(light_yields)
        
        print(f'RESULT:average_light_yield_pe_per_kev={avg_light_yield:.2f}')
        print(f'RESULT:light_yield_std={light_yield_std:.2f}')
    
    # Plot energy resolution vs energy
    resolutions = [r['pe_resolution'] for r in results if 'pe_resolution' in r]
    if resolutions:
        plt.figure(figsize=(10, 6))
        plt.plot(energies, resolutions, 'ro-', linewidth=2, markersize=8)
        plt.xlabel('Electron Energy (MeV)')
        plt.ylabel('Energy Resolution (σ/μ)')
        plt.title('Energy Resolution vs Energy')
        plt.grid(True, alpha=0.3)
        plt.savefig('energy_resolution_vs_energy.png', dpi=150, bbox_inches='tight')
        plt.savefig('energy_resolution_vs_energy.pdf', bbox_inches='tight')
        plt.close()
        
        # Best resolution (typically at highest energy)
        best_resolution = min(resolutions)
        print(f'RESULT:best_energy_resolution={best_resolution:.4f}')
    
    # Save detailed results
    with open('scintillation_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print('RESULT:s1_plots=s1_distribution_*.png')
    print('RESULT:light_yield_plot=light_yield_vs_energy.png')
    print('RESULT:resolution_plot=energy_resolution_vs_energy.png')
    print('RESULT:analysis_complete=True')
    print(f'RESULT:num_energy_points={len(results)}')
    
else:
    print('ERROR: No valid simulation data found')
    print('RESULT:analysis_complete=False')