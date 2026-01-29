import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Analysis parameters from required input
light_yield = 10000
path_length_m = 1.06

# Energy points and file paths from Previous Step Outputs
energy_points = [1.0, 2.0, 3.0]
particle_types = ['pip', 'kaonp', 'proton']
particle_names = ['pion', 'kaon', 'proton']

# File path template from Previous Step Outputs
base_path = '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/tof_claude-sonnet-4-20250514_20260128_020039'

results = {}
all_results = []

# Analyze each particle type and energy
for i, (particle_type, particle_name) in enumerate(zip(particle_types, particle_names)):
    particle_results = []
    
    for energy_gev in energy_points:
        # Construct file path based on Previous Step Outputs pattern
        events_file = f'{base_path}/energy_{energy_gev:.3f}GeV/cylindrical_tof_detector_{particle_type}_events.parquet'
        hits_file = f'{base_path}/energy_{energy_gev:.3f}GeV/cylindrical_tof_detector_{particle_type}_hits_data.parquet'
        
        try:
            # Read events data (contains totalEdep per event)
            events_df = pd.read_parquet(events_file)
            
            if len(events_df) == 0:
                print(f'Warning: No events found for {particle_name} at {energy_gev} GeV')
                continue
                
            # Calculate performance metrics
            mean_edep = events_df['totalEdep'].mean()
            std_edep = events_df['totalEdep'].std()
            num_events = len(events_df)
            
            # Energy resolution (sigma/mean)
            energy_resolution = std_edep / mean_edep if mean_edep > 0 else 0
            resolution_err = energy_resolution / np.sqrt(2 * num_events) if num_events > 0 else 0
            
            # Beam energy for linearity calculation
            beam_energy_mev = energy_gev * 1000
            linearity = mean_edep / beam_energy_mev if beam_energy_mev > 0 else 0
            
            # Store results
            result = {
                'particle': particle_name,
                'energy_gev': energy_gev,
                'beam_energy_mev': beam_energy_mev,
                'mean_edep_mev': mean_edep,
                'std_edep_mev': std_edep,
                'energy_resolution': energy_resolution,
                'resolution_err': resolution_err,
                'linearity': linearity,
                'num_events': num_events
            }
            
            particle_results.append(result)
            all_results.append(result)
            
            print(f'{particle_name.capitalize()} at {energy_gev} GeV: Resolution = {energy_resolution:.4f} ± {resolution_err:.4f}')
            
        except Exception as e:
            print(f'Error processing {particle_name} at {energy_gev} GeV: {e}')
            continue
    
    results[particle_name] = particle_results

# Create performance plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Energy Resolution vs Energy
for particle_name in particle_names:
    if particle_name in results and len(results[particle_name]) > 0:
        particle_data = results[particle_name]
        energies = [r['energy_gev'] for r in particle_data]
        resolutions = [r['energy_resolution'] for r in particle_data]
        errors = [r['resolution_err'] for r in particle_data]
        
        ax1.errorbar(energies, resolutions, yerr=errors, marker='o', label=particle_name.capitalize(), capsize=5)

ax1.set_xlabel('Beam Energy (GeV)')
ax1.set_ylabel('Energy Resolution (σ/E)')
ax1.set_title('Cylindrical Detector: Energy Resolution vs Energy')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Plot 2: Mean Energy Deposit vs Beam Energy (Linearity)
for particle_name in particle_names:
    if particle_name in results and len(results[particle_name]) > 0:
        particle_data = results[particle_name]
        beam_energies = [r['beam_energy_mev'] for r in particle_data]
        mean_edeps = [r['mean_edep_mev'] for r in particle_data]
        
        ax2.plot(beam_energies, mean_edeps, marker='o', label=particle_name.capitalize())
        
        # Add ideal linearity line for reference
        if len(beam_energies) > 0:
            ideal_line = np.array(beam_energies) * (mean_edeps[0] / beam_energies[0])
            ax2.plot(beam_energies, ideal_line, '--', alpha=0.5, color='gray')

ax2.set_xlabel('Beam Energy (MeV)')
ax2.set_ylabel('Mean Energy Deposit (MeV)')
ax2.set_title('Cylindrical Detector: Energy Linearity')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Raw Energy Distributions for 2 GeV
for particle_name in particle_names:
    if particle_name in results:
        # Find 2 GeV data point
        particle_data = [r for r in results[particle_name] if r['energy_gev'] == 2.0]
        if len(particle_data) > 0:
            # Re-read the events file for histogram
            particle_type = particle_types[particle_names.index(particle_name)]
            events_file = f'{base_path}/energy_2.000GeV/cylindrical_tof_detector_{particle_type}_events.parquet'
            try:
                events_df = pd.read_parquet(events_file)
                ax3.hist(events_df['totalEdep'], bins=50, alpha=0.7, label=f'{particle_name.capitalize()} (2 GeV)', density=True)
            except:
                pass

ax3.set_xlabel('Total Energy Deposit (MeV)')
ax3.set_ylabel('Normalized Frequency')
ax3.set_title('Energy Deposit Distributions at 2 GeV')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Resolution comparison across particles
if len(all_results) > 0:
    # Group by particle type and calculate average resolution
    particle_avg_res = {}
    for particle_name in particle_names:
        particle_data = [r for r in all_results if r['particle'] == particle_name]
        if len(particle_data) > 0:
            avg_res = np.mean([r['energy_resolution'] for r in particle_data])
            avg_res_err = np.sqrt(np.sum([r['resolution_err']**2 for r in particle_data])) / len(particle_data)
            particle_avg_res[particle_name] = {'avg_res': avg_res, 'err': avg_res_err}
    
    if len(particle_avg_res) > 0:
        particles = list(particle_avg_res.keys())
        avg_resolutions = [particle_avg_res[p]['avg_res'] for p in particles]
        avg_errors = [particle_avg_res[p]['err'] for p in particles]
        
        bars = ax4.bar(particles, avg_resolutions, yerr=avg_errors, capsize=5, alpha=0.7)
        ax4.set_ylabel('Average Energy Resolution (σ/E)')
        ax4.set_title('Average Resolution by Particle Type')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, res in zip(bars, avg_resolutions):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{res:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('cylindrical_detector_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('cylindrical_detector_analysis.pdf', bbox_inches='tight')
print('RESULT:analysis_plot=cylindrical_detector_analysis.png')

# Save detailed analysis results
analysis_summary = {
    'detector_type': 'cylindrical',
    'light_yield_per_mev': light_yield,
    'path_length_m': path_length_m,
    'particles_analyzed': len(particle_names),
    'energy_points': energy_points,
    'total_events_analyzed': sum([r['num_events'] for r in all_results]),
    'results_by_particle': results,
    'average_resolutions': {p: np.mean([r['energy_resolution'] for r in results[p]]) for p in results if len(results[p]) > 0}
}

with open('cylindrical_detector_analysis.json', 'w') as f:
    json.dump(analysis_summary, f, indent=2)

# Output key metrics for workflow
print(f'RESULT:light_yield_per_mev={light_yield}')
print(f'RESULT:path_length_m={path_length_m}')
print('RESULT:analysis_file=cylindrical_detector_analysis.json')
print(f'RESULT:particles_analyzed={len(particle_names)}')
print(f'RESULT:total_events_analyzed={sum([r["num_events"] for r in all_results])}')

# Output average performance metrics for comparison
if len(all_results) > 0:
    overall_avg_resolution = np.mean([r['energy_resolution'] for r in all_results])
    overall_avg_linearity = np.mean([r['linearity'] for r in all_results])
    print(f'RESULT:overall_avg_resolution={overall_avg_resolution:.4f}')
    print(f'RESULT:overall_avg_linearity={overall_avg_linearity:.4f}')

print('RESULT:success=True')
print('Cylindrical detector analysis completed successfully!')