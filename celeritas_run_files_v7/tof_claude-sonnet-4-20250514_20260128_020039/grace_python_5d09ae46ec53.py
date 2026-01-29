import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Energy points and file paths from Previous Step Outputs
energies = [1.0, 2.0, 3.0]  # GeV
base_path = '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/tof_claude-sonnet-4-20250514_20260128_020039'

# Parameters from Required Input
light_yield = 10000  # photons/MeV
path_length = 1.0    # meters

# Speed of light in m/ns
c = 0.299792458  # m/ns

# Particle masses in MeV/c^2
mass_pion = 139.57
mass_kaon = 493.68
mass_proton = 938.27

results = {}
all_results = []

print("Analyzing planar detector results for all particle types...")

# Analyze each particle type
for particle, mass in [('pip', mass_pion), ('kaonp', mass_kaon), ('proton', mass_proton)]:
    particle_results = []
    
    for energy_gev in energies:
        energy_dir = f'energy_{energy_gev:.3f}GeV'
        events_file = f'{base_path}/{energy_dir}/planar_tof_detector_{particle}_events.parquet'
        hits_file = f'{base_path}/{energy_dir}/planar_tof_detector_{particle}_hits_data.parquet'
        
        try:
            # Load events data
            events_df = pd.read_parquet(events_file)
            
            if len(events_df) == 0:
                print(f"Warning: No events found for {particle} at {energy_gev} GeV")
                continue
                
            # Calculate basic metrics
            mean_edep = events_df['totalEdep'].mean()  # MeV
            std_edep = events_df['totalEdep'].std()
            num_events = len(events_df)
            
            # Calculate light yield (photons)
            mean_photons = mean_edep * light_yield
            
            # Calculate velocity and time of flight
            momentum_gev = np.sqrt(energy_gev**2 - (mass/1000)**2)  # GeV/c
            beta = momentum_gev / energy_gev  # v/c
            velocity = beta * c  # m/ns
            tof_ns = path_length / velocity  # time of flight in ns
            
            result = {
                'particle': particle,
                'energy_gev': energy_gev,
                'momentum_gev': momentum_gev,
                'beta': beta,
                'tof_ns': tof_ns,
                'mean_edep_mev': mean_edep,
                'std_edep_mev': std_edep,
                'mean_photons': mean_photons,
                'num_events': num_events
            }
            
            particle_results.append(result)
            all_results.append(result)
            
            print(f"{particle} at {energy_gev} GeV: TOF = {tof_ns:.3f} ns, Edep = {mean_edep:.2f} ± {std_edep:.2f} MeV, Photons = {mean_photons:.0f}")
            
        except Exception as e:
            print(f"Error processing {particle} at {energy_gev} GeV: {e}")
            continue
    
    results[particle] = particle_results

# Create summary plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Time of Flight vs Energy
for particle in ['pip', 'kaonp', 'proton']:
    if particle in results and len(results[particle]) > 0:
        data = results[particle]
        energies_plot = [r['energy_gev'] for r in data]
        tofs = [r['tof_ns'] for r in data]
        ax1.plot(energies_plot, tofs, 'o-', label=particle, linewidth=2, markersize=8)

ax1.set_xlabel('Energy (GeV)')
ax1.set_ylabel('Time of Flight (ns)')
ax1.set_title('Time of Flight vs Energy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Energy Deposits
for particle in ['pip', 'kaonp', 'proton']:
    if particle in results and len(results[particle]) > 0:
        data = results[particle]
        energies_plot = [r['energy_gev'] for r in data]
        edeps = [r['mean_edep_mev'] for r in data]
        ax2.plot(energies_plot, edeps, 'o-', label=particle, linewidth=2, markersize=8)

ax2.set_xlabel('Energy (GeV)')
ax2.set_ylabel('Mean Energy Deposit (MeV)')
ax2.set_title('Energy Deposits in Planar Detector')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Light Yield
for particle in ['pip', 'kaonp', 'proton']:
    if particle in results and len(results[particle]) > 0:
        data = results[particle]
        energies_plot = [r['energy_gev'] for r in data]
        photons = [r['mean_photons'] for r in data]
        ax3.plot(energies_plot, photons, 'o-', label=particle, linewidth=2, markersize=8)

ax3.set_xlabel('Energy (GeV)')
ax3.set_ylabel('Light Yield (photons)')
ax3.set_title('Light Yield vs Energy')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Timing Separations
if len(all_results) >= 6:  # Need at least 2 particles with data
    # Calculate timing separations at each energy
    energy_separations = {}
    
    for energy_gev in energies:
        energy_data = [r for r in all_results if r['energy_gev'] == energy_gev]
        if len(energy_data) >= 2:
            # Sort by TOF
            energy_data.sort(key=lambda x: x['tof_ns'])
            
            separations = []
            labels = []
            for i in range(len(energy_data)-1):
                sep = energy_data[i+1]['tof_ns'] - energy_data[i]['tof_ns']
                separations.append(sep)
                labels.append(f"{energy_data[i]['particle']}-{energy_data[i+1]['particle']}")
            
            energy_separations[energy_gev] = {'separations': separations, 'labels': labels}
    
    # Plot timing separations
    x_pos = np.arange(len(energies))
    width = 0.35
    
    if energy_separations:
        # Get unique separation pairs
        all_pairs = set()
        for data in energy_separations.values():
            all_pairs.update(data['labels'])
        
        colors = ['blue', 'red', 'green']
        for i, pair in enumerate(sorted(all_pairs)):
            pair_seps = []
            for energy_gev in energies:
                if energy_gev in energy_separations:
                    data = energy_separations[energy_gev]
                    if pair in data['labels']:
                        idx = data['labels'].index(pair)
                        pair_seps.append(data['separations'][idx])
                    else:
                        pair_seps.append(0)
                else:
                    pair_seps.append(0)
            
            ax4.bar(x_pos + i*width/len(all_pairs), pair_seps, width/len(all_pairs), 
                   label=pair, color=colors[i % len(colors)], alpha=0.7)
    
    ax4.set_xlabel('Energy (GeV)')
    ax4.set_ylabel('Timing Separation (ns)')
    ax4.set_title('Timing Separations Between Particles')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'{e:.1f}' for e in energies])
    ax4.legend()
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('planar_detector_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('planar_detector_analysis.pdf', bbox_inches='tight')

# Calculate average timing separations
if all_results:
    # Group by energy and calculate separations
    timing_separations = {}
    
    for energy_gev in energies:
        energy_data = [r for r in all_results if r['energy_gev'] == energy_gev]
        if len(energy_data) >= 2:
            # Find pion-kaon and kaon-proton separations
            pion_data = [r for r in energy_data if r['particle'] == 'pip']
            kaon_data = [r for r in energy_data if r['particle'] == 'kaonp']
            proton_data = [r for r in energy_data if r['particle'] == 'proton']
            
            if pion_data and kaon_data:
                pi_k_sep = abs(kaon_data[0]['tof_ns'] - pion_data[0]['tof_ns'])
                timing_separations[f'pi_kaon_{energy_gev}GeV'] = pi_k_sep
            
            if kaon_data and proton_data:
                k_p_sep = abs(proton_data[0]['tof_ns'] - kaon_data[0]['tof_ns'])
                timing_separations[f'kaon_proton_{energy_gev}GeV'] = k_p_sep

# Save results to JSON
output_data = {
    'analysis_type': 'planar_detector_performance',
    'detector_parameters': {
        'light_yield_per_mev': light_yield,
        'path_length_m': path_length
    },
    'particle_results': results,
    'timing_separations': timing_separations if 'timing_separations' in locals() else {},
    'summary': {
        'total_particles_analyzed': len(set(r['particle'] for r in all_results)),
        'energy_points': len(energies),
        'total_events': sum(r['num_events'] for r in all_results)
    }
}

with open('planar_detector_analysis.json', 'w') as f:
    json.dump(output_data, f, indent=2)

# Output results
print("\n=== PLANAR DETECTOR ANALYSIS SUMMARY ===")
print(f"Light yield: {light_yield} photons/MeV")
print(f"Path length: {path_length} m")
print(f"Particles analyzed: {len(set(r['particle'] for r in all_results))}")
print(f"Energy points: {len(energies)}")
print(f"Total events: {sum(r['num_events'] for r in all_results)}")

if 'timing_separations' in locals():
    print("\nTiming separations:")
    for pair, separation in timing_separations.items():
        print(f"  {pair}: {separation:.3f} ns")

# Return values for downstream steps
print(f"RESULT:light_yield_per_mev={light_yield}")
print(f"RESULT:path_length_m={path_length}")
print(f"RESULT:analysis_file=planar_detector_analysis.json")
print(f"RESULT:analysis_plot=planar_detector_analysis.png")
print(f"RESULT:particles_analyzed={len(set(r['particle'] for r in all_results))}")
print(f"RESULT:total_events_analyzed={sum(r['num_events'] for r in all_results)}")
print("RESULT:success=True")

print("\nPlanar detector analysis completed successfully!")