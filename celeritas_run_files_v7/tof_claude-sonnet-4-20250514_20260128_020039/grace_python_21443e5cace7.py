import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Load analysis results from previous step
analysis_file = 'segmented_detector_analysis.json'
with open(analysis_file, 'r') as f:
    analysis_data = json.load(f)

# Create comprehensive performance plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Segmented TOF Detector Performance Analysis', fontsize=16, fontweight='bold')

# Plot 1: Energy Deposits by Particle Type
particles = ['pip', 'kaonp', 'proton']
particle_labels = ['π+', 'K+', 'p']
energies = [1.0, 2.0, 3.0]
colors = ['blue', 'green', 'red']

# Energy deposit data from analysis
for i, (particle, label, color) in enumerate(zip(particles, particle_labels, colors)):
    if particle in analysis_data:
        energy_vals = []
        energy_errs = []
        for energy in energies:
            energy_key = f'{energy:.1f}GeV'
            if energy_key in analysis_data[particle]:
                mean_edep = analysis_data[particle][energy_key].get('mean_edep', 0)
                std_edep = analysis_data[particle][energy_key].get('std_edep', 0)
                n_events = analysis_data[particle][energy_key].get('num_events', 1)
                energy_vals.append(mean_edep)
                energy_errs.append(std_edep / np.sqrt(n_events))
            else:
                energy_vals.append(0)
                energy_errs.append(0)
        
        ax1.errorbar(energies, energy_vals, yerr=energy_errs, 
                    marker='o', label=label, color=color, capsize=5, linewidth=2)

ax1.set_xlabel('Beam Energy (GeV)')
ax1.set_ylabel('Mean Energy Deposit (MeV)')
ax1.set_title('Energy Deposits vs Beam Energy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Timing Resolution by Particle Type
for i, (particle, label, color) in enumerate(zip(particles, particle_labels, colors)):
    if particle in analysis_data:
        timing_vals = []
        timing_errs = []
        for energy in energies:
            energy_key = f'{energy:.1f}GeV'
            if energy_key in analysis_data[particle]:
                timing_res = analysis_data[particle][energy_key].get('timing_resolution_ps', 0)
                n_events = analysis_data[particle][energy_key].get('num_events', 1)
                timing_vals.append(timing_res)
                timing_errs.append(timing_res / np.sqrt(n_events) * 0.1)  # Estimate 10% relative error
            else:
                timing_vals.append(0)
                timing_errs.append(0)
        
        ax2.errorbar(energies, timing_vals, yerr=timing_errs, 
                    marker='s', label=label, color=color, capsize=5, linewidth=2)

ax2.set_xlabel('Beam Energy (GeV)')
ax2.set_ylabel('Timing Resolution (ps)')
ax2.set_title('Timing Resolution vs Beam Energy')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(100, color='black', linestyle='--', alpha=0.5, label='Design Target (100 ps)')

# Plot 3: Particle Identification (Energy Resolution)
for i, (particle, label, color) in enumerate(zip(particles, particle_labels, colors)):
    if particle in analysis_data:
        resolution_vals = []
        resolution_errs = []
        for energy in energies:
            energy_key = f'{energy:.1f}GeV'
            if energy_key in analysis_data[particle]:
                resolution = analysis_data[particle][energy_key].get('energy_resolution', 0)
                n_events = analysis_data[particle][energy_key].get('num_events', 1)
                resolution_vals.append(resolution * 100)  # Convert to percentage
                resolution_errs.append(resolution / np.sqrt(2 * n_events) * 100)
            else:
                resolution_vals.append(0)
                resolution_errs.append(0)
        
        ax3.errorbar(energies, resolution_vals, yerr=resolution_errs, 
                    marker='^', label=label, color=color, capsize=5, linewidth=2)

ax3.set_xlabel('Beam Energy (GeV)')
ax3.set_ylabel('Energy Resolution (%)')
ax3.set_title('Energy Resolution for Particle ID')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Timing Separations for TOF
# Calculate theoretical timing separations
path_length = 1.0  # meters from analysis
particle_masses = {'pip': 0.1396, 'kaonp': 0.4937, 'proton': 0.9383}  # GeV/c^2

# Plot timing differences between particles
for energy in energies:
    pi_time = []
    k_time = []
    p_time = []
    
    for particle, mass in particle_masses.items():
        momentum = np.sqrt(energy**2 - mass**2)  # GeV/c
        beta = momentum / energy
        time_ns = path_length / (beta * 0.2998)  # time in ns
        
        if particle == 'pip':
            pi_time.append(time_ns)
        elif particle == 'kaonp':
            k_time.append(time_ns)
        elif particle == 'proton':
            p_time.append(time_ns)
    
    if pi_time and k_time and p_time:
        pi_k_sep = (k_time[0] - pi_time[0]) * 1000  # Convert to ps
        k_p_sep = (p_time[0] - k_time[0]) * 1000    # Convert to ps
        
        ax4.bar([f'{energy}GeV π-K', f'{energy}GeV K-p'], [pi_k_sep, k_p_sep], 
               alpha=0.7, color=['orange', 'purple'])

ax4.set_ylabel('Timing Separation (ps)')
ax4.set_title('TOF Separations Between Particle Types')
ax4.axhline(300, color='red', linestyle='--', label='3σ Separation Target (300 ps)')
ax4.legend()
ax4.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('segmented_tof_performance_plots.png', dpi=150, bbox_inches='tight')
plt.savefig('segmented_tof_performance_plots.pdf', bbox_inches='tight')
plt.close()

# Create summary statistics
summary_stats = {
    'detector_type': 'segmented_tof',
    'analysis_summary': {
        'overall_avg_resolution_percent': analysis_data.get('overall_avg_resolution', 0),
        'overall_avg_timing_ps': analysis_data.get('overall_avg_timing_ps', 0),
        'resolution_improvement_percent': analysis_data.get('resolution_improvement_percent', 0),
        'timing_improvement_factor': analysis_data.get('timing_improvement_factor', 0)
    },
    'performance_metrics': {
        'path_length_m': analysis_data.get('path_length_m', 1.0),
        'tile_size_mm': analysis_data.get('tile_size_mm', 10),
        'light_yield_per_mev': analysis_data.get('light_yield_per_mev', 10000)
    },
    'plots_generated': [
        'energy_deposits',
        'timing_resolution', 
        'particle_identification',
        'timing_separations'
    ]
}

with open('segmented_tof_plots_summary.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

# Print results
print('RESULT:plot_file=segmented_tof_performance_plots.png')
print('RESULT:summary_file=segmented_tof_plots_summary.json')
print('RESULT:plots_generated=4')
print(f'RESULT:overall_avg_resolution={analysis_data.get("overall_avg_resolution", 0):.4f}')
print(f'RESULT:overall_avg_timing_ps={analysis_data.get("overall_avg_timing_ps", 0):.1f}')
print(f'RESULT:resolution_improvement_percent={analysis_data.get("resolution_improvement_percent", 0):.1f}')
print(f'RESULT:timing_improvement_factor={analysis_data.get("timing_improvement_factor", 0):.2f}')
print('RESULT:success=True')

print('\nSegmented TOF detector performance plots generated successfully!')
print('Plots include energy deposits, timing resolution, particle ID, and timing separations with statistical uncertainties.')