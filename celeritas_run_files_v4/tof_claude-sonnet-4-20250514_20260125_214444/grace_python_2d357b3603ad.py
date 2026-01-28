import matplotlib
matplotlib.use('Agg')
import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load simulation data from all three particle types
with uproot.open('/u/jhill5/grace/work/benchmarks/celeritas_20260125_192529/tof_claude-sonnet-4-20250514_20260125_214444/baseline_planar_tof_pip_hits.root') as f:
    pion_events = f['events'].arrays(library='pd')
    pion_hits = f['hits'].arrays(['x', 'y', 'z', 'edep', 'time'], library='np', entry_stop=100000)

with uproot.open('/u/jhill5/grace/work/benchmarks/celeritas_20260125_192529/tof_claude-sonnet-4-20250514_20260125_214444/baseline_planar_tof_kaonp_hits.root') as f:
    kaon_events = f['events'].arrays(library='pd')
    kaon_hits = f['hits'].arrays(['x', 'y', 'z', 'edep', 'time'], library='np', entry_stop=100000)

with uproot.open('/u/jhill5/grace/work/benchmarks/celeritas_20260125_192529/tof_claude-sonnet-4-20250514_20260125_214444/baseline_planar_tof_proton_hits.root') as f:
    proton_events = f['events'].arrays(library='pd')
    proton_hits = f['hits'].arrays(['x', 'y', 'z', 'edep', 'time'], library='np', entry_stop=100000)

# Calculate energy deposit statistics for each particle type
light_yield_per_mev = 10000  # photons per MeV

# Energy deposit analysis
particles = ['Pion', 'Kaon', 'Proton']
events_data = [pion_events, kaon_events, proton_events]

results = {}
for i, (particle, events) in enumerate(zip(particles, events_data)):
    mean_edep = events['totalEdep'].mean()
    std_edep = events['totalEdep'].std()
    resolution = std_edep / mean_edep if mean_edep > 0 else 0
    light_yield = mean_edep * light_yield_per_mev
    
    results[particle.lower()] = {
        'mean_edep': mean_edep,
        'std_edep': std_edep,
        'resolution': resolution,
        'light_yield': light_yield
    }
    
    print(f'{particle} - Mean Energy: {mean_edep:.3f} MeV, Resolution: {resolution:.4f}, Light Yield: {light_yield:.0f} photons')

# Plot energy distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (particle, events) in enumerate(zip(particles, events_data)):
    axes[i].hist(events['totalEdep'], bins=50, alpha=0.7, color=['blue', 'green', 'red'][i])
    axes[i].set_xlabel('Energy Deposit (MeV)')
    axes[i].set_ylabel('Events')
    axes[i].set_title(f'{particle} Energy Distribution')
    axes[i].axvline(results[particle.lower()]['mean_edep'], color='black', linestyle='--', label=f'Mean: {results[particle.lower()]["mean_edep"]:.3f} MeV')
    axes[i].legend()

plt.tight_layout()
plt.savefig('baseline_energy_distributions.png', dpi=150, bbox_inches='tight')
plt.savefig('baseline_energy_distributions.pdf', bbox_inches='tight')

# Timing analysis
hits_data = [pion_hits, kaon_hits, proton_hits]
timing_results = {}
for i, (particle, hits) in enumerate(zip(particles, hits_data)):
    if len(hits['time']) > 0:
        mean_time = np.mean(hits['time'])
        std_time = np.std(hits['time'])
        timing_results[particle.lower()] = {'mean_time': mean_time, 'time_resolution': std_time}
        print(f'{particle} - Mean Time: {mean_time:.3f} ns, Time Resolution: {std_time:.3f} ns')

# Particle separation metrics
separation_metrics = {}
for i in range(len(particles)):
    for j in range(i+1, len(particles)):
        p1, p2 = particles[i], particles[j]
        mean1 = results[p1.lower()]['mean_edep']
        mean2 = results[p2.lower()]['mean_edep']
        std1 = results[p1.lower()]['std_edep']
        std2 = results[p2.lower()]['std_edep']
        
        # Calculate separation power (signal/noise ratio)
        separation = abs(mean1 - mean2) / np.sqrt(0.5 * (std1**2 + std2**2)) if (std1 > 0 and std2 > 0) else 0
        separation_metrics[f'{p1}_{p2}'] = separation
        print(f'{p1}-{p2} Separation Power: {separation:.3f}')

# Summary comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Light yield comparison
particle_names = list(results.keys())
light_yields = [results[p]['light_yield'] for p in particle_names]
ax1.bar(particle_names, light_yields, color=['blue', 'green', 'red'])
ax1.set_ylabel('Light Yield (photons)')
ax1.set_title('Light Yield by Particle Type')
for i, v in enumerate(light_yields):
    ax1.text(i, v + max(light_yields)*0.01, f'{v:.0f}', ha='center')

# Energy resolution comparison
resolutions = [results[p]['resolution'] for p in particle_names]
ax2.bar(particle_names, resolutions, color=['blue', 'green', 'red'])
ax2.set_ylabel('Energy Resolution (σ/E)')
ax2.set_title('Energy Resolution by Particle Type')
for i, v in enumerate(resolutions):
    ax2.text(i, v + max(resolutions)*0.01, f'{v:.4f}', ha='center')

plt.tight_layout()
plt.savefig('baseline_performance_summary.png', dpi=150, bbox_inches='tight')
plt.savefig('baseline_performance_summary.pdf', bbox_inches='tight')

# Output results for downstream steps
print(f'RESULT:pion_light_yield={results["pion"]["light_yield"]:.0f}')
print(f'RESULT:kaon_light_yield={results["kaon"]["light_yield"]:.0f}')
print(f'RESULT:proton_light_yield={results["proton"]["light_yield"]:.0f}')
print(f'RESULT:pion_resolution={results["pion"]["resolution"]:.4f}')
print(f'RESULT:kaon_resolution={results["kaon"]["resolution"]:.4f}')
print(f'RESULT:proton_resolution={results["proton"]["resolution"]:.4f}')
print(f'RESULT:pion_kaon_separation={separation_metrics.get("Pion_Kaon", 0):.3f}')
print(f'RESULT:kaon_proton_separation={separation_metrics.get("Kaon_Proton", 0):.3f}')
print(f'RESULT:pion_proton_separation={separation_metrics.get("Pion_Proton", 0):.3f}')
print('RESULT:energy_distributions_plot=baseline_energy_distributions.png')
print('RESULT:performance_summary_plot=baseline_performance_summary.png')