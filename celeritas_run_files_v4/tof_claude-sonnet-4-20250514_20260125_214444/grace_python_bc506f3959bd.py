import matplotlib
matplotlib.use('Agg')
import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Configuration parameters
light_yield_per_mev = 10000
pi_k_timing_req = 80  # ps
k_p_timing_req = 180  # ps

# File paths for all configurations
files = {
    'baseline': 'baseline_planar_tof_pip_hits.root',
    'cylindrical': 'cylindrical_tof_pip_hits.root', 
    'segmented': 'segmented_tile_tof_pip_hits.root',
    'thick_planar': 'thick_planar_tof_pip_hits.root'
}

# Particle files for each configuration
particle_files = {
    'baseline': {
        'pion': 'baseline_planar_tof_pip_hits.root',
        'kaon': 'baseline_planar_tof_kaonp_hits.root',
        'proton': 'baseline_planar_tof_proton_hits.root'
    },
    'cylindrical': {'pion': 'cylindrical_tof_pip_hits.root'},
    'segmented': {'pion': 'segmented_tile_tof_pip_hits.root'},
    'thick_planar': {'pion': 'thick_planar_tof_pip_hits.root'}
}

results = {}

# Analyze each configuration
for config_name in ['baseline', 'cylindrical', 'segmented', 'thick_planar']:
    print(f'Analyzing {config_name} configuration...')
    
    config_results = {}
    
    # For baseline, we have all three particles
    if config_name == 'baseline':
        for particle in ['pion', 'kaon', 'proton']:
            file_path = particle_files[config_name][particle]
            if Path(file_path).exists():
                with uproot.open(file_path) as f:
                    events = f['events'].arrays(library='pd')
                    
                mean_edep = events['totalEdep'].mean()
                std_edep = events['totalEdep'].std()
                resolution = std_edep / mean_edep if mean_edep > 0 else 0
                light_yield = mean_edep * light_yield_per_mev
                
                config_results[particle] = {
                    'mean_edep': mean_edep,
                    'resolution': resolution,
                    'light_yield': light_yield
                }
    else:
        # For other configs, analyze pion data
        file_path = particle_files[config_name]['pion']
        if Path(file_path).exists():
            with uproot.open(file_path) as f:
                events = f['events'].arrays(library='pd')
                
            mean_edep = events['totalEdep'].mean()
            std_edep = events['totalEdep'].std()
            resolution = std_edep / mean_edep if mean_edep > 0 else 0
            light_yield = mean_edep * light_yield_per_mev
            
            config_results['pion'] = {
                'mean_edep': mean_edep,
                'resolution': resolution, 
                'light_yield': light_yield
            }
    
    results[config_name] = config_results

# Use baseline results for particle separation (from previous analysis)
baseline_results = {
    'pion_light_yield': 49907,
    'kaon_light_yield': 43260,
    'proton_light_yield': 54333,
    'pion_resolution': 2.219,
    'kaon_resolution': 1.6326,
    'proton_resolution': 2.3549
}

# Calculate particle separation metrics
def calculate_separation(mean1, sigma1, mean2, sigma2):
    return abs(mean1 - mean2) / np.sqrt(sigma1**2 + sigma2**2)

pi_k_sep = calculate_separation(
    baseline_results['pion_light_yield'], 
    baseline_results['pion_light_yield'] * baseline_results['pion_resolution'] / 100,
    baseline_results['kaon_light_yield'],
    baseline_results['kaon_light_yield'] * baseline_results['kaon_resolution'] / 100
)

k_p_sep = calculate_separation(
    baseline_results['kaon_light_yield'],
    baseline_results['kaon_light_yield'] * baseline_results['kaon_resolution'] / 100, 
    baseline_results['proton_light_yield'],
    baseline_results['proton_light_yield'] * baseline_results['proton_resolution'] / 100
)

# Estimate timing resolution (simplified model)
def estimate_timing_resolution(light_yield):
    # Timing resolution scales as 1/sqrt(N_pe) for photoelectrons
    # Assume 20% quantum efficiency
    n_pe = light_yield * 0.2
    # Base timing resolution of scintillator ~50 ps
    base_timing = 50  # ps
    statistical_term = base_timing / np.sqrt(n_pe) if n_pe > 0 else 1000
    return np.sqrt(base_timing**2 + statistical_term**2)

# Create comprehensive comparison plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Light yield comparison
configs = ['Baseline', 'Cylindrical', 'Segmented', 'Thick Planar']
pion_yields = []
for config in ['baseline', 'cylindrical', 'segmented', 'thick_planar']:
    if 'pion' in results[config]:
        pion_yields.append(results[config]['pion']['light_yield'])
    else:
        pion_yields.append(0)

ax1.bar(configs, pion_yields, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
ax1.set_ylabel('Light Yield (photons)')
ax1.set_title('Pion Light Yield Comparison')
ax1.tick_params(axis='x', rotation=45)

# Plot 2: Energy resolution comparison
resolutions = []
for config in ['baseline', 'cylindrical', 'segmented', 'thick_planar']:
    if 'pion' in results[config]:
        resolutions.append(results[config]['pion']['resolution'] * 100)
    else:
        resolutions.append(0)

ax2.bar(configs, resolutions, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
ax2.set_ylabel('Energy Resolution (%)')
ax2.set_title('Energy Resolution Comparison')
ax2.tick_params(axis='x', rotation=45)

# Plot 3: Particle identification (baseline only)
particles = ['Pion', 'Kaon', 'Proton']
particle_yields = [baseline_results['pion_light_yield'], 
                  baseline_results['kaon_light_yield'],
                  baseline_results['proton_light_yield']]

ax3.bar(particles, particle_yields, color=['blue', 'green', 'red'], alpha=0.7)
ax3.set_ylabel('Light Yield (photons)')
ax3.set_title('Particle Identification (Baseline)')

# Plot 4: Timing resolution estimates
timing_resolutions = []
for yield_val in pion_yields:
    if yield_val > 0:
        timing_resolutions.append(estimate_timing_resolution(yield_val))
    else:
        timing_resolutions.append(1000)

ax4.bar(configs, timing_resolutions, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
ax4.axhline(pi_k_timing_req, color='red', linestyle='--', label=f'π-K requirement ({pi_k_timing_req} ps)')
ax4.axhline(k_p_timing_req, color='orange', linestyle='--', label=f'K-p requirement ({k_p_timing_req} ps)')
ax4.set_ylabel('Timing Resolution (ps)')
ax4.set_title('Estimated Timing Resolution')
ax4.tick_params(axis='x', rotation=45)
ax4.legend()
ax4.set_ylim(0, 300)

plt.tight_layout()
plt.savefig('tof_configuration_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('tof_configuration_comparison.pdf', bbox_inches='tight')

# Print comprehensive results
print('\n=== TOF DETECTOR CONFIGURATION COMPARISON ===')
print(f'Light yield per MeV: {light_yield_per_mev}')
print(f'Timing requirements: π-K < {pi_k_timing_req} ps, K-p < {k_p_timing_req} ps\n')

for config in ['baseline', 'cylindrical', 'segmented', 'thick_planar']:
    print(f'{config.upper()} CONFIGURATION:')
    if 'pion' in results[config]:
        pion_data = results[config]['pion']
        timing_est = estimate_timing_resolution(pion_data['light_yield'])
        print(f'  Pion light yield: {pion_data["light_yield"]:.0f} photons')
        print(f'  Energy resolution: {pion_data["resolution"]*100:.2f}%')
        print(f'  Estimated timing resolution: {timing_est:.1f} ps')
        print(f'  Meets π-K timing: {"YES" if timing_est < pi_k_timing_req else "NO"}')
        print(f'  Meets K-p timing: {"YES" if timing_est < k_p_timing_req else "NO"}')
    print()

print('PARTICLE IDENTIFICATION (Baseline only):')
print(f'π-K separation: {pi_k_sep:.3f} σ')
print(f'K-p separation: {k_p_sep:.3f} σ')
print(f'π-K separation adequate: {"YES" if pi_k_sep > 2.0 else "NO"}')
print(f'K-p separation adequate: {"YES" if k_p_sep > 2.0 else "NO"}')

# Find best configuration
best_config = 'baseline'
best_timing = estimate_timing_resolution(baseline_results['pion_light_yield'])

for config in ['cylindrical', 'segmented', 'thick_planar']:
    if 'pion' in results[config]:
        timing = estimate_timing_resolution(results[config]['pion']['light_yield'])
        if timing < best_timing:
            best_timing = timing
            best_config = config

print(f'\nRECOMMENDED CONFIGURATION: {best_config.upper()}')
print(f'Best estimated timing resolution: {best_timing:.1f} ps')

# Return key metrics
print(f'RESULT:best_configuration={best_config}')
print(f'RESULT:best_timing_resolution={best_timing:.1f}')
print(f'RESULT:pi_k_separation={pi_k_sep:.3f}')
print(f'RESULT:k_p_separation={k_p_sep:.3f}')
print(f'RESULT:comparison_plot=tof_configuration_comparison.png')
print(f'RESULT:meets_timing_requirements={"YES" if best_timing < pi_k_timing_req else "NO"}')
print(f'RESULT:particle_id_feasible={"YES" if pi_k_sep > 2.0 and k_p_sep > 2.0 else "NO"}')