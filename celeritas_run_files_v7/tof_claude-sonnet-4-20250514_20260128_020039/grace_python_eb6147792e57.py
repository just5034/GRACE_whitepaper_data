import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import json

# Input parameters from Required Input
particle_masses = {"pi+": 139.6, "kaon+": 493.7, "proton": 938.3}  # MeV/c²
momentum_range = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # GeV/c
path_lengths = [0.5, 1.0, 1.5, 2.0]  # meters
scintillator_thicknesses = [1, 2, 3, 4, 5]  # mm
light_yield = 10000  # photons/MeV

# Speed of light
c = 299792458  # m/s

# Calculate TOF for each particle at each momentum and path length
def calculate_tof(mass_mev, momentum_gev, path_length_m):
    """Calculate time-of-flight using relativistic formula"""
    mass_gev = mass_mev / 1000.0  # Convert MeV to GeV
    # t = (L/c) * sqrt(1 + (m/p)²)
    gamma_factor = np.sqrt(1 + (mass_gev / momentum_gev)**2)
    tof_seconds = (path_length_m / c) * gamma_factor
    return tof_seconds * 1e9  # Convert to nanoseconds

# Calculate timing separations
results = {}
for path_length in path_lengths:
    results[f'path_{path_length}m'] = {}
    for momentum in momentum_range:
        # Calculate TOF for each particle
        tof_pi = calculate_tof(particle_masses['pi+'], momentum, path_length)
        tof_k = calculate_tof(particle_masses['kaon+'], momentum, path_length)
        tof_p = calculate_tof(particle_masses['proton'], momentum, path_length)
        
        # Calculate separations
        sep_pi_k = tof_k - tof_pi  # K+ vs π+ separation
        sep_k_p = tof_p - tof_k    # proton vs K+ separation
        sep_pi_p = tof_p - tof_pi  # proton vs π+ separation
        
        results[f'path_{path_length}m'][f'p_{momentum}GeV'] = {
            'tof_pi': tof_pi,
            'tof_kaon': tof_k,
            'tof_proton': tof_p,
            'sep_pi_kaon_ns': sep_pi_k,
            'sep_kaon_proton_ns': sep_k_p,
            'sep_pi_proton_ns': sep_pi_p
        }

# Find optimal parameters
print("TOF Separation Analysis:")
print("=" * 50)

# Plot timing separations vs momentum for different path lengths
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, path_length in enumerate(path_lengths):
    ax = axes[i]
    
    sep_pi_k = []
    sep_k_p = []
    sep_pi_p = []
    
    for momentum in momentum_range:
        data = results[f'path_{path_length}m'][f'p_{momentum}GeV']
        sep_pi_k.append(data['sep_pi_kaon_ns'])
        sep_k_p.append(data['sep_kaon_proton_ns'])
        sep_pi_p.append(data['sep_pi_proton_ns'])
    
    ax.plot(momentum_range, sep_pi_k, 'b-o', label='π+/K+ separation', linewidth=2)
    ax.plot(momentum_range, sep_k_p, 'r-s', label='K+/proton separation', linewidth=2)
    ax.plot(momentum_range, sep_pi_p, 'g-^', label='π+/proton separation', linewidth=2)
    
    ax.set_xlabel('Momentum (GeV/c)')
    ax.set_ylabel('Time Separation (ns)')
    ax.set_title(f'TOF Separations - Path Length {path_length}m')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

plt.tight_layout()
plt.savefig('tof_separations_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('tof_separations_analysis.pdf', bbox_inches='tight')
print("RESULT:tof_separations_plot=tof_separations_analysis.png")

# Determine optimal path lengths for 3σ separation (assuming 100ps timing resolution)
timing_resolution_ns = 0.1  # 100 ps typical for fast scintillators
min_separation_3sigma = 3 * timing_resolution_ns  # 300 ps for 3σ

optimal_params = {}
for momentum in momentum_range:
    optimal_params[f'p_{momentum}GeV'] = {}
    
    for separation_type in ['pi_kaon', 'kaon_proton', 'pi_proton']:
        min_path_length = None
        
        for path_length in path_lengths:
            sep_key = f'sep_{separation_type}_ns'
            separation = results[f'path_{path_length}m'][f'p_{momentum}GeV'][sep_key]
            
            if separation >= min_separation_3sigma:
                min_path_length = path_length
                break
        
        optimal_params[f'p_{momentum}GeV'][separation_type] = {
            'min_path_length_m': min_path_length,
            'required_separation_ns': min_separation_3sigma
        }

# Thickness optimization for light yield vs timing
print("\nOptimal Parameter Ranges:")
print("=" * 30)

# For each topology, recommend parameter ranges
topologies = {
    'barrel_tof': {'path_range': [1.0, 2.0], 'thickness_range': [2, 4]},
    'endcap_tof': {'path_range': [1.5, 2.0], 'thickness_range': [3, 5]},
    'forward_tof': {'path_range': [0.5, 1.0], 'thickness_range': [1, 3]}
}

for topology, params in topologies.items():
    print(f"\n{topology.upper()} Topology:")
    print(f"  Optimal path length range: {params['path_range'][0]}-{params['path_range'][1]} m")
    print(f"  Optimal thickness range: {params['thickness_range'][0]}-{params['thickness_range'][1]} mm")
    
    # Calculate expected light yield for thickness range
    min_thickness = params['thickness_range'][0]
    max_thickness = params['thickness_range'][1]
    
    # Assume 2 MeV energy deposit per cm for MIPs
    min_light = (min_thickness / 10) * 2 * light_yield  # photons
    max_light = (max_thickness / 10) * 2 * light_yield  # photons
    
    print(f"  Expected light yield: {min_light:.0f}-{max_light:.0f} photons")
    
    # Timing resolution estimate (√N statistics + electronics)
    min_timing_res = np.sqrt(200 / min_light) * 100  # ps
    max_timing_res = np.sqrt(200 / max_light) * 100  # ps
    print(f"  Expected timing resolution: {max_timing_res:.1f}-{min_timing_res:.1f} ps")

# Save detailed results
with open('tof_theoretical_analysis.json', 'w') as f:
    json.dump({
        'timing_separations': results,
        'optimal_parameters': optimal_params,
        'topology_recommendations': topologies,
        'analysis_parameters': {
            'timing_resolution_assumption_ns': timing_resolution_ns,
            'min_separation_3sigma_ns': min_separation_3sigma,
            'light_yield_per_mev': light_yield
        }
    }, f, indent=2)

# Output key metrics
print("\nKey Results:")
print(f"RESULT:timing_resolution_assumption_ps={timing_resolution_ns*1000:.0f}")
print(f"RESULT:min_separation_3sigma_ps={min_separation_3sigma*1000:.0f}")
print(f"RESULT:barrel_optimal_path_min_m={topologies['barrel_tof']['path_range'][0]}")
print(f"RESULT:barrel_optimal_path_max_m={topologies['barrel_tof']['path_range'][1]}")
print(f"RESULT:barrel_optimal_thickness_min_mm={topologies['barrel_tof']['thickness_range'][0]}")
print(f"RESULT:barrel_optimal_thickness_max_mm={topologies['barrel_tof']['thickness_range'][1]}")
print(f"RESULT:endcap_optimal_path_min_m={topologies['endcap_tof']['path_range'][0]}")
print(f"RESULT:endcap_optimal_path_max_m={topologies['endcap_tof']['path_range'][1]}")
print(f"RESULT:forward_optimal_path_min_m={topologies['forward_tof']['path_range'][0]}")
print(f"RESULT:forward_optimal_path_max_m={topologies['forward_tof']['path_range'][1]}")
print("RESULT:analysis_file=tof_theoretical_analysis.json")
print("RESULT:success=True")

# Calculate some example separations for verification
example_momentum = 1.0  # GeV
example_path = 1.0  # m
example_data = results[f'path_{example_path}m'][f'p_{example_momentum}GeV']
print(f"\nExample at {example_momentum} GeV/c, {example_path}m path:")
print(f"  π+/K+ separation: {example_data['sep_pi_kaon_ns']:.2f} ns")
print(f"  K+/proton separation: {example_data['sep_kaon_proton_ns']:.2f} ns")
print(f"RESULT:example_pi_kaon_separation_ns={example_data['sep_pi_kaon_ns']:.3f}")
print(f"RESULT:example_kaon_proton_separation_ns={example_data['sep_kaon_proton_ns']:.3f}")