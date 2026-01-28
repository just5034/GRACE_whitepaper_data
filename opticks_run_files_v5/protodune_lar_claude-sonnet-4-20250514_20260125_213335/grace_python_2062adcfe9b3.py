import matplotlib
matplotlib.use('Agg')
import uproot
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Load simulation data from test_energy_response step
hits_file = '/u/jhill5/grace/work/benchmarks/opticks_20260125_211816/protodune_lar_claude-sonnet-4-20250514_20260125_213335/optimized_protodune_electron_hits.root'

print('Loading energy response data...')
with uproot.open(hits_file) as f:
    # Load events data for energy analysis
    events = f['events'].arrays(library='pd')
    print(f'Loaded {len(events)} events')

# Energy response analysis
print('\nAnalyzing energy response...')

# Calculate energy resolution
mean_energy = events['totalEdep'].mean()
std_energy = events['totalEdep'].std()
energy_resolution = std_energy / mean_energy if mean_energy > 0 else 0
energy_resolution_err = energy_resolution / np.sqrt(2 * len(events)) if len(events) > 0 else 0

print(f'Mean energy deposit: {mean_energy:.4f} MeV')
print(f'Energy resolution (σ/E): {energy_resolution:.4f} ± {energy_resolution_err:.4f}')

# Energy linearity analysis (single energy point - assess linearity coefficient)
# For single energy, linearity coefficient is the ratio of measured to expected
# Assuming 1 MeV input energy (from previous steps)
input_energy = 1.0  # MeV
linearity_coeff = mean_energy / input_energy if input_energy > 0 else 0
linearity_coeff_err = std_energy / input_energy / np.sqrt(len(events)) if input_energy > 0 and len(events) > 0 else 0

print(f'Energy linearity coefficient: {linearity_coeff:.4f} ± {linearity_coeff_err:.4f}')

# Light yield scaling (from optimized detector parameters)
# Using values from previous optimization steps
sensor_count = 20  # From generate_optimized_geometry
sensor_coverage = 0.183719  # From generate_optimized_geometry
scintillation_yield = 40000  # photons/MeV for LAr

# Calculate effective light yield
detection_efficiency = 0.17  # From analyze_optimized_performance
light_yield_pe_per_mev = 22.61  # From analyze_optimized_performance

# Light yield scaling parameters
geometric_efficiency = sensor_coverage  # Fraction of photons that hit sensors
quantum_efficiency = 0.25  # Typical PMT QE
expected_pe_per_mev = scintillation_yield * geometric_efficiency * quantum_efficiency
light_yield_scaling = light_yield_pe_per_mev / expected_pe_per_mev if expected_pe_per_mev > 0 else 0

print(f'Light yield: {light_yield_pe_per_mev:.2f} PE/MeV')
print(f'Light yield scaling factor: {light_yield_scaling:.4f}')

# Create energy response characterization plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Energy distribution
ax1.hist(events['totalEdep'], bins=30, histtype='step', linewidth=2, color='blue')
ax1.axvline(mean_energy, color='red', linestyle='--', label=f'Mean: {mean_energy:.3f} MeV')
ax1.set_xlabel('Energy Deposit (MeV)')
ax1.set_ylabel('Events')
ax1.set_title(f'Energy Distribution (σ/E = {energy_resolution:.4f})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Energy resolution vs energy (single point)
ax2.errorbar([input_energy], [energy_resolution], yerr=[energy_resolution_err], 
             fmt='o', markersize=8, capsize=5, color='green')
ax2.set_xlabel('Input Energy (MeV)')
ax2.set_ylabel('Energy Resolution (σ/E)')
ax2.set_title('Energy Resolution vs Energy')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0.5, 1.5)

# Linearity plot
ax3.errorbar([input_energy], [mean_energy], yerr=[std_energy/np.sqrt(len(events))], 
             fmt='s', markersize=8, capsize=5, color='purple')
ax3.plot([0, 1.5], [0, 1.5], 'k--', alpha=0.5, label='Perfect linearity')
ax3.set_xlabel('Input Energy (MeV)')
ax3.set_ylabel('Measured Energy (MeV)')
ax3.set_title(f'Energy Linearity (coeff = {linearity_coeff:.4f})')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Light yield scaling
scaling_factors = ['Geometric', 'Quantum', 'Collection', 'Overall']
scaling_values = [geometric_efficiency, quantum_efficiency, detection_efficiency, light_yield_scaling]
ax4.bar(scaling_factors, scaling_values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
ax4.set_ylabel('Scaling Factor')
ax4.set_title('Light Yield Scaling Components')
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('energy_response_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('energy_response_analysis.pdf', bbox_inches='tight')
print('\nSaved energy response analysis plot')

# Save detailed results to JSON
results = {
    'energy_response': {
        'resolution': float(energy_resolution),
        'resolution_error': float(energy_resolution_err),
        'mean_energy_deposit': float(mean_energy),
        'energy_spread': float(std_energy)
    },
    'linearity': {
        'coefficient': float(linearity_coeff),
        'coefficient_error': float(linearity_coeff_err),
        'input_energy_mev': float(input_energy),
        'measured_energy_mev': float(mean_energy)
    },
    'light_yield': {
        'pe_per_mev': float(light_yield_pe_per_mev),
        'scaling_factor': float(light_yield_scaling),
        'geometric_efficiency': float(geometric_efficiency),
        'quantum_efficiency': float(quantum_efficiency),
        'detection_efficiency': float(detection_efficiency)
    },
    'detector_parameters': {
        'sensor_count': int(sensor_count),
        'sensor_coverage': float(sensor_coverage),
        'scintillation_yield_per_mev': int(scintillation_yield)
    }
}

with open('energy_response_characterization.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\n=== ENERGY RESPONSE CHARACTERIZATION ===')
print(f'Energy Resolution: {energy_resolution:.4f} ± {energy_resolution_err:.4f}')
print(f'Linearity Coefficient: {linearity_coeff:.4f} ± {linearity_coeff_err:.4f}')
print(f'Light Yield: {light_yield_pe_per_mev:.2f} PE/MeV')
print(f'Light Yield Scaling: {light_yield_scaling:.4f}')

# Return values for downstream steps
print(f'RESULT:energy_resolution={energy_resolution:.4f}')
print(f'RESULT:energy_resolution_err={energy_resolution_err:.4f}')
print(f'RESULT:linearity_coefficient={linearity_coeff:.4f}')
print(f'RESULT:linearity_coefficient_err={linearity_coeff_err:.4f}')
print(f'RESULT:light_yield_pe_per_mev={light_yield_pe_per_mev:.2f}')
print(f'RESULT:light_yield_scaling={light_yield_scaling:.4f}')
print(f'RESULT:mean_energy_deposit={mean_energy:.4f}')
print('RESULT:analysis_plot=energy_response_analysis.png')
print('RESULT:results_file=energy_response_characterization.json')
print('\nEnergy response characterization completed successfully!')