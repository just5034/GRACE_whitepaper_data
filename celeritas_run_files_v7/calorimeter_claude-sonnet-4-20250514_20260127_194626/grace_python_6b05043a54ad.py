import numpy as np
import json

# Design parameters from input specification
absorber_material = 'steel'
interaction_length_cm = 16.8  # cm
total_depth_interaction_lengths = 10
sampling_fraction = 0.03
energy_range_gev = [10, 30, 50]

# Steel material properties
steel_density = 7.87  # g/cm3
steel_radiation_length = 1.76  # cm (X0)
steel_nuclear_interaction_length = 16.8  # cm (lambda_I)

# Calculate total calorimeter depth
total_depth_cm = total_depth_interaction_lengths * interaction_length_cm
total_depth_m = total_depth_cm / 100

print(f'Total calorimeter depth: {total_depth_cm} cm ({total_depth_m} m)')

# Design layer structure for sampling fraction = 3%
# Sampling fraction = active_thickness / (absorber_thickness + active_thickness)
# For steel absorber with plastic scintillator active layers

# Typical steel absorber thickness for hadronic calorimeter: 20mm
absorber_thickness_mm = 20
absorber_thickness_m = absorber_thickness_mm / 1000

# Calculate active thickness from sampling fraction
# SF = t_active / (t_absorber + t_active)
# t_active = SF * t_absorber / (1 - SF)
active_thickness_m = sampling_fraction * absorber_thickness_m / (1 - sampling_fraction)
active_thickness_mm = active_thickness_m * 1000

print(f'Absorber thickness: {absorber_thickness_mm} mm ({absorber_thickness_m} m)')
print(f'Active thickness: {active_thickness_mm:.2f} mm ({active_thickness_m:.5f} m)')

# Calculate number of layers needed
layer_thickness_m = absorber_thickness_m + active_thickness_m
num_layers = int(total_depth_m / layer_thickness_m)
actual_depth_m = num_layers * layer_thickness_m
actual_depth_cm = actual_depth_m * 100

print(f'Layer thickness: {layer_thickness_m*1000:.2f} mm')
print(f'Number of layers: {num_layers}')
print(f'Actual depth: {actual_depth_cm:.1f} cm ({actual_depth_m:.3f} m)')

# Calculate lateral dimensions for shower containment
# Hadronic showers: ~95% containment at 2 lambda_I radius
# For 50 GeV hadrons, need ~1.5 lambda_I radius for good containment
containment_radius_cm = 1.5 * interaction_length_cm
lateral_diameter_cm = 2 * containment_radius_cm
lateral_diameter_m = lateral_diameter_cm / 100

print(f'Containment radius: {containment_radius_cm:.1f} cm')
print(f'Lateral diameter: {lateral_diameter_cm:.1f} cm ({lateral_diameter_m:.3f} m)')

# Calculate sampling configuration
actual_sampling_fraction = active_thickness_m / layer_thickness_m
print(f'Actual sampling fraction: {actual_sampling_fraction:.4f} ({actual_sampling_fraction*100:.2f}%)')

# Calculate total material budget in radiation lengths
total_absorber_thickness_cm = num_layers * absorber_thickness_mm / 10  # convert mm to cm
total_x0 = total_absorber_thickness_cm / steel_radiation_length
print(f'Total absorber material: {total_absorber_thickness_cm:.1f} cm steel')
print(f'Total material budget: {total_x0:.1f} X0')

# Energy resolution estimate (Stochastic term dominant for sampling calorimeters)
# sigma/E = a/sqrt(E) + b + c/E, where a ~ 50-100% for hadronic
# For steel/scintillator: a ~ 80%
stochastic_term = 0.80  # 80% stochastic term
resolution_estimates = []
for energy in energy_range_gev:
    resolution = stochastic_term / np.sqrt(energy)
    resolution_estimates.append(resolution)
    print(f'Estimated energy resolution at {energy} GeV: {resolution:.3f} ({resolution*100:.1f}%)')

# Summary of baseline configuration
baseline_config = {
    'detector_name': 'baseline_steel_calorimeter',
    'topology': 'box',
    'absorber_material': 'steel',
    'active_material': 'plastic_scintillator',
    'absorber_thickness_m': absorber_thickness_m,
    'active_thickness_m': active_thickness_m,
    'num_layers': num_layers,
    'total_depth_m': actual_depth_m,
    'lateral_diameter_m': lateral_diameter_m,
    'sampling_fraction': actual_sampling_fraction,
    'total_interaction_lengths': actual_depth_cm / interaction_length_cm,
    'total_radiation_lengths': total_x0
}

# Save configuration
with open('baseline_calorimeter_config.json', 'w') as f:
    json.dump(baseline_config, f, indent=2)

# Output results for downstream steps
print(f'RESULT:absorber_thickness_m={absorber_thickness_m}')
print(f'RESULT:active_thickness_m={active_thickness_m:.5f}')
print(f'RESULT:num_layers={num_layers}')
print(f'RESULT:total_depth_m={actual_depth_m:.3f}')
print(f'RESULT:lateral_diameter_m={lateral_diameter_m:.3f}')
print(f'RESULT:lateral_diameter_cm={lateral_diameter_cm:.1f}')
print(f'RESULT:sampling_fraction={actual_sampling_fraction:.4f}')
print(f'RESULT:total_interaction_lengths={actual_depth_cm/interaction_length_cm:.1f}')
print(f'RESULT:estimated_resolution_10gev={resolution_estimates[0]:.4f}')
print(f'RESULT:estimated_resolution_30gev={resolution_estimates[1]:.4f}')
print(f'RESULT:estimated_resolution_50gev={resolution_estimates[2]:.4f}')
print('RESULT:config_file=baseline_calorimeter_config.json')
print('RESULT:success=True')

print('\nBaseline planar sampling calorimeter design completed!')
print(f'Steel absorber, plastic scintillator active layers')
print(f'Configuration saved to baseline_calorimeter_config.json')