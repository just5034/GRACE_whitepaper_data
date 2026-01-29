import numpy as np
import json

# Material properties for homogeneous calorimeters
materials = {
    'BGO': {
        'radiation_length_cm': 1.12,
        'moliere_radius_cm': 2.23,
        'density_g_cm3': 7.13
    },
    'PbWO4': {
        'radiation_length_cm': 0.89,
        'moliere_radius_cm': 2.19,
        'density_g_cm3': 8.28
    },
    'CsI': {
        'radiation_length_cm': 1.86,
        'moliere_radius_cm': 3.57,
        'density_g_cm3': 4.51
    }
}

# Target parameters
target_depth_x0 = 20  # radiation lengths
containment_requirement = 0.95  # 95% containment
energy_range_gev = [0.5, 2, 5]  # GeV

print('Homogeneous Calorimeter Design Parameters')
print('=' * 50)

results = {}

for material_name, props in materials.items():
    print(f'\n{material_name} Calorimeter:')
    
    # Calculate crystal depth for 20 X0
    depth_cm = target_depth_x0 * props['radiation_length_cm']
    depth_m = depth_cm / 100.0
    
    # Calculate lateral dimensions for 95% containment
    # Use 2.5 * Moliere radius for 95% radial containment
    containment_radius_cm = 2.5 * props['moliere_radius_cm']
    lateral_diameter_cm = 2 * containment_radius_cm
    lateral_diameter_m = lateral_diameter_cm / 100.0
    
    # Calculate mass for reference
    volume_cm3 = np.pi * (containment_radius_cm**2) * depth_cm
    mass_kg = volume_cm3 * props['density_g_cm3'] / 1000.0
    
    print(f'  Radiation length: {props["radiation_length_cm"]:.2f} cm')
    print(f'  Moliere radius: {props["moliere_radius_cm"]:.2f} cm')
    print(f'  Crystal depth (20 X0): {depth_cm:.1f} cm ({depth_m:.3f} m)')
    print(f'  95% containment radius: {containment_radius_cm:.1f} cm')
    print(f'  Lateral diameter: {lateral_diameter_cm:.1f} cm ({lateral_diameter_m:.3f} m)')
    print(f'  Estimated mass: {mass_kg:.1f} kg')
    
    # Store results with standardized naming
    material_key = material_name.lower()
    results[f'{material_key}_depth_cm'] = depth_cm
    results[f'{material_key}_depth_m'] = depth_m
    results[f'{material_key}_lateral_diameter_cm'] = lateral_diameter_cm
    results[f'{material_key}_lateral_diameter_m'] = lateral_diameter_m
    results[f'{material_key}_containment_radius_cm'] = containment_radius_cm
    results[f'{material_key}_mass_kg'] = mass_kg
    results[f'{material_key}_20x0_depth_cm'] = depth_cm
    results[f'{material_key}_20x0_diameter_cm'] = lateral_diameter_cm

# Energy-dependent shower parameters
print('\nEnergy-dependent Analysis:')
print('-' * 30)
for energy_gev in energy_range_gev:
    # Critical energy scaling (approximate)
    # Shower maximum scales as log(E/Ec)
    shower_max_x0 = np.log(energy_gev / 0.01)  # Rough approximation
    print(f'  {energy_gev} GeV: Shower max ~ {shower_max_x0:.1f} X0')
    results[f'shower_max_{energy_gev}gev_x0'] = shower_max_x0

# Summary comparison
print('\nMaterial Comparison Summary:')
print('-' * 40)
print('Material | Depth (cm) | Diameter (cm) | Mass (kg)')
print('-' * 40)
for material in ['BGO', 'PbWO4', 'CsI']:
    key = material.lower()
    depth = results[f'{key}_depth_cm']
    diameter = results[f'{key}_lateral_diameter_cm']
    mass = results[f'{key}_mass_kg']
    print(f'{material:8s} | {depth:8.1f} | {diameter:11.1f} | {mass:7.1f}')

# Output results for downstream steps
for key, value in results.items():
    print(f'RESULT:{key}={value:.3f}')

print('RESULT:target_depth_x0=20')
print('RESULT:containment_requirement=0.95')
print('RESULT:success=True')

# Save detailed results to JSON
with open('calorimeter_design_parameters.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\nDetailed parameters saved to calorimeter_design_parameters.json')