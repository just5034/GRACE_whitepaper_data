import numpy as np
import json

# Input parameters
energy_range = [5, 20, 50]  # GeV
iron_interaction_length = 16.8  # cm
absorber_thickness_options = [15, 20, 30]  # cm
layer_count_options = [3, 4, 5]
topologies = ['box', 'cylinder_barrel']

# Physics constants
iron_density = 7.87  # g/cm3
iron_radiation_length = 1.76  # cm
muon_critical_energy = 460  # GeV (for iron)

# Calculate optimal configurations
configurations = {}

for i, topology in enumerate(['box', 'cylinder_barrel']):
    config_name = f'config_{i+1}_{topology}'
    
    # For muon spectrometers, optimize for momentum resolution
    # Higher energy muons need more absorber thickness
    max_energy = max(energy_range)
    
    if max_energy <= 10:
        optimal_thickness = absorber_thickness_options[0]  # 15 cm
        optimal_layers = layer_count_options[0]  # 3 layers
    elif max_energy <= 30:
        optimal_thickness = absorber_thickness_options[1]  # 20 cm
        optimal_layers = layer_count_options[1]  # 4 layers
    else:
        optimal_thickness = absorber_thickness_options[2]  # 30 cm
        optimal_layers = layer_count_options[2]  # 5 layers
    
    # Calculate total absorber depth
    total_depth_cm = optimal_thickness * optimal_layers
    
    # Calculate interaction lengths
    interaction_lengths = total_depth_cm / iron_interaction_length
    
    # Calculate transverse size based on shower containment
    # Muon showers are narrower than EM showers
    moliere_radius_cm = iron_radiation_length * (21.2 / (iron_density * 0.001))  # Approximation
    transverse_size_cm = 4 * moliere_radius_cm  # 4 Moliere radii for 95% containment
    
    # For cylindrical geometry, calculate radius
    if topology == 'cylinder_barrel':
        inner_radius_cm = 50  # Typical for muon spectrometer
        outer_radius_cm = inner_radius_cm + total_depth_cm
        half_length_cm = 200  # 4m total length
    
    configurations[config_name] = {
        'topology': topology,
        'absorber_material': 'iron',
        'active_material': 'plastic_scintillator',
        'absorber_thickness_cm': optimal_thickness,
        'active_thickness_cm': 1.0,  # Standard scintillator thickness
        'num_layers': optimal_layers,
        'total_depth_cm': total_depth_cm,
        'interaction_lengths': interaction_lengths,
        'transverse_size_cm': transverse_size_cm,
        'energy_range_gev': energy_range
    }
    
    if topology == 'cylinder_barrel':
        configurations[config_name].update({
            'inner_radius_cm': inner_radius_cm,
            'outer_radius_cm': outer_radius_cm,
            'half_length_cm': half_length_cm
        })

# Add thick absorber variant (config 3)
config_3_name = 'config_3_thick_absorber'
configurations[config_3_name] = {
    'topology': 'box',
    'absorber_material': 'iron',
    'active_material': 'plastic_scintillator',
    'absorber_thickness_cm': 40,  # Thick absorber variant
    'active_thickness_cm': 1.0,
    'num_layers': 4,
    'total_depth_cm': 160,  # 40cm * 4 layers
    'interaction_lengths': 160 / iron_interaction_length,
    'transverse_size_cm': transverse_size_cm,
    'energy_range_gev': energy_range
}

# Save configurations to JSON
with open('detector_configurations.json', 'w') as f:
    json.dump(configurations, f, indent=2)

# Print summary
print('Muon Spectrometer Detector Configurations:')
print('=' * 50)
for name, config in configurations.items():
    print(f'\n{name.upper()}:')
    print(f'  Topology: {config["topology"]}')
    print(f'  Absorber: {config["absorber_material"]} ({config["absorber_thickness_cm"]} cm thick)')
    print(f'  Active: {config["active_material"]} ({config["active_thickness_cm"]} cm thick)')
    print(f'  Layers: {config["num_layers"]}')
    print(f'  Total depth: {config["total_depth_cm"]} cm ({config["interaction_lengths"]:.2f} λ_int)')
    print(f'  Transverse size: {config["transverse_size_cm"]:.1f} cm')
    if 'inner_radius_cm' in config:
        print(f'  Inner radius: {config["inner_radius_cm"]} cm')
        print(f'  Half length: {config["half_length_cm"]} cm')

# Output results for next steps
print(f'\nRESULT:config_1_topology=box')
print(f'RESULT:config_1_absorber_thickness_cm={configurations["config_1_box"]["absorber_thickness_cm"]}')
print(f'RESULT:config_1_num_layers={configurations["config_1_box"]["num_layers"]}')
print(f'RESULT:config_1_total_depth_cm={configurations["config_1_box"]["total_depth_cm"]}')
print(f'RESULT:config_1_transverse_size_cm={configurations["config_1_box"]["transverse_size_cm"]:.1f}')

print(f'\nRESULT:config_2_topology=cylinder_barrel')
print(f'RESULT:config_2_absorber_thickness_cm={configurations["config_2_cylinder_barrel"]["absorber_thickness_cm"]}')
print(f'RESULT:config_2_num_layers={configurations["config_2_cylinder_barrel"]["num_layers"]}')
print(f'RESULT:config_2_inner_radius_cm={configurations["config_2_cylinder_barrel"]["inner_radius_cm"]}')
print(f'RESULT:config_2_half_length_cm={configurations["config_2_cylinder_barrel"]["half_length_cm"]}')

print(f'\nRESULT:config_3_topology=box')
print(f'RESULT:config_3_absorber_thickness_cm={configurations["config_3_thick_absorber"]["absorber_thickness_cm"]}')
print(f'RESULT:config_3_num_layers={configurations["config_3_thick_absorber"]["num_layers"]}')
print(f'RESULT:config_3_total_depth_cm={configurations["config_3_thick_absorber"]["total_depth_cm"]}')

print('\nRESULT:configurations_file=detector_configurations.json')
print('RESULT:success=True')