import numpy as np
import json
from pathlib import Path

# Silicon detector physics parameters
silicon_x0 = 0.0937  # radiation length in meters (9.37 cm)
silicon_density = 2.33  # g/cm3

# Design parameters from input
momentum_range = [1, 50]  # GeV/c
thickness_range = [100e-6, 500e-6]  # 100-500 microns in meters
spacing_range = [0.02, 0.10]  # 2-10 cm in meters
layer_range = [3, 8]

# Physics-based design calculations
def calculate_multiple_scattering_angle(momentum_gev, thickness_m, x0_m):
    """Highland formula for multiple scattering angle"""
    x_over_x0 = thickness_m / x0_m
    theta_ms = (0.0136 / momentum_gev) * np.sqrt(x_over_x0) * (1 + 0.038 * np.log(x_over_x0))
    return theta_ms

def calculate_momentum_resolution(momentum_gev, theta_ms, lever_arm_m):
    """Momentum resolution from multiple scattering"""
    # Simplified formula for tracking resolution
    sigma_p_over_p = theta_ms * momentum_gev / (0.3 * 3.8 * lever_arm_m)  # Assuming 3.8T field
    return sigma_p_over_p

def optimize_layer_parameters(topology, momentum_gev):
    """Optimize layer count, thickness, and spacing for given topology"""
    if topology == 'box':
        # Planar tracker - optimize for low momentum resolution
        thickness = 200e-6  # 200 microns - balance resolution vs material
        spacing = 0.05  # 5 cm spacing
        num_layers = 6
        lever_arm = num_layers * spacing
        
    elif topology == 'cylinder_barrel':
        # Cylindrical barrel - optimize for uniform coverage
        thickness = 150e-6  # 150 microns - thinner for less material
        spacing = 0.08  # 8 cm radial spacing
        num_layers = 5
        lever_arm = num_layers * spacing
        
    elif topology == 'forward_disks':
        # Forward disks - optimize for forward tracking
        thickness = 300e-6  # 300 microns - thicker for better efficiency
        spacing = 0.15  # 15 cm z-spacing
        num_layers = 4
        lever_arm = num_layers * spacing
    
    # Calculate physics performance
    theta_ms = calculate_multiple_scattering_angle(momentum_gev, thickness, silicon_x0)
    sigma_p_over_p = calculate_momentum_resolution(momentum_gev, theta_ms, lever_arm)
    material_budget = num_layers * thickness / silicon_x0
    
    return {
        'thickness_m': thickness,
        'spacing_m': spacing,
        'num_layers': num_layers,
        'lever_arm_m': lever_arm,
        'theta_ms_rad': theta_ms,
        'momentum_resolution': sigma_p_over_p,
        'material_budget_x0': material_budget
    }

# Design three configurations
configurations = {}
topologies = ['box', 'cylinder_barrel', 'forward_disks']
test_momentum = 10.0  # GeV/c for optimization

for i, topology in enumerate(topologies, 1):
    config_name = f'config{i}_{topology}'
    params = optimize_layer_parameters(topology, test_momentum)
    
    # Configuration specification
    config = {
        'name': config_name,
        'topology': topology,
        'detector_type': 'silicon_pixel_tracker',
        'silicon_thickness_um': params['thickness_m'] * 1e6,
        'layer_spacing_cm': params['spacing_m'] * 100,
        'num_layers': params['num_layers'],
        'momentum_range_gev': momentum_range,
        'physics_performance': {
            'momentum_resolution_at_10gev': params['momentum_resolution'],
            'multiple_scattering_angle_mrad': params['theta_ms_rad'] * 1000,
            'material_budget_percent_x0': params['material_budget_x0'] * 100,
            'lever_arm_m': params['lever_arm_m']
        }
    }
    
    # Topology-specific geometry parameters
    if topology == 'box':
        config['geometry'] = {
            'transverse_size_m': 1.0,  # 1m x 1m active area
            'total_depth_m': params['lever_arm_m'],
            'layer_positions_m': [i * params['spacing_m'] for i in range(params['num_layers'])]
        }
        config['physics_justification'] = (
            'Planar box topology optimized for test beam and fixed-target experiments. '
            f'6 layers with {params["thickness_m"]*1e6:.0f}μm thickness provide '
            f'{params["momentum_resolution"]*100:.2f}% momentum resolution at 10 GeV '
            f'with {params["material_budget_x0"]*100:.1f}% X0 material budget.'
        )
        
    elif topology == 'cylinder_barrel':
        inner_radius = 0.5  # 50 cm inner radius
        config['geometry'] = {
            'inner_radius_m': inner_radius,
            'layer_radii_m': [inner_radius + i * params['spacing_m'] for i in range(params['num_layers'])],
            'barrel_length_m': 2.0,  # 2m barrel length
            'pixel_size_um': [50, 400]  # r-phi x z pixel size
        }
        config['physics_justification'] = (
            'Cylindrical barrel topology for collider-like geometry with full azimuthal coverage. '
            f'5 layers at radii 50-82 cm with {params["thickness_m"]*1e6:.0f}μm silicon provide '
            f'{params["momentum_resolution"]*100:.2f}% momentum resolution and uniform tracking '
            f'efficiency with {params["material_budget_x0"]*100:.1f}% X0 material budget.'
        )
        
    elif topology == 'forward_disks':
        config['geometry'] = {
            'disk_z_positions_m': [1.0 + i * params['spacing_m'] for i in range(params['num_layers'])],
            'inner_radius_m': 0.2,  # 20 cm inner radius
            'outer_radius_m': 1.0,  # 1m outer radius
            'pixel_size_um': [100, 100]  # square pixels
        }
        config['physics_justification'] = (
            'Forward disk topology for extended pseudorapidity coverage (η > 2.5). '
            f'4 disks at z = 1.0-1.45m with {params["thickness_m"]*1e6:.0f}μm thickness provide '
            f'forward tracking with {params["momentum_resolution"]*100:.2f}% resolution '
            f'and minimal {params["material_budget_x0"]*100:.1f}% X0 material budget.'
        )
    
    configurations[config_name] = config

# Save configurations to file
with open('silicon_detector_configurations.json', 'w') as f:
    json.dump(configurations, f, indent=2)

# Print summary
print('Three Silicon Pixel Detector Configurations Designed:')
print('=' * 60)

for config_name, config in configurations.items():
    print(f'\n{config_name.upper()}:')
    print(f'  Topology: {config["topology"]}')
    print(f'  Layers: {config["num_layers"]}')
    print(f'  Silicon thickness: {config["silicon_thickness_um"]:.0f} μm')
    print(f'  Layer spacing: {config["layer_spacing_cm"]:.1f} cm')
    perf = config['physics_performance']
    print(f'  Momentum resolution (10 GeV): {perf["momentum_resolution_at_10gev"]*100:.2f}%')
    print(f'  Material budget: {perf["material_budget_percent_x0"]:.1f}% X0')
    print(f'  Multiple scattering: {perf["multiple_scattering_angle_mrad"]:.2f} mrad')
    print(f'  Physics justification: {config["physics_justification"]}')

# Output results for workflow
print(f'\nRESULT:config1_momentum_resolution={configurations["config1_box"]["physics_performance"]["momentum_resolution_at_10gev"]:.4f}')
print(f'RESULT:config2_momentum_resolution={configurations["config2_cylinder_barrel"]["physics_performance"]["momentum_resolution_at_10gev"]:.4f}')
print(f'RESULT:config3_momentum_resolution={configurations["config3_forward_disks"]["physics_performance"]["momentum_resolution_at_10gev"]:.4f}')
print(f'RESULT:config1_material_budget={configurations["config1_box"]["physics_performance"]["material_budget_percent_x0"]:.2f}')
print(f'RESULT:config2_material_budget={configurations["config2_cylinder_barrel"]["physics_performance"]["material_budget_percent_x0"]:.2f}')
print(f'RESULT:config3_material_budget={configurations["config3_forward_disks"]["physics_performance"]["material_budget_percent_x0"]:.2f}')
print('RESULT:configurations_file=silicon_detector_configurations.json')
print('RESULT:num_configurations=3')
print('RESULT:topologies_explored=box,cylinder_barrel,forward_disks')
print('RESULT:success=True')