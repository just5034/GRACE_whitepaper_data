import numpy as np
import json
import math

# Input parameters
detector_medium = 'liquid argon'
detector_diameter = 3.0  # meters
detector_height = 2.5    # meters
sensor_count_target = 75
sensor_type = 'SiPM with TPB coating'
optimization_goal = 'maximize light collection efficiency'

# Calculate detector geometry
radius = detector_diameter / 2.0
surface_area_barrel = 2 * math.pi * radius * detector_height
surface_area_endcaps = 2 * math.pi * radius**2
total_surface_area = surface_area_barrel + surface_area_endcaps

print(f'Detector radius: {radius:.2f} m')
print(f'Barrel surface area: {surface_area_barrel:.2f} m²')
print(f'Endcap surface area: {surface_area_endcaps:.2f} m²')
print(f'Total surface area: {total_surface_area:.2f} m²')

# Design three configurations
configurations = {}

# Configuration 1: Endcap-heavy (60% endcaps, 40% barrel)
endcap_fraction_1 = 0.6
barrel_fraction_1 = 0.4
endcap_sensors_1 = int(sensor_count_target * endcap_fraction_1)
barrel_sensors_1 = sensor_count_target - endcap_sensors_1

configurations['endcap_heavy'] = {
    'name': 'endcap_heavy',
    'strategy': 'endcap-heavy placement',
    'total_sensors': sensor_count_target,
    'endcap_sensors': endcap_sensors_1,
    'barrel_sensors': barrel_sensors_1,
    'endcap_coverage': endcap_sensors_1 / (endcap_sensors_1 + barrel_sensors_1),
    'barrel_coverage': barrel_sensors_1 / (endcap_sensors_1 + barrel_sensors_1),
    'sensor_positions': {
        'endcap_top': {'count': endcap_sensors_1 // 2, 'z_position': detector_height / 2},
        'endcap_bottom': {'count': endcap_sensors_1 // 2, 'z_position': -detector_height / 2},
        'barrel': {'count': barrel_sensors_1, 'z_range': [-detector_height/2, detector_height/2]}
    }
}

# Configuration 2: Barrel-heavy (60% barrel, 40% endcaps)
barrel_fraction_2 = 0.6
endcap_fraction_2 = 0.4
barrel_sensors_2 = int(sensor_count_target * barrel_fraction_2)
endcap_sensors_2 = sensor_count_target - barrel_sensors_2

configurations['barrel_heavy'] = {
    'name': 'barrel_heavy',
    'strategy': 'barrel-heavy placement',
    'total_sensors': sensor_count_target,
    'endcap_sensors': endcap_sensors_2,
    'barrel_sensors': barrel_sensors_2,
    'endcap_coverage': endcap_sensors_2 / (endcap_sensors_2 + barrel_sensors_2),
    'barrel_coverage': barrel_sensors_2 / (endcap_sensors_2 + barrel_sensors_2),
    'sensor_positions': {
        'endcap_top': {'count': endcap_sensors_2 // 2, 'z_position': detector_height / 2},
        'endcap_bottom': {'count': endcap_sensors_2 // 2, 'z_position': -detector_height / 2},
        'barrel': {'count': barrel_sensors_2, 'z_range': [-detector_height/2, detector_height/2]}
    }
}

# Configuration 3: Uniform distribution (50% barrel, 50% endcaps)
uniform_fraction = 0.5
barrel_sensors_3 = int(sensor_count_target * uniform_fraction)
endcap_sensors_3 = sensor_count_target - barrel_sensors_3

configurations['uniform'] = {
    'name': 'uniform',
    'strategy': 'uniform distribution',
    'total_sensors': sensor_count_target,
    'endcap_sensors': endcap_sensors_3,
    'barrel_sensors': barrel_sensors_3,
    'endcap_coverage': endcap_sensors_3 / (endcap_sensors_3 + barrel_sensors_3),
    'barrel_coverage': barrel_sensors_3 / (endcap_sensors_3 + barrel_sensors_3),
    'sensor_positions': {
        'endcap_top': {'count': endcap_sensors_3 // 2, 'z_position': detector_height / 2},
        'endcap_bottom': {'count': endcap_sensors_3 // 2, 'z_position': -detector_height / 2},
        'barrel': {'count': barrel_sensors_3, 'z_range': [-detector_height/2, detector_height/2]}
    }
}

# Calculate geometry parameters for each configuration
for config_name, config in configurations.items():
    # Calculate sensor densities
    endcap_density = config['endcap_sensors'] / surface_area_endcaps
    barrel_density = config['barrel_sensors'] / surface_area_barrel
    
    config['geometry_parameters'] = {
        'detector_diameter': detector_diameter,
        'detector_height': detector_height,
        'detector_radius': radius,
        'endcap_sensor_density': endcap_density,
        'barrel_sensor_density': barrel_density,
        'sensor_type': sensor_type,
        'fill_material': 'liquid_argon'
    }
    
    print(f'\nConfiguration: {config_name}')
    print(f'  Strategy: {config["strategy"]}')
    print(f'  Total sensors: {config["total_sensors"]}')
    print(f'  Endcap sensors: {config["endcap_sensors"]} ({config["endcap_coverage"]:.1%})')
    print(f'  Barrel sensors: {config["barrel_sensors"]} ({config["barrel_coverage"]:.1%})')
    print(f'  Endcap density: {endcap_density:.2f} sensors/m²')
    print(f'  Barrel density: {barrel_density:.2f} sensors/m²')

# Save configurations to JSON file
with open('lar_detector_configurations.json', 'w') as f:
    json.dump(configurations, f, indent=2)

print(f'\nRESULT:configurations_defined=3')
print(f'RESULT:endcap_heavy_sensors={configurations["endcap_heavy"]["total_sensors"]}')
print(f'RESULT:barrel_heavy_sensors={configurations["barrel_heavy"]["total_sensors"]}')
print(f'RESULT:uniform_sensors={configurations["uniform"]["total_sensors"]}')
print(f'RESULT:detector_diameter_m={detector_diameter}')
print(f'RESULT:detector_height_m={detector_height}')
print(f'RESULT:configuration_file=lar_detector_configurations.json')
print('RESULT:success=True')