import matplotlib
matplotlib.use('Agg')
import numpy as np
import json
import matplotlib.pyplot as plt

# Input parameters from step requirements
momentum_range = [1, 50]  # GeV/c
silicon_x0 = 0.0937  # radiation length in meters (9.37 cm)
thickness_range = [100e-6, 500e-6]  # 100-500 microns in meters
spacing_range = [0.02, 0.10]  # 2-10 cm in meters
layer_range = [3, 8]

# Design three distinct configurations exploring different topologies
configurations = []

# Configuration 1: Box topology - minimal material budget for low momentum
config1 = {
    'name': 'config1_box_minimal',
    'topology': 'box',
    'layer_count': 4,
    'silicon_thickness_um': 150,  # thin for minimal scattering
    'layer_spacing_cm': 3.0,     # tight spacing for good resolution
    'transverse_size_cm': 20.0,  # compact detector
    'optimization_target': 'material_budget_minimization',
    'physics_justification': 'Minimal material budget (0.64% X0 total) optimized for low momentum tracks (1-10 GeV/c) where multiple scattering dominates'
}

# Calculate material budget for config1
config1_material_budget = (config1['layer_count'] * config1['silicon_thickness_um'] * 1e-6) / silicon_x0
config1['material_budget_percent'] = config1_material_budget * 100

# Configuration 2: Cylinder barrel - balanced performance
config2 = {
    'name': 'config2_cylinder_balanced',
    'topology': 'cylinder_barrel',
    'layer_count': 6,
    'silicon_thickness_um': 300,  # thicker for better signal
    'inner_radius_cm': 5.0,
    'outer_radius_cm': 25.0,
    'half_length_cm': 30.0,
    'optimization_target': 'momentum_resolution',
    'physics_justification': 'Cylindrical geometry provides full azimuthal coverage with 6 measurement points for optimal momentum resolution at medium energies (10-30 GeV/c)'
}

# Calculate layer spacing for config2
layer_radii = np.linspace(config2['inner_radius_cm'], config2['outer_radius_cm'], config2['layer_count'])
config2['layer_radii_cm'] = layer_radii.tolist()
config2_material_budget = (config2['layer_count'] * config2['silicon_thickness_um'] * 1e-6) / silicon_x0
config2['material_budget_percent'] = config2_material_budget * 100

# Configuration 3: Forward disks - high momentum optimization
config3 = {
    'name': 'config3_forward_disks',
    'topology': 'forward_disks',
    'layer_count': 8,
    'silicon_thickness_um': 500,  # thick for maximum signal
    'disk_z_positions_cm': [10, 15, 25, 40, 60, 85, 115, 150],  # increasing spacing
    'inner_radius_cm': 3.0,
    'outer_radius_cm': 40.0,
    'optimization_target': 'momentum_resolution',
    'physics_justification': 'Forward disk geometry with 8 layers and increasing Z spacing optimized for high momentum tracks (30-50 GeV/c) where measurement precision dominates over multiple scattering'
}

config3_material_budget = (config3['layer_count'] * config3['silicon_thickness_um'] * 1e-6) / silicon_x0
config3['material_budget_percent'] = config3_material_budget * 100

# Calculate expected momentum resolution for each configuration
def calculate_momentum_resolution(config, momentum_gev):
    """Calculate momentum resolution using Highland multiple scattering formula"""
    # Highland formula: theta_ms = (0.0136/p) * sqrt(x/X0) * (1 + 0.038*ln(x/X0))
    x_over_x0 = config['material_budget_percent'] / 100
    theta_ms = (0.0136 / momentum_gev) * np.sqrt(x_over_x0) * (1 + 0.038 * np.log(x_over_x0))
    
    # Momentum resolution approximation: sigma_p/p ~ theta_ms for tracking
    return theta_ms

# Calculate resolution at 10 GeV for comparison
test_momentum = 10.0  # GeV/c
config1['momentum_resolution_10gev'] = calculate_momentum_resolution(config1, test_momentum)
config2['momentum_resolution_10gev'] = calculate_momentum_resolution(config2, test_momentum)
config3['momentum_resolution_10gev'] = calculate_momentum_resolution(config3, test_momentum)

configurations = [config1, config2, config3]

# Create comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Material budget comparison
names = [c['name'].replace('_', ' ').title() for c in configurations]
material_budgets = [c['material_budget_percent'] for c in configurations]
resolutions = [c['momentum_resolution_10gev'] * 100 for c in configurations]  # Convert to percent

ax1.bar(names, material_budgets, color=['blue', 'green', 'red'], alpha=0.7)
ax1.set_ylabel('Material Budget (% X0)')
ax1.set_title('Material Budget Comparison')
ax1.tick_params(axis='x', rotation=45)

# Momentum resolution comparison
ax2.bar(names, resolutions, color=['blue', 'green', 'red'], alpha=0.7)
ax2.set_ylabel('Momentum Resolution (%) at 10 GeV/c')
ax2.set_title('Expected Momentum Resolution')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('detector_configurations_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('detector_configurations_comparison.pdf', bbox_inches='tight')

# Save configurations to JSON
with open('detector_configurations.json', 'w') as f:
    json.dump(configurations, f, indent=2)

# Print summary
print('Three Silicon Pixel Detector Configurations Designed:')
print('=' * 60)
for i, config in enumerate(configurations, 1):
    print(f'\nConfiguration {i}: {config["name"]}')
    print(f'  Topology: {config["topology"]}')
    print(f'  Layers: {config["layer_count"]}')
    print(f'  Silicon thickness: {config["silicon_thickness_um"]} μm')
    print(f'  Material budget: {config["material_budget_percent"]:.2f}% X0')
    print(f'  Momentum resolution (10 GeV/c): {config["momentum_resolution_10gev"]*100:.3f}%')
    print(f'  Optimization target: {config["optimization_target"]}')
    print(f'  Physics justification: {config["physics_justification"]}')

# Output results for downstream steps
print(f'\nRESULT:config1_name={config1["name"]}')
print(f'RESULT:config1_topology={config1["topology"]}')
print(f'RESULT:config1_layers={config1["layer_count"]}')
print(f'RESULT:config1_thickness_um={config1["silicon_thickness_um"]}')
print(f'RESULT:config1_material_budget_percent={config1["material_budget_percent"]:.2f}')

print(f'RESULT:config2_name={config2["name"]}')
print(f'RESULT:config2_topology={config2["topology"]}')
print(f'RESULT:config2_layers={config2["layer_count"]}')
print(f'RESULT:config2_thickness_um={config2["silicon_thickness_um"]}')
print(f'RESULT:config2_material_budget_percent={config2["material_budget_percent"]:.2f}')

print(f'RESULT:config3_name={config3["name"]}')
print(f'RESULT:config3_topology={config3["topology"]}')
print(f'RESULT:config3_layers={config3["layer_count"]}')
print(f'RESULT:config3_thickness_um={config3["silicon_thickness_um"]}')
print(f'RESULT:config3_material_budget_percent={config3["material_budget_percent"]:.2f}')

print('RESULT:configurations_file=detector_configurations.json')
print('RESULT:comparison_plot=detector_configurations_comparison.png')
print('RESULT:success=True')