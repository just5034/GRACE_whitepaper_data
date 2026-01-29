import json
import numpy as np
import math

# Input parameters
momentum_range = [1, 50]  # GeV/c
silicon_x0 = 0.0937  # m (9.37 cm radiation length)
thickness_range = [100e-6, 500e-6]  # m (100-500 microns)
spacing_range = [0.02, 0.10]  # m (2-10 cm)
layer_range = [3, 8]

# Physics calculations for tracking performance
def calculate_multiple_scattering_angle(momentum_gev, thickness_m, x0_m):
    """Highland formula for RMS scattering angle"""
    x_over_x0 = thickness_m / x0_m
    theta_ms = (0.0136 / momentum_gev) * np.sqrt(x_over_x0) * (1 + 0.038 * np.log(x_over_x0))
    return theta_ms

def calculate_momentum_resolution(momentum_gev, theta_ms, lever_arm_m):
    """Momentum resolution from multiple scattering"""
    # Simplified formula: sigma_p/p ~ theta_ms for single point measurement
    # For tracker: sigma_p/p ~ p * theta_ms / (0.3 * B * L), assume B=1T
    sigma_p_over_p = momentum_gev * theta_ms / (0.3 * 1.0 * lever_arm_m)
    return sigma_p_over_p

def calculate_material_budget(num_layers, thickness_m, x0_m):
    """Total material budget in radiation lengths"""
    return num_layers * thickness_m / x0_m

# Design Configuration 1: Box Topology - Minimal Material Budget
print("=== CONFIGURATION 1: BOX TOPOLOGY (Minimal Material) ===")
config1 = {
    "name": "config1_box_minimal",
    "topology": "box",
    "num_layers": 4,
    "layer_thickness_um": 150,  # microns
    "layer_spacing_cm": 5.0,    # cm
    "transverse_size_cm": 20.0, # cm
    "physics_justification": "Minimal material budget design for high momentum tracking"
}

# Physics calculations for config1
thickness_m = config1["layer_thickness_um"] * 1e-6
lever_arm = (config1["num_layers"] - 1) * config1["layer_spacing_cm"] * 0.01
material_budget = calculate_material_budget(config1["num_layers"], thickness_m, silicon_x0)
theta_ms_1gev = calculate_multiple_scattering_angle(1.0, thickness_m, silicon_x0)
theta_ms_50gev = calculate_multiple_scattering_angle(50.0, thickness_m, silicon_x0)
sigma_p_1gev = calculate_momentum_resolution(1.0, theta_ms_1gev, lever_arm)
sigma_p_50gev = calculate_momentum_resolution(50.0, theta_ms_50gev, lever_arm)

config1["calculated_parameters"] = {
    "material_budget_x0": round(material_budget, 4),
    "lever_arm_m": round(lever_arm, 3),
    "momentum_resolution_1gev": round(sigma_p_1gev, 4),
    "momentum_resolution_50gev": round(sigma_p_50gev, 4)
}

print(f"Layers: {config1['num_layers']}, Thickness: {config1['layer_thickness_um']} μm")
print(f"Spacing: {config1['layer_spacing_cm']} cm, Size: {config1['transverse_size_cm']} cm")
print(f"Material budget: {material_budget:.4f} X0")
print(f"Momentum resolution at 1 GeV: {sigma_p_1gev:.4f}")
print(f"Momentum resolution at 50 GeV: {sigma_p_50gev:.4f}")
print()

# Design Configuration 2: Cylindrical Barrel - Balanced Performance
print("=== CONFIGURATION 2: CYLINDRICAL BARREL (Balanced) ===")
config2 = {
    "name": "config2_cylinder_balanced",
    "topology": "cylinder_barrel",
    "num_layers": 6,
    "layer_thickness_um": 300,  # microns
    "inner_radius_cm": 10.0,    # cm
    "outer_radius_cm": 35.0,    # cm
    "half_length_cm": 50.0,     # cm
    "physics_justification": "Cylindrical geometry for uniform azimuthal coverage and moderate material budget"
}

# Physics calculations for config2
thickness_m = config2["layer_thickness_um"] * 1e-6
lever_arm = (config2["outer_radius_cm"] - config2["inner_radius_cm"]) * 0.01
material_budget = calculate_material_budget(config2["num_layers"], thickness_m, silicon_x0)
theta_ms_1gev = calculate_multiple_scattering_angle(1.0, thickness_m, silicon_x0)
theta_ms_50gev = calculate_multiple_scattering_angle(50.0, thickness_m, silicon_x0)
sigma_p_1gev = calculate_momentum_resolution(1.0, theta_ms_1gev, lever_arm)
sigma_p_50gev = calculate_momentum_resolution(50.0, theta_ms_50gev, lever_arm)

config2["calculated_parameters"] = {
    "material_budget_x0": round(material_budget, 4),
    "lever_arm_m": round(lever_arm, 3),
    "momentum_resolution_1gev": round(sigma_p_1gev, 4),
    "momentum_resolution_50gev": round(sigma_p_50gev, 4)
}

print(f"Layers: {config2['num_layers']}, Thickness: {config2['layer_thickness_um']} μm")
print(f"Inner radius: {config2['inner_radius_cm']} cm, Outer: {config2['outer_radius_cm']} cm")
print(f"Half length: {config2['half_length_cm']} cm")
print(f"Material budget: {material_budget:.4f} X0")
print(f"Momentum resolution at 1 GeV: {sigma_p_1gev:.4f}")
print(f"Momentum resolution at 50 GeV: {sigma_p_50gev:.4f}")
print()

# Design Configuration 3: Forward Disks - High Precision
print("=== CONFIGURATION 3: FORWARD DISKS (High Precision) ===")
config3 = {
    "name": "config3_forward_precision",
    "topology": "forward_disks",
    "num_layers": 8,
    "layer_thickness_um": 200,  # microns
    "disk_positions_cm": [15, 25, 35, 50, 70, 95, 125, 160],  # z positions
    "inner_radius_cm": 5.0,     # cm
    "outer_radius_cm": 30.0,    # cm
    "physics_justification": "Forward disk geometry for high precision momentum measurement with optimized spacing"
}

# Physics calculations for config3
thickness_m = config3["layer_thickness_um"] * 1e-6
lever_arm = (config3["disk_positions_cm"][-1] - config3["disk_positions_cm"][0]) * 0.01
material_budget = calculate_material_budget(config3["num_layers"], thickness_m, silicon_x0)
theta_ms_1gev = calculate_multiple_scattering_angle(1.0, thickness_m, silicon_x0)
theta_ms_50gev = calculate_multiple_scattering_angle(50.0, thickness_m, silicon_x0)
sigma_p_1gev = calculate_momentum_resolution(1.0, theta_ms_1gev, lever_arm)
sigma_p_50gev = calculate_momentum_resolution(50.0, theta_ms_50gev, lever_arm)

config3["calculated_parameters"] = {
    "material_budget_x0": round(material_budget, 4),
    "lever_arm_m": round(lever_arm, 3),
    "momentum_resolution_1gev": round(sigma_p_1gev, 4),
    "momentum_resolution_50gev": round(sigma_p_50gev, 4)
}

print(f"Layers: {config3['num_layers']}, Thickness: {config3['layer_thickness_um']} μm")
print(f"Disk positions: {config3['disk_positions_cm']} cm")
print(f"Radial coverage: {config3['inner_radius_cm']}-{config3['outer_radius_cm']} cm")
print(f"Material budget: {material_budget:.4f} X0")
print(f"Momentum resolution at 1 GeV: {sigma_p_1gev:.4f}")
print(f"Momentum resolution at 50 GeV: {sigma_p_50gev:.4f}")
print()

# Summary comparison
print("=== CONFIGURATION COMPARISON ===")
configs = [config1, config2, config3]
for i, config in enumerate(configs, 1):
    params = config['calculated_parameters']
    print(f"Config {i} ({config['topology']}):") 
    print(f"  Material budget: {params['material_budget_x0']:.4f} X0")
    print(f"  Lever arm: {params['lever_arm_m']:.3f} m")
    print(f"  σ_p/p at 1 GeV: {params['momentum_resolution_1gev']:.4f}")
    print(f"  σ_p/p at 50 GeV: {params['momentum_resolution_50gev']:.4f}")
    print()

# Save configurations to JSON
detector_configurations = {
    "config1_box_minimal": config1,
    "config2_cylinder_balanced": config2, 
    "config3_forward_precision": config3,
    "design_summary": {
        "momentum_range_gev": momentum_range,
        "silicon_radiation_length_m": silicon_x0,
        "design_philosophy": "Three configurations exploring topology and material budget trade-offs"
    }
}

with open('silicon_detector_configurations.json', 'w') as f:
    json.dump(detector_configurations, f, indent=2)

# Output results for workflow
print("RESULT:config1_name=config1_box_minimal")
print("RESULT:config2_name=config2_cylinder_balanced")
print("RESULT:config3_name=config3_forward_precision")
print(f"RESULT:config1_material_budget={config1['calculated_parameters']['material_budget_x0']:.4f}")
print(f"RESULT:config2_material_budget={config2['calculated_parameters']['material_budget_x0']:.4f}")
print(f"RESULT:config3_material_budget={config3['calculated_parameters']['material_budget_x0']:.4f}")
print(f"RESULT:config1_momentum_res_1gev={config1['calculated_parameters']['momentum_resolution_1gev']:.4f}")
print(f"RESULT:config2_momentum_res_1gev={config2['calculated_parameters']['momentum_resolution_1gev']:.4f}")
print(f"RESULT:config3_momentum_res_1gev={config3['calculated_parameters']['momentum_resolution_1gev']:.4f}")
print("RESULT:configurations_file=silicon_detector_configurations.json")
print("RESULT:success=True")