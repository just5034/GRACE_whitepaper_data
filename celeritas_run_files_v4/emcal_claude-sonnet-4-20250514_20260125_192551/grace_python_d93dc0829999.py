import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

# Performance data from previous step outputs
configurations = {
    'Baseline (CsI box)': {
        'resolution': 0.034363,
        'resolution_err': 0.000768,
        'linearity': 0.467007,
        'linearity_err': 0.000507,
        'containment_90': 40.82,
        'containment_95': 53.06,
        'material': 'CsI',
        'topology': 'box'
    },
    'Projective (PbWO4)': {
        'resolution': 0.021387,
        'resolution_err': 0.000478,
        'linearity': 0.978699,
        'linearity_err': 0.000662,
        'containment_90': 100,
        'containment_95': 100,
        'material': 'PbWO4',
        'topology': 'projective_tower'
    },
    'Shashlik (BGO)': {
        'resolution': 0.036765,
        'resolution_err': 0.000822,
        'linearity': 4.72437,
        'linearity_err': 0.005493,
        'containment_90': 24.2,
        'containment_95': 36.4,
        'material': 'BGO',
        'topology': 'shashlik'
    },
    'Accordion (CsI)': {
        'resolution': 0.021184,
        'resolution_err': 0.000474,
        'linearity': 0.980013,
        'linearity_err': 0.02076,
        'containment_90': 200,
        'containment_95': 200,
        'material': 'CsI',
        'topology': 'accordion'
    }
}

# Target criteria
target_resolution_min = 0.02  # 2%/sqrt(E)
target_resolution_max = 0.03  # 3%/sqrt(E)
target_linearity_max = 1.0    # Good linearity (close to 1.0)
min_containment_90 = 30       # Minimum acceptable containment

print("=== CALORIMETER CONFIGURATION OPTIMIZATION ANALYSIS ===")
print(f"Target Resolution: {target_resolution_min*100:.1f}% - {target_resolution_max*100:.1f}%")
print(f"Target Linearity: < {target_linearity_max:.1f}")
print(f"Minimum Containment (90%): > {min_containment_90} mm")
print()

# Scoring system for each criterion
scores = {}
for name, config in configurations.items():
    score = 0
    details = []
    
    # Resolution score (40% weight) - meets 2-3% target?
    res_pct = config['resolution'] * 100
    if target_resolution_min <= config['resolution'] <= target_resolution_max:
        res_score = 10  # Perfect score for meeting target
        details.append(f"Resolution: {res_pct:.2f}% (MEETS TARGET) +10")
    elif config['resolution'] < target_resolution_min:
        res_score = 8   # Very good, better than target
        details.append(f"Resolution: {res_pct:.2f}% (EXCEEDS TARGET) +8")
    else:
        # Penalty for missing target
        excess = (config['resolution'] - target_resolution_max) / target_resolution_max
        res_score = max(0, 5 - 10 * excess)
        details.append(f"Resolution: {res_pct:.2f}% (MISSES TARGET) +{res_score:.1f}")
    
    score += 0.4 * res_score
    
    # Linearity score (35% weight) - closer to 1.0 is better
    lin_score = max(0, 10 - 5 * abs(config['linearity'] - 1.0))
    details.append(f"Linearity: {config['linearity']:.3f} +{lin_score:.1f}")
    score += 0.35 * lin_score
    
    # Containment score (25% weight)
    if config['containment_90'] >= min_containment_90:
        cont_score = min(10, config['containment_90'] / 50)  # Normalize to reasonable scale
        details.append(f"Containment: {config['containment_90']:.1f}mm +{cont_score:.1f}")
    else:
        cont_score = 0
        details.append(f"Containment: {config['containment_90']:.1f}mm (TOO LOW) +0")
    
    score += 0.25 * cont_score
    
    scores[name] = {'total_score': score, 'details': details}
    
    print(f"{name}:")
    print(f"  Resolution: {res_pct:.2f}% ± {config['resolution_err']*100:.2f}%")
    print(f"  Linearity: {config['linearity']:.3f} ± {config['linearity_err']:.3f}")
    print(f"  Containment (90%): {config['containment_90']:.1f} mm")
    print(f"  TOTAL SCORE: {score:.2f}/10")
    for detail in details:
        print(f"    {detail}")
    print()

# Find optimal configuration
best_config = max(scores.keys(), key=lambda x: scores[x]['total_score'])
best_score = scores[best_config]['total_score']

print(f"=== OPTIMAL CONFIGURATION IDENTIFIED ===")
print(f"WINNER: {best_config}")
print(f"Score: {best_score:.2f}/10")
print()

# Detailed justification
optimal_data = configurations[best_config]
print("QUANTITATIVE JUSTIFICATION:")
print(f"1. RESOLUTION CRITERION: {optimal_data['resolution']*100:.2f}% ± {optimal_data['resolution_err']*100:.2f}%")
if target_resolution_min <= optimal_data['resolution'] <= target_resolution_max:
    print(f"   ✓ MEETS 2-3%/√E target ({target_resolution_min*100}%-{target_resolution_max*100}%)")
elif optimal_data['resolution'] < target_resolution_min:
    print(f"   ✓ EXCEEDS 2-3%/√E target (better than required)")
else:
    print(f"   ✗ Does not meet 2-3%/√E target")

print(f"2. LINEARITY CRITERION: {optimal_data['linearity']:.3f} ± {optimal_data['linearity_err']:.3f}")
if abs(optimal_data['linearity'] - 1.0) < 0.1:
    print(f"   ✓ EXCELLENT linearity (within 10% of ideal)")
elif abs(optimal_data['linearity'] - 1.0) < 0.5:
    print(f"   ✓ GOOD linearity (within 50% of ideal)")
else:
    print(f"   ⚠ Poor linearity (deviation > 50%)")

print(f"3. CONTAINMENT CRITERION: {optimal_data['containment_90']:.1f} mm (90%)")
if optimal_data['containment_90'] >= min_containment_90:
    print(f"   ✓ ADEQUATE containment (> {min_containment_90} mm)")
else:
    print(f"   ✗ Poor containment (< {min_containment_30} mm)")

# Create comprehensive comparison plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

config_names = list(configurations.keys())
config_short = ['Baseline\n(CsI box)', 'Projective\n(PbWO4)', 'Shashlik\n(BGO)', 'Accordion\n(CsI)']
colors = ['blue', 'green', 'red', 'purple']

# Resolution comparison
resolutions = [configurations[name]['resolution'] * 100 for name in config_names]
res_errors = [configurations[name]['resolution_err'] * 100 for name in config_names]
ax1.bar(config_short, resolutions, yerr=res_errors, capsize=5, color=colors, alpha=0.7)
ax1.axhspan(target_resolution_min*100, target_resolution_max*100, alpha=0.2, color='green', label='Target Range')
ax1.set_ylabel('Energy Resolution (%)')
ax1.set_title('Energy Resolution Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Linearity comparison  
linearities = [configurations[name]['linearity'] for name in config_names]
lin_errors = [configurations[name]['linearity_err'] for name in config_names]
ax2.bar(config_short, linearities, yerr=lin_errors, capsize=5, color=colors, alpha=0.7)
ax2.axhline(1.0, color='green', linestyle='--', alpha=0.7, label='Ideal Linearity')
ax2.set_ylabel('Linearity Parameter')
ax2.set_title('Linearity Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Containment comparison
containments = [configurations[name]['containment_90'] for name in config_names]
ax3.bar(config_short, containments, color=colors, alpha=0.7)
ax3.axhline(min_containment_90, color='red', linestyle='--', alpha=0.7, label=f'Min Acceptable ({min_containment_90} mm)')
ax3.set_ylabel('90% Containment Radius (mm)')
ax3.set_title('Containment Comparison')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Overall scores
scores_list = [scores[name]['total_score'] for name in config_names]
ax4.bar(config_short, scores_list, color=colors, alpha=0.7)
ax4.set_ylabel('Overall Score (/10)')
ax4.set_title('Overall Performance Score')
ax4.grid(True, alpha=0.3)

# Highlight optimal configuration
optimal_idx = config_names.index(best_config)
for ax in [ax1, ax2, ax3, ax4]:
    ax.patches[optimal_idx].set_edgecolor('black')
    ax.patches[optimal_idx].set_linewidth(3)

plt.tight_layout()
plt.savefig('optimal_configuration_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('optimal_configuration_analysis.pdf', bbox_inches='tight')
print()
print("RESULT:analysis_plot=optimal_configuration_analysis.png")

# Save detailed results
results = {
    'optimal_configuration': best_config,
    'optimal_score': best_score,
    'target_criteria': {
        'resolution_range_percent': [target_resolution_min*100, target_resolution_max*100],
        'max_linearity_deviation': target_linearity_max,
        'min_containment_90_mm': min_containment_90
    },
    'optimal_performance': {
        'resolution_percent': optimal_data['resolution'] * 100,
        'resolution_error_percent': optimal_data['resolution_err'] * 100,
        'linearity': optimal_data['linearity'],
        'linearity_error': optimal_data['linearity_err'],
        'containment_90_mm': optimal_data['containment_90'],
        'material': optimal_data['material'],
        'topology': optimal_data['topology']
    },
    'meets_resolution_target': target_resolution_min <= optimal_data['resolution'] <= target_resolution_max,
    'meets_linearity_target': abs(optimal_data['linearity'] - 1.0) < 0.5,
    'meets_containment_target': optimal_data['containment_90'] >= min_containment_90,
    'all_configurations_scores': scores
}

with open('optimal_configuration_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"RESULT:optimal_configuration={best_config}")
print(f"RESULT:optimal_score={best_score:.2f}")
print(f"RESULT:resolution_percent={optimal_data['resolution']*100:.2f}")
print(f"RESULT:linearity={optimal_data['linearity']:.3f}")
print(f"RESULT:containment_90_mm={optimal_data['containment_90']:.1f}")
print(f"RESULT:meets_resolution_target={target_resolution_min <= optimal_data['resolution'] <= target_resolution_max}")
print(f"RESULT:meets_linearity_target={abs(optimal_data['linearity'] - 1.0) < 0.5}")
print(f"RESULT:meets_containment_target={optimal_data['containment_90'] >= min_containment_90}")
print("RESULT:analysis_file=optimal_configuration_analysis.json")
print("RESULT:success=True")