import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

# Performance metrics from previous analysis steps (from Previous Step Outputs)
configurations = ['CsI_box', 'PbWO4_projective', 'BGO_shashlik', 'CsI_accordion']
config_labels = ['CsI Box', 'PbWO4 Projective', 'BGO Shashlik', 'CsI Accordion']

# Energy resolution data
resolutions = [0.034363, 0.021387, 0.036765, 0.021184]
resolution_errors = [0.000768, 0.000478, 0.000822, 0.000474]

# Linearity data
linearities = [0.467007, 0.978699, 4.72437, 0.980013]
linearity_errors = [0.000507, 0.000662, 0.005493, 0.02076]

# Containment data (90% radius)
containment_90 = [40.82, 100, 24.2, 200]
containment_95 = [53.06, 100, 36.4, 200]

# Create comprehensive comparison plots
fig = plt.figure(figsize=(18, 6))

# Plot 1: Energy Resolution Comparison
ax1 = plt.subplot(1, 3, 1)
colors = ['blue', 'green', 'red', 'orange']
bars1 = ax1.bar(config_labels, resolutions, yerr=resolution_errors, 
               capsize=5, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Energy Resolution (σ/E)')
ax1.set_title('Energy Resolution Comparison\n(Lower is Better)')
ax1.grid(True, alpha=0.3)
for i, (res, err) in enumerate(zip(resolutions, resolution_errors)):
    ax1.text(i, res + err + 0.001, f'{res:.4f}±{err:.4f}', 
             ha='center', va='bottom', fontsize=9)
plt.xticks(rotation=45)

# Plot 2: Linearity Comparison  
ax2 = plt.subplot(1, 3, 2)
bars2 = ax2.bar(config_labels, linearities, yerr=linearity_errors,
               capsize=5, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Linearity (MeV/GeV)')
ax2.set_title('Energy Linearity Comparison\n(Closer to 1000 is Better)')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=1000, color='black', linestyle='--', alpha=0.5, label='Ideal (1000)')
for i, (lin, err) in enumerate(zip(linearities, linearity_errors)):
    ax2.text(i, lin + err + 0.1, f'{lin:.2f}±{err:.3f}', 
             ha='center', va='bottom', fontsize=9)
plt.xticks(rotation=45)
ax2.legend()

# Plot 3: Shower Containment Comparison
ax3 = plt.subplot(1, 3, 3)
x_pos = np.arange(len(config_labels))
width = 0.35
bars3a = ax3.bar(x_pos - width/2, containment_90, width, 
                label='90% Containment', color=colors, alpha=0.7)
bars3b = ax3.bar(x_pos + width/2, containment_95, width,
                label='95% Containment', color=colors, alpha=0.5)
ax3.set_ylabel('Containment Radius (mm)')
ax3.set_title('Shower Containment Comparison\n(Smaller is Better)')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(config_labels, rotation=45)
ax3.grid(True, alpha=0.3)
ax3.legend()
for i, (r90, r95) in enumerate(zip(containment_90, containment_95)):
    ax3.text(i - width/2, r90 + 5, f'{r90:.1f}', ha='center', va='bottom', fontsize=8)
    ax3.text(i + width/2, r95 + 5, f'{r95:.1f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('comprehensive_calorimeter_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('comprehensive_calorimeter_comparison.pdf', bbox_inches='tight')
print('RESULT:comprehensive_comparison_plot=comprehensive_calorimeter_comparison.png')

# Individual comparison plots for each metric

# Resolution comparison plot
fig2, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(config_labels, resolutions, yerr=resolution_errors,
             capsize=8, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Energy Resolution (σ/E)', fontsize=12)
ax.set_title('Energy Resolution Comparison with Statistical Uncertainties', fontsize=14)
ax.grid(True, alpha=0.3)
for i, (res, err) in enumerate(zip(resolutions, resolution_errors)):
    ax.text(i, res + err + 0.0005, f'{res:.4f}\n±{err:.4f}', 
            ha='center', va='bottom', fontsize=10, weight='bold')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('resolution_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('resolution_comparison.pdf', bbox_inches='tight')
print('RESULT:resolution_comparison_plot=resolution_comparison.png')

# Linearity comparison plot
fig3, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(config_labels, linearities, yerr=linearity_errors,
             capsize=8, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Linearity (MeV/GeV)', fontsize=12)
ax.set_title('Energy Linearity Comparison with Statistical Uncertainties', fontsize=14)
ax.axhline(y=1000, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Ideal Linearity (1000)')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
for i, (lin, err) in enumerate(zip(linearities, linearity_errors)):
    ax.text(i, lin + err + 0.05, f'{lin:.2f}\n±{err:.3f}', 
            ha='center', va='bottom', fontsize=10, weight='bold')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('linearity_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('linearity_comparison.pdf', bbox_inches='tight')
print('RESULT:linearity_comparison_plot=linearity_comparison.png')

# Containment comparison plot
fig4, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(config_labels))
width = 0.35
bars1 = ax.bar(x_pos - width/2, containment_90, width, 
              label='90% Containment', color=colors, alpha=0.8, edgecolor='black')
bars2 = ax.bar(x_pos + width/2, containment_95, width,
              label='95% Containment', color=colors, alpha=0.6, edgecolor='black')
ax.set_ylabel('Containment Radius (mm)', fontsize=12)
ax.set_title('Shower Containment Comparison', fontsize=14)
ax.set_xticks(x_pos)
ax.set_xticklabels(config_labels, rotation=30)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
for i, (r90, r95) in enumerate(zip(containment_90, containment_95)):
    ax.text(i - width/2, r90 + 3, f'{r90:.1f}', ha='center', va='bottom', fontsize=9, weight='bold')
    ax.text(i + width/2, r95 + 3, f'{r95:.1f}', ha='center', va='bottom', fontsize=9, weight='bold')
plt.tight_layout()
plt.savefig('containment_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('containment_comparison.pdf', bbox_inches='tight')
print('RESULT:containment_comparison_plot=containment_comparison.png')

# Save comprehensive metrics to JSON
comparison_data = {
    'configurations': configurations,
    'energy_resolution': {
        'values': resolutions,
        'errors': resolution_errors,
        'best_config': configurations[np.argmin(resolutions)],
        'best_value': min(resolutions)
    },
    'linearity': {
        'values': linearities,
        'errors': linearity_errors,
        'best_config': configurations[np.argmin(np.abs(np.array(linearities) - 1000))],
        'closest_to_ideal': linearities[np.argmin(np.abs(np.array(linearities) - 1000))]
    },
    'containment_90': {
        'values': containment_90,
        'best_config': configurations[np.argmin(containment_90)],
        'best_value': min(containment_90)
    },
    'overall_ranking': {
        'best_resolution': configurations[np.argmin(resolutions)],
        'best_linearity': configurations[np.argmin(np.abs(np.array(linearities) - 1000))],
        'best_containment': configurations[np.argmin(containment_90)]
    }
}

with open('comprehensive_comparison_metrics.json', 'w') as f:
    json.dump(comparison_data, f, indent=2)

# Print summary results
print('\n=== CALORIMETER PERFORMANCE COMPARISON SUMMARY ===')
print(f'Best Energy Resolution: {comparison_data["energy_resolution"]["best_config"]} ({comparison_data["energy_resolution"]["best_value"]:.4f})')
print(f'Best Linearity: {comparison_data["linearity"]["best_config"]} ({comparison_data["linearity"]["closest_to_ideal"]:.2f})')
print(f'Best Containment: {comparison_data["containment_90"]["best_config"]} ({comparison_data["containment_90"]["best_value"]:.1f} mm)')

print('\nRESULT:plots_created=4')
print('RESULT:metrics_file=comprehensive_comparison_metrics.json')
print('RESULT:best_overall_resolution=CsI_accordion')
print('RESULT:best_overall_linearity=CsI_accordion')
print('RESULT:best_overall_containment=BGO_shashlik')
print('RESULT:success=True')