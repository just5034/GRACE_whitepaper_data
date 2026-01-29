import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import json

# Extract performance metrics from previous step outputs
# BGO results
bgo_resolutions = [0.026309, 0.021725, 0.017925]  # 0.5, 2.0, 5.0 GeV
bgo_energies = [0.5, 2.0, 5.0]
bgo_linearity = [0.9625, 0.9614, 0.959]
bgo_stochastic = 0.82  # percent
bgo_constant = 1.496   # percent

# PbWO4 results
pbwo4_resolutions = [0.033044, 0.014936, 0.019751]  # 0.5, 2.0, 5.0 GeV
pbwo4_energies = [0.5, 2.0, 5.0]
pbwo4_linearity = [0.9678, 0.9674, 0.9637]
pbwo4_stochastic = 1.62  # percent
pbwo4_constant = 0.87    # percent

# CsI results
csi_resolutions = [0.031228, 0.019525, 0.016829]  # 0.5, 2.0, 5.0 GeV
csi_energies = [0.5, 2.0, 5.0]
csi_linearity = [0.9553, 0.9548, 0.9527]
csi_stochastic = 1.48  # percent
csi_constant = 0.98    # percent

# Target resolution: 1-3% / sqrt(E)
# Calculate target at each energy
target_min = np.array([1.0 / np.sqrt(E) / 100 for E in [0.5, 2.0, 5.0]])
target_max = np.array([3.0 / np.sqrt(E) / 100 for E in [0.5, 2.0, 5.0]])

print("=== MATERIAL PERFORMANCE COMPARISON ===")
print(f"Target resolution: 1-3% / sqrt(E)")
print(f"At 0.5 GeV: {target_min[0]:.4f} - {target_max[0]:.4f}")
print(f"At 2.0 GeV: {target_min[1]:.4f} - {target_max[1]:.4f}")
print(f"At 5.0 GeV: {target_min[2]:.4f} - {target_max[2]:.4f}")
print()

# Check which materials meet target at each energy
print("Energy Resolution Performance:")
for i, energy in enumerate([0.5, 2.0, 5.0]):
    print(f"\nAt {energy} GeV (target: {target_min[i]:.4f}-{target_max[i]:.4f}):")
    print(f"  BGO:   {bgo_resolutions[i]:.4f} {'✓' if target_min[i] <= bgo_resolutions[i] <= target_max[i] else '✗'}")
    print(f"  PbWO4: {pbwo4_resolutions[i]:.4f} {'✓' if target_min[i] <= pbwo4_resolutions[i] <= target_max[i] else '✗'}")
    print(f"  CsI:   {csi_resolutions[i]:.4f} {'✓' if target_min[i] <= csi_resolutions[i] <= target_max[i] else '✗'}")

# Calculate average linearity
bgo_avg_linearity = np.mean(bgo_linearity)
pbwo4_avg_linearity = np.mean(pbwo4_linearity)
csi_avg_linearity = np.mean(csi_linearity)

print("\nLinearity Performance:")
print(f"  BGO:   {bgo_avg_linearity:.4f}")
print(f"  PbWO4: {pbwo4_avg_linearity:.4f}")
print(f"  CsI:   {csi_avg_linearity:.4f}")

# Resolution parameterization: σ/E = a/√E ⊕ b
print("\nResolution Parameterization (a/√E ⊕ b):")
print(f"  BGO:   {bgo_stochastic:.2f}%/√E ⊕ {bgo_constant:.2f}%")
print(f"  PbWO4: {pbwo4_stochastic:.2f}%/√E ⊕ {pbwo4_constant:.2f}%")
print(f"  CsI:   {csi_stochastic:.2f}%/√E ⊕ {csi_constant:.2f}%")

# Create comparison plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Energy resolution vs energy
ax1.errorbar(bgo_energies, bgo_resolutions, fmt='o-', label='BGO', linewidth=2, markersize=8)
ax1.errorbar(pbwo4_energies, pbwo4_resolutions, fmt='s-', label='PbWO4', linewidth=2, markersize=8)
ax1.errorbar(csi_energies, csi_resolutions, fmt='^-', label='CsI', linewidth=2, markersize=8)
ax1.fill_between([0.4, 5.5], target_min, target_max, alpha=0.2, color='gray', label='Target Range')
ax1.set_xlabel('Energy (GeV)')
ax1.set_ylabel('Energy Resolution (σ/E)')
ax1.set_title('Energy Resolution vs Energy')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0.4, 5.5)

# Plot 2: Linearity comparison
materials = ['BGO', 'PbWO4', 'CsI']
linearity_values = [bgo_avg_linearity, pbwo4_avg_linearity, csi_avg_linearity]
colors = ['blue', 'red', 'green']
ax2.bar(materials, linearity_values, color=colors, alpha=0.7)
ax2.set_ylabel('Average Linearity')
ax2.set_title('Linearity Comparison')
ax2.set_ylim(0.94, 0.98)
for i, v in enumerate(linearity_values):
    ax2.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')

# Plot 3: Resolution at 2 GeV (typical benchmark)
res_2gev = [bgo_resolutions[1], pbwo4_resolutions[1], csi_resolutions[1]]
target_2gev = [target_min[1], target_max[1]]
ax3.bar(materials, res_2gev, color=colors, alpha=0.7)
ax3.axhspan(target_2gev[0], target_2gev[1], alpha=0.2, color='gray', label='Target')
ax3.set_ylabel('Energy Resolution at 2 GeV')
ax3.set_title('Resolution at 2 GeV Benchmark')
ax3.legend()
for i, v in enumerate(res_2gev):
    ax3.text(i, v + 0.0005, f'{v:.4f}', ha='center', va='bottom')

# Plot 4: Stochastic vs Constant terms
ax4.scatter([bgo_stochastic], [bgo_constant], s=150, label='BGO', color='blue')
ax4.scatter([pbwo4_stochastic], [pbwo4_constant], s=150, label='PbWO4', color='red')
ax4.scatter([csi_stochastic], [csi_constant], s=150, label='CsI', color='green')
ax4.set_xlabel('Stochastic Term (%/√E)')
ax4.set_ylabel('Constant Term (%)')
ax4.set_title('Resolution Terms Comparison')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('material_performance_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('material_performance_comparison.pdf', bbox_inches='tight')

# Determine best material
print("\n=== MATERIAL RANKING ===")

# Score based on target achievement and performance
scores = {}
for material, resolutions in [('BGO', bgo_resolutions), ('PbWO4', pbwo4_resolutions), ('CsI', csi_resolutions)]:
    score = 0
    # Points for meeting target at each energy
    for i, res in enumerate(resolutions):
        if target_min[i] <= res <= target_max[i]:
            score += 10
        elif res < target_min[i]:  # Better than target
            score += 15
        else:  # Worse than target
            score += max(0, 5 - (res - target_max[i]) * 1000)
    
    scores[material] = score

# Add linearity bonus
linearity_scores = {'BGO': bgo_avg_linearity, 'PbWO4': pbwo4_avg_linearity, 'CsI': csi_avg_linearity}
for material in scores:
    scores[material] += linearity_scores[material] * 10

print("Performance Scores:")
for material, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
    print(f"  {material}: {score:.2f}")

best_material = max(scores.keys(), key=lambda x: scores[x])
print(f"\nBEST PERFORMING MATERIAL: {best_material}")

# Detailed analysis of best material
if best_material == 'BGO':
    best_res = bgo_resolutions
    best_lin = bgo_avg_linearity
    best_stoch = bgo_stochastic
    best_const = bgo_constant
elif best_material == 'PbWO4':
    best_res = pbwo4_resolutions
    best_lin = pbwo4_avg_linearity
    best_stoch = pbwo4_stochastic
    best_const = pbwo4_constant
else:
    best_res = csi_resolutions
    best_lin = csi_avg_linearity
    best_stoch = csi_stochastic
    best_const = csi_constant

print(f"\n{best_material} Performance Summary:")
print(f"  Energy Resolution: {best_res[0]:.4f} (0.5 GeV), {best_res[1]:.4f} (2 GeV), {best_res[2]:.4f} (5 GeV)")
print(f"  Average Linearity: {best_lin:.4f}")
print(f"  Resolution Formula: {best_stoch:.2f}%/√E ⊕ {best_const:.2f}%")

# Save results to JSON
results = {
    'best_material': best_material,
    'target_resolution': '1-3% / sqrt(E)',
    'bgo_performance': {
        'resolutions': bgo_resolutions,
        'avg_linearity': bgo_avg_linearity,
        'stochastic_term_percent': bgo_stochastic,
        'constant_term_percent': bgo_constant,
        'score': scores['BGO']
    },
    'pbwo4_performance': {
        'resolutions': pbwo4_resolutions,
        'avg_linearity': pbwo4_avg_linearity,
        'stochastic_term_percent': pbwo4_stochastic,
        'constant_term_percent': pbwo4_constant,
        'score': scores['PbWO4']
    },
    'csi_performance': {
        'resolutions': csi_resolutions,
        'avg_linearity': csi_avg_linearity,
        'stochastic_term_percent': csi_stochastic,
        'constant_term_percent': csi_constant,
        'score': scores['CsI']
    }
}

with open('material_comparison_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nRESULT:best_material={best_material}")
print(f"RESULT:best_material_score={scores[best_material]:.2f}")
print(f"RESULT:comparison_plot=material_performance_comparison.png")
print(f"RESULT:results_json=material_comparison_results.json")
print(f"RESULT:target_resolution=1-3% / sqrt(E)")
print("RESULT:success=True")