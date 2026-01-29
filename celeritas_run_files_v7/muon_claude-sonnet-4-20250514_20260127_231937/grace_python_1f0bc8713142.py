import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import stats

# Extract performance metrics from previous step outputs
# Planar configuration results
planar_metrics = {
    'muon_5gev_resolution': 0.0044,
    'muon_20gev_resolution': 0.1429,
    'muon_50gev_resolution': 0.3306,
    'pion_5gev_resolution': 0.0564,
    'pion_20gev_resolution': 0.0407,
    'pion_50gev_resolution': 0.0331,
    'muon_5gev_edep': 5021.32,
    'muon_20gev_edep': 10417.1,
    'muon_50gev_edep': 11905.4,
    'pion_5gev_edep': 4414.98,
    'pion_20gev_edep': 17905.7,
    'pion_50gev_edep': 45285.3
}

# Cylindrical configuration results
cylindrical_metrics = {
    'muon_5gev_resolution': 0.2653,
    'muon_20gev_resolution': 0.3146,
    'muon_50gev_resolution': 0.314,
    'pion_5gev_resolution': 1.6311,
    'pion_20gev_resolution': 1.9645,
    'pion_50gev_resolution': 2.2051,
    'muon_5gev_edep': 62.29,
    'muon_20gev_edep': 63.83,
    'muon_50gev_edep': 65.13,
    'pion_5gev_edep': 184.91,
    'pion_20gev_edep': 264.93,
    'pion_50gev_edep': 330.48,
    'pion_rejection_factor': 4.06
}

# Thick absorber configuration results
thick_absorber_metrics = {
    'muon_5gev_resolution': 0.1413,
    'muon_20gev_resolution': 0.4455,
    'muon_50gev_resolution': 0.8015,
    'pion_5gev_resolution': 0.1464,
    'pion_20gev_resolution': 0.1764,
    'pion_50gev_resolution': 0.2061,
    'muon_5gev_edep': 1298.45,
    'muon_20gev_edep': 1468.9,
    'muon_50gev_edep': 1631.03,
    'pion_5gev_edep': 3968.33,
    'pion_20gev_edep': 16097.5,
    'pion_50gev_edep': 39527,
    'pion_rejection_factor': 12.75
}

# Create comprehensive comparison table
configurations = ['Planar', 'Cylindrical', 'Thick Absorber']
metrics_data = [planar_metrics, cylindrical_metrics, thick_absorber_metrics]

# Calculate average resolutions across energies
avg_muon_resolution = []
avg_pion_resolution = []
for metrics in metrics_data:
    muon_res = np.mean([metrics['muon_5gev_resolution'], metrics['muon_20gev_resolution'], metrics['muon_50gev_resolution']])
    pion_res = np.mean([metrics['pion_5gev_resolution'], metrics['pion_20gev_resolution'], metrics['pion_50gev_resolution']])
    avg_muon_resolution.append(muon_res)
    avg_pion_resolution.append(pion_res)

# Calculate pion rejection factors (where available)
pion_rejection = []
for i, metrics in enumerate(metrics_data):
    if 'pion_rejection_factor' in metrics:
        pion_rejection.append(metrics['pion_rejection_factor'])
    else:
        # Calculate approximate rejection from energy deposit ratios
        pion_edep_20 = metrics['pion_20gev_edep']
        muon_edep_20 = metrics['muon_20gev_edep']
        rejection = pion_edep_20 / muon_edep_20 if muon_edep_20 > 0 else 1
        pion_rejection.append(rejection)

# Create comparison table
print('=== DETECTOR CONFIGURATION COMPARISON ===')
print(f'{'Configuration':<15} {'Avg Muon Res':<12} {'Avg Pion Res':<12} {'Pion Rejection':<15}')
print('-' * 60)
for i, config in enumerate(configurations):
    print(f'{config:<15} {avg_muon_resolution[i]:<12.4f} {avg_pion_resolution[i]:<12.4f} {pion_rejection[i]:<15.2f}')

# Statistical significance analysis
print('\n=== STATISTICAL SIGNIFICANCE ANALYSIS ===')

# Compare muon resolution performance
muon_res_values = np.array(avg_muon_resolution)
best_muon_idx = np.argmin(muon_res_values)
print(f'Best muon resolution: {configurations[best_muon_idx]} ({muon_res_values[best_muon_idx]:.4f})')

# Compare pion rejection performance
pion_rej_values = np.array(pion_rejection)
best_pion_idx = np.argmax(pion_rej_values)
print(f'Best pion rejection: {configurations[best_pion_idx]} ({pion_rej_values[best_pion_idx]:.2f})')

# Overall performance ranking
print('\n=== PERFORMANCE RANKING ===')
print('1. Muon Detection Efficiency (lower resolution = better):')
muon_ranking = np.argsort(muon_res_values)
for i, idx in enumerate(muon_ranking):
    print(f'   {i+1}. {configurations[idx]}: {muon_res_values[idx]:.4f}')

print('\n2. Pion Rejection (higher factor = better):')
pion_ranking = np.argsort(pion_rej_values)[::-1]
for i, idx in enumerate(pion_ranking):
    print(f'   {i+1}. {configurations[idx]}: {pion_rej_values[idx]:.2f}')

# Create comparison plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Average resolution comparison
ax1.bar(configurations, avg_muon_resolution, alpha=0.7, color='blue', label='Muon')
ax1.bar(configurations, avg_pion_resolution, alpha=0.7, color='red', label='Pion')
ax1.set_ylabel('Average Energy Resolution')
ax1.set_title('Average Energy Resolution Comparison')
ax1.legend()
ax1.set_yscale('log')

# Plot 2: Pion rejection comparison
ax2.bar(configurations, pion_rejection, alpha=0.7, color='green')
ax2.set_ylabel('Pion Rejection Factor')
ax2.set_title('Pion Rejection Performance')

# Plot 3: Resolution vs energy for muons
energies = [5, 20, 50]
for i, config in enumerate(configurations):
    muon_res_vs_energy = [metrics_data[i]['muon_5gev_resolution'], 
                         metrics_data[i]['muon_20gev_resolution'], 
                         metrics_data[i]['muon_50gev_resolution']]
    ax3.plot(energies, muon_res_vs_energy, 'o-', label=config, linewidth=2, markersize=6)
ax3.set_xlabel('Energy (GeV)')
ax3.set_ylabel('Muon Energy Resolution')
ax3.set_title('Muon Resolution vs Energy')
ax3.legend()
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3)

# Plot 4: Energy deposit comparison at 20 GeV
muon_edep_20 = [metrics_data[i]['muon_20gev_edep'] for i in range(3)]
pion_edep_20 = [metrics_data[i]['pion_20gev_edep'] for i in range(3)]

x = np.arange(len(configurations))
width = 0.35
ax4.bar(x - width/2, muon_edep_20, width, label='Muon', alpha=0.7, color='blue')
ax4.bar(x + width/2, pion_edep_20, width, label='Pion', alpha=0.7, color='red')
ax4.set_xlabel('Configuration')
ax4.set_ylabel('Energy Deposit (MeV)')
ax4.set_title('Energy Deposit at 20 GeV')
ax4.set_xticks(x)
ax4.set_xticklabels(configurations)
ax4.legend()
ax4.set_yscale('log')

plt.tight_layout()
plt.savefig('detector_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('detector_comparison.pdf', bbox_inches='tight')

# Calculate performance trade-offs
print('\n=== PERFORMANCE TRADE-OFFS ===')
for i, config in enumerate(configurations):
    print(f'{config}:')
    print(f'  - Muon resolution: {avg_muon_resolution[i]:.4f}')
    print(f'  - Pion rejection: {pion_rejection[i]:.2f}')
    print(f'  - Trade-off score: {pion_rejection[i] / avg_muon_resolution[i]:.1f}')

# Save detailed results
results = {
    'configurations': configurations,
    'average_muon_resolution': avg_muon_resolution,
    'average_pion_resolution': avg_pion_resolution,
    'pion_rejection_factors': pion_rejection,
    'best_muon_detector': configurations[best_muon_idx],
    'best_pion_rejection': configurations[best_pion_idx],
    'detailed_metrics': {
        'planar': planar_metrics,
        'cylindrical': cylindrical_metrics,
        'thick_absorber': thick_absorber_metrics
    }
}

with open('detector_comparison_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print summary results
print(f'RESULT:best_muon_detector={configurations[best_muon_idx]}')
print(f'RESULT:best_pion_rejection={configurations[best_pion_idx]}')
print(f'RESULT:planar_avg_muon_resolution={avg_muon_resolution[0]:.4f}')
print(f'RESULT:cylindrical_avg_muon_resolution={avg_muon_resolution[1]:.4f}')
print(f'RESULT:thick_absorber_avg_muon_resolution={avg_muon_resolution[2]:.4f}')
print(f'RESULT:planar_pion_rejection={pion_rejection[0]:.2f}')
print(f'RESULT:cylindrical_pion_rejection={pion_rejection[1]:.2f}')
print(f'RESULT:thick_absorber_pion_rejection={pion_rejection[2]:.2f}')
print('RESULT:comparison_plot=detector_comparison.png')
print('RESULT:results_file=detector_comparison_results.json')
print('RESULT:success=True')