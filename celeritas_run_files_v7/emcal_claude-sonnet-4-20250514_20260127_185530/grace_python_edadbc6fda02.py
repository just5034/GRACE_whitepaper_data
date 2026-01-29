import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Energy sweep file paths from Previous Step Outputs
energy_files = {
    0.5: "/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/emcal_claude-sonnet-4-20250514_20260127_185530/energy_0.500GeV/pbwo4_box_calorimeter_electron_events.parquet",
    2.0: "/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/emcal_claude-sonnet-4-20250514_20260127_185530/energy_2.000GeV/pbwo4_box_calorimeter_electron_events.parquet",
    5.0: "/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/emcal_claude-sonnet-4-20250514_20260127_185530/energy_5.000GeV/pbwo4_box_calorimeter_electron_events.parquet"
}

# Geometry parameters from Previous Step Outputs
sampling_fraction = 0.994413
containment_radius_cm = 5.475

# Analyze each energy point
results = []
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (energy_gev, filepath) in enumerate(energy_files.items()):
    # Read events data
    events = pd.read_parquet(filepath)
    
    # Calculate beam energy in MeV for linearity
    beam_energy_mev = energy_gev * 1000
    
    # Calculate energy resolution metrics
    mean_deposit_mev = events['totalEdep'].mean()
    std_deposit_mev = events['totalEdep'].std()
    energy_resolution = std_deposit_mev / mean_deposit_mev
    resolution_err = energy_resolution / np.sqrt(2 * len(events))
    
    # Calculate linearity (response/beam_energy)
    linearity = mean_deposit_mev / beam_energy_mev
    
    # Store results
    results.append({
        'energy_gev': energy_gev,
        'beam_energy_mev': beam_energy_mev,
        'mean_deposit_mev': mean_deposit_mev,
        'std_deposit_mev': std_deposit_mev,
        'energy_resolution': energy_resolution,
        'resolution_err': resolution_err,
        'linearity': linearity,
        'num_events': len(events)
    })
    
    # Plot energy distribution for this energy
    axes[i].hist(events['totalEdep'], bins=50, histtype='step', linewidth=2, alpha=0.8)
    axes[i].axvline(mean_deposit_mev, color='r', linestyle='--', 
                   label=f'Mean: {mean_deposit_mev:.1f} MeV')
    axes[i].set_xlabel('Energy Deposit (MeV)')
    axes[i].set_ylabel('Events')
    axes[i].set_title(f'{energy_gev} GeV: σ/E = {energy_resolution:.4f}')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.suptitle('PbWO4 Energy Distributions', fontsize=14)
plt.tight_layout()
plt.savefig('pbwo4_energy_distributions.png', dpi=150, bbox_inches='tight')
plt.savefig('pbwo4_energy_distributions.pdf', bbox_inches='tight')
plt.close()

# Create resolution curve plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Energy resolution vs energy
energies = [r['energy_gev'] for r in results]
resolutions = [r['energy_resolution'] for r in results]
resolution_errs = [r['resolution_err'] for r in results]

ax1.errorbar(energies, resolutions, yerr=resolution_errs, 
            marker='o', linewidth=2, markersize=8, capsize=5)
ax1.set_xlabel('Beam Energy (GeV)')
ax1.set_ylabel('Energy Resolution (σ/E)')
ax1.set_title('PbWO4 Energy Resolution')
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# Linearity plot
linearities = [r['linearity'] for r in results]
ax2.plot(energies, linearities, 'o-', linewidth=2, markersize=8)
ax2.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='Perfect linearity')
ax2.set_xlabel('Beam Energy (GeV)')
ax2.set_ylabel('Linearity (Response/Energy)')
ax2.set_title('PbWO4 Linearity')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

plt.tight_layout()
plt.savefig('pbwo4_resolution_curve.png', dpi=150, bbox_inches='tight')
plt.savefig('pbwo4_resolution_curve.pdf', bbox_inches='tight')
plt.close()

# Save detailed results to JSON
results_dict = {
    'material': 'PbWO4',
    'sampling_fraction': sampling_fraction,
    'containment_radius_cm': containment_radius_cm,
    'energy_points': results,
    'plots': {
        'energy_distributions': 'pbwo4_energy_distributions.png',
        'resolution_curve': 'pbwo4_resolution_curve.png'
    }
}

with open('pbwo4_performance_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

# Output individual energy point results for workflow
for r in results:
    energy_str = f"{r['energy_gev']:.1f}gev".replace('.', '_')
    print(f"RESULT:pbwo4_{energy_str}_energy_resolution={r['energy_resolution']:.6f}")
    print(f"RESULT:pbwo4_{energy_str}_resolution_err={r['resolution_err']:.6f}")
    print(f"RESULT:pbwo4_{energy_str}_mean_deposit_mev={r['mean_deposit_mev']:.2f}")
    print(f"RESULT:pbwo4_{energy_str}_linearity={r['linearity']:.4f}")
    print(f"RESULT:pbwo4_{energy_str}_num_events={r['num_events']}")

# Output summary results
print(f"RESULT:pbwo4_resolution_curve_plot=pbwo4_resolution_curve.png")
print(f"RESULT:pbwo4_energy_distributions_plot=pbwo4_energy_distributions.png")
print(f"RESULT:pbwo4_results_json=pbwo4_performance_results.json")
print("RESULT:success=True")

print("\nPbWO4 Performance Analysis Complete:")
for r in results:
    print(f"  {r['energy_gev']} GeV: σ/E = {r['energy_resolution']:.4f} ± {r['resolution_err']:.4f}, Linearity = {r['linearity']:.4f}")