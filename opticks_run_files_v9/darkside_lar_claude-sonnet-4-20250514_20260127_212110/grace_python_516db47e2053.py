import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load nuclear recoil data (proton simulation)
nuclear_files = {
    0.1: '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/darkside_lar_claude-sonnet-4-20250514_20260127_212110/energy_0.000GeV/darkside_detector_proton_events.parquet',
    1.0: '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/darkside_lar_claude-sonnet-4-20250514_20260127_212110/energy_0.001GeV/darkside_detector_proton_events.parquet',
    5.0: '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/darkside_lar_claude-sonnet-4-20250514_20260127_212110/energy_0.005GeV/darkside_detector_events.parquet'
}

# Load electronic recoil data (electron simulation)
electronic_files = {
    0.1: '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/darkside_lar_claude-sonnet-4-20250514_20260127_212110/energy_0.000GeV/darkside_detector_electron_events.parquet',
    1.0: '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/darkside_lar_claude-sonnet-4-20250514_20260127_212110/energy_0.001GeV/darkside_detector_electron_events.parquet',
    5.0: '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/darkside_lar_claude-sonnet-4-20250514_20260127_212110/energy_0.005GeV/darkside_detector_events.parquet'
}

# Analyze each energy point
nuclear_results = []
electronic_results = []
quenching_factors = []

for energy_mev in [100, 1000, 5000]:  # Convert GeV to MeV
    energy_gev = energy_mev / 1000.0
    
    # Load nuclear recoil data
    nuclear_df = pd.read_parquet(nuclear_files[energy_gev])
    nuclear_photons = nuclear_df['nPhotons'].values
    nuclear_mean = np.mean(nuclear_photons)
    nuclear_std = np.std(nuclear_photons)
    nuclear_light_yield = nuclear_mean / energy_mev  # photons per MeV
    
    # Load electronic recoil data
    electronic_df = pd.read_parquet(electronic_files[energy_gev])
    electronic_photons = electronic_df['nPhotons'].values
    electronic_mean = np.mean(electronic_photons)
    electronic_std = np.std(electronic_photons)
    electronic_light_yield = electronic_mean / energy_mev  # photons per MeV
    
    # Calculate quenching factor
    quenching_factor = nuclear_light_yield / electronic_light_yield if electronic_light_yield > 0 else 0
    
    nuclear_results.append({
        'energy_mev': energy_mev,
        'mean_photons': nuclear_mean,
        'std_photons': nuclear_std,
        'light_yield': nuclear_light_yield,
        'num_events': len(nuclear_photons)
    })
    
    electronic_results.append({
        'energy_mev': energy_mev,
        'mean_photons': electronic_mean,
        'std_photons': electronic_std,
        'light_yield': electronic_light_yield,
        'num_events': len(electronic_photons)
    })
    
    quenching_factors.append(quenching_factor)
    
    print(f'Energy: {energy_mev} MeV')
    print(f'Nuclear recoil light yield: {nuclear_light_yield:.2f} photons/MeV')
    print(f'Electronic recoil light yield: {electronic_light_yield:.2f} photons/MeV')
    print(f'Quenching factor: {quenching_factor:.3f}')
    print()

# Calculate average quenching factor
average_quenching = np.mean(quenching_factors)
quenching_std = np.std(quenching_factors)

# Plot comparison of light yields
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Light yield vs energy
energies = [r['energy_mev'] for r in nuclear_results]
nuclear_ly = [r['light_yield'] for r in nuclear_results]
electronic_ly = [r['light_yield'] for r in electronic_results]

ax1.plot(energies, nuclear_ly, 'ro-', label='Nuclear recoils (protons)', linewidth=2, markersize=8)
ax1.plot(energies, electronic_ly, 'bo-', label='Electronic recoils (electrons)', linewidth=2, markersize=8)
ax1.set_xlabel('Energy (MeV)')
ax1.set_ylabel('Light Yield (photons/MeV)')
ax1.set_title('Light Yield Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# Plot 2: Quenching factor vs energy
ax2.plot(energies, quenching_factors, 'go-', linewidth=2, markersize=8)
ax2.axhline(average_quenching, color='r', linestyle='--', label=f'Average: {average_quenching:.3f}')
ax2.set_xlabel('Energy (MeV)')
ax2.set_ylabel('Quenching Factor (Nuclear/Electronic)')
ax2.set_title('Nuclear Recoil Quenching Factor')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

plt.tight_layout()
plt.savefig('nuclear_recoil_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('nuclear_recoil_analysis.pdf', bbox_inches='tight')

# Plot discrimination distributions for highest energy
high_energy_nuclear = pd.read_parquet(nuclear_files[5.0])['nPhotons']
high_energy_electronic = pd.read_parquet(electronic_files[5.0])['nPhotons']

plt.figure(figsize=(10, 6))
plt.hist(high_energy_nuclear, bins=30, alpha=0.7, label='Nuclear recoils', density=True, color='red')
plt.hist(high_energy_electronic, bins=30, alpha=0.7, label='Electronic recoils', density=True, color='blue')
plt.xlabel('Number of Photoelectrons')
plt.ylabel('Normalized Frequency')
plt.title('Discrimination Power at 5 MeV')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('recoil_discrimination.png', dpi=150, bbox_inches='tight')
plt.savefig('recoil_discrimination.pdf', bbox_inches='tight')

# Calculate discrimination power (separation between means in units of combined width)
nuclear_mean_5mev = np.mean(high_energy_nuclear)
electronic_mean_5mev = np.mean(high_energy_electronic)
nuclear_std_5mev = np.std(high_energy_nuclear)
electronic_std_5mev = np.std(high_energy_electronic)
combined_std = np.sqrt(nuclear_std_5mev**2 + electronic_std_5mev**2)
discrimination_power = abs(nuclear_mean_5mev - electronic_mean_5mev) / combined_std

print(f'RESULT:average_quenching_factor={average_quenching:.4f}')
print(f'RESULT:quenching_factor_std={quenching_std:.4f}')
print(f'RESULT:discrimination_power={discrimination_power:.2f}')
print(f'RESULT:nuclear_light_yield_5mev={nuclear_results[2]["light_yield"]:.2f}')
print(f'RESULT:electronic_light_yield_5mev={electronic_results[2]["light_yield"]:.2f}')
print(f'RESULT:quenching_analysis_plot=nuclear_recoil_analysis.png')
print(f'RESULT:discrimination_plot=recoil_discrimination.png')
print('RESULT:success=True')