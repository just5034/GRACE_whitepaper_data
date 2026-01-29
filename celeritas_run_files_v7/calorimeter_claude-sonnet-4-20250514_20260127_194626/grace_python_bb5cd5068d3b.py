import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Analysis results from previous step (analyze_baseline_performance)
# Energy points and their results
energies = [10.0, 30.0, 50.0]
resolutions = [0.095581, 0.077267, 0.070718]
resolution_errors = [0.000956, 0.000773, 0.000707]
mean_deposits = [8175.35, 25111.1, 42250.1]
linearities = [0.817535, 0.837036, 0.845002]
num_events = [5000, 5000, 5000]

# Set up publication-quality plotting parameters
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (10, 8)
})

# Create figure with subplots for all plot types
fig = plt.figure(figsize=(16, 12))

# Plot 1: Energy Resolution vs Energy
ax1 = plt.subplot(2, 2, 1)
plt.errorbar(energies, resolutions, yerr=resolution_errors, 
             marker='o', markersize=8, linewidth=2, capsize=5, capthick=2)
plt.xlabel('Beam Energy (GeV)')
plt.ylabel('Energy Resolution (σ/E)')
plt.title('Baseline Calorimeter Energy Resolution')
plt.grid(True, alpha=0.3)
plt.xlim(5, 55)
plt.ylim(0.06, 0.10)

# Add fit line for resolution scaling
fit_energies = np.linspace(10, 50, 100)
# Typical calorimeter resolution: σ/E = a/√E + b
a_fit = np.mean([r * np.sqrt(e) for r, e in zip(resolutions, energies)])
fit_resolution = a_fit / np.sqrt(fit_energies)
plt.plot(fit_energies, fit_resolution, '--', color='red', alpha=0.7, label=f'σ/E ∝ 1/√E fit')
plt.legend()

# Plot 2: Linearity Response
ax2 = plt.subplot(2, 2, 2)
expected_deposits = [e * 1000 for e in energies]  # Expected: E_beam * 1000 MeV/GeV
plt.plot(expected_deposits, mean_deposits, 'o-', markersize=8, linewidth=2, label='Measured')
plt.plot([0, 50000], [0, 50000], '--', color='red', alpha=0.7, label='Perfect linearity')
plt.xlabel('Expected Energy Deposit (MeV)')
plt.ylabel('Measured Energy Deposit (MeV)')
plt.title('Baseline Calorimeter Linearity Response')
plt.grid(True, alpha=0.3)
plt.legend()

# Add linearity values as text
for i, (x, y, lin) in enumerate(zip(expected_deposits, mean_deposits, linearities)):
    plt.annotate(f'L={lin:.3f}', (x, y), xytext=(5, 5), textcoords='offset points')

# Plot 3: Energy Distributions (load from parquet files)
ax3 = plt.subplot(2, 2, 3)

# File paths from simulate_baseline_energy_sweep step
file_paths = {
    10.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/calorimeter_claude-sonnet-4-20250514_20260127_194626/energy_10.000GeV/baseline_calorimeter_pip_events.parquet',
    30.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/calorimeter_claude-sonnet-4-20250514_20260127_194626/energy_30.000GeV/baseline_calorimeter_pip_events.parquet',
    50.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/calorimeter_claude-sonnet-4-20250514_20260127_194626/energy_50.000GeV/baseline_calorimeter_pip_events.parquet'
}

colors = ['blue', 'green', 'red']
for i, (energy, filepath) in enumerate(file_paths.items()):
    if Path(filepath).exists():
        df = pd.read_parquet(filepath)
        plt.hist(df['totalEdep'], bins=50, alpha=0.6, histtype='step', 
                linewidth=2, color=colors[i], label=f'{energy} GeV')
        # Add mean line
        mean_val = df['totalEdep'].mean()
        plt.axvline(mean_val, color=colors[i], linestyle='--', alpha=0.8)

plt.xlabel('Total Energy Deposit (MeV)')
plt.ylabel('Events')
plt.title('Energy Distributions for Different Beam Energies')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Shower Profiles (longitudinal)
ax4 = plt.subplot(2, 2, 4)

# Load hits data for shower profile analysis
hits_files = {
    10.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/calorimeter_claude-sonnet-4-20250514_20260127_194626/energy_10.000GeV/baseline_calorimeter_pip_hits_data.parquet',
    30.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/calorimeter_claude-sonnet-4-20250514_20260127_194626/energy_30.000GeV/baseline_calorimeter_pip_hits_data.parquet',
    50.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/calorimeter_claude-sonnet-4-20250514_20260127_194626/energy_50.000GeV/baseline_calorimeter_pip_hits_data.parquet'
}

z_bins = np.linspace(0, 1700, 50)  # 0 to 1.7m depth in mm
z_centers = (z_bins[:-1] + z_bins[1:]) / 2

for i, (energy, filepath) in enumerate(hits_files.items()):
    if Path(filepath).exists():
        # Sample first 100k hits to avoid timeout
        df_hits = pd.read_parquet(filepath)
        if len(df_hits) > 100000:
            df_hits = df_hits.sample(100000)
        
        # Create longitudinal profile
        z_hist, _ = np.histogram(df_hits['z'], bins=z_bins, weights=df_hits['edep'])
        # Normalize by total energy for comparison
        z_hist = z_hist / np.sum(z_hist) if np.sum(z_hist) > 0 else z_hist
        
        plt.step(z_centers, z_hist, where='mid', linewidth=2, 
                color=colors[i], label=f'{energy} GeV')

plt.xlabel('Depth (mm)')
plt.ylabel('Normalized Energy Deposit')
plt.title('Longitudinal Shower Profiles')
plt.legend()
plt.grid(True, alpha=0.3)

# Adjust layout and save
plt.tight_layout()
plt.savefig('baseline_performance_plots.png', dpi=300, bbox_inches='tight')
plt.savefig('baseline_performance_plots.pdf', bbox_inches='tight')

# Create summary statistics table
print('\n=== BASELINE CALORIMETER PERFORMANCE SUMMARY ===')
print(f'Energy Range: {min(energies)}-{max(energies)} GeV')
print(f'Mean Resolution: {np.mean(resolutions):.4f} ± {np.std(resolutions):.4f}')
print(f'Resolution at 10 GeV: {resolutions[0]:.4f} ± {resolution_errors[0]:.4f}')
print(f'Resolution at 50 GeV: {resolutions[2]:.4f} ± {resolution_errors[2]:.4f}')
print(f'Mean Linearity: {np.mean(linearities):.4f}')
print(f'Linearity Spread: {np.std(linearities):.4f}')

# Output results for workflow
print(f'RESULT:baseline_mean_resolution={np.mean(resolutions):.6f}')
print(f'RESULT:baseline_resolution_spread={np.std(resolutions):.6f}')
print(f'RESULT:baseline_mean_linearity={np.mean(linearities):.6f}')
print(f'RESULT:baseline_linearity_spread={np.std(linearities):.6f}')
print('RESULT:baseline_performance_plots=baseline_performance_plots.png')
print('RESULT:baseline_performance_plots_pdf=baseline_performance_plots.pdf')
print('RESULT:success=True')

print('\nPublication-quality baseline performance plots generated successfully!')
print('Files saved: baseline_performance_plots.png, baseline_performance_plots.pdf')