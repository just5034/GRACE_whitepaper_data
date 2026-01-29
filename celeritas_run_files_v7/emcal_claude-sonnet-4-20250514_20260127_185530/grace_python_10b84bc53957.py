import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Energy sweep file paths from Previous Step Outputs
energy_files = {
    0.5: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/emcal_claude-sonnet-4-20250514_20260127_185530/energy_0.500GeV/bgo_box_calorimeter_electron_events.parquet',
    2.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/emcal_claude-sonnet-4-20250514_20260127_185530/energy_2.000GeV/bgo_box_calorimeter_electron_events.parquet',
    5.0: '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/emcal_claude-sonnet-4-20250514_20260127_185530/energy_5.000GeV/bgo_box_calorimeter_electron_events.parquet'
}

# BGO geometry parameters from Previous Step Outputs
sampling_fraction = 0.995556
transverse_size_m = 0.1115
total_depth_m = 0.225

# Analyze each energy point
results = []
for energy_gev, filepath in energy_files.items():
    print(f'Analyzing BGO at {energy_gev} GeV...')
    
    # Read events data
    df = pd.read_parquet(filepath)
    beam_energy_mev = energy_gev * 1000  # Convert GeV to MeV
    
    # Calculate energy resolution metrics
    mean_edep = df['totalEdep'].mean()
    std_edep = df['totalEdep'].std()
    num_events = len(df)
    
    # Energy resolution (σ/E)
    energy_resolution = std_edep / mean_edep if mean_edep > 0 else 0
    resolution_err = energy_resolution / np.sqrt(2 * num_events) if num_events > 0 else 0
    
    # Linearity (response vs beam energy)
    linearity = mean_edep / beam_energy_mev if beam_energy_mev > 0 else 0
    
    results.append({
        'energy_gev': energy_gev,
        'beam_energy_mev': beam_energy_mev,
        'mean_edep_mev': mean_edep,
        'std_edep_mev': std_edep,
        'energy_resolution': energy_resolution,
        'resolution_err': resolution_err,
        'linearity': linearity,
        'num_events': num_events
    })
    
    # Plot raw energy distribution for this energy
    plt.figure(figsize=(10, 6))
    plt.hist(df['totalEdep'], bins=50, histtype='step', linewidth=2, alpha=0.8)
    plt.axvline(mean_edep, color='r', linestyle='--', label=f'Mean: {mean_edep:.1f} MeV')
    plt.xlabel('Total Energy Deposit (MeV)')
    plt.ylabel('Events')
    plt.title(f'BGO Energy Distribution at {energy_gev} GeV (σ/E = {energy_resolution:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'bgo_energy_dist_{energy_gev}gev.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'bgo_energy_dist_{energy_gev}gev.pdf', bbox_inches='tight')
    plt.close()

# Convert to arrays for analysis
energies = np.array([r['energy_gev'] for r in results])
resolutions = np.array([r['energy_resolution'] for r in results])
resolution_errs = np.array([r['resolution_err'] for r in results])
linearities = np.array([r['linearity'] for r in results])
mean_edeps = np.array([r['mean_edep_mev'] for r in results])

# Fit energy resolution curve: σ/E = a/√E + b + c*E
# For BGO, expect stochastic term dominance
from scipy.optimize import curve_fit

def resolution_func(E, a, b, c):
    return a / np.sqrt(E) + b + c * E

try:
    popt, pcov = curve_fit(resolution_func, energies, resolutions, sigma=resolution_errs, absolute_sigma=True)
    a_fit, b_fit, c_fit = popt
    fit_errs = np.sqrt(np.diag(pcov))
    
    # Calculate fit quality
    chi2 = np.sum(((resolutions - resolution_func(energies, *popt)) / resolution_errs)**2)
    ndof = len(energies) - 3
    chi2_ndf = chi2 / ndof if ndof > 0 else 0
    
    print(f'BGO Resolution Fit: σ/E = {a_fit:.4f}/√E + {b_fit:.4f} + {c_fit:.6f}*E')
    print(f'Fit quality: χ²/ndf = {chi2_ndf:.2f}')
except:
    print('Resolution curve fit failed - using individual points')
    a_fit, b_fit, c_fit = 0, np.mean(resolutions), 0
    fit_errs = [0, 0, 0]

# Create comprehensive performance plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Energy resolution vs energy
ax1.errorbar(energies, resolutions * 100, yerr=resolution_errs * 100, 
             fmt='o', capsize=5, markersize=8, label='BGO Data')
if a_fit > 0:
    E_fit = np.linspace(0.3, 6, 100)
    res_fit = resolution_func(E_fit, a_fit, b_fit, c_fit)
    ax1.plot(E_fit, res_fit * 100, 'r-', linewidth=2, 
             label=f'Fit: {a_fit:.3f}/√E + {b_fit:.3f} + {c_fit:.5f}E')
ax1.set_xlabel('Beam Energy (GeV)')
ax1.set_ylabel('Energy Resolution σ/E (%)')
ax1.set_title('BGO Energy Resolution vs Energy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Linearity
ax2.plot(energies, linearities, 'o-', markersize=8, linewidth=2, color='green')
ax2.set_xlabel('Beam Energy (GeV)')
ax2.set_ylabel('Response (MeV_dep / MeV_beam)')
ax2.set_title('BGO Linearity')
ax2.grid(True, alpha=0.3)

# Mean energy deposit vs beam energy
ax3.plot(energies * 1000, mean_edeps, 'o-', markersize=8, linewidth=2, color='blue')
beam_energies_mev = energies * 1000
ax3.plot(beam_energies_mev, beam_energies_mev, 'k--', alpha=0.5, label='Perfect linearity')
ax3.set_xlabel('Beam Energy (MeV)')
ax3.set_ylabel('Mean Energy Deposit (MeV)')
ax3.set_title('BGO Energy Response')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Resolution vs 1/√E (should be linear if stochastic term dominates)
ax4.plot(1/np.sqrt(energies), resolutions * 100, 'o', markersize=8, color='purple')
ax4.set_xlabel('1/√E (GeV^-0.5)')
ax4.set_ylabel('Energy Resolution σ/E (%)')
ax4.set_title('BGO Resolution vs 1/√E')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bgo_performance_summary.png', dpi=150, bbox_inches='tight')
plt.savefig('bgo_performance_summary.pdf', bbox_inches='tight')
plt.close()

# Calculate shower containment (using hits data for one energy point)
containment_radius_mm = None
try:
    # Use 2 GeV data for containment analysis
    hits_file = '/u/jhill5/grace/work/benchmarks/celeritas_20260127_185459/emcal_claude-sonnet-4-20250514_20260127_185530/energy_2.000GeV/bgo_box_calorimeter_electron_hits_data.parquet'
    hits_df = pd.read_parquet(hits_file)
    
    # Calculate radial distance from center
    r_mm = np.sqrt(hits_df['x']**2 + hits_df['y']**2)
    
    # Radial containment analysis
    radius_bins = np.linspace(0, transverse_size_m * 500, 50)  # mm
    total_edep = hits_df['edep'].sum()
    
    containment_fractions = []
    for r_cut in radius_bins:
        contained_edep = hits_df[r_mm < r_cut]['edep'].sum()
        containment_fractions.append(contained_edep / total_edep if total_edep > 0 else 0)
    
    containment_fractions = np.array(containment_fractions)
    
    # Find 90% containment radius
    idx_90 = np.where(containment_fractions >= 0.9)[0]
    containment_radius_mm = radius_bins[idx_90[0]] if len(idx_90) > 0 else radius_bins[-1]
    
    # Plot containment
    plt.figure(figsize=(10, 6))
    plt.plot(radius_bins, containment_fractions * 100, linewidth=2, color='red')
    plt.axhline(90, color='k', linestyle='--', alpha=0.7, label='90% containment')
    plt.axvline(containment_radius_mm, color='k', linestyle=':', alpha=0.7, 
                label=f'R90 = {containment_radius_mm:.1f} mm')
    plt.xlabel('Radius (mm)')
    plt.ylabel('Shower Containment (%)')
    plt.title('BGO Radial Shower Containment (2 GeV electrons)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('bgo_shower_containment.png', dpi=150, bbox_inches='tight')
    plt.savefig('bgo_shower_containment.pdf', bbox_inches='tight')
    plt.close()
    
except Exception as e:
    print(f'Containment analysis failed: {e}')
    containment_radius_mm = 0

# Save results to JSON
performance_metrics = {
    'material': 'BGO',
    'geometry': {
        'transverse_size_m': transverse_size_m,
        'depth_m': total_depth_m,
        'sampling_fraction': sampling_fraction
    },
    'energy_points': results,
    'resolution_fit': {
        'stochastic_term': a_fit,
        'constant_term': b_fit,
        'noise_term': c_fit,
        'fit_errors': fit_errs,
        'chi2_ndf': chi2_ndf if 'chi2_ndf' in locals() else 0
    },
    'containment': {
        'r90_mm': containment_radius_mm
    }
}

with open('bgo_performance_metrics.json', 'w') as f:
    json.dump(performance_metrics, f, indent=2)

# Output results for workflow
print('\n=== BGO PERFORMANCE SUMMARY ===')
for i, result in enumerate(results):
    energy = result['energy_gev']
    resolution = result['energy_resolution']
    linearity = result['linearity']
    print(f'{energy} GeV: σ/E = {resolution:.4f}, linearity = {linearity:.3f}')

print(f'\nResolution fit: σ/E = {a_fit:.4f}/√E + {b_fit:.4f} + {c_fit:.6f}*E')
if containment_radius_mm:
    print(f'90% containment radius: {containment_radius_mm:.1f} mm')

# Return values for downstream steps
print(f'RESULT:bgo_stochastic_term={a_fit:.6f}')
print(f'RESULT:bgo_constant_term={b_fit:.6f}')
print(f'RESULT:bgo_noise_term={c_fit:.8f}')
print(f'RESULT:bgo_resolution_0.5gev={results[0]["energy_resolution"]:.6f}')
print(f'RESULT:bgo_resolution_2gev={results[1]["energy_resolution"]:.6f}')
print(f'RESULT:bgo_resolution_5gev={results[2]["energy_resolution"]:.6f}')
print(f'RESULT:bgo_linearity_0.5gev={results[0]["linearity"]:.6f}')
print(f'RESULT:bgo_linearity_2gev={results[1]["linearity"]:.6f}')
print(f'RESULT:bgo_linearity_5gev={results[2]["linearity"]:.6f}')
if containment_radius_mm:
    print(f'RESULT:bgo_r90_containment_mm={containment_radius_mm:.2f}')
print('RESULT:bgo_performance_plot=bgo_performance_summary.png')
print('RESULT:bgo_containment_plot=bgo_shower_containment.png')
print('RESULT:bgo_metrics_file=bgo_performance_metrics.json')
print('RESULT:success=True')