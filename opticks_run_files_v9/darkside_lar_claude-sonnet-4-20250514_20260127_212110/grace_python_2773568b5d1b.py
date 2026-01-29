import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# File paths from Previous Step Outputs - baseline data
baseline_files = {
    0.0001: '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/darkside_lar_claude-sonnet-4-20250514_20260127_212110/energy_0.000GeV/darkside_detector_electron_hits_data.parquet',
    0.001: '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/darkside_lar_claude-sonnet-4-20250514_20260127_212110/energy_0.001GeV/darkside_detector_electron_hits_data.parquet',
    0.005: '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/darkside_lar_claude-sonnet-4-20250514_20260127_212110/energy_0.005GeV/darkside_detector_hits_data.parquet'
}

# File paths from Previous Step Outputs - optimized data
optimized_files = {
    0.0001: '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/darkside_lar_claude-sonnet-4-20250514_20260127_212110/energy_0.000GeV/darkside_optimized_electron_hits_data.parquet',
    0.001: '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/darkside_lar_claude-sonnet-4-20250514_20260127_212110/energy_0.001GeV/darkside_optimized_electron_hits_data.parquet',
    0.005: '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/darkside_lar_claude-sonnet-4-20250514_20260127_212110/energy_0.005GeV/darkside_optimized_hits_data.parquet'
}

# Previous step results for baseline (from analyze_scintillation_response)
baseline_light_yield = 0.29  # pe/keV
baseline_resolution = 0.0258
baseline_num_pmts = 75
baseline_coverage = 0.0408163

# Previous step results for optimized (from optimize_pmt_coverage)
optimized_num_pmts = 100
optimized_coverage = 0.0544218

# Analysis results storage
results = {'baseline': {}, 'optimized': {}}

# Analyze baseline configuration
print('Analyzing baseline configuration...')
total_hits_baseline = 0
total_edep_baseline = 0
for energy_gev, filepath in baseline_files.items():
    if Path(filepath).exists():
        df = pd.read_parquet(filepath)
        total_hits_baseline += len(df)
        total_edep_baseline += df['edep'].sum() if 'edep' in df.columns else 0
        print(f'Baseline {energy_gev} GeV: {len(df)} hits')

# Analyze optimized configuration
print('Analyzing optimized configuration...')
total_hits_optimized = 0
total_edep_optimized = 0
for energy_gev, filepath in optimized_files.items():
    if Path(filepath).exists():
        df = pd.read_parquet(filepath)
        total_hits_optimized += len(df)
        total_edep_optimized += df['edep'].sum() if 'edep' in df.columns else 0
        print(f'Optimized {energy_gev} GeV: {len(df)} hits')

# Calculate light yield improvement (hits per unit energy)
baseline_light_yield_calc = total_hits_baseline / total_edep_baseline if total_edep_baseline > 0 else 0
optimized_light_yield_calc = total_hits_optimized / total_edep_optimized if total_edep_optimized > 0 else 0

# Calculate improvements
light_yield_improvement = (optimized_light_yield_calc - baseline_light_yield_calc) / baseline_light_yield_calc * 100 if baseline_light_yield_calc > 0 else 0
coverage_improvement = (optimized_coverage - baseline_coverage) / baseline_coverage * 100
pmt_count_improvement = (optimized_num_pmts - baseline_num_pmts) / baseline_num_pmts * 100

# Statistical significance calculation
n_baseline = total_hits_baseline
n_optimized = total_hits_optimized
if n_baseline > 0 and n_optimized > 0:
    # Standard error for light yield ratio
    se_baseline = np.sqrt(baseline_light_yield_calc / n_baseline)
    se_optimized = np.sqrt(optimized_light_yield_calc / n_optimized)
    combined_se = np.sqrt(se_baseline**2 + se_optimized**2)
    
    # Z-score for significance test
    z_score = abs(optimized_light_yield_calc - baseline_light_yield_calc) / combined_se if combined_se > 0 else 0
    p_value = 2 * (1 - 0.5 * (1 + np.sign(z_score) * np.sqrt(1 - np.exp(-2 * z_score**2 / np.pi))))
    significant = p_value < 0.05
else:
    z_score = 0
    p_value = 1.0
    significant = False

# Create comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Light yield comparison
configs = ['Baseline\n(75 PMTs)', 'Optimized\n(100 PMTs)']
light_yields = [baseline_light_yield_calc, optimized_light_yield_calc]
coverages = [baseline_coverage * 100, optimized_coverage * 100]

ax1.bar(configs, light_yields, color=['blue', 'green'], alpha=0.7)
ax1.set_ylabel('Light Yield (hits/MeV)')
ax1.set_title('Light Yield Comparison')
for i, v in enumerate(light_yields):
    ax1.text(i, v + 0.01 * max(light_yields), f'{v:.3f}', ha='center', va='bottom')

# Coverage comparison
ax2.bar(configs, coverages, color=['blue', 'green'], alpha=0.7)
ax2.set_ylabel('PMT Coverage (%)')
ax2.set_title('PMT Coverage Comparison')
for i, v in enumerate(coverages):
    ax2.text(i, v + 0.1, f'{v:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('optimization_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('optimization_comparison.pdf', bbox_inches='tight')

# Summary statistics
print('\n=== OPTIMIZATION RESULTS ===')
print(f'Baseline PMT count: {baseline_num_pmts}')
print(f'Optimized PMT count: {optimized_num_pmts}')
print(f'PMT count improvement: {pmt_count_improvement:.1f}%')
print(f'Coverage improvement: {coverage_improvement:.1f}%')
print(f'Light yield improvement: {light_yield_improvement:.1f}%')
print(f'Statistical significance: p = {p_value:.4f} (significant: {significant})')

# Save detailed results
results_summary = {
    'baseline': {
        'pmt_count': baseline_num_pmts,
        'coverage': baseline_coverage,
        'light_yield': float(baseline_light_yield_calc),
        'total_hits': int(total_hits_baseline),
        'energy_resolution': baseline_resolution
    },
    'optimized': {
        'pmt_count': optimized_num_pmts,
        'coverage': optimized_coverage,
        'light_yield': float(optimized_light_yield_calc),
        'total_hits': int(total_hits_optimized)
    },
    'improvements': {
        'pmt_count_percent': float(pmt_count_improvement),
        'coverage_percent': float(coverage_improvement),
        'light_yield_percent': float(light_yield_improvement)
    },
    'statistics': {
        'z_score': float(z_score),
        'p_value': float(p_value),
        'significant': bool(significant)
    }
}

with open('optimization_analysis.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

# Output results for workflow
print(f'RESULT:pmt_count_improvement={pmt_count_improvement:.2f}')
print(f'RESULT:coverage_improvement={coverage_improvement:.2f}')
print(f'RESULT:light_yield_improvement={light_yield_improvement:.2f}')
print(f'RESULT:statistical_significance={significant}')
print(f'RESULT:p_value={p_value:.4f}')
print(f'RESULT:comparison_plot=optimization_comparison.png')
print(f'RESULT:analysis_file=optimization_analysis.json')
print('RESULT:success=True')