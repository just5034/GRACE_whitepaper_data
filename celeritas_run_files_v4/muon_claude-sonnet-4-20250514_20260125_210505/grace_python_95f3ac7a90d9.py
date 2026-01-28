import matplotlib
matplotlib.use('Agg')
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Configuration names and their corresponding file paths
configurations = {
    'baseline_planar': 'baseline_planar_muon_spectrometer',
    'cylindrical_barrel': 'cylindrical_barrel_muon_spectrometer', 
    'thick_absorber': 'thick_absorber_muon_spectrometer',
    'thin_absorber': 'thin_absorber_muon_spectrometer'
}

# Energy thresholds for detection (MeV)
muon_detection_threshold = 1.0  # Minimum energy deposit to count as detected
pion_stopping_threshold = 50.0  # Energy threshold to distinguish stopping vs penetrating

results = {}
all_metrics = {}

print('Analyzing all detector configurations...')

for config_name, file_prefix in configurations.items():
    print(f'\nAnalyzing {config_name}...')
    
    # Load muon data
    muon_file = f'{file_prefix}_mum_hits.root'
    pion_file = f'{file_prefix}_pim_hits.root'
    
    config_results = {}
    
    try:
        # Analyze muon efficiency
        with uproot.open(muon_file) as f:
            muon_events = f['events'].arrays(library='pd')
            
        total_muons = len(muon_events)
        detected_muons = len(muon_events[muon_events['totalEdep'] > muon_detection_threshold])
        muon_efficiency = detected_muons / total_muons if total_muons > 0 else 0
        muon_eff_err = np.sqrt(muon_efficiency * (1 - muon_efficiency) / total_muons) if total_muons > 0 else 0
        
        config_results['muon_efficiency'] = muon_efficiency
        config_results['muon_efficiency_err'] = muon_eff_err
        config_results['total_muons'] = total_muons
        config_results['detected_muons'] = detected_muons
        
        print(f'  Muon efficiency: {muon_efficiency:.4f} ± {muon_eff_err:.4f}')
        
        # Analyze pion rejection
        with uproot.open(pion_file) as f:
            pion_events = f['events'].arrays(library='pd')
            
        total_pions = len(pion_events)
        # Pions that deposit significant energy (not rejected)
        penetrating_pions = len(pion_events[pion_events['totalEdep'] > pion_stopping_threshold])
        stopped_pions = total_pions - penetrating_pions
        pion_rejection = stopped_pions / total_pions if total_pions > 0 else 0
        pion_rej_err = np.sqrt(pion_rejection * (1 - pion_rejection) / total_pions) if total_pions > 0 else 0
        
        config_results['pion_rejection'] = pion_rejection
        config_results['pion_rejection_err'] = pion_rej_err
        config_results['total_pions'] = total_pions
        config_results['stopped_pions'] = stopped_pions
        
        print(f'  Pion rejection: {pion_rejection:.4f} ± {pion_rej_err:.4f}')
        
        # Calculate combined performance metric
        performance_score = muon_efficiency * pion_rejection
        config_results['performance_score'] = performance_score
        
        print(f'  Performance score: {performance_score:.4f}')
        
    except Exception as e:
        print(f'  Error analyzing {config_name}: {e}')
        config_results = {'error': str(e)}
    
    results[config_name] = config_results
    all_metrics[config_name] = config_results

# Create comprehensive comparison plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Extract data for plotting
config_names = []
muon_effs = []
muon_errs = []
pion_rejs = []
pion_errs = []
perf_scores = []

for config, data in results.items():
    if 'error' not in data:
        config_names.append(config.replace('_', ' ').title())
        muon_effs.append(data['muon_efficiency'])
        muon_errs.append(data['muon_efficiency_err'])
        pion_rejs.append(data['pion_rejection'])
        pion_errs.append(data['pion_rejection_err'])
        perf_scores.append(data['performance_score'])

# Plot 1: Muon Efficiency Comparison
colors = ['blue', 'green', 'red', 'orange']
ax1.bar(range(len(config_names)), muon_effs, yerr=muon_errs, 
        capsize=5, color=colors[:len(config_names)], alpha=0.7)
ax1.set_xlabel('Detector Configuration')
ax1.set_ylabel('Muon Detection Efficiency')
ax1.set_title('Muon Detection Efficiency Comparison')
ax1.set_xticks(range(len(config_names)))
ax1.set_xticklabels(config_names, rotation=45, ha='right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.1)

# Plot 2: Pion Rejection Comparison
ax2.bar(range(len(config_names)), pion_rejs, yerr=pion_errs,
        capsize=5, color=colors[:len(config_names)], alpha=0.7)
ax2.set_xlabel('Detector Configuration')
ax2.set_ylabel('Pion Rejection Fraction')
ax2.set_title('Pion Rejection Performance Comparison')
ax2.set_xticks(range(len(config_names)))
ax2.set_xticklabels(config_names, rotation=45, ha='right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.1)

# Plot 3: Performance Score
ax3.bar(range(len(config_names)), perf_scores,
        color=colors[:len(config_names)], alpha=0.7)
ax3.set_xlabel('Detector Configuration')
ax3.set_ylabel('Performance Score (Efficiency × Rejection)')
ax3.set_title('Overall Performance Comparison')
ax3.set_xticks(range(len(config_names)))
ax3.set_xticklabels(config_names, rotation=45, ha='right')
ax3.grid(True, alpha=0.3)

# Plot 4: Efficiency vs Rejection scatter
ax4.errorbar(muon_effs, pion_rejs, xerr=muon_errs, yerr=pion_errs,
             fmt='o', markersize=10, capsize=5)
for i, name in enumerate(config_names):
    ax4.annotate(name, (muon_effs[i], pion_rejs[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=10)
ax4.set_xlabel('Muon Detection Efficiency')
ax4.set_ylabel('Pion Rejection Fraction')
ax4.set_title('Efficiency vs Rejection Trade-off')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 1.1)
ax4.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('all_configurations_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('all_configurations_comparison.pdf', bbox_inches='tight')
plt.close()

# Find optimal configuration
if perf_scores:
    best_idx = np.argmax(perf_scores)
    best_config = config_names[best_idx]
    best_score = perf_scores[best_idx]
    best_muon_eff = muon_effs[best_idx]
    best_pion_rej = pion_rejs[best_idx]
else:
    best_config = 'None'
    best_score = 0
    best_muon_eff = 0
    best_pion_rej = 0

print(f'\n=== COMPREHENSIVE ANALYSIS RESULTS ===')
print(f'Best performing configuration: {best_config}')
print(f'Performance score: {best_score:.4f}')
print(f'Muon efficiency: {best_muon_eff:.4f}')
print(f'Pion rejection: {best_pion_rej:.4f}')

# Save detailed results
with open('all_configurations_analysis.json', 'w') as f:
    json.dump(all_metrics, f, indent=2, default=str)

# Print results for workflow
print(f'RESULT:best_configuration={best_config.lower().replace(" ", "_")}')
print(f'RESULT:best_performance_score={best_score:.4f}')
print(f'RESULT:best_muon_efficiency={best_muon_eff:.4f}')
print(f'RESULT:best_pion_rejection={best_pion_rej:.4f}')
print('RESULT:comparison_plot=all_configurations_comparison.png')
print('RESULT:analysis_complete=True')

print('\nComprehensive analysis completed successfully!')