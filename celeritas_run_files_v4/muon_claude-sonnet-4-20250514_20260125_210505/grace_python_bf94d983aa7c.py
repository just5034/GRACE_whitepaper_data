import matplotlib
matplotlib.use('Agg')
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Configuration parameters from requirements
min_muon_efficiency = 0.95
weight_efficiency = 0.6
weight_rejection = 0.4
energy_range = [5, 100]  # GeV

# Define configurations to analyze
configurations = {
    'baseline_planar': 'baseline_planar_muon_spectrometer',
    'cylindrical_barrel': 'cylindrical_barrel_muon_spectrometer', 
    'thick_absorber': 'thick_absorber_muon_spectrometer',
    'thin_absorber': 'thin_absorber_muon_spectrometer'
}

results = {}
print('Analyzing detector configurations for optimal performance...')

# Analyze each configuration
for config_name, file_prefix in configurations.items():
    print(f'\nAnalyzing {config_name}...')
    
    try:
        # Load muon data
        muon_file = f'{file_prefix}_mum_hits.root'
        if Path(muon_file).exists():
            with uproot.open(muon_file) as f:
                muon_events = f['events'].arrays(library='pd')
        else:
            print(f'Warning: {muon_file} not found')
            continue
            
        # Load pion data  
        pion_file = f'{file_prefix}_pim_hits.root'
        if Path(pion_file).exists():
            with uproot.open(pion_file) as f:
                pion_events = f['events'].arrays(library='pd')
        else:
            print(f'Warning: {pion_file} not found')
            continue
            
        # Calculate muon efficiency (fraction with hits above threshold)
        muon_threshold = 0.1  # MeV minimum energy deposit
        total_muons = len(muon_events)
        detected_muons = len(muon_events[muon_events['totalEdep'] > muon_threshold])
        muon_efficiency = detected_muons / total_muons if total_muons > 0 else 0
        
        # Calculate pion rejection (fraction that deposit significant energy - shower)
        pion_threshold = 10.0  # MeV - pions shower and deposit more energy
        total_pions = len(pion_events)
        showering_pions = len(pion_events[pion_events['totalEdep'] > pion_threshold])
        pion_rejection_rate = showering_pions / total_pions if total_pions > 0 else 0
        
        # Store results
        results[config_name] = {
            'muon_efficiency': muon_efficiency,
            'pion_rejection_rate': pion_rejection_rate,
            'total_muons': total_muons,
            'total_pions': total_pions,
            'detected_muons': detected_muons,
            'showering_pions': showering_pions
        }
        
        print(f'  Muon efficiency: {muon_efficiency:.4f} ({detected_muons}/{total_muons})')
        print(f'  Pion rejection rate: {pion_rejection_rate:.4f} ({showering_pions}/{total_pions})')
        
    except Exception as e:
        print(f'Error analyzing {config_name}: {e}')
        continue

# Apply optimization criteria
print('\n' + '='*60)
print('OPTIMIZATION ANALYSIS')
print('='*60)

valid_configs = {}
for config_name, data in results.items():
    muon_eff = data['muon_efficiency']
    pion_rej = data['pion_rejection_rate']
    
    # Check minimum efficiency requirement
    meets_requirement = muon_eff >= min_muon_efficiency
    
    # Calculate weighted performance score
    performance_score = weight_efficiency * muon_eff + weight_rejection * pion_rej
    
    data['meets_requirement'] = meets_requirement
    data['performance_score'] = performance_score
    
    if meets_requirement:
        valid_configs[config_name] = data
        
    print(f'\n{config_name}:')
    print(f'  Muon efficiency: {muon_eff:.4f} (req: >= {min_muon_efficiency})')
    print(f'  Pion rejection: {pion_rej:.4f}')
    print(f'  Performance score: {performance_score:.4f}')
    print(f'  Meets requirements: {meets_requirement}')

# Identify optimal configuration
if valid_configs:
    optimal_config = max(valid_configs.keys(), key=lambda k: valid_configs[k]['performance_score'])
    optimal_data = valid_configs[optimal_config]
    
    print(f'\n' + '='*60)
    print('OPTIMAL CONFIGURATION IDENTIFIED')
    print('='*60)
    print(f'Configuration: {optimal_config}')
    print(f'Muon efficiency: {optimal_data["muon_efficiency"]:.4f}')
    print(f'Pion rejection rate: {optimal_data["pion_rejection_rate"]:.4f}')
    print(f'Performance score: {optimal_data["performance_score"]:.4f}')
    print(f'\nJustification:')
    print(f'- Meets minimum muon efficiency requirement (>= {min_muon_efficiency})')
    print(f'- Highest weighted performance score ({weight_efficiency:.1f}*eff + {weight_rejection:.1f}*rej)')
    print(f'- Optimal balance of muon detection and pion rejection')
else:
    print('\nWARNING: No configurations meet the minimum efficiency requirement!')
    # Find best available option
    if results:
        optimal_config = max(results.keys(), key=lambda k: results[k]['performance_score'])
        optimal_data = results[optimal_config]
        print(f'Best available option: {optimal_config}')
    else:
        optimal_config = 'none'
        optimal_data = {}

# Generate comparison plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

config_names = list(results.keys())
efficiencies = [results[c]['muon_efficiency'] for c in config_names]
rejections = [results[c]['pion_rejection_rate'] for c in config_names]
scores = [results[c]['performance_score'] for c in config_names]

# Efficiency comparison
colors = ['red' if eff < min_muon_efficiency else 'green' for eff in efficiencies]
ax1.bar(config_names, efficiencies, color=colors, alpha=0.7)
ax1.axhline(min_muon_efficiency, color='red', linestyle='--', label=f'Min requirement ({min_muon_efficiency})')
ax1.set_ylabel('Muon Efficiency')
ax1.set_title('Muon Detection Efficiency')
ax1.legend()
ax1.tick_params(axis='x', rotation=45)

# Rejection comparison
ax2.bar(config_names, rejections, color='blue', alpha=0.7)
ax2.set_ylabel('Pion Rejection Rate')
ax2.set_title('Pion Rejection Performance')
ax2.tick_params(axis='x', rotation=45)

# Performance score comparison
colors = ['gold' if c == optimal_config else 'lightblue' for c in config_names]
ax3.bar(config_names, scores, color=colors, alpha=0.7)
ax3.set_ylabel('Weighted Performance Score')
ax3.set_title('Overall Performance Score\n(0.6*eff + 0.4*rej)')
ax3.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('optimal_detector_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('optimal_detector_analysis.pdf', bbox_inches='tight')
print('\nSaved comparison plot: optimal_detector_analysis.png')

# Export detailed results
output_data = {
    'optimization_criteria': {
        'min_muon_efficiency': min_muon_efficiency,
        'weight_efficiency': weight_efficiency,
        'weight_rejection': weight_rejection,
        'energy_range_gev': energy_range
    },
    'configuration_results': results,
    'optimal_configuration': {
        'name': optimal_config,
        'data': optimal_data if optimal_data else {}
    },
    'valid_configurations': list(valid_configs.keys()) if valid_configs else []
}

with open('optimal_detector_selection.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print('\nSaved detailed analysis: optimal_detector_selection.json')

# Return values for workflow
if optimal_data:
    print(f'RESULT:optimal_configuration={optimal_config}')
    print(f'RESULT:optimal_muon_efficiency={optimal_data["muon_efficiency"]:.4f}')
    print(f'RESULT:optimal_pion_rejection={optimal_data["pion_rejection_rate"]:.4f}')
    print(f'RESULT:optimal_performance_score={optimal_data["performance_score"]:.4f}')
else:
    print('RESULT:optimal_configuration=none')
    print('RESULT:optimal_muon_efficiency=0.0')
    print('RESULT:optimal_pion_rejection=0.0')
    print('RESULT:optimal_performance_score=0.0')

print('RESULT:analysis_plot=optimal_detector_analysis.png')
print('RESULT:analysis_complete=True')
print('RESULT:summary_file=optimal_detector_selection.json')