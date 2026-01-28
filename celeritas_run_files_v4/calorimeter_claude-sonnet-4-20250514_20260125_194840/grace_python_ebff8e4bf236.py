import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import json

# Define topology options and physics considerations from input
topology_options = ['box', 'cylinder_barrel', 'projective_tower']
physics_considerations = ['hermeticity', 'crack_effects', 'jet_energy_measurement', 'shower_containment']

# Physics analysis of each topology
print('=== CALORIMETER TOPOLOGY PHYSICS ANALYSIS ===')
print()

# Define physics performance matrix (0-5 scale, 5=best)
physics_matrix = {
    'box': {
        'hermeticity': 2,  # Poor - gaps at edges, limited solid angle
        'crack_effects': 1,  # Worst - planar gaps between modules
        'jet_energy_measurement': 2,  # Poor - shower spreading across boundaries
        'shower_containment': 4,  # Good - uniform depth
        'description': 'Planar layers - simple but limited coverage'
    },
    'cylinder_barrel': {
        'hermeticity': 4,  # Good - full azimuthal coverage
        'crack_effects': 3,  # Moderate - gaps at barrel ends only
        'jet_energy_measurement': 3,  # Moderate - uniform phi response
        'shower_containment': 4,  # Good - uniform material thickness
        'description': 'Cylindrical layers - excellent for barrel region'
    },
    'projective_tower': {
        'hermeticity': 5,  # Excellent - pointing geometry minimizes gaps
        'crack_effects': 5,  # Excellent - towers point to interaction vertex
        'jet_energy_measurement': 5,  # Excellent - optimal for jet reconstruction
        'shower_containment': 3,  # Moderate - varying path lengths
        'description': 'Towers pointing to IP - optimal for collider physics'
    }
}

# Detailed physics analysis for each topology
for topology in topology_options:
    print(f'--- {topology.upper()} TOPOLOGY ---')
    print(f'Description: {physics_matrix[topology]["description"]}')
    print('Physics Performance:')
    
    total_score = 0
    for consideration in physics_considerations:
        score = physics_matrix[topology][consideration]
        total_score += score
        print(f'  {consideration}: {score}/5')
    
    avg_score = total_score / len(physics_considerations)
    print(f'  Overall Score: {avg_score:.2f}/5')
    print()

# Physics rationale for each consideration
print('=== DETAILED PHYSICS RATIONALE ===')
print()

rationale = {
    'hermeticity': {
        'definition': 'Solid angle coverage around interaction point',
        'box': 'Limited to forward direction, large uncovered regions',
        'cylinder_barrel': 'Full azimuthal coverage in barrel region (|eta| < 1.5)',
        'projective_tower': 'Excellent coverage with minimal gaps due to pointing geometry'
    },
    'crack_effects': {
        'definition': 'Energy loss in gaps between detector modules',
        'box': 'Severe - planar boundaries create large dead regions',
        'cylinder_barrel': 'Moderate - only gaps at barrel-endcap boundaries',
        'projective_tower': 'Minimal - towers point toward vertex, reducing shower leakage'
    },
    'jet_energy_measurement': {
        'definition': 'Accuracy of measuring jet energy and direction',
        'box': 'Poor - jets spread across module boundaries, energy loss',
        'cylinder_barrel': 'Good - uniform response in phi, but eta boundaries problematic',
        'projective_tower': 'Excellent - jet axis aligned with tower pointing, minimal spreading'
    },
    'shower_containment': {
        'definition': 'Fraction of shower energy captured within detector',
        'box': 'Good - uniform material thickness, predictable containment',
        'cylinder_barrel': 'Good - consistent radial depth for all particles',
        'projective_tower': 'Variable - path length depends on eta, requires careful design'
    }
}

for consideration in physics_considerations:
    print(f'--- {consideration.upper()} ---')
    print(f'Definition: {rationale[consideration]["definition"]}')
    for topology in topology_options:
        print(f'{topology}: {rationale[consideration][topology]}')
    print()

# Create comparison visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Performance radar chart
angles = np.linspace(0, 2*np.pi, len(physics_considerations), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))  # Close the plot

colors = ['blue', 'green', 'red']
for i, topology in enumerate(topology_options):
    values = [physics_matrix[topology][consideration] for consideration in physics_considerations]
    values += [values[0]]  # Close the plot
    ax1.plot(angles, values, 'o-', linewidth=2, label=topology, color=colors[i])
    ax1.fill(angles, values, alpha=0.1, color=colors[i])

ax1.set_xticks(angles[:-1])
ax1.set_xticklabels([c.replace('_', '\n') for c in physics_considerations])
ax1.set_ylim(0, 5)
ax1.set_title('Physics Performance Comparison')
ax1.legend()
ax1.grid(True)

# Overall scores bar chart
topologies = [t.replace('_', '\n') for t in topology_options]
scores = [sum(physics_matrix[topology][consideration] for consideration in physics_considerations) / len(physics_considerations) for topology in topology_options]

ax2.bar(topologies, scores, color=colors, alpha=0.7)
ax2.set_ylabel('Average Score')
ax2.set_title('Overall Physics Performance')
ax2.set_ylim(0, 5)
for i, score in enumerate(scores):
    ax2.text(i, score + 0.1, f'{score:.2f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('topology_physics_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('topology_physics_comparison.pdf', bbox_inches='tight')
print('RESULT:physics_comparison_plot=topology_physics_comparison.png')

# Generate selection recommendation
best_topology = max(topology_options, key=lambda t: sum(physics_matrix[t][c] for c in physics_considerations))
best_score = sum(physics_matrix[best_topology][c] for c in physics_considerations) / len(physics_considerations)

print('=== TOPOLOGY SELECTION RECOMMENDATION ===')
print(f'Recommended topology: {best_topology}')
print(f'Overall score: {best_score:.2f}/5')
print(f'Primary advantages: {physics_matrix[best_topology]["description"]}')
print()

# Trade-off analysis
print('=== TRADE-OFF ANALYSIS ===')
for topology in topology_options:
    strengths = [c for c in physics_considerations if physics_matrix[topology][c] >= 4]
    weaknesses = [c for c in physics_considerations if physics_matrix[topology][c] <= 2]
    print(f'{topology}:')
    print(f'  Strengths: {strengths if strengths else "None"}')
    print(f'  Weaknesses: {weaknesses if weaknesses else "None"}')
print()

# Save results to JSON for downstream use
results = {
    'topology_analysis': physics_matrix,
    'recommended_topology': best_topology,
    'physics_rationale': rationale,
    'selection_criteria': {
        'primary': 'jet_energy_measurement',
        'secondary': ['hermeticity', 'crack_effects'],
        'rationale': 'Hadronic shower measurement and jet reconstruction require minimal energy loss and accurate direction measurement'
    }
}

with open('topology_selection_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'RESULT:recommended_topology={best_topology}')
print(f'RESULT:topology_score={best_score:.2f}')
print('RESULT:analysis_file=topology_selection_analysis.json')
print('RESULT:physics_rationale_established=True')