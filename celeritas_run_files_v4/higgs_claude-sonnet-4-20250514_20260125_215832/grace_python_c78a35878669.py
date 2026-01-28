import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
import json
import matplotlib.pyplot as plt

# Load enhanced features data
features_df = pd.read_csv('enhanced_physics_features.csv')
print(f'Loaded dataset with shape: {features_df.shape}')

# Separate features, target, and weights
feature_cols = [col for col in features_df.columns if col not in ['Label', 'Weight', 'EventId']]
X = features_df[feature_cols].values
y = (features_df['Label'] == 's').astype(int)  # Convert s/b to 1/0
weights = features_df['Weight'].values

print(f'Features: {len(feature_cols)}, Signal events: {np.sum(y)}, Background events: {np.sum(1-y)}')

# Define AMS metric function
def ams_score(y_true, y_pred, sample_weight=None):
    if sample_weight is None:
        sample_weight = np.ones(len(y_true))
    
    # Calculate weighted signal and background
    s = np.sum(sample_weight[(y_true == 1) & (y_pred == 1)])
    b = np.sum(sample_weight[(y_true == 0) & (y_pred == 1)])
    
    # AMS formula with regularization
    br = 10.0  # Background regularization
    if b + br <= 0:
        return 0
    
    ams = np.sqrt(2 * ((s + b + br) * np.log(1 + s/(b + br)) - s))
    return ams

# Create AMS scorer for cross-validation
ams_scorer = make_scorer(ams_score, greater_is_better=True, needs_proba=False)

# Scale features for neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define models
models = {
    'gradient_boosting': GradientBoostingClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    ),
    'random_forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    ),
    'neural_network': MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        alpha=0.001,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Train and evaluate models
results = {}
best_score = -np.inf
best_model_name = None
best_model = None

print('Training and evaluating models...')
for name, model in models.items():
    print(f'Training {name}...')
    
    # Use scaled features for neural network, original for tree-based
    X_train = X_scaled if name == 'neural_network' else X
    
    # Cross-validation with sample weights
    cv_scores = []
    for train_idx, val_idx in cv.split(X_train, y):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        w_tr, w_val = weights[train_idx], weights[val_idx]
        
        # Train model with sample weights
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        
        # Predict and calculate AMS
        y_pred = model.predict(X_val)
        ams = ams_score(y_val, y_pred, w_val)
        cv_scores.append(ams)
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    
    results[name] = {
        'cv_scores': cv_scores,
        'mean_ams': mean_score,
        'std_ams': std_score
    }
    
    print(f'{name}: AMS = {mean_score:.4f} ± {std_score:.4f}')
    
    # Track best model
    if mean_score > best_score:
        best_score = mean_score
        best_model_name = name
        # Train final model on full dataset
        best_model = model
        best_model.fit(X_train, y, sample_weight=weights)

# Create comparison plot
fig, ax = plt.subplots(figsize=(10, 6))
model_names = list(results.keys())
mean_scores = [results[name]['mean_ams'] for name in model_names]
std_scores = [results[name]['std_ams'] for name in model_names]

bars = ax.bar(model_names, mean_scores, yerr=std_scores, capsize=5, 
              color=['blue', 'green', 'orange'], alpha=0.7)
ax.set_ylabel('AMS Score')
ax.set_title('Model Comparison - AMS Performance with Cross-Validation')
ax.grid(True, alpha=0.3)

# Highlight best model
for i, (name, bar) in enumerate(zip(model_names, bars)):
    if name == best_model_name:
        bar.set_color('red')
        bar.set_alpha(0.9)
        ax.text(i, mean_scores[i] + std_scores[i] + 0.01, 'BEST', 
                ha='center', va='bottom', fontweight='bold')

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_comparison_ams.png', dpi=150, bbox_inches='tight')
plt.savefig('model_comparison_ams.pdf', bbox_inches='tight')

# Feature importance for best model (if available)
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15 features
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), [feature_cols[i] for i in indices], rotation=45)
    plt.ylabel('Feature Importance')
    plt.title(f'Top 15 Feature Importances - {best_model_name}')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.savefig('feature_importance.pdf', bbox_inches='tight')

# Save results
results_summary = {
    'best_model': best_model_name,
    'best_ams_score': float(best_score),
    'best_ams_std': float(results[best_model_name]['std_ams']),
    'model_results': {name: {'mean_ams': float(res['mean_ams']), 
                            'std_ams': float(res['std_ams'])} 
                     for name, res in results.items()},
    'feature_count': len(feature_cols),
    'training_samples': len(X)
}

with open('classifier_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

# Save best model (simplified - just the name and parameters)
best_model_info = {
    'model_type': best_model_name,
    'parameters': best_model.get_params(),
    'feature_columns': feature_cols,
    'scaler_used': best_model_name == 'neural_network'
}

with open('best_model_info.json', 'w') as f:
    json.dump(best_model_info, f, indent=2)

print(f'\nBest model: {best_model_name}')
print(f'Best AMS score: {best_score:.4f} ± {results[best_model_name]["std_ams"]:.4f}')
print(f'Results saved to classifier_results.json')

# Return values for downstream steps
print(f'RESULT:best_model={best_model_name}')
print(f'RESULT:best_ams_score={best_score:.4f}')
print(f'RESULT:best_ams_std={results[best_model_name]["std_ams"]:.4f}')
print(f'RESULT:models_trained={len(models)}')
print(f'RESULT:cv_folds=5')
print(f'RESULT:training_samples={len(X)}')
print('RESULT:model_comparison_plot=model_comparison_ams.png')
print('RESULT:results_file=classifier_results.json')
print('RESULT:best_model_file=best_model_info.json')
if hasattr(best_model, 'feature_importances_'):
    print('RESULT:feature_importance_plot=feature_importance.png')