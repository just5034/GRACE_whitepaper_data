import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import json

# Load enhanced features and create target/weights
print('Loading enhanced physics features...')
features_df = pd.read_csv('enhanced_physics_features.csv')

# Create binary target (signal=1, background=0)
y = (features_df['Label'] == 's').astype(int)
weights = features_df['Weight'].values

# Remove non-feature columns
feature_cols = [col for col in features_df.columns if col not in ['EventId', 'Label', 'Weight']]
X = features_df[feature_cols].values

print(f'Training data: {X.shape[0]} events, {X.shape[1]} features')
print(f'Signal events: {y.sum()}, Background events: {(y==0).sum()}')

# AMS metric calculation
def ams_score(y_true, y_pred_proba, weights, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    s = np.sum(weights[(y_true == 1) & (y_pred == 1)])
    b = np.sum(weights[(y_true == 0) & (y_pred == 1)])
    br = 10.0  # Background regularization
    return np.sqrt(2 * ((s + b + br) * np.log(1 + s/(b + br)) - s)) if b > 0 else 0

# Optimize threshold for AMS
def optimize_ams_threshold(y_true, y_pred_proba, weights):
    thresholds = np.linspace(0.1, 0.9, 81)
    best_ams = 0
    best_thresh = 0.5
    for thresh in thresholds:
        ams = ams_score(y_true, y_pred_proba, weights, thresh)
        if ams > best_ams:
            best_ams = ams
            best_thresh = thresh
    return best_thresh, best_ams

# Define classifiers
classifiers = {
    'gradient_boosting': GradientBoostingClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
    'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
}

# Cross-validation with AMS optimization
print('\nPerforming cross-validation with AMS optimization...')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, clf in classifiers.items():
    print(f'\nTraining {name}...')
    cv_ams_scores = []
    cv_auc_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train, w_val = weights[train_idx], weights[val_idx]
        
        # Scale features for neural network
        if name == 'neural_network':
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
        
        # Train with sample weights
        if name == 'neural_network':
            clf.fit(X_train, y_train)  # MLPClassifier doesn't support sample_weight
        else:
            clf.fit(X_train, y_train, sample_weight=w_train)
        
        # Predict probabilities
        y_pred_proba = clf.predict_proba(X_val)[:, 1]
        
        # Optimize AMS threshold
        best_thresh, best_ams = optimize_ams_threshold(y_val, y_pred_proba, w_val)
        cv_ams_scores.append(best_ams)
        
        # Calculate AUC for comparison
        auc = roc_auc_score(y_val, y_pred_proba, sample_weight=w_val)
        cv_auc_scores.append(auc)
        
        print(f'  Fold {fold+1}: AMS={best_ams:.4f}, AUC={auc:.4f}, Threshold={best_thresh:.3f}')
    
    # Store results
    mean_ams = np.mean(cv_ams_scores)
    std_ams = np.std(cv_ams_scores)
    mean_auc = np.mean(cv_auc_scores)
    
    results[name] = {
        'mean_ams': mean_ams,
        'std_ams': std_ams,
        'mean_auc': mean_auc,
        'cv_ams_scores': cv_ams_scores,
        'cv_auc_scores': cv_auc_scores
    }
    
    print(f'  {name} CV Results: AMS={mean_ams:.4f}±{std_ams:.4f}, AUC={mean_auc:.4f}')

# Select best model based on AMS score
best_model_name = max(results.keys(), key=lambda x: results[x]['mean_ams'])
best_ams = results[best_model_name]['mean_ams']
best_ams_std = results[best_model_name]['std_ams']

print(f'\nBest model: {best_model_name}')
print(f'Best AMS score: {best_ams:.4f} ± {best_ams_std:.4f}')

# Train final model on full dataset
print(f'\nTraining final {best_model_name} on full dataset...')
final_clf = classifiers[best_model_name]

if best_model_name == 'neural_network':
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    final_clf.fit(X_scaled, y)
else:
    final_clf.fit(X, y, sample_weight=weights)

# Get final predictions and optimize threshold
if best_model_name == 'neural_network':
    final_proba = final_clf.predict_proba(X_scaled)[:, 1]
else:
    final_proba = final_clf.predict_proba(X)[:, 1]

final_threshold, final_ams = optimize_ams_threshold(y, final_proba, weights)

print(f'Final model AMS score: {final_ams:.4f}')
print(f'Optimal threshold: {final_threshold:.3f}')

# Save results
results_summary = {
    'best_model': best_model_name,
    'best_ams_score': float(best_ams),
    'best_ams_std': float(best_ams_std),
    'final_ams_score': float(final_ams),
    'optimal_threshold': float(final_threshold),
    'cv_results': {name: {'mean_ams': float(res['mean_ams']), 'std_ams': float(res['std_ams']), 'mean_auc': float(res['mean_auc'])} for name, res in results.items()},
    'num_features': X.shape[1],
    'num_events': X.shape[0]
}

with open('higgs_classifier_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print('\nClassifier training completed successfully!')
print('RESULT:best_model=' + best_model_name)
print(f'RESULT:best_ams_score={best_ams:.4f}')
print(f'RESULT:ams_uncertainty={best_ams_std:.4f}')
print(f'RESULT:final_ams_score={final_ams:.4f}')
print(f'RESULT:optimal_threshold={final_threshold:.3f}')
print(f'RESULT:num_trained_features={X.shape[1]}')
print('RESULT:classifier_status=success')