import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Load enhanced dataset from previous step
df = pd.read_parquet('atlas_higgs_enhanced_features.parquet')
print(f'Loaded dataset with shape: {df.shape}')

# Prepare features and target
feature_cols = [col for col in df.columns if col not in ['Label', 'Weight', 'EventId']]
X = df[feature_cols].copy()
y = (df['Label'] == 's').astype(int)  # Convert s/b to 1/0
weights = df['Weight'].values

print(f'Features: {len(feature_cols)}')
print(f'Signal events: {y.sum()}, Background events: {(y==0).sum()}')
print(f'Weight range: {weights.min():.6f} to {weights.max():.6f}')

# Handle missing values (-999)
X = X.replace(-999.0, np.nan)
X = X.fillna(X.median())

# AMS metric calculation
def calculate_ams(y_true, y_pred, weights, threshold=0.5):
    """Calculate Approximate Median Significance (AMS) score"""
    predictions = (y_pred >= threshold).astype(int)
    
    # True positives (signal correctly identified)
    tp_mask = (y_true == 1) & (predictions == 1)
    s = weights[tp_mask].sum() if tp_mask.any() else 0
    
    # False positives (background misidentified as signal)
    fp_mask = (y_true == 0) & (predictions == 1)
    b = weights[fp_mask].sum() if fp_mask.any() else 0
    
    # AMS formula with regularization
    b_reg = 10.0  # Regularization term
    if s <= 0:
        return 0.0
    if b + b_reg <= 0:
        return 0.0
    
    ams = np.sqrt(2 * ((s + b + b_reg) * np.log(1 + s/(b + b_reg)) - s))
    return ams

# Optimize threshold for AMS
def optimize_ams_threshold(y_true, y_pred_proba, weights):
    """Find optimal threshold for AMS score"""
    thresholds = np.linspace(0.1, 0.9, 50)
    best_ams = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        ams = calculate_ams(y_true, y_pred_proba, weights, threshold)
        if ams > best_ams:
            best_ams = ams
            best_threshold = threshold
    
    return best_threshold, best_ams

# Scale features for neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define models with AMS-friendly configurations
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
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    ),
    'neural_network': MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        learning_rate_init=0.001,
        random_state=42
    )
}

# Cross-validation with AMS optimization
print('\nPerforming 5-fold cross-validation...')
cv_results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models.items():
    print(f'\nTraining {model_name}...')
    
    fold_ams_scores = []
    fold_auc_scores = []
    fold_thresholds = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        w_train, w_val = weights[train_idx], weights[val_idx]
        
        # Use scaled features for neural network
        if model_name == 'neural_network':
            X_train_model = X_scaled[train_idx]
            X_val_model = X_scaled[val_idx]
        else:
            X_train_model = X_train
            X_val_model = X_val
        
        # Train model with sample weights
        if model_name == 'neural_network':
            # MLPClassifier doesn't support sample_weight directly
            model.fit(X_train_model, y_train)
        else:
            model.fit(X_train_model, y_train, sample_weight=w_train)
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_val_model)[:, 1]
        
        # Optimize threshold for AMS
        best_threshold, best_ams = optimize_ams_threshold(y_val.values, y_pred_proba, w_val)
        
        # Calculate AUC for comparison
        auc_score = roc_auc_score(y_val, y_pred_proba, sample_weight=w_val)
        
        fold_ams_scores.append(best_ams)
        fold_auc_scores.append(auc_score)
        fold_thresholds.append(best_threshold)
        
        print(f'  Fold {fold+1}: AMS={best_ams:.4f}, AUC={auc_score:.4f}, Threshold={best_threshold:.3f}')
    
    # Store results
    cv_results[model_name] = {
        'ams_scores': fold_ams_scores,
        'auc_scores': fold_auc_scores,
        'thresholds': fold_thresholds,
        'mean_ams': np.mean(fold_ams_scores),
        'std_ams': np.std(fold_ams_scores),
        'mean_auc': np.mean(fold_auc_scores),
        'std_auc': np.std(fold_auc_scores),
        'mean_threshold': np.mean(fold_thresholds)
    }
    
    print(f'  Mean AMS: {np.mean(fold_ams_scores):.4f} ± {np.std(fold_ams_scores):.4f}')
    print(f'  Mean AUC: {np.mean(fold_auc_scores):.4f} ± {np.std(fold_auc_scores):.4f}')

# Select best model based on AMS score
best_model_name = max(cv_results.keys(), key=lambda k: cv_results[k]['mean_ams'])
best_model = models[best_model_name]
best_ams = cv_results[best_model_name]['mean_ams']
best_ams_std = cv_results[best_model_name]['std_ams']

print(f'\nBest model: {best_model_name}')
print(f'Best AMS score: {best_ams:.4f} ± {best_ams_std:.4f}')

# Train final model on full dataset
print(f'\nTraining final {best_model_name} model on full dataset...')
if best_model_name == 'neural_network':
    best_model.fit(X_scaled, y)
    final_predictions = best_model.predict_proba(X_scaled)[:, 1]
else:
    best_model.fit(X, y, sample_weight=weights)
    final_predictions = best_model.predict_proba(X)[:, 1]

# Optimize final threshold
final_threshold, final_ams = optimize_ams_threshold(y.values, final_predictions, weights)

print(f'Final model AMS: {final_ams:.4f}')
print(f'Optimal threshold: {final_threshold:.3f}')

# Save results
results = {
    'best_model': best_model_name,
    'cv_results': cv_results,
    'final_ams_score': final_ams,
    'optimal_threshold': final_threshold,
    'num_features': len(feature_cols),
    'training_samples': len(y)
}

with open('higgs_classifier_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f'\nRESULT:best_model={best_model_name}')
print(f'RESULT:best_ams_score={best_ams:.4f}')
print(f'RESULT:ams_std_error={best_ams_std:.4f}')
print(f'RESULT:optimal_threshold={final_threshold:.3f}')
print(f'RESULT:final_ams_score={final_ams:.4f}')
print(f'RESULT:cv_folds=5')
print(f'RESULT:num_features={len(feature_cols)}')
print('RESULT:results_file=higgs_classifier_results.json')
print('\nHiggs classifier training completed successfully!')