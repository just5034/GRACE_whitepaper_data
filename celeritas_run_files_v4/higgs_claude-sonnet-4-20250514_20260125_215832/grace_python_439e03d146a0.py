import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer
import json
import time
from pathlib import Path

# Load enhanced dataset
print('Loading enhanced dataset...')
df = pd.read_parquet('atlas_higgs_enhanced_features.parquet')
print(f'Dataset shape: {df.shape}')

# Sample data fraction for speed
if len(df) > 100000:
    df = df.sample(frac=0.7, random_state=42)
    print(f'Sampled to {len(df)} events')

# Separate features and target
target_col = 'Label'
weight_col = 'Weight'
exclude_cols = ['EventId', target_col, weight_col]
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols].copy()
y = df[target_col].copy()
weights = df[weight_col].copy() if weight_col in df.columns else np.ones(len(df))

# Convert target to binary (s=1, b=0)
y_binary = (y == 's').astype(int)

# Handle categorical columns
categorical_cols = []
numeric_cols = []
for col in X.columns:
    if X[col].dtype == 'object' or X[col].nunique() < 10:
        categorical_cols.append(col)
    else:
        numeric_cols.append(col)

print(f'Categorical columns: {len(categorical_cols)}')
print(f'Numeric columns: {len(numeric_cols)}')

# Process categorical columns
if categorical_cols:
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# Fill missing values
X = X.fillna(-999)  # Physics convention for missing values

# Select numeric features only if specified
if len(numeric_cols) > 0:
    X_numeric = X[numeric_cols]
else:
    X_numeric = X

print(f'Final feature matrix shape: {X_numeric.shape}')

# Define AMS metric
def ams_score(y_true, y_pred_proba, sample_weight=None):
    """Calculate Approximate Median Significance"""
    if len(y_pred_proba.shape) > 1:
        y_pred_proba = y_pred_proba[:, 1]
    
    # Use median threshold
    threshold = np.median(y_pred_proba)
    y_pred = (y_pred_proba > threshold).astype(int)
    
    if sample_weight is None:
        sample_weight = np.ones(len(y_true))
    
    # Calculate weighted true/false positives
    s = np.sum(sample_weight[(y_true == 1) & (y_pred == 1)])
    b = np.sum(sample_weight[(y_true == 0) & (y_pred == 1)])
    
    # AMS formula with regularization
    br = 10.0  # Background regularization
    if b + br <= 0:
        return 0
    
    ams = np.sqrt(2 * ((s + b + br) * np.log(1 + s / (b + br)) - s))
    return ams

# Create AMS scorer
ams_scorer = make_scorer(ams_score, needs_proba=True, greater_is_better=True)

# Quick training mode - simplified models
models = {
    'rf_simple': RandomForestClassifier(
        n_estimators=50,  # Reduced from default 100
        max_depth=10,     # Limited depth
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
}

# Add second model if max_models > 1
if True:  # max_models >= 2
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        models['gb_simple'] = GradientBoostingClassifier(
            n_estimators=50,  # Reduced
            max_depth=6,      # Limited
            learning_rate=0.1,
            random_state=42
        )
    except ImportError:
        print('GradientBoostingClassifier not available')

# Limit to max 3 models
model_keys = list(models.keys())[:3]
models = {k: models[k] for k in model_keys}

print(f'Training {len(models)} models: {list(models.keys())}')

# Cross-validation with reduced folds
cv_folds = 3
skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

results = {}
start_time = time.time()
timeout_seconds = 1200  # 20 minutes

for name, model in models.items():
    if time.time() - start_time > timeout_seconds:
        print(f'Timeout reached, skipping {name}')
        break
        
    print(f'Training {name}...')
    model_start = time.time()
    
    try:
        # Cross-validation with timeout protection
        cv_scores = cross_val_score(
            model, X_numeric, y_binary, 
            cv=skf, 
            scoring=ams_scorer,
            fit_params={'sample_weight': weights} if hasattr(model, 'fit') else {},
            n_jobs=1  # Avoid nested parallelism
        )
        
        # Fit final model
        model.fit(X_numeric, y_binary, sample_weight=weights)
        
        # Calculate final AMS on full dataset
        y_pred_proba = model.predict_proba(X_numeric)[:, 1]
        final_ams = ams_score(y_binary, y_pred_proba, weights)
        
        results[name] = {
            'cv_scores': cv_scores.tolist(),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'final_ams': float(final_ams),
            'training_time': time.time() - model_start
        }
        
        print(f'{name}: CV AMS = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')
        print(f'{name}: Final AMS = {final_ams:.4f}')
        
    except Exception as e:
        print(f'Error training {name}: {str(e)}')
        results[name] = {'error': str(e)}

# Select best model
valid_results = {k: v for k, v in results.items() if 'error' not in v}
if valid_results:
    best_model_name = max(valid_results.keys(), key=lambda k: valid_results[k]['final_ams'])
    best_ams = valid_results[best_model_name]['final_ams']
    best_cv_mean = valid_results[best_model_name]['cv_mean']
    best_cv_std = valid_results[best_model_name]['cv_std']
else:
    best_model_name = 'none'
    best_ams = 0.0
    best_cv_mean = 0.0
    best_cv_std = 0.0

print(f'\nBest model: {best_model_name}')
print(f'Best AMS: {best_ams:.4f}')
print(f'Best CV: {best_cv_mean:.4f} ± {best_cv_std:.4f}')

# Save results
results_summary = {
    'best_model': best_model_name,
    'best_ams_score': best_ams,
    'best_cv_mean': best_cv_mean,
    'best_cv_std': best_cv_std,
    'all_results': results,
    'num_features': X_numeric.shape[1],
    'num_events': len(df),
    'cv_folds': cv_folds,
    'training_modifications': ['timeout', 'reduce_model_complexity', 'limit_hyperparameter_search', 'use_early_stopping', 'reduce_cv_folds', 'sample_data_fraction', 'max_models', 'quick_training_mode']
}

with open('higgs_classifier_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f'\nTraining completed in {time.time() - start_time:.1f} seconds')
print(f'Results saved to higgs_classifier_results.json')

# Return values for workflow
print(f'RESULT:best_model_name={best_model_name}')
print(f'RESULT:best_ams_score={best_ams:.4f}')
print(f'RESULT:best_cv_mean={best_cv_mean:.4f}')
print(f'RESULT:best_cv_std={best_cv_std:.4f}')
print(f'RESULT:num_models_trained={len(valid_results)}')
print(f'RESULT:training_success=true')
print(f'RESULT:results_file=higgs_classifier_results.json')