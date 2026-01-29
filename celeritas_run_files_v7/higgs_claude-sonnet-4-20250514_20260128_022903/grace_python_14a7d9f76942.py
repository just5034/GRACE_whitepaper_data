import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
import json
from pathlib import Path
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

# Load preprocessed data
df = pd.read_parquet('preprocessed_atlas_data.parquet')

# Prepare features and target
feature_cols = [col for col in df.columns if col not in ['Label', 'KaggleSet', 'KaggleWeight']]
X = df[feature_cols].values
y = (df['Label'] == 's').astype(int)
weights = df['KaggleWeight'].values

# AMS metric implementation
def ams_score(y_true, y_pred_proba, weights):
    """Calculate AMS score with optimal threshold"""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_ams = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate weighted signal and background
        s = np.sum(weights[(y_true == 1) & (y_pred == 1)])
        b = np.sum(weights[(y_true == 0) & (y_pred == 1)])
        
        # AMS formula with regularization
        if b > 0:
            ams = np.sqrt(2 * ((s + b + 10) * np.log(1 + s/(b + 10)) - s))
        else:
            ams = 0
        
        if ams > best_ams:
            best_ams = ams
            best_threshold = threshold
    
    return best_ams, best_threshold

# Cross-validation function
def cv_ams_score(params, X, y, weights, cv_folds=5):
    """Cross-validation with AMS scoring"""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    ams_scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train, w_val = weights[train_idx], weights[val_idx]
        
        # Train model with current parameters
        model = GradientBoostingClassifier(
            n_estimators=int(params['n_estimators']),
            learning_rate=params['learning_rate'],
            max_depth=int(params['max_depth']),
            min_samples_split=int(params['min_samples_split']),
            min_samples_leaf=int(params['min_samples_leaf']),
            subsample=params['subsample'],
            random_state=42
        )
        
        model.fit(X_train, y_train, sample_weight=w_train)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        fold_ams, _ = ams_score(y_val, y_pred_proba, w_val)
        ams_scores.append(fold_ams)
    
    return np.mean(ams_scores)

# Baseline performance from previous step
baseline_ams = 58.443
print(f'Baseline AMS score: {baseline_ams:.3f}')

# Define search space
if SKOPT_AVAILABLE:
    search_space = [
        Integer(50, 500, name='n_estimators'),
        Real(0.01, 0.3, name='learning_rate'),
        Integer(3, 10, name='max_depth'),
        Integer(2, 20, name='min_samples_split'),
        Integer(1, 10, name='min_samples_leaf'),
        Real(0.6, 1.0, name='subsample')
    ]
    
    @use_named_args(search_space)
    def objective(**params):
        score = cv_ams_score(params, X, y, weights, cv_folds=5)
        return -score  # Minimize negative AMS
    
    print('Starting Bayesian optimization...')
    result = gp_minimize(objective, search_space, n_calls=30, random_state=42)
    
    best_params = {
        'n_estimators': result.x[0],
        'learning_rate': result.x[1],
        'max_depth': result.x[2],
        'min_samples_split': result.x[3],
        'min_samples_leaf': result.x[4],
        'subsample': result.x[5]
    }
    optimized_ams = -result.fun
    
else:
    # Fallback grid search if skopt not available
    print('Scikit-optimize not available, using grid search...')
    param_grid = [
        {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'min_samples_split': 5, 'min_samples_leaf': 2, 'subsample': 0.8},
        {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 8, 'min_samples_split': 10, 'min_samples_leaf': 1, 'subsample': 0.9},
        {'n_estimators': 150, 'learning_rate': 0.15, 'max_depth': 5, 'min_samples_split': 8, 'min_samples_leaf': 3, 'subsample': 0.85}
    ]
    
    best_score = 0
    best_params = param_grid[0]
    
    for params in param_grid:
        score = cv_ams_score(params, X, y, weights, cv_folds=5)
        if score > best_score:
            best_score = score
            best_params = params
    
    optimized_ams = best_score

# Calculate improvement
ams_improvement = optimized_ams - baseline_ams
relative_improvement = (ams_improvement / baseline_ams) * 100

print(f'Optimized AMS score: {optimized_ams:.3f}')
print(f'AMS improvement: {ams_improvement:.3f} ({relative_improvement:.1f}%)')
print(f'Best parameters: {best_params}')

# Save results
results = {
    'baseline_ams': baseline_ams,
    'optimized_ams': optimized_ams,
    'ams_improvement': ams_improvement,
    'relative_improvement_percent': relative_improvement,
    'best_hyperparameters': best_params,
    'cv_folds': 5,
    'optimization_strategy': 'bayesian' if SKOPT_AVAILABLE else 'grid_search',
    'search_space': 'comprehensive'
}

with open('optimized_hyperparameters.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'RESULT:optimized_ams_score={optimized_ams:.3f}')
print(f'RESULT:ams_improvement={ams_improvement:.3f}')
print(f'RESULT:relative_improvement_percent={relative_improvement:.1f}')
print(f'RESULT:best_n_estimators={best_params["n_estimators"]}')
print(f'RESULT:best_learning_rate={best_params["learning_rate"]:.3f}')
print(f'RESULT:best_max_depth={best_params["max_depth"]}')
print('RESULT:optimization_complete=True')
print('RESULT:results_file=optimized_hyperparameters.json')
print('RESULT:success=True')