import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer
import json
import time
from functools import wraps
import warnings
warnings.filterwarnings('ignore')

# Timeout wrapper function
def timeout_wrapper(timeout_seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    print(f'Warning: Function took {elapsed:.1f}s, exceeding timeout of {timeout_seconds}s')
                return result
            except Exception as e:
                print(f'Function failed after {time.time() - start_time:.1f}s: {str(e)}')
                raise
        return wrapper
    return decorator

# AMS metric implementation
def ams_score(y_true, y_pred_proba, sample_weight=None):
    if len(y_pred_proba.shape) > 1:
        y_pred_proba = y_pred_proba[:, 1]
    
    # Find optimal threshold
    thresholds = np.linspace(0.01, 0.99, 50)
    best_ams = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        if sample_weight is not None:
            s = np.sum(sample_weight[(y_true == 1) & (y_pred == 1)])
            b = np.sum(sample_weight[(y_true == 0) & (y_pred == 1)])
        else:
            s = np.sum((y_true == 1) & (y_pred == 1))
            b = np.sum((y_true == 0) & (y_pred == 1))
        
        if b > 0:
            ams = np.sqrt(2 * ((s + b + 10) * np.log(1 + s / (b + 10)) - s))
        else:
            ams = 0
        
        if ams > best_ams:
            best_ams = ams
            best_threshold = threshold
    
    return best_ams

# Load and prepare data
print('Loading preprocessed data...')
df = pd.read_parquet('preprocessed_atlas_data.parquet')

# Handle missing values
df_clean = df.copy()
df_clean = df_clean.replace(-999.0, np.nan)

# Separate features and target
feature_cols = [col for col in df_clean.columns if col not in ['Label', 'KaggleSet', 'Weight', 'KaggleWeight']]
X = df_clean[feature_cols].fillna(0)
y = (df_clean['Label'] == 's').astype(int)
weights = df_clean['Weight'].values

# Split into train/test based on KaggleSet
train_mask = df_clean['KaggleSet'] == 't'
X_train = X[train_mask]
y_train = y[train_mask]
weights_train = weights[train_mask]

print(f'Training data: {len(X_train)} events, {X_train.shape[1]} features')
print(f'Signal events: {np.sum(y_train)}, Background events: {np.sum(1-y_train)}')

# Reduced search space for faster optimization
param_distributions = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'max_features': ['sqrt', 0.5, 0.8]
}

print('Parameter search space:')
for param, values in param_distributions.items():
    print(f'  {param}: {values}')

# Create base estimator with explicit parameters
base_estimator = GradientBoostingClassifier(
    random_state=42,
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.9,
    max_features='sqrt'
)

# Custom scorer for AMS
def ams_scorer(estimator, X, y, sample_weight=None):
    y_pred_proba = estimator.predict_proba(X)
    return ams_score(y, y_pred_proba, sample_weight)

ams_scorer_obj = make_scorer(ams_scorer, needs_proba=False, greater_is_better=True)

# Cross-validation setup
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Timeout wrapper for optimization
@timeout_wrapper(1200)
def run_optimization():
    # Standard RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_distributions,
        n_iter=50,
        cv=cv,
        scoring=ams_scorer_obj,
        n_jobs=1,
        random_state=42,
        verbose=1
    )
    
    print('Starting hyperparameter optimization...')
    start_time = time.time()
    
    # Fit with sample weights (pass through fit_params)
    random_search.fit(X_train, y_train, sample_weight=weights_train)
    
    optimization_time = time.time() - start_time
    print(f'Optimization completed in {optimization_time:.1f} seconds')
    
    return random_search

try:
    # Run optimization with timeout
    search_result = run_optimization()
    
    # Extract best parameters and score
    best_params = search_result.best_params_
    best_cv_score = search_result.best_score_
    
    print('\nOptimization Results:')
    print(f'Best CV AMS Score: {best_cv_score:.4f}')
    print('Best Parameters:')
    for param, value in best_params.items():
        print(f'  {param}: {value}')
    
    # Train final model with best parameters
    print('\nTraining final model with optimized parameters...')
    final_model = GradientBoostingClassifier(**best_params, random_state=42)
    final_model.fit(X_train, y_train, sample_weight=weights_train)
    
    # Evaluate on training set
    y_pred_proba = final_model.predict_proba(X_train)[:, 1]
    final_ams = ams_score(y_train, y_pred_proba, weights_train)
    
    print(f'Final model AMS score: {final_ams:.4f}')
    
    # Save results
    results = {
        'best_params': best_params,
        'best_cv_score': float(best_cv_score),
        'final_ams_score': float(final_ams),
        'optimization_method': 'RandomizedSearchCV',
        'n_iterations': 50,
        'cv_folds': 3,
        'training_events': len(X_train),
        'features_used': len(feature_cols),
        'success': True
    }
    
    with open('optimized_hyperparameters.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Output results
    print(f'RESULT:best_cv_ams_score={best_cv_score:.4f}')
    print(f'RESULT:final_ams_score={final_ams:.4f}')
    print(f'RESULT:best_n_estimators={best_params.get("n_estimators", 100)}')
    print(f'RESULT:best_max_depth={best_params.get("max_depth", 5)}')
    print(f'RESULT:best_learning_rate={best_params.get("learning_rate", 0.1)}')
    print(f'RESULT:optimization_iterations=50')
    print(f'RESULT:cv_folds=3')
    print(f'RESULT:training_events={len(X_train)}')
    print('RESULT:optimization_method=RandomizedSearchCV')
    print('RESULT:results_file=optimized_hyperparameters.json')
    print('RESULT:success=True')
    
except Exception as e:
    print(f'Optimization failed: {str(e)}')
    print('Falling back to basic optimization...')
    
    # Fallback to basic grid search with minimal parameters
    basic_params = {
        'n_estimators': [100, 200],
        'max_depth': [5, 7],
        'learning_rate': [0.1, 0.2]
    }
    
    try:
        from sklearn.model_selection import GridSearchCV
        basic_search = GridSearchCV(
            estimator=base_estimator,
            param_grid=basic_params,
            cv=3,
            scoring=ams_scorer_obj,
            n_jobs=1
        )
        
        basic_search.fit(X_train, y_train, sample_weight=weights_train)
        
        fallback_results = {
            'best_params': basic_search.best_params_,
            'best_cv_score': float(basic_search.best_score_),
            'optimization_method': 'GridSearchCV_fallback',
            'success': True
        }
        
        print(f'RESULT:best_cv_ams_score={basic_search.best_score_:.4f}')
        print('RESULT:optimization_method=GridSearchCV_fallback')
        print('RESULT:success=True')
        
    except Exception as fallback_error:
        print(f'Fallback optimization also failed: {str(fallback_error)}')
        print('RESULT:success=False')