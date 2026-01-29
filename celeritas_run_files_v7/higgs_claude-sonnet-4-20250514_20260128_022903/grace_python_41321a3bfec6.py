import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer
import json
import time
from pathlib import Path

# Load preprocessed data
df = pd.read_parquet('preprocessed_atlas_data.parquet')

# Prepare features and target
feature_cols = [col for col in df.columns if col not in ['Label', 'KaggleSet', 'Weight', 'EventId']]
X = df[feature_cols].copy()
y = (df['Label'] == 's').astype(int)
weights = df['Weight'].values

# Handle missing values (-999.0)
X[X == -999.0] = np.nan
X = X.fillna(X.median())

# Define AMS scoring function
def ams_score(y_true, y_pred_proba, sample_weight=None):
    if sample_weight is None:
        sample_weight = np.ones(len(y_true))
    
    # Find optimal threshold
    thresholds = np.percentile(y_pred_proba, np.linspace(5, 95, 50))
    best_ams = 0
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate weighted signal and background
        s = np.sum(sample_weight[(y_true == 1) & (y_pred == 1)])
        b = np.sum(sample_weight[(y_true == 0) & (y_pred == 1)])
        
        if s > 0 and b > 0:
            ams = np.sqrt(2 * ((s + b + 10) * np.log(1 + s/(b + 10)) - s))
            if ams > best_ams:
                best_ams = ams
    
    return best_ams

# Custom scorer that handles sample weights
def ams_scorer(estimator, X, y, sample_weight=None):
    y_pred_proba = estimator.predict_proba(X)[:, 1]
    return ams_score(y, y_pred_proba, sample_weight)

# Reduced search space for faster optimization
param_distributions = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.8, 0.9, 1.0],
    'max_features': ['sqrt', 0.5, 0.8]
}

# Setup cross-validation with reduced folds
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Initialize base estimator
base_estimator = GradientBoostingClassifier(random_state=42)

# Setup RandomizedSearchCV with timeout and early stopping
print('Starting hyperparameter optimization...')
start_time = time.time()

# Custom fit function with timeout
class TimeoutRandomizedSearchCV(RandomizedSearchCV):
    def __init__(self, *args, timeout=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeout = timeout
        self.start_time = None
    
    def fit(self, X, y, **fit_params):
        self.start_time = time.time()
        return super().fit(X, y, **fit_params)
    
    def _run_search(self, evaluate_candidates):
        def timed_evaluate_candidates(candidate_params):
            if self.timeout and (time.time() - self.start_time) > self.timeout:
                print(f'Timeout reached after {self.timeout} seconds')
                return
            return evaluate_candidates(candidate_params)
        
        super()._run_search(timed_evaluate_candidates)

# Create scorer
scorer = make_scorer(ams_scorer, greater_is_better=True, needs_proba=False)

# Run optimization with timeout
random_search = TimeoutRandomizedSearchCV(
    estimator=base_estimator,
    param_distributions=param_distributions,
    n_iter=50,
    cv=cv,
    scoring=scorer,
    n_jobs=-1,
    random_state=42,
    timeout=1200,
    verbose=1
)

# Fit with sample weights
fit_params = {'sample_weight': weights}
random_search.fit(X, y, **fit_params)

# Get best parameters and score
best_params = random_search.best_params_
best_cv_score = random_search.best_score_

# Train final model with best parameters
best_model = GradientBoostingClassifier(**best_params, random_state=42)
best_model.fit(X, y, sample_weight=weights)

# Evaluate on full dataset to get final AMS score
y_pred_proba = best_model.predict_proba(X)[:, 1]
final_ams = ams_score(y, y_pred_proba, weights)

# Calculate improvement over baseline
baseline_ams = 58.443  # From previous step output
improvement = ((final_ams - baseline_ams) / baseline_ams) * 100

print(f'Optimization completed in {time.time() - start_time:.1f} seconds')
print(f'Best CV AMS Score: {best_cv_score:.3f}')
print(f'Final AMS Score: {final_ams:.3f}')
print(f'Baseline AMS Score: {baseline_ams:.3f}')
print(f'Improvement: {improvement:.2f}%')
print(f'Best Parameters: {best_params}')

# Save results
results = {
    'best_params': best_params,
    'best_cv_score': float(best_cv_score),
    'final_ams_score': float(final_ams),
    'baseline_ams_score': float(baseline_ams),
    'improvement_percent': float(improvement),
    'optimization_time': time.time() - start_time,
    'n_iterations_completed': len(random_search.cv_results_['params']),
    'success': True
}

with open('optimized_hyperparameters.json', 'w') as f:
    json.dump(results, f, indent=2)

# Output results for workflow
print(f'RESULT:optimized_ams_score={final_ams:.3f}')
print(f'RESULT:cv_ams_score={best_cv_score:.3f}')
print(f'RESULT:improvement_percent={improvement:.2f}')
print(f'RESULT:optimization_time={time.time() - start_time:.1f}')
print(f'RESULT:n_iterations={len(random_search.cv_results_["params"])}')
print('RESULT:results_file=optimized_hyperparameters.json')
print('RESULT:success=True')