import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer
import json
import warnings
warnings.filterwarnings('ignore')

# Load preprocessed data
df = pd.read_parquet('preprocessed_atlas_data.parquet')
print(f'Loaded {len(df)} events with {df.shape[1]} features')

# Prepare features and target
feature_cols = [col for col in df.columns if col not in ['Label', 'KaggleSet', 'Weight', 'EventId']]
X = df[feature_cols]
y = (df['Label'] == 's').astype(int)  # Convert to binary
weights = df['Weight']

print(f'Using {len(feature_cols)} features for optimization')
print(f'Signal events: {y.sum()}, Background events: {(y==0).sum()}')

# Define AMS scoring function
def ams_score(y_true, y_pred_proba, sample_weight=None):
    if sample_weight is None:
        sample_weight = np.ones(len(y_true))
    
    # Find optimal threshold
    thresholds = np.percentile(y_pred_proba, np.linspace(5, 95, 50))
    best_ams = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate weighted signal and background
        s = np.sum(sample_weight[y_true == 1] * y_pred[y_true == 1])
        b = np.sum(sample_weight[y_true == 0] * y_pred[y_true == 0])
        
        if b > 0:
            ams = np.sqrt(2 * ((s + b + 10) * np.log(1 + s / (b + 10)) - s))
            if ams > best_ams:
                best_ams = ams
                best_threshold = threshold
    
    return best_ams

# Custom scorer for RandomizedSearchCV
def ams_scorer(estimator, X, y, sample_weight=None):
    y_pred_proba = estimator.predict_proba(X)[:, 1]
    return ams_score(y, y_pred_proba, sample_weight)

ams_scoring = make_scorer(ams_scorer, needs_proba=False, greater_is_better=True)

# Define comprehensive search space
param_distributions = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

print('Starting randomized hyperparameter search...')
print(f'Search space size: {np.prod([len(v) for v in param_distributions.values()])} combinations')

# Setup cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize RandomForest and RandomizedSearchCV
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# Use RandomizedSearchCV with 50 iterations
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=50,
    cv=cv,
    scoring=ams_scoring,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

# Fit with sample weights
print('Fitting RandomizedSearchCV...')
random_search.fit(X, y, sample_weight=weights)

# Get best parameters and score
best_params = random_search.best_params_
best_cv_score = random_search.best_score_

print(f'Best cross-validation AMS score: {best_cv_score:.3f}')
print('Best parameters:')
for param, value in best_params.items():
    print(f'  {param}: {value}')

# Train final model with best parameters
print('Training final optimized model...')
best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
best_model.fit(X, y, sample_weight=weights)

# Evaluate on full dataset to get final AMS
y_pred_proba = best_model.predict_proba(X)[:, 1]
final_ams = ams_score(y, y_pred_proba, weights)

# Find optimal threshold
thresholds = np.percentile(y_pred_proba, np.linspace(5, 95, 100))
best_ams = 0
best_threshold = 0.5

for threshold in thresholds:
    y_pred = (y_pred_proba >= threshold).astype(int)
    s = np.sum(weights[y == 1] * y_pred[y == 1])
    b = np.sum(weights[y == 0] * y_pred[y == 0])
    
    if b > 0:
        ams = np.sqrt(2 * ((s + b + 10) * np.log(1 + s / (b + 10)) - s))
        if ams > best_ams:
            best_ams = ams
            best_threshold = threshold

# Calculate improvement over baseline
baseline_ams = 58.443  # From previous step
improvement = ((best_ams - baseline_ams) / baseline_ams) * 100

print(f'\nOptimization Results:')
print(f'Baseline AMS score: {baseline_ams:.3f}')
print(f'Optimized AMS score: {best_ams:.3f}')
print(f'Improvement: {improvement:.1f}%')
print(f'Optimal threshold: {best_threshold:.3f}')

# Save results
results = {
    'best_parameters': best_params,
    'best_cv_score': float(best_cv_score),
    'final_ams_score': float(best_ams),
    'optimal_threshold': float(best_threshold),
    'baseline_ams': baseline_ams,
    'improvement_percent': float(improvement),
    'search_iterations': 50,
    'cv_folds': 5
}

with open('optimized_hyperparameters.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\nRESULT:optimized_ams_score={best_ams:.3f}')
print(f'RESULT:cv_ams_score={best_cv_score:.3f}')
print(f'RESULT:optimal_threshold={best_threshold:.3f}')
print(f'RESULT:improvement_percent={improvement:.1f}')
print(f'RESULT:search_iterations=50')
print(f'RESULT:best_n_estimators={best_params["n_estimators"]}')
print(f'RESULT:best_max_depth={best_params["max_depth"]}')
print('RESULT:results_file=optimized_hyperparameters.json')
print('RESULT:success=True')