import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import json
import warnings
warnings.filterwarnings('ignore')

# Load preprocessed data
print('Loading preprocessed data...')
df = pd.read_parquet('preprocessed_atlas_data.parquet')
print(f'Loaded {len(df)} events with {df.shape[1]} columns')

# Inspect dataframe columns
print('\nDataframe columns:')
print(df.columns.tolist())
print(f'\nDataframe shape: {df.shape}')
print(f'Data types: {df.dtypes.value_counts().to_dict()}')

# Fix column names - remove spaces and special characters
original_columns = df.columns.tolist()
df.columns = df.columns.str.replace(' ', '_').str.replace('-', '_').str.replace('(', '').str.replace(')', '')
print(f'\nFixed column names: {len([c for c in original_columns if c != df.columns[original_columns.index(c)]])} columns renamed')

# Add column validation
required_columns = ['Label', 'Weight']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f'Missing required columns: {missing_columns}')
print(f'Column validation passed: all required columns present')

# Prepare features and target
y = (df['Label'] == 's').astype(int)
weights = df['Weight'].values
feature_cols = [col for col in df.columns if col not in ['Label', 'Weight', 'KaggleSet', 'KaggleWeight']]
X = df[feature_cols].copy()

# Handle missing values (-999)
X = X.replace(-999.0, np.nan)
X = X.fillna(X.median())

print(f'\nFeatures: {len(feature_cols)} columns')
print(f'Target distribution: {np.bincount(y)} (0=background, 1=signal)')
print(f'Weight range: [{weights.min():.3f}, {weights.max():.3f}]')
print(f'Using weight column: Weight')

# AMS metric implementation
def ams_score(y_true, y_pred, sample_weight=None):
    if sample_weight is None:
        sample_weight = np.ones(len(y_true))
    
    s = np.sum(sample_weight[(y_true == 1) & (y_pred == 1)])
    b = np.sum(sample_weight[(y_true == 0) & (y_pred == 1)])
    
    if b <= 0:
        return 0
    
    br = 10.0  # Background regularization
    return np.sqrt(2 * ((s + b + br) * np.log(1 + s / (b + br)) - s))

# Custom scorer for cross-validation
ams_scorer = make_scorer(ams_score, greater_is_better=True, needs_proba=False)

# Define search space for comprehensive optimization
search_space = [
    Integer(50, 300, name='n_estimators'),
    Real(0.01, 0.3, name='learning_rate'),
    Integer(3, 10, name='max_depth'),
    Real(0.1, 1.0, name='subsample'),
    Real(0.1, 1.0, name='max_features'),
    Integer(5, 50, name='min_samples_split'),
    Integer(1, 20, name='min_samples_leaf')
]

# Objective function for Bayesian optimization
@use_named_args(search_space)
def objective(**params):
    model = GradientBoostingClassifier(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        subsample=params['subsample'],
        max_features=params['max_features'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        random_state=42
    )
    
    # 5-fold cross-validation with stratification and weights
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train, w_val = weights[train_idx], weights[val_idx]
        
        model.fit(X_train, y_train, sample_weight=w_train)
        y_pred = model.predict(X_val)
        score = ams_score(y_val, y_pred, w_val)
        scores.append(score)
    
    mean_score = np.mean(scores)
    return -mean_score  # Negative because gp_minimize minimizes

# Run Bayesian optimization
print('\nStarting Bayesian hyperparameter optimization...')
result = gp_minimize(
    func=objective,
    dimensions=search_space,
    n_calls=50,  # Reasonable number for T0 tier
    random_state=42,
    acq_func='EI'  # Expected Improvement
)

# Extract optimal parameters
optimal_params = {
    'n_estimators': result.x[0],
    'learning_rate': result.x[1],
    'max_depth': result.x[2],
    'subsample': result.x[3],
    'max_features': result.x[4],
    'min_samples_split': result.x[5],
    'min_samples_leaf': result.x[6]
}

optimal_ams = -result.fun

print(f'\nOptimization completed!')
print(f'Optimal AMS score: {optimal_ams:.3f}')
print(f'Baseline AMS score: 58.443')
print(f'Improvement: {optimal_ams - 58.443:.3f}')

print('\nOptimal hyperparameters:')
for param, value in optimal_params.items():
    print(f'  {param}: {value}')

# Train final model with optimal parameters
final_model = GradientBoostingClassifier(**optimal_params, random_state=42)
final_model.fit(X, y, sample_weight=weights)

# Save results
results = {
    'optimal_hyperparameters': optimal_params,
    'optimal_ams_score': float(optimal_ams),
    'baseline_ams_score': 58.443,
    'improvement': float(optimal_ams - 58.443),
    'optimization_strategy': 'bayesian',
    'cv_folds': 5,
    'search_iterations': 50,
    'features_used': len(feature_cols),
    'training_events': len(df),
    'weight_column_used': 'Weight',
    'column_validation_passed': True,
    'columns_fixed': len([c for c in original_columns if c != df.columns[original_columns.index(c)]])
}

with open('optimized_hyperparameters.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\nRESULT:optimal_ams_score={optimal_ams:.3f}')
print(f'RESULT:improvement_over_baseline={optimal_ams - 58.443:.3f}')
print(f'RESULT:optimization_iterations=50')
print(f'RESULT:cv_folds=5')
print(f'RESULT:features_used={len(feature_cols)}')
print(f'RESULT:weight_column_used=Weight')
print(f'RESULT:columns_validated=True')
print(f'RESULT:results_file=optimized_hyperparameters.json')
print('RESULT:success=True')