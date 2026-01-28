import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
import json
import time
from pathlib import Path

# Load enhanced features from previous step
enhanced_dataset = 'enhanced_higgs_features.csv'
if not Path(enhanced_dataset).exists():
    print(f'ERROR: {enhanced_dataset} not found')
    exit(1)

print('Loading enhanced Higgs dataset...')
df = pd.read_csv(enhanced_dataset)
print(f'Dataset shape: {df.shape}')

# Separate features and target
feature_cols = [col for col in df.columns if col not in ['Label', 'Weight', 'EventId']]
X = df[feature_cols]
y = (df['Label'] == 's').astype(int)  # Convert s/b to 1/0
weights = df['Weight'] if 'Weight' in df.columns else None

print(f'Features: {len(feature_cols)}')
print(f'Signal events: {sum(y)}, Background events: {len(y) - sum(y)}')

# Sample training data for speed (70% as specified)
sample_size = int(0.7 * len(df))
np.random.seed(42)
sample_idx = np.random.choice(len(df), sample_size, replace=False)
X_sample = X.iloc[sample_idx]
y_sample = y.iloc[sample_idx]
weights_sample = weights.iloc[sample_idx] if weights is not None else None

print(f'Training on {len(X_sample)} samples ({sample_size/len(df)*100:.1f}% of data)')

# Define AMS metric calculation
def calculate_ams(y_true, y_pred, weights=None, s_reg=10.0):
    if weights is None:
        weights = np.ones(len(y_true))
    
    s = np.sum(weights[(y_true == 1) & (y_pred == 1)])  # True positives weighted
    b = np.sum(weights[(y_true == 0) & (y_pred == 1)])  # False positives weighted
    
    if b == 0:
        return 0
    
    ams = np.sqrt(2 * ((s + b + s_reg) * np.log(1 + s / (b + s_reg)) - s))
    return ams

# Simplified model configurations for speed
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=50,  # Reduced from default 100
        max_depth=10,     # Limited depth
        min_samples_split=10,
        random_state=42,
        n_jobs=1  # Single thread as parallel training disabled
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=50,  # Reduced from default 100
        max_depth=6,      # Limited depth
        learning_rate=0.1,
        random_state=42
    )
}

# Cross-validation with reduced folds (3 as specified)
cv_folds = 3
skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

results = {}
best_ams = 0
best_model = None
best_model_name = ''

print('\nTraining models with optimizations...')
start_time = time.time()

for name, model in models.items():
    print(f'\nTraining {name}...')
    model_start = time.time()
    
    # Fit model
    model.fit(X_sample, y_sample, sample_weight=weights_sample)
    
    # Cross-validation predictions for AMS calculation
    cv_predictions = np.zeros(len(y_sample))
    for train_idx, val_idx in skf.split(X_sample, y_sample):
        X_train, X_val = X_sample.iloc[train_idx], X_sample.iloc[val_idx]
        y_train, y_val = y_sample.iloc[train_idx], y_sample.iloc[val_idx]
        w_train = weights_sample.iloc[train_idx] if weights_sample is not None else None
        
        temp_model = type(model)(**model.get_params())
        temp_model.fit(X_train, y_train, sample_weight=w_train)
        cv_predictions[val_idx] = temp_model.predict(X_val)
    
    # Calculate AMS score
    ams_score = calculate_ams(y_sample, cv_predictions, weights_sample)
    
    # Standard accuracy for comparison
    accuracy = np.mean(cv_predictions == y_sample)
    
    model_time = time.time() - model_start
    
    results[name] = {
        'ams_score': ams_score,
        'accuracy': accuracy,
        'training_time': model_time,
        'cv_folds': cv_folds
    }
    
    print(f'{name} - AMS: {ams_score:.4f}, Accuracy: {accuracy:.4f}, Time: {model_time:.1f}s')
    
    # Track best model
    if ams_score > best_ams:
        best_ams = ams_score
        best_model = model
        best_model_name = name

total_time = time.time() - start_time

# Feature importance from best model
if hasattr(best_model, 'feature_importances_'):
    feature_importance = dict(zip(feature_cols, best_model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f'\nTop 10 features from {best_model_name}:')
    for feat, imp in top_features:
        print(f'  {feat}: {imp:.4f}')
else:
    top_features = []

# Save results
results_summary = {
    'best_model': best_model_name,
    'best_ams_score': best_ams,
    'total_training_time': total_time,
    'sample_size': sample_size,
    'cv_folds': cv_folds,
    'models': results,
    'top_features': top_features[:5],  # Top 5 for summary
    'optimization_settings': {
        'timeout': 1200,
        'reduce_model_complexity': True,
        'limit_hyperparameter_search': True,
        'use_early_stopping': True,
        'reduce_cv_folds': cv_folds,
        'sample_training_data': 0.7,
        'max_iterations': 100,
        'enable_parallel_training': False,
        'optimize_for_speed': True
    }
}

with open('higgs_classifier_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)

print(f'\n=== TRAINING COMPLETE ===')
print(f'Best Model: {best_model_name}')
print(f'Best AMS Score: {best_ams:.4f}')
print(f'Total Training Time: {total_time:.1f}s')
print(f'Results saved to: higgs_classifier_results.json')

# Return values for downstream steps
print(f'RESULT:best_model={best_model_name}')
print(f'RESULT:best_ams_score={best_ams:.4f}')
print(f'RESULT:training_time={total_time:.1f}')
print(f'RESULT:sample_size={sample_size}')
print(f'RESULT:cv_folds={cv_folds}')
print(f'RESULT:models_trained={len(models)}')
print(f'RESULT:classifier_status=success')
print(f'RESULT:results_file=higgs_classifier_results.json')