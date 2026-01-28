import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import json
import pickle
from pathlib import Path

# Load enhanced dataset from feature engineering step
enhanced_data = pd.read_parquet('atlas_higgs_enhanced_features.parquet')
print(f'Loaded enhanced dataset: {enhanced_data.shape}')

# Handle categorical columns - select numeric only
numeric_columns = enhanced_data.select_dtypes(include=[np.number]).columns.tolist()
print(f'Numeric columns found: {len(numeric_columns)}')

# Remove EventId and Weight columns from features if present
feature_cols = [col for col in numeric_columns if col not in ['EventId', 'Weight', 'Label']]
X = enhanced_data[feature_cols].copy()
y = enhanced_data['Label'].copy()
weights = enhanced_data.get('Weight', np.ones(len(enhanced_data)))

print(f'Features shape: {X.shape}')
print(f'Target distribution: {y.value_counts()}')

# Handle missing values with fallback categorical fill (mode for remaining non-numeric)
X_numeric = X.select_dtypes(include=[np.number])
X_categorical = X.select_dtypes(exclude=[np.number])

if not X_categorical.empty:
    print(f'Found {X_categorical.shape[1]} categorical columns, filling with mode')
    for col in X_categorical.columns:
        mode_val = X_categorical[col].mode()[0] if not X_categorical[col].mode().empty else 'unknown'
        X_categorical[col] = X_categorical[col].fillna(mode_val)
    # Convert categorical to numeric using label encoding
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in X_categorical.columns:
        X_categorical[col] = le.fit_transform(X_categorical[col].astype(str))
    X = pd.concat([X_numeric, X_categorical], axis=1)
else:
    X = X_numeric

# Fill remaining numeric NaN values
X = X.fillna(X.median())

# Define AMS metric function
def ams_score(y_true, y_pred_proba, weights):
    """Calculate Approximate Median Significance (AMS) score"""
    s = np.sum(weights[(y_true == 1) & (y_pred_proba > 0.5)])
    b = np.sum(weights[(y_true == 0) & (y_pred_proba > 0.5)])
    br = 10.0  # regularization term
    return np.sqrt(2 * ((s + b + br) * np.log(1 + s/(b + br)) - s))

# Model configurations
models = {
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42
    )
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print('\nTraining classifiers with cross-validation...')
for name, model in models.items():
    print(f'\nTraining {name}...')
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    
    # Fit full model for AMS evaluation
    model.fit(X, y, sample_weight=weights)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Calculate AMS score
    ams = ams_score(y, y_pred_proba, weights)
    
    # Store results
    results[name] = {
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'ams_score': ams,
        'model': model
    }
    
    print(f'{name} - CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')
    print(f'{name} - AMS Score: {ams:.4f}')

# Select best model based on AMS score
best_model_name = max(results.keys(), key=lambda k: results[k]['ams_score'])
best_model = results[best_model_name]['model']
best_ams = results[best_model_name]['ams_score']

print(f'\nBest model: {best_model_name} (AMS: {best_ams:.4f})')

# Feature importance for best model
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print('\nTop 10 most important features:')
    print(feature_importance.head(10))
    feature_importance.to_csv('feature_importance.csv', index=False)

# Save best model
with open('best_higgs_classifier.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save results summary
summary = {
    'best_model': best_model_name,
    'best_ams_score': float(best_ams),
    'num_features': len(feature_cols),
    'training_samples': len(X),
    'model_results': {name: {
        'cv_auc_mean': float(res['cv_auc_mean']),
        'cv_auc_std': float(res['cv_auc_std']),
        'ams_score': float(res['ams_score'])
    } for name, res in results.items()}
}

with open('higgs_classifier_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('\nTraining completed successfully!')
print(f'RESULT:best_model={best_model_name}')
print(f'RESULT:best_ams_score={best_ams:.4f}')
print(f'RESULT:num_features_used={len(feature_cols)}')
print(f'RESULT:training_samples={len(X)}')
print('RESULT:model_file=best_higgs_classifier.pkl')
print('RESULT:results_file=higgs_classifier_results.json')
print('RESULT:feature_importance_file=feature_importance.csv')