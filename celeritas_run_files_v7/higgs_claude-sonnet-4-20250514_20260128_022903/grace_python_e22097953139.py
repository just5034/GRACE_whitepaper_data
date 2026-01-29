import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Load preprocessed data
df = pd.read_parquet('atlas_preprocessed.parquet')
print(f'Loaded {len(df)} events with {df.shape[1]} features')

# Separate features and target
# Remove non-feature columns
feature_cols = [col for col in df.columns if col not in ['Label', 'KaggleSet', 'KaggleWeight', 'Weight']]
X = df[feature_cols]
y = (df['Label'] == 's').astype(int)  # Convert s/b to 1/0
weights = df['Weight'] if 'Weight' in df.columns else df.get('KaggleWeight', np.ones(len(df)))

print(f'Features: {len(feature_cols)}')
print(f'Signal events: {y.sum()}, Background events: {(y==0).sum()}')
print(f'Mean event weight: {weights.mean():.4f}')

# Train/test split with stratification
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights, test_size=0.3, random_state=42, stratify=y
)

# Build gradient boosting classifier
# Optimize for AMS metric by using appropriate parameters
gb_classifier = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    random_state=42,
    verbose=1
)

print('Training gradient boosting classifier...')
gb_classifier.fit(X_train, y_train, sample_weight=w_train)

# Make predictions
y_pred_proba = gb_classifier.predict_proba(X_test)[:, 1]
y_pred = gb_classifier.predict(X_test)

# Calculate AMS score
def calculate_ams(y_true, y_pred_proba, weights, threshold=0.5, b_reg=10):
    """Calculate Approximate Median Significance (AMS) score"""
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    # True positives (signal correctly identified)
    s = np.sum(weights[(y_true == 1) & (y_pred_binary == 1)])
    # False positives (background incorrectly identified as signal)
    b = np.sum(weights[(y_true == 0) & (y_pred_binary == 1)])
    
    # AMS formula with regularization
    if b == 0:
        return 0
    ams = np.sqrt(2 * ((s + b + b_reg) * np.log(1 + s/(b + b_reg)) - s))
    return ams

# Calculate AMS at different thresholds to find optimal
thresholds = np.linspace(0.1, 0.9, 17)
ams_scores = []
for thresh in thresholds:
    ams = calculate_ams(y_test, y_pred_proba, w_test, threshold=thresh, b_reg=10)
    ams_scores.append(ams)

optimal_idx = np.argmax(ams_scores)
optimal_threshold = thresholds[optimal_idx]
optimal_ams = ams_scores[optimal_idx]

print(f'Optimal threshold: {optimal_threshold:.3f}')
print(f'Optimal AMS score: {optimal_ams:.4f}')

# Calculate additional metrics
roc_auc = roc_auc_score(y_test, y_pred_proba, sample_weight=w_test)
print(f'Weighted ROC AUC: {roc_auc:.4f}')

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': gb_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print('\nTop 10 most important features:')
print(feature_importance.head(10))

# Plot AMS vs threshold
plt.figure(figsize=(10, 6))
plt.plot(thresholds, ams_scores, 'b-', linewidth=2, marker='o')
plt.axvline(optimal_threshold, color='r', linestyle='--', label=f'Optimal: {optimal_threshold:.3f}')
plt.axhline(optimal_ams, color='r', linestyle='--', alpha=0.5)
plt.xlabel('Classification Threshold')
plt.ylabel('AMS Score')
plt.title(f'AMS Score vs Classification Threshold (Max: {optimal_ams:.4f})')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('ams_threshold_optimization.png', dpi=150, bbox_inches='tight')
plt.savefig('ams_threshold_optimization.pdf', bbox_inches='tight')

# Plot feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 15 Feature Importances (Gradient Boosting)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.savefig('feature_importance.pdf', bbox_inches='tight')

# Plot prediction distributions
plt.figure(figsize=(10, 6))
signal_mask = y_test == 1
background_mask = y_test == 0

plt.hist(y_pred_proba[background_mask], bins=50, alpha=0.7, label='Background', 
         weights=w_test[background_mask], density=True, histtype='step', linewidth=2)
plt.hist(y_pred_proba[signal_mask], bins=50, alpha=0.7, label='Signal',
         weights=w_test[signal_mask], density=True, histtype='step', linewidth=2)
plt.axvline(optimal_threshold, color='r', linestyle='--', label=f'Optimal Threshold: {optimal_threshold:.3f}')
plt.xlabel('Prediction Probability')
plt.ylabel('Density (weighted)')
plt.title('Signal vs Background Prediction Distributions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('prediction_distributions.png', dpi=150, bbox_inches='tight')
plt.savefig('prediction_distributions.pdf', bbox_inches='tight')

# Save model results
results = {
    'model_type': 'gradient_boosting',
    'n_estimators': 100,
    'optimal_threshold': float(optimal_threshold),
    'ams_score': float(optimal_ams),
    'roc_auc': float(roc_auc),
    'n_train_events': len(X_train),
    'n_test_events': len(X_test),
    'n_features': len(feature_cols),
    'feature_importance': feature_importance.to_dict('records')[:20]
}

with open('baseline_classifier_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Output results for downstream steps
print(f'RESULT:ams_score={optimal_ams:.4f}')
print(f'RESULT:optimal_threshold={optimal_threshold:.4f}')
print(f'RESULT:roc_auc={roc_auc:.4f}')
print(f'RESULT:n_features={len(feature_cols)}')
print(f'RESULT:n_train_events={len(X_train)}')
print(f'RESULT:n_test_events={len(X_test)}')
print('RESULT:ams_plot=ams_threshold_optimization.png')
print('RESULT:importance_plot=feature_importance.png')
print('RESULT:distribution_plot=prediction_distributions.png')
print('RESULT:model_file=baseline_classifier_results.json')
print('RESULT:success=True')

print('\nBaseline classifier training completed successfully!')
print(f'Final AMS Score: {optimal_ams:.4f} at threshold {optimal_threshold:.3f}')