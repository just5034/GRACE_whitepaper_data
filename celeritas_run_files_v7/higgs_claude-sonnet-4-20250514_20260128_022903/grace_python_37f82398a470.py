import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json
import matplotlib.pyplot as plt

# Load preprocessed data
df = pd.read_parquet('preprocessed_atlas_data.parquet')
print(f'Loaded {len(df)} events with {df.shape[1]} features')

# Separate features and target
feature_cols = [col for col in df.columns if col not in ['Label', 'KaggleSet', 'KaggleWeight', 'Weight']]
X = df[feature_cols]
y = (df['Label'] == 's').astype(int)  # Convert s/b to 1/0
weights = df['Weight'].values

print(f'Features: {len(feature_cols)}')
print(f'Signal events: {np.sum(y)}')
print(f'Background events: {np.sum(1-y)}')

# Train/test split maintaining event weights
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights, test_size=0.3, random_state=42, stratify=y
)

# Build gradient boosting classifier
print('Training gradient boosting classifier...')
gb_classifier = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    verbose=1
)

# Train with sample weights
gb_classifier.fit(X_train, y_train, sample_weight=w_train)

# Get predictions and probabilities
y_pred = gb_classifier.predict(X_test)
y_prob = gb_classifier.predict_proba(X_test)[:, 1]

# AMS metric calculation function
def calculate_ams(y_true, y_prob, weights, b_reg=10):
    """Calculate AMS score for different probability thresholds"""
    thresholds = np.linspace(0.1, 0.9, 50)
    best_ams = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred_thresh = (y_prob >= threshold).astype(int)
        
        # Calculate weighted signal and background
        s = np.sum(weights[(y_true == 1) & (y_pred_thresh == 1)])
        b = np.sum(weights[(y_true == 0) & (y_pred_thresh == 1)])
        
        # AMS formula: sqrt(2 * ((s+b+b_reg) * ln(1 + s/(b+b_reg)) - s))
        if b > 0:
            ams = np.sqrt(2 * ((s + b + b_reg) * np.log(1 + s/(b + b_reg)) - s))
        else:
            ams = 0
            
        if ams > best_ams:
            best_ams = ams
            best_threshold = threshold
    
    return best_ams, best_threshold

# Calculate AMS score
print('Calculating AMS score...')
ams_score, optimal_threshold = calculate_ams(y_test, y_prob, w_test, b_reg=10)

# Calculate additional metrics
y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
s_weighted = np.sum(w_test[(y_test == 1) & (y_pred_optimal == 1)])
b_weighted = np.sum(w_test[(y_test == 0) & (y_pred_optimal == 1)])
signal_efficiency = np.sum((y_test == 1) & (y_pred_optimal == 1)) / np.sum(y_test == 1)
background_rejection = 1 - np.sum((y_test == 0) & (y_pred_optimal == 1)) / np.sum(y_test == 0)

print(f'\nBaseline Classifier Results:')
print(f'AMS Score: {ams_score:.4f}')
print(f'Optimal threshold: {optimal_threshold:.3f}')
print(f'Signal efficiency: {signal_efficiency:.3f}')
print(f'Background rejection: {background_rejection:.3f}')
print(f'Weighted signal events: {s_weighted:.1f}')
print(f'Weighted background events: {b_weighted:.1f}')

# Feature importance plot
feature_importance = gb_classifier.feature_importances_
top_features_idx = np.argsort(feature_importance)[-10:]
top_features = [feature_cols[i] for i in top_features_idx]
top_importance = feature_importance[top_features_idx]

plt.figure(figsize=(10, 6))
plt.barh(range(len(top_features)), top_importance)
plt.yticks(range(len(top_features)), top_features)
plt.xlabel('Feature Importance')
plt.title('Top 10 Most Important Features - Baseline Classifier')
plt.tight_layout()
plt.savefig('baseline_feature_importance.png', dpi=150, bbox_inches='tight')
plt.savefig('baseline_feature_importance.pdf', bbox_inches='tight')

# Score distribution plot
plt.figure(figsize=(10, 6))
signal_scores = y_prob[y_test == 1]
background_scores = y_prob[y_test == 0]
plt.hist(background_scores, bins=50, alpha=0.7, label='Background', density=True, weights=w_test[y_test == 0])
plt.hist(signal_scores, bins=50, alpha=0.7, label='Signal', density=True, weights=w_test[y_test == 1])
plt.axvline(optimal_threshold, color='red', linestyle='--', label=f'Optimal threshold ({optimal_threshold:.3f})')
plt.xlabel('Classifier Score')
plt.ylabel('Weighted Density')
plt.title(f'Baseline Classifier Score Distribution (AMS = {ams_score:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('baseline_score_distribution.png', dpi=150, bbox_inches='tight')
plt.savefig('baseline_score_distribution.pdf', bbox_inches='tight')

# Save model and results
results = {
    'model_type': 'gradient_boosting',
    'ams_score': float(ams_score),
    'optimal_threshold': float(optimal_threshold),
    'signal_efficiency': float(signal_efficiency),
    'background_rejection': float(background_rejection),
    'weighted_signal': float(s_weighted),
    'weighted_background': float(b_weighted),
    'n_estimators': gb_classifier.n_estimators,
    'feature_importance': {feature_cols[i]: float(importance) for i, importance in enumerate(feature_importance)},
    'training_events': len(X_train),
    'test_events': len(X_test)
}

with open('baseline_classifier_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Output results for workflow
print(f'RESULT:ams_score={ams_score:.4f}')
print(f'RESULT:optimal_threshold={optimal_threshold:.3f}')
print(f'RESULT:signal_efficiency={signal_efficiency:.3f}')
print(f'RESULT:background_rejection={background_rejection:.3f}')
print(f'RESULT:weighted_signal={s_weighted:.1f}')
print(f'RESULT:weighted_background={b_weighted:.1f}')
print(f'RESULT:training_events={len(X_train)}')
print(f'RESULT:test_events={len(X_test)}')
print('RESULT:feature_importance_plot=baseline_feature_importance.png')
print('RESULT:score_distribution_plot=baseline_score_distribution.png')
print('RESULT:results_file=baseline_classifier_results.json')
print('RESULT:success=True')