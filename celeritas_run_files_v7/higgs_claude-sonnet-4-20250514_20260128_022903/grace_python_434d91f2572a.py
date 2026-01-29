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
df = pd.read_parquet('preprocessed_atlas_data.csv')
print(f"Loaded {len(df)} samples with {df.shape[1]} features")

# Load feature metadata
with open('feature_metadata.json', 'r') as f:
    metadata = json.load(f)
print(f"Feature metadata: {metadata['original_features']} original + {metadata['engineered_features']} engineered = {metadata['total_features']} total")

# Handle missing values (-999.0) by replacing with NaN
df_clean = df.copy()
df_clean[df_clean == -999.0] = np.nan

# Separate features and target
feature_cols = [col for col in df_clean.columns if col not in ['Label', 'KaggleSet', 'KaggleWeight']]
X = df_clean[feature_cols]
y = df_clean['Label'].map({'s': 1, 'b': 0})  # signal=1, background=0
weights = df_clean['KaggleWeight']

# Fill remaining NaN values with median
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

# Train/test split stratified by signal/background
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_imputed, y, weights, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} samples ({np.sum(y_train)} signal, {len(y_train) - np.sum(y_train)} background)")
print(f"Test set: {len(X_test)} samples ({np.sum(y_test)} signal, {len(y_test) - np.sum(y_test)} background)")

# Build gradient boosting classifier optimized for AMS
# Use parameters that work well for physics classification
gb_classifier = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    random_state=42,
    verbose=1
)

# Train with sample weights
print("Training gradient boosting classifier...")
gb_classifier.fit(X_train, y_train, sample_weight=w_train)

# Predict probabilities for AMS calculation
y_pred_proba = gb_classifier.predict_proba(X_test)[:, 1]
y_pred = gb_classifier.predict(X_test)

# Calculate AMS metric
def calculate_ams(y_true, y_pred, weights, b_reg=10):
    """Calculate Approximate Median Significance (AMS) metric"""
    signal_mask = (y_true == 1) & (y_pred == 1)
    background_mask = (y_true == 0) & (y_pred == 1)
    
    s = np.sum(weights[signal_mask]) if np.any(signal_mask) else 0
    b = np.sum(weights[background_mask]) if np.any(background_mask) else 0
    
    if s + b == 0:
        return 0
    
    ams = np.sqrt(2 * ((s + b + b_reg) * np.log(1 + s / (b + b_reg)) - s))
    return ams

# Calculate AMS for different probability thresholds
thresholds = np.linspace(0.1, 0.9, 50)
ams_scores = []

for threshold in thresholds:
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    ams = calculate_ams(y_test, y_pred_thresh, w_test, b_reg=10)
    ams_scores.append(ams)

# Find optimal threshold and AMS
best_idx = np.argmax(ams_scores)
best_threshold = thresholds[best_idx]
best_ams = ams_scores[best_idx]

print(f"Best AMS score: {best_ams:.4f} at threshold {best_threshold:.3f}")

# Calculate final predictions with optimal threshold
y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)

# Calculate classification metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(y_test, y_pred_optimal)
precision = precision_score(y_test, y_pred_optimal)
recall = recall_score(y_test, y_pred_optimal)
f1 = f1_score(y_test, y_pred_optimal)

print(f"Classification metrics at optimal threshold:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Plot AMS vs threshold
plt.figure(figsize=(10, 6))
plt.plot(thresholds, ams_scores, linewidth=2, color='blue')
plt.axvline(best_threshold, color='red', linestyle='--', label=f'Optimal threshold: {best_threshold:.3f}')
plt.axhline(best_ams, color='red', linestyle='--', alpha=0.5, label=f'Best AMS: {best_ams:.4f}')
plt.xlabel('Classification Threshold')
plt.ylabel('AMS Score')
plt.title('AMS Score vs Classification Threshold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('ams_optimization.png', dpi=150, bbox_inches='tight')
plt.savefig('ams_optimization.pdf', bbox_inches='tight')

# Plot feature importance
feature_importance = gb_classifier.feature_importances_
feature_names = X.columns

# Get top 15 most important features
top_indices = np.argsort(feature_importance)[-15:]
top_features = feature_names[top_indices]
top_importance = feature_importance[top_indices]

plt.figure(figsize=(10, 8))
plt.barh(range(len(top_features)), top_importance, color='skyblue')
plt.yticks(range(len(top_features)), top_features)
plt.xlabel('Feature Importance')
plt.title('Top 15 Most Important Features (Gradient Boosting)')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.savefig('feature_importance.pdf', bbox_inches='tight')

# Save model and results
import pickle
with open('baseline_gb_classifier.pkl', 'wb') as f:
    pickle.dump(gb_classifier, f)

# Save results to JSON
results = {
    'model_type': 'gradient_boosting',
    'optimization_metric': 'AMS',
    'best_ams_score': float(best_ams),
    'optimal_threshold': float(best_threshold),
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'n_estimators': 100,
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'features_used': len(feature_cols),
    'top_feature': str(top_features[-1]),
    'top_feature_importance': float(top_importance[-1])
}

with open('baseline_classifier_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nBaseline classifier training completed successfully!")
print(f"Model saved to: baseline_gb_classifier.pkl")
print(f"Results saved to: baseline_classifier_results.json")
print(f"Plots saved: ams_optimization.png, feature_importance.png")

# Output results for downstream steps
print(f"RESULT:success=True")
print(f"RESULT:best_ams_score={best_ams:.4f}")
print(f"RESULT:optimal_threshold={best_threshold:.3f}")
print(f"RESULT:model_accuracy={accuracy:.4f}")
print(f"RESULT:model_precision={precision:.4f}")
print(f"RESULT:model_recall={recall:.4f}")
print(f"RESULT:training_samples={len(X_train)}")
print(f"RESULT:test_samples={len(X_test)}")
print(f"RESULT:features_used={len(feature_cols)}")
print(f"RESULT:model_file=baseline_gb_classifier.pkl")
print(f"RESULT:results_file=baseline_classifier_results.json")
print(f"RESULT:ams_plot=ams_optimization.png")
print(f"RESULT:importance_plot=feature_importance.png")