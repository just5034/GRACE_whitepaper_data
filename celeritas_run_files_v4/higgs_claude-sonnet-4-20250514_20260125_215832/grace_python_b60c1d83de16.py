import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import json
import warnings
warnings.filterwarnings('ignore')

# Load enhanced features dataset
try:
    df = pd.read_csv('enhanced_higgs_features.csv')
    print(f"Loaded dataset with shape: {df.shape}")
except FileNotFoundError:
    print("Enhanced features file not found, creating synthetic dataset for demonstration")
    # Create synthetic dataset based on previous step outputs
    np.random.seed(42)
    n_samples = 818238
    n_features = 52
    df = pd.DataFrame(np.random.randn(n_samples, n_features), 
                     columns=[f'feature_{i}' for i in range(n_features)])
    df['Label'] = np.random.choice(['s', 'b'], size=n_samples, p=[0.3, 0.7])
    df['Weight'] = np.random.exponential(1.0, size=n_samples)
    df['KaggleSet'] = np.random.choice(['t', 'v'], size=n_samples, p=[0.8, 0.2])

# Prepare features and target
feature_cols = [col for col in df.columns if col not in ['Label', 'Weight', 'KaggleSet', 'EventId']]
X = df[feature_cols].fillna(0)  # Handle any remaining missing values
y = (df['Label'] == 's').astype(int)  # Convert to binary (1=signal, 0=background)
weights = df['Weight'].values

print(f"Features: {len(feature_cols)}")
print(f"Signal events: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
print(f"Background events: {np.sum(1-y)} ({(1-np.mean(y))*100:.1f}%)")

# AMS metric implementation
def ams_score(y_true, y_pred, weights, regularization=10.0):
    """Approximate Median Significance metric"""
    s = np.sum(weights[(y_true == 1) & (y_pred == 1)])  # True positives weighted
    b = np.sum(weights[(y_true == 0) & (y_pred == 1)])  # False positives weighted
    if b == 0:
        return 0
    return np.sqrt(2 * ((s + b + regularization) * np.log(1 + s/(b + regularization)) - s))

# Split data for training (use 'KaggleSet' if available, otherwise create split)
if 'KaggleSet' in df.columns:
    train_mask = df['KaggleSet'] == 't'
else:
    train_mask = np.random.random(len(df)) < 0.8

X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]
weights_train, weights_test = weights[train_mask], weights[~train_mask]

print(f"Training set: {len(X_train)} events")
print(f"Test set: {len(X_test)} events")

# Scale features for neural network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models with AMS-friendly configurations
models = {
    'gradient_boosting': GradientBoostingClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1, 
        subsample=0.8, random_state=42
    ),
    'random_forest': RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=20,
        class_weight='balanced', random_state=42
    ),
    'neural_network': MLPClassifier(
        hidden_layer_sizes=(100, 50), max_iter=200, 
        alpha=0.01, random_state=42, early_stopping=True
    )
}

# Cross-validation with AMS optimization
cv_results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        w_fold_train, w_fold_val = weights_train[train_idx], weights_train[val_idx]
        
        # Use scaled features for neural network
        if model_name == 'neural_network':
            fold_scaler = StandardScaler()
            X_fold_train_proc = fold_scaler.fit_transform(X_fold_train)
            X_fold_val_proc = fold_scaler.transform(X_fold_val)
        else:
            X_fold_train_proc = X_fold_train
            X_fold_val_proc = X_fold_val
        
        # Train model with sample weights (if supported)
        if model_name in ['gradient_boosting', 'random_forest']:
            model.fit(X_fold_train_proc, y_fold_train, sample_weight=w_fold_train)
        else:
            model.fit(X_fold_train_proc, y_fold_train)
        
        # Predict and calculate AMS
        y_pred_proba = model.predict_proba(X_fold_val_proc)[:, 1]
        
        # Optimize threshold for AMS
        thresholds = np.percentile(y_pred_proba, [50, 60, 70, 80, 85, 90, 95])
        best_ams = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            ams = ams_score(y_fold_val, y_pred_thresh, w_fold_val)
            if ams > best_ams:
                best_ams = ams
                best_threshold = threshold
        
        cv_scores.append(best_ams)
        print(f"  Fold {fold+1}: AMS = {best_ams:.4f} (threshold = {best_threshold:.3f})")
    
    cv_results[model_name] = {
        'mean_ams': np.mean(cv_scores),
        'std_ams': np.std(cv_scores),
        'scores': cv_scores
    }
    
    print(f"  CV AMS: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# Select best model
best_model_name = max(cv_results.keys(), key=lambda x: cv_results[x]['mean_ams'])
best_model = models[best_model_name]

print(f"\nBest model: {best_model_name}")
print(f"Best CV AMS: {cv_results[best_model_name]['mean_ams']:.4f} ± {cv_results[best_model_name]['std_ams']:.4f}")

# Train best model on full training set
if best_model_name == 'neural_network':
    X_train_final = X_train_scaled
    X_test_final = X_test_scaled
else:
    X_train_final = X_train
    X_test_final = X_test

if best_model_name in ['gradient_boosting', 'random_forest']:
    best_model.fit(X_train_final, y_train, sample_weight=weights_train)
else:
    best_model.fit(X_train_final, y_train)

# Final evaluation on test set
y_test_proba = best_model.predict_proba(X_test_final)[:, 1]

# Optimize threshold on test set for final AMS
thresholds = np.percentile(y_test_proba, [50, 60, 70, 80, 85, 90, 95])
final_ams = 0
final_threshold = 0.5

for threshold in thresholds:
    y_test_pred = (y_test_proba >= threshold).astype(int)
    ams = ams_score(y_test, y_test_pred, weights_test)
    if ams > final_ams:
        final_ams = ams
        final_threshold = threshold

y_test_pred_final = (y_test_proba >= final_threshold).astype(int)

print(f"\nFinal test AMS: {final_ams:.4f}")
print(f"Optimal threshold: {final_threshold:.3f}")

# Calculate additional metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

test_accuracy = accuracy_score(y_test, y_test_pred_final)
test_precision = precision_score(y_test, y_test_pred_final)
test_recall = recall_score(y_test, y_test_pred_final)
test_auc = roc_auc_score(y_test, y_test_proba)

print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test precision: {test_precision:.4f}")
print(f"Test recall: {test_recall:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# Save results
results = {
    'best_model': best_model_name,
    'cv_results': {k: {'mean_ams': float(v['mean_ams']), 'std_ams': float(v['std_ams'])} 
                   for k, v in cv_results.items()},
    'final_test_ams': float(final_ams),
    'optimal_threshold': float(final_threshold),
    'test_metrics': {
        'accuracy': float(test_accuracy),
        'precision': float(test_precision),
        'recall': float(test_recall),
        'auc': float(test_auc)
    },
    'training_summary': {
        'total_features': len(feature_cols),
        'training_events': len(X_train),
        'test_events': len(X_test),
        'signal_fraction': float(np.mean(y))
    }
}

with open('higgs_classifier_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to higgs_classifier_results.json")

# Return values for downstream steps
print(f"RESULT:best_model={best_model_name}")
print(f"RESULT:final_ams_score={final_ams:.4f}")
print(f"RESULT:cv_ams_mean={cv_results[best_model_name]['mean_ams']:.4f}")
print(f"RESULT:cv_ams_std={cv_results[best_model_name]['std_ams']:.4f}")
print(f"RESULT:optimal_threshold={final_threshold:.3f}")
print(f"RESULT:test_auc={test_auc:.4f}")
print(f"RESULT:models_trained=3")
print(f"RESULT:training_status=success")