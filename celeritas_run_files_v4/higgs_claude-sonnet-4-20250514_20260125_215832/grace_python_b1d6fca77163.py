import pandas as pd
import numpy as np
from pathlib import Path
import json

# Load enhanced features from previous step
features_file = 'atlas_higgs_features_enhanced.parquet'
print(f'Loading enhanced features from {features_file}')
df = pd.read_parquet(features_file)

# Load metadata for feature information
with open('atlas_higgs_features_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f'Loaded {len(df)} events with {len(df.columns)} features')
print(f'Features: {list(df.columns)}')

# Separate features, target, and weights
feature_cols = [col for col in df.columns if col not in ['Label', 'Weight', 'EventId']]
X = df[feature_cols].values
y = (df['Label'] == 's').astype(int)  # Convert to binary (1=signal, 0=background)
weights = df['Weight'].values

print(f'Feature matrix shape: {X.shape}')
print(f'Signal events: {np.sum(y)}, Background events: {np.sum(1-y)}')
print(f'Weight range: [{weights.min():.6f}, {weights.max():.6f}]')

# Define AMS metric function
def ams_score(y_true, y_pred_proba, weights, threshold=0.5):
    """Calculate Approximate Median Significance (AMS) score"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate weighted true/false positives
    s = np.sum(weights[(y_true == 1) & (y_pred == 1)])  # Signal correctly identified
    b = np.sum(weights[(y_true == 0) & (y_pred == 1)])  # Background misidentified
    
    # AMS formula with regularization
    br = 10.0  # Regularization term
    if b + br <= 0:
        return 0
    
    ams = np.sqrt(2 * ((s + b + br) * np.log(1 + s/(b + br)) - s))
    return ams

# Optimize threshold for AMS
def optimize_ams_threshold(y_true, y_pred_proba, weights):
    """Find optimal threshold that maximizes AMS score"""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_ams = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        ams = ams_score(y_true, y_pred_proba, weights, threshold)
        if ams > best_ams:
            best_ams = ams
            best_threshold = threshold
    
    return best_threshold, best_ams

# Simple stratified split for cross-validation
def stratified_split(X, y, weights, test_size=0.2, random_state=42):
    """Simple stratified train/test split"""
    np.random.seed(random_state)
    
    # Get indices for signal and background
    signal_idx = np.where(y == 1)[0]
    background_idx = np.where(y == 0)[0]
    
    # Sample test indices
    n_signal_test = int(len(signal_idx) * test_size)
    n_bg_test = int(len(background_idx) * test_size)
    
    test_signal_idx = np.random.choice(signal_idx, n_signal_test, replace=False)
    test_bg_idx = np.random.choice(background_idx, n_bg_test, replace=False)
    
    test_idx = np.concatenate([test_signal_idx, test_bg_idx])
    train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
    
    return train_idx, test_idx

# Simple logistic regression implementation
class SimpleLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
    
    def fit(self, X, y, sample_weight=None):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        
        for i in range(self.max_iter):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            
            # Weighted gradients
            dw = (1/n_samples) * np.dot(X.T, sample_weight * (y_pred - y))
            db = (1/n_samples) * np.sum(sample_weight * (y_pred - y))
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        proba = self.sigmoid(z)
        return np.column_stack([1-proba, proba])

# Perform cross-validation
print('\nPerforming 3-fold cross-validation...')
cv_results = []

for fold in range(3):
    print(f'\nFold {fold + 1}:')
    
    # Split data
    train_idx, val_idx = stratified_split(X, y, weights, test_size=0.33, random_state=42+fold)
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    w_train, w_val = weights[train_idx], weights[val_idx]
    
    # Normalize features
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0) + 1e-8
    X_train_norm = (X_train - X_mean) / X_std
    X_val_norm = (X_val - X_mean) / X_std
    
    # Train simple logistic regression (as baseline)
    model = SimpleLogisticRegression(learning_rate=0.1, max_iter=500)
    model.fit(X_train_norm, y_train, sample_weight=w_train)
    
    # Predict on validation set
    y_val_proba = model.predict_proba(X_val_norm)[:, 1]
    
    # Optimize threshold for AMS
    best_threshold, best_ams = optimize_ams_threshold(y_val, y_val_proba, w_val)
    
    print(f'  Best threshold: {best_threshold:.3f}')
    print(f'  Best AMS score: {best_ams:.4f}')
    
    cv_results.append({
        'fold': fold + 1,
        'ams_score': best_ams,
        'threshold': best_threshold,
        'model': model,
        'normalization': (X_mean, X_std)
    })

# Select best model based on AMS score
best_fold = max(cv_results, key=lambda x: x['ams_score'])
print(f'\nBest model from fold {best_fold["fold"]} with AMS = {best_fold["ams_score"]:.4f}')

# Calculate cross-validation statistics
ams_scores = [result['ams_score'] for result in cv_results]
ams_mean = np.mean(ams_scores)
ams_std = np.std(ams_scores)

print(f'\nCross-validation results:')
print(f'Mean AMS: {ams_mean:.4f} ± {ams_std:.4f}')
print(f'AMS scores: {ams_scores}')

# Feature importance (from best model weights)
best_model = best_fold['model']
feature_importance = np.abs(best_model.weights)
top_features_idx = np.argsort(feature_importance)[-10:][::-1]

print('\nTop 10 most important features:')
for i, idx in enumerate(top_features_idx):
    print(f'{i+1:2d}. {feature_cols[idx]:30s} {feature_importance[idx]:.4f}')

# Save results
results = {
    'cv_mean_ams': float(ams_mean),
    'cv_std_ams': float(ams_std),
    'cv_scores': [float(score) for score in ams_scores],
    'best_fold': int(best_fold['fold']),
    'best_ams': float(best_fold['ams_score']),
    'best_threshold': float(best_fold['threshold']),
    'num_features': len(feature_cols),
    'feature_names': feature_cols,
    'top_features': [feature_cols[idx] for idx in top_features_idx],
    'feature_importance': [float(feature_importance[idx]) for idx in top_features_idx]
}

with open('higgs_classifier_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\nRESULT:cv_mean_ams={ams_mean:.4f}')
print(f'RESULT:cv_std_ams={ams_std:.4f}')
print(f'RESULT:best_ams_score={best_fold["ams_score"]:.4f}')
print(f'RESULT:best_threshold={best_fold["threshold"]:.3f}')
print(f'RESULT:num_features_used={len(feature_cols)}')
print(f'RESULT:classifier_trained=True')
print('RESULT:results_file=higgs_classifier_results.json')
print('Training completed successfully!')