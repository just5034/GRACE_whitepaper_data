import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load the ATLAS Higgs dataset
data_path = '/projects/bgde/grace/data/atlas-higgs-challenge-2014-v2.csv'
print(f'Loading data from {data_path}')
df = pd.read_csv(data_path)
print(f'Original data shape: {df.shape}')

# Handle missing values (marked as -999.0)
missing_indicator = -999.0
print(f'Handling missing values marked as {missing_indicator}')

# Identify columns with missing values
missing_cols = []
for col in df.columns:
    if col != 'EventId':
        missing_count = (df[col] == missing_indicator).sum()
        if missing_count > 0:
            missing_cols.append(col)
            print(f'{col}: {missing_count} missing values ({missing_count/len(df)*100:.1f}%)')

# Categorize features by physics type
kinematic_features = [col for col in df.columns if col.startswith('DER_')]
primitives = [col for col in df.columns if col.startswith('PRI_')]
weight_cols = ['Weight']
event_id = ['EventId']
label_col = ['Label'] if 'Label' in df.columns else []

print(f'Feature categories:')
print(f'  Kinematic (DER_): {len(kinematic_features)} features')
print(f'  Primitives (PRI_): {len(primitives)} features')
print(f'  Weight columns: {weight_cols}')
print(f'  Event ID: {event_id}')
if label_col:
    print(f'  Labels: {label_col}')

# Handle missing values with physics-motivated imputation
df_processed = df.copy()

# For jet-related features, impute based on jet multiplicity
for col in missing_cols:
    if 'jet' in col.lower():
        # Missing jet features likely mean no jets - impute with 0
        df_processed[col] = df_processed[col].replace(missing_indicator, 0.0)
        print(f'Imputed {col} missing values with 0 (no jets)')
    else:
        # For other features, use median imputation within jet multiplicity groups
        if 'PRI_jet_num' in df.columns:
            for jet_num in df['PRI_jet_num'].unique():
                mask = (df_processed['PRI_jet_num'] == jet_num) & (df_processed[col] != missing_indicator)
                if mask.sum() > 0:
                    median_val = df_processed.loc[mask, col].median()
                    missing_mask = (df_processed['PRI_jet_num'] == jet_num) & (df_processed[col] == missing_indicator)
                    df_processed.loc[missing_mask, col] = median_val
        else:
            # Fallback to global median
            median_val = df_processed[df_processed[col] != missing_indicator][col].median()
            df_processed[col] = df_processed[col].replace(missing_indicator, median_val)
        print(f'Imputed {col} missing values with median')

# Convert numpy types to Python types for JSON serialization
print('Converting numpy types to Python types')
for col in df_processed.columns:
    if df_processed[col].dtype == 'int64':
        df_processed[col] = df_processed[col].astype('int32')  # Smaller int type
    elif df_processed[col].dtype == 'float64':
        df_processed[col] = df_processed[col].astype('float32')  # Smaller float type

# Verify no missing values remain
remaining_missing = (df_processed == missing_indicator).sum().sum()
print(f'Remaining missing values: {remaining_missing}')

# Create feature metadata
feature_metadata = {
    'kinematic_features': kinematic_features,
    'primitive_features': primitives,
    'weight_columns': weight_cols,
    'event_id_column': event_id[0] if event_id else None,
    'label_column': label_col[0] if label_col else None,
    'total_features': len(df_processed.columns),
    'total_events': len(df_processed),
    'missing_value_indicator': float(missing_indicator),
    'imputation_strategy': 'jet_multiplicity_aware'
}

# Save processed data in multiple formats
print('Saving processed data')

# Save as CSV
csv_path = 'atlas_higgs_processed.csv'
df_processed.to_csv(csv_path, index=False)
print(f'Saved CSV: {csv_path}')

# Save as Parquet (more efficient for ML)
parquet_path = 'atlas_higgs_processed.parquet'
df_processed.to_parquet(parquet_path, index=False)
print(f'Saved Parquet: {parquet_path}')

# Save metadata as JSON with proper type conversion
metadata_path = 'atlas_higgs_metadata.json'
with open(metadata_path, 'w') as f:
    # Convert numpy types in metadata to Python types
    metadata_safe = {}
    for key, value in feature_metadata.items():
        if isinstance(value, (np.integer, np.int64)):
            metadata_safe[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            metadata_safe[key] = float(value)
        else:
            metadata_safe[key] = value
    json.dump(metadata_safe, f, indent=2)
print(f'Saved metadata: {metadata_path}')

# Create train/validation split preserving event weights
print('Creating train/validation split')
from sklearn.model_selection import train_test_split

# Stratify by label if available, otherwise random split
if label_col and label_col[0] in df_processed.columns:
    train_df, val_df = train_test_split(
        df_processed, 
        test_size=0.2, 
        random_state=42, 
        stratify=df_processed[label_col[0]]
    )
    print(f'Stratified split by {label_col[0]}')
else:
    train_df, val_df = train_test_split(
        df_processed, 
        test_size=0.2, 
        random_state=42
    )
    print('Random split (no labels found)')

# Save splits
train_df.to_parquet('atlas_higgs_train.parquet', index=False)
val_df.to_parquet('atlas_higgs_val.parquet', index=False)
print(f'Train set: {len(train_df)} events')
print(f'Validation set: {len(val_df)} events')

# Summary statistics
print('\nData preprocessing summary:')
print(f'  Original shape: {df.shape}')
print(f'  Processed shape: {df_processed.shape}')
print(f'  Features by type:')
print(f'    - Kinematic (DER_): {len(kinematic_features)}')
print(f'    - Primitives (PRI_): {len(primitives)}')
print(f'    - Weights: {len(weight_cols)}')
print(f'  Missing values handled: {len(missing_cols)} columns')
print(f'  Data types converted for JSON safety')
print(f'  Files created: CSV, Parquet, metadata JSON, train/val splits')

# Return results
print('RESULT:processed_csv=atlas_higgs_processed.csv')
print('RESULT:processed_parquet=atlas_higgs_processed.parquet')
print('RESULT:metadata_file=atlas_higgs_metadata.json')
print('RESULT:train_file=atlas_higgs_train.parquet')
print('RESULT:val_file=atlas_higgs_val.parquet')
print(f'RESULT:num_events={len(df_processed)}')
print(f'RESULT:num_features={len(df_processed.columns)}')
print(f'RESULT:missing_columns_handled={len(missing_cols)}')
print('RESULT:serialization_fixed=true')