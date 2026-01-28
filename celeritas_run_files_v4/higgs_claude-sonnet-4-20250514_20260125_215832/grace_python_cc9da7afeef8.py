
import pandas as pd
import json
from pathlib import Path

data_path = Path('/projects/bgde/grace/data/atlas-higgs-challenge-2014-v2.csv')

try:
    # Determine file type and load
    if data_path.suffix.lower() == '.csv':
        df = pd.read_csv(data_path, nrows=1000)  # Sample for speed
    elif data_path.suffix.lower() in ['.parquet', '.pq']:
        df = pd.read_parquet(data_path).head(1000)
    elif data_path.suffix.lower() == '.json':
        df = pd.read_json(data_path).head(1000)
    else:
        print("RESULT:error=Unsupported file type")
        raise SystemExit(1)

    # Get basic info
    info = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }

    # Get stats for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    stats = {}
    for col in numeric_cols:
        stats[col] = {
            "min": float(df[col].min()) if pd.notna(df[col].min()) else None,
            "max": float(df[col].max()) if pd.notna(df[col].max()) else None,
            "mean": float(df[col].mean()) if pd.notna(df[col].mean()) else None,
            "std": float(df[col].std()) if pd.notna(df[col].std()) else None,
            "null_count": int(df[col].isnull().sum()),
        }
    info["numeric_stats"] = stats

    # Get preview
    info["preview"] = df.head(5).to_dict(orient='records')

    # Output results
    print(f"RESULT:shape={info['shape']}")
    print(f"RESULT:n_columns={len(info['columns'])}")
    print(f"RESULT:columns={','.join(info['columns'][:20])}")  # First 20

    # Save full info to file
    with open("data_profile.json", "w") as f:
        json.dump(info, f, indent=2, default=str)
    print("RESULT:profile_saved=true")

except Exception as e:
    print(f"RESULT:error={str(e)}")
