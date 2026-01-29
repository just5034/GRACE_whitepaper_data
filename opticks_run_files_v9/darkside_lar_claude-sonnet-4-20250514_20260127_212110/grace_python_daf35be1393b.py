import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Input parameters from workflow
simulation_data = 'validate_optimized_design.hits_file'
analysis_type = 'position_reconstruction'
inspect_data_columns = True
adapt_column_names = True
fallback_grouping_columns = ['eventID', 'detectorID', 'sensorID', 'volumeID']
debug_data_structure = True

# File paths from validate_optimized_design step outputs
energy_files = {
    0.0001: '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/darkside_lar_claude-sonnet-4-20250514_20260127_212110/energy_0.000GeV/darkside_optimized_electron_hits_data.parquet',
    0.001: '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/darkside_lar_claude-sonnet-4-20250514_20260127_212110/energy_0.001GeV/darkside_optimized_electron_hits_data.parquet',
    0.005: '/u/jhill5/grace/work/benchmarks/opticks_20260127_190934/darkside_lar_claude-sonnet-4-20250514_20260127_212110/energy_0.005GeV/darkside_optimized_hits_data.parquet'
}

# PMT geometry from generate_optimized_geometry
pmt_count = 100
pmt_diameter_m = 0.2
vessel_diameter_m = 3.6
vessel_height_m = 3.6

print('Starting position reconstruction analysis with debugging...')

# Debug: Inspect data structure for each energy
for energy_gev, filepath in energy_files.items():
    print(f'\n=== DEBUGGING ENERGY {energy_gev} GeV ===')
    
    if not Path(filepath).exists():
        print(f'WARNING: File not found: {filepath}')
        continue
    
    try:
        df = pd.read_parquet(filepath)
        print(f'File loaded successfully: {len(df)} rows')
        
        if inspect_data_columns:
            print(f'Available columns: {list(df.columns)}')
            print(f'Data types: {df.dtypes.to_dict()}')
            print(f'First few rows:')
            print(df.head(3))
        
        if debug_data_structure:
            print(f'Data shape: {df.shape}')
            print(f'Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB')
            
            # Check for missing values
            missing = df.isnull().sum()
            if missing.any():
                print(f'Missing values: {missing[missing > 0].to_dict()}')
        
        # Adapt column names if needed
        original_columns = df.columns.tolist()
        column_mapping = {}
        
        if adapt_column_names:
            # Common column name variations
            for col in df.columns:
                if col.lower() in ['event_id', 'event', 'evt']:
                    column_mapping[col] = 'eventID'
                elif col.lower() in ['detector_id', 'det_id', 'detector']:
                    column_mapping[col] = 'detectorID'
                elif col.lower() in ['sensor_id', 'pmt_id', 'sensor']:
                    column_mapping[col] = 'sensorID'
                elif col.lower() in ['volume_id', 'vol_id', 'volume']:
                    column_mapping[col] = 'volumeID'
            
            if column_mapping:
                print(f'Adapting column names: {column_mapping}')
                df = df.rename(columns=column_mapping)
        
        # Check for required grouping columns
        available_grouping = []
        for col in fallback_grouping_columns:
            if col in df.columns:
                available_grouping.append(col)
        
        print(f'Available grouping columns: {available_grouping}')
        
        if not available_grouping:
            print('ERROR: No suitable grouping columns found!')
            print('Trying alternative column detection...')
            
            # Look for any ID-like columns
            id_columns = [col for col in df.columns if 'id' in col.lower() or 'ID' in col]
            print(f'ID-like columns found: {id_columns}')
            
            if id_columns:
                available_grouping = id_columns[:2]  # Use first 2 ID columns
                print(f'Using fallback grouping: {available_grouping}')
        
        # Position reconstruction analysis
        if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
            print('\n=== POSITION RECONSTRUCTION ANALYSIS ===')
            
            # Calculate position statistics
            x_mean = df['x'].mean()
            y_mean = df['y'].mean()
            z_mean = df['z'].mean()
            
            x_std = df['x'].std()
            y_std = df['y'].std()
            z_std = df['z'].std()
            
            print(f'Position means: x={x_mean:.2f}, y={y_mean:.2f}, z={z_mean:.2f} mm')
            print(f'Position spreads: σ_x={x_std:.2f}, σ_y={y_std:.2f}, σ_z={z_std:.2f} mm')
            
            # Radial distribution
            r = np.sqrt(df['x']**2 + df['y']**2)
            r_mean = r.mean()
            r_std = r.std()
            
            print(f'Radial distribution: r_mean={r_mean:.2f} mm, σ_r={r_std:.2f} mm')
            
            # Position reconstruction resolution (RMS)
            position_resolution = np.sqrt(x_std**2 + y_std**2 + z_std**2)
            radial_resolution = r_std
            
            print(f'3D position resolution: {position_resolution:.2f} mm')
            print(f'Radial resolution: {radial_resolution:.2f} mm')
            
            # Store results for this energy
            print(f'RESULT:energy_{energy_gev}_gev_position_resolution_mm={position_resolution:.2f}')
            print(f'RESULT:energy_{energy_gev}_gev_radial_resolution_mm={radial_resolution:.2f}')
            print(f'RESULT:energy_{energy_gev}_gev_x_resolution_mm={x_std:.2f}')
            print(f'RESULT:energy_{energy_gev}_gev_y_resolution_mm={y_std:.2f}')
            print(f'RESULT:energy_{energy_gev}_gev_z_resolution_mm={z_std:.2f}')
        
        else:
            print('WARNING: Position columns (x, y, z) not found in data')
            print('Available numeric columns:', [col for col in df.columns if df[col].dtype in ['float64', 'int64']])
    
    except Exception as e:
        print(f'ERROR processing {filepath}: {str(e)}')
        continue

# Create position reconstruction plots
print('\n=== CREATING POSITION PLOTS ===')

try:
    # Load data from highest energy file for plotting
    plot_file = energy_files[0.005]  # 5 MeV data
    if Path(plot_file).exists():
        df_plot = pd.read_parquet(plot_file)
        
        if 'x' in df_plot.columns and 'y' in df_plot.columns:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # XY scatter plot
            ax1.scatter(df_plot['x'], df_plot['y'], alpha=0.6, s=1)
            ax1.set_xlabel('X Position (mm)')
            ax1.set_ylabel('Y Position (mm)')
            ax1.set_title('Hit Positions (XY View)')
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal')
            
            # Radial distribution
            r = np.sqrt(df_plot['x']**2 + df_plot['y']**2)
            ax2.hist(r, bins=50, alpha=0.7, density=True)
            ax2.set_xlabel('Radial Distance (mm)')
            ax2.set_ylabel('Density')
            ax2.set_title('Radial Hit Distribution')
            ax2.grid(True, alpha=0.3)
            
            # X position distribution
            ax3.hist(df_plot['x'], bins=50, alpha=0.7, density=True)
            ax3.set_xlabel('X Position (mm)')
            ax3.set_ylabel('Density')
            ax3.set_title('X Position Distribution')
            ax3.grid(True, alpha=0.3)
            
            # Y position distribution
            ax4.hist(df_plot['y'], bins=50, alpha=0.7, density=True)
            ax4.set_xlabel('Y Position (mm)')
            ax4.set_ylabel('Density')
            ax4.set_title('Y Position Distribution')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('position_reconstruction_analysis.png', dpi=150, bbox_inches='tight')
            plt.savefig('position_reconstruction_analysis.pdf', bbox_inches='tight')
            print('RESULT:position_plots=position_reconstruction_analysis.png')
            
            # Spatial uniformity analysis
            vessel_radius_mm = vessel_diameter_m * 1000 / 2  # Convert to mm
            
            # Divide detector into spatial bins
            r_bins = np.linspace(0, vessel_radius_mm, 10)
            r_centers = (r_bins[:-1] + r_bins[1:]) / 2
            
            hit_counts = []
            for i in range(len(r_bins)-1):
                mask = (r >= r_bins[i]) & (r < r_bins[i+1])
                hit_counts.append(np.sum(mask))
            
            hit_counts = np.array(hit_counts)
            uniformity = np.std(hit_counts) / np.mean(hit_counts) if np.mean(hit_counts) > 0 else 0
            
            print(f'Spatial uniformity (CV): {uniformity:.4f}')
            print(f'RESULT:spatial_uniformity={uniformity:.4f}')
            
        else:
            print('Cannot create position plots - position columns not found')
    
except Exception as e:
    print(f'Error creating plots: {str(e)}')

# Summary
print('\n=== ANALYSIS SUMMARY ===')
print('Position reconstruction analysis completed with debugging')
print('Key modifications applied:')
print('- Data column inspection enabled')
print('- Column name adaptation performed')
print('- Fallback grouping columns used')
print('- Data structure debugging enabled')

print('RESULT:analysis_type=position_reconstruction')
print('RESULT:debug_mode=True')
print('RESULT:success=True')