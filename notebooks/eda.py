"""
Exploratory Data Analysis (EDA) Script

This script performs comprehensive EDA on AQI data from Hopsworks.
Can be run as a standalone script or imported as a module.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hopsworks
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_data_from_hopsworks():
    """Load data from Hopsworks feature store."""
    load_dotenv()
    HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
    PROJECT_NAME = os.getenv("PROJECT_NAME")
    HOST = os.getenv("HOST")
    
    project = hopsworks.login(
        host=HOST,
        project=PROJECT_NAME,
        api_key_value=HOPSWORKS_API_KEY
    )
    
    fs = project.get_feature_store(name='aqi_predictor_lahore_featurestore')
    fg = fs.get_feature_group('aqi_features', version=1)
    
    print("Loading data from Hopsworks...")
    df = fg.read(online=True)
    print(f"Data loaded successfully! Shape: {df.shape}")
    
    return df


def data_overview(df):
    """Display basic data overview."""
    print("\n" + "="*60)
    print("DATA OVERVIEW")
    print("="*60)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nData Types:")
    print(df.dtypes)
    print(f"\nStatistical Summary:")
    print(df.describe())
    print("="*60 + "\n")


def missing_values_analysis(df):
    """Analyze missing values."""
    print("\n" + "="*60)
    print("MISSING VALUES ANALYSIS")
    print("="*60)
    
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Count': missing_values,
        'Missing Percentage': missing_percent
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_df) > 0:
        print("Missing Values:")
        print(missing_df)
        
        # Visualize missing values
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.tight_layout()
        plt.savefig('notebooks/plots/missing_values_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\n✓ Saved plot: notebooks/plots/missing_values_heatmap.png")
    else:
        print("✅ No missing values found!")
    
    print("="*60 + "\n")


def target_variable_analysis(df, target_col='aqi', save_plots=True):
    """Analyze the target variable (AQI)."""
    print("\n" + "="*60)
    print("TARGET VARIABLE ANALYSIS (AQI)")
    print("="*60)
    
    if target_col not in df.columns:
        print(f"❌ Target column '{target_col}' not found!")
        return
    
    # Statistics
    print(f"Mean: {df[target_col].mean():.2f}")
    print(f"Median: {df[target_col].median():.2f}")
    print(f"Std: {df[target_col].std():.2f}")
    print(f"Min: {df[target_col].min():.2f}")
    print(f"Max: {df[target_col].max():.2f}")
    print(f"Skewness: {df[target_col].skew():.2f}")
    print(f"Kurtosis: {df[target_col].kurtosis():.2f}")
    
    if save_plots:
        # Histogram and box plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        axes[0].hist(df[target_col], bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_title('Distribution of AQI')
        axes[0].set_xlabel('AQI')
        axes[0].set_ylabel('Frequency')
        axes[0].axvline(df[target_col].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df[target_col].mean():.2f}')
        axes[0].axvline(df[target_col].median(), color='green', linestyle='--', 
                       label=f'Median: {df[target_col].median():.2f}')
        axes[0].legend()
        
        # Box plot
        axes[1].boxplot(df[target_col], vert=True)
        axes[1].set_title('Box Plot of AQI')
        axes[1].set_ylabel('AQI')
        
        plt.tight_layout()
        os.makedirs('notebooks/plots', exist_ok=True)
        plt.savefig('notebooks/plots/target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\n✓ Saved plot: notebooks/plots/target_distribution.png")
    
    print("="*60 + "\n")


def feature_distributions(df, pollutant_cols=None, save_plots=True):
    """Analyze distributions of pollutant features."""
    print("\n" + "="*60)
    print("FEATURE DISTRIBUTIONS")
    print("="*60)
    
    if pollutant_cols is None:
        pollutant_cols = ['pm25', 'pm10', 'no2', 'so2', 'o3']
    
    pollutants = [p for p in pollutant_cols if p in df.columns]
    
    if len(pollutants) == 0:
        print("❌ No pollutant columns found!")
        return
    
    if save_plots:
        # Histograms
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for i, pollutant in enumerate(pollutants):
            axes[i].hist(df[pollutant], bins=30, edgecolor='black', alpha=0.7)
            axes[i].set_title(f'Distribution of {pollutant.upper()}')
            axes[i].set_xlabel(pollutant.upper())
            axes[i].set_ylabel('Frequency')
            axes[i].axvline(df[pollutant].mean(), color='red', linestyle='--', 
                           label=f'Mean: {df[pollutant].mean():.2f}')
            axes[i].legend()
        
        # Hide extra subplots
        for i in range(len(pollutants), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        os.makedirs('notebooks/plots', exist_ok=True)
        plt.savefig('notebooks/plots/pollutant_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved plot: notebooks/plots/pollutant_distributions.png")
        
        # Box plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for i, pollutant in enumerate(pollutants):
            axes[i].boxplot(df[pollutant], vert=True)
            axes[i].set_title(f'Box Plot of {pollutant.upper()}')
            axes[i].set_ylabel(pollutant.upper())
        
        # Hide extra subplots
        for i in range(len(pollutants), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('notebooks/plots/pollutant_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved plot: notebooks/plots/pollutant_boxplots.png")
    
    print("="*60 + "\n")


def correlation_analysis(df, target_col='aqi', save_plots=True):
    """Analyze correlations between features and target."""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    # Select numeric features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'event_id' in numeric_features:
        numeric_features.remove('event_id')
    
    correlation_features = [f for f in numeric_features if f != target_col]
    correlation_features.append(target_col)
    
    corr_matrix = df[correlation_features].corr()
    
    print("\nCorrelation with AQI:")
    if target_col in corr_matrix.columns:
        correlations = corr_matrix[target_col].sort_values(ascending=False)
        correlations = correlations[correlations.index != target_col]
        print(correlations)
    
    if save_plots:
        # Correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix of Features', fontsize=16, pad=20)
        plt.tight_layout()
        os.makedirs('notebooks/plots', exist_ok=True)
        plt.savefig('notebooks/plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\n✓ Saved plot: notebooks/plots/correlation_heatmap.png")
        
        # Correlation bar chart
        if target_col in corr_matrix.columns:
            correlations = corr_matrix[target_col].sort_values(ascending=False)
            correlations = correlations[correlations.index != target_col]
            
            plt.figure(figsize=(10, 6))
            correlations.plot(kind='barh', color='steelblue')
            plt.title('Correlation of Features with AQI', fontsize=14)
            plt.xlabel('Correlation Coefficient')
            plt.ylabel('Features')
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            plt.tight_layout()
            plt.savefig('notebooks/plots/correlation_with_target.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved plot: notebooks/plots/correlation_with_target.png")
    
    print("="*60 + "\n")


def time_series_analysis(df, timestamp_col='timestamp', target_col='aqi', save_plots=True):
    """Analyze time series patterns."""
    print("\n" + "="*60)
    print("TIME SERIES ANALYSIS")
    print("="*60)
    
    if timestamp_col not in df.columns:
        print(f"❌ Timestamp column '{timestamp_col}' not found!")
        return
    
    # Convert timestamp to datetime
    if df[timestamp_col].dtype == 'object' or 'datetime' in str(df[timestamp_col].dtype):
        df['datetime'] = pd.to_datetime(df[timestamp_col])
    else:
        df['datetime'] = pd.to_datetime(df[timestamp_col], unit='ms')
    
    df_sorted = df.sort_values('datetime')
    
    if save_plots:
        # AQI over time
        plt.figure(figsize=(15, 6))
        plt.plot(df_sorted['datetime'], df_sorted[target_col], marker='o', markersize=3, linewidth=1)
        plt.title('AQI Over Time', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('AQI')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs('notebooks/plots', exist_ok=True)
        plt.savefig('notebooks/plots/aqi_time_series.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved plot: notebooks/plots/aqi_time_series.png")
    
    print("="*60 + "\n")


def feature_relationships(df, pollutant_cols=None, target_col='aqi', save_plots=True):
    """Analyze relationships between features and target."""
    print("\n" + "="*60)
    print("FEATURE RELATIONSHIPS WITH TARGET")
    print("="*60)
    
    if pollutant_cols is None:
        pollutant_cols = ['pm25', 'pm10', 'no2', 'so2', 'o3']
    
    pollutants = [p for p in pollutant_cols if p in df.columns]
    
    if len(pollutants) == 0:
        print("❌ No pollutant columns found!")
        return
    
    if save_plots:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for i, pollutant in enumerate(pollutants):
            axes[i].scatter(df[pollutant], df[target_col], alpha=0.6, s=50)
            axes[i].set_xlabel(pollutant.upper())
            axes[i].set_ylabel('AQI')
            axes[i].set_title(f'{pollutant.upper()} vs AQI')
            
            # Add trend line
            z = np.polyfit(df[pollutant], df[target_col], 1)
            p = np.poly1d(z)
            axes[i].plot(df[pollutant], p(df[pollutant]), "r--", alpha=0.8, linewidth=2)
            axes[i].grid(True, alpha=0.3)
        
        # Hide extra subplots
        for i in range(len(pollutants), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        os.makedirs('notebooks/plots', exist_ok=True)
        plt.savefig('notebooks/plots/feature_relationships.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved plot: notebooks/plots/feature_relationships.png")
    
    print("="*60 + "\n")


def run_full_eda(df=None, save_plots=True):
    """Run complete EDA pipeline."""
    print("\n" + "="*80)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*80)
    
    # Load data if not provided
    if df is None:
        df = load_data_from_hopsworks()
    
    # Run all analyses
    data_overview(df)
    missing_values_analysis(df)
    target_variable_analysis(df, save_plots=save_plots)
    feature_distributions(df, save_plots=save_plots)
    correlation_analysis(df, save_plots=save_plots)
    time_series_analysis(df, save_plots=save_plots)
    feature_relationships(df, save_plots=save_plots)
    
    print("\n" + "="*80)
    print("EDA COMPLETED!")
    print("="*80 + "\n")
    
    return df


if __name__ == "__main__":
    # Run full EDA
    df = run_full_eda(save_plots=True)
    
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
