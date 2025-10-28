import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from datetime import datetime

# Function to read the raw data from inputs/ip_raw.csv and return a cleaned-up dataframe
def read_raw_data():
    df = pd.read_csv('inputs/ip_raw.csv', skiprows=4)
    df.columns = ['Date', 'Actual_IP', 'BlueChip_Forecast', 'Oxford_Forecast', 'MA4CAST_Forecast']
    
    # Convert date column to month-end dates
    df['Date'] = pd.to_datetime(df['Date'], format='%b %Y')
    # Convert to month-end by adding one month and subtracting one day
    df['Date'] = df['Date'] + pd.offsets.MonthEnd(0)
    
    # Convert numeric columns, replacing '---' with NaN
    numeric_cols = ['Actual_IP', 'BlueChip_Forecast', 'Oxford_Forecast', 'MA4CAST_Forecast']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.sort_values('Date')
    return df


def fill_forecast_missing_values(df):
    """
    Fill missing values in columns ending with 'Forecast' using three-month 
    simple moving average of Actual_IP column.
    
    Parameters:
    df (pd.DataFrame): DataFrame with Date, Actual_IP, and forecast columns
    
    Returns:
    pd.DataFrame: DataFrame with filled forecast values
    """
    # Create a copy to avoid modifying the original
    df_filled = df.copy()
    
    # Calculate 3-month simple moving average of Actual_IP
    df_filled['Actual_IP_3MA'] = df_filled['Actual_IP'].rolling(window=3, min_periods=3).mean()
    
    # Find columns ending with 'Forecast'
    forecast_columns = [col for col in df_filled.columns if col.endswith('Forecast')]
    
    # Fill missing values in forecast columns with the 3-month moving average
    for col in forecast_columns:
        # Find index of last non-null value in Actual_IP column
        last_non_null_idx = df_filled['Actual_IP'].last_valid_index()
        # Fill all values up to last_non_null_idx with the 3-month moving average
        df_filled.loc[:last_non_null_idx, col] = df_filled.loc[:last_non_null_idx, 'Actual_IP_3MA']
    
    # Drop the temporary moving average column
    df_filled = df_filled.drop('Actual_IP_3MA', axis=1)
    
    return df_filled


def fill_forecast_with_spline(df):
    """
    Fill remaining missing values in forecast columns using cubic spline interpolation.
    Does not extrapolate beyond the last available data point.
    
    Parameters:
    df (pd.DataFrame): DataFrame with Date, Actual_IP, and forecast columns
    
    Returns:
    pd.DataFrame: DataFrame with filled forecast values using spline interpolation
    """
    # Create a copy to avoid modifying the original
    df_filled = df.copy()
    
    # Find columns ending with 'Forecast'
    forecast_columns = [col for col in df_filled.columns if col.endswith('Forecast')]
    
    # Create a numeric index for interpolation
    df_filled['index'] = range(len(df_filled))
    
    for col in forecast_columns:
        # Get non-null values for this column
        valid_mask = df_filled[col].notna()
        
        if valid_mask.sum() < 2:  # Need at least 2 points for interpolation
            print(f"Warning: Not enough data points for {col} interpolation. Skipping.")
            continue
            
        # Get valid indices and values
        valid_indices = df_filled.loc[valid_mask, 'index'].values
        valid_values = df_filled.loc[valid_mask, col].values
        
        # Find the range of valid data (don't extrapolate beyond last data point)
        first_valid_idx = valid_indices.min()
        last_valid_idx = valid_indices.max()
        
        # Only interpolate within the range of valid data
        interpolate_mask = (df_filled['index'] >= first_valid_idx) & (df_filled['index'] <= last_valid_idx)
        
        if interpolate_mask.sum() == 0:
            continue
            
        # Get indices where we need to interpolate
        interpolate_indices = df_filled.loc[interpolate_mask & df_filled[col].isna(), 'index'].values
        
        if len(interpolate_indices) == 0:
            continue
            
        try:
            # Create cubic spline interpolator
            cs = CubicSpline(valid_indices, valid_values)
            
            # Interpolate only at the missing indices within valid range
            interpolated_values = cs(interpolate_indices)
            
            # Fill the missing values
            df_filled.loc[df_filled['index'].isin(interpolate_indices), col] = interpolated_values
            
        except Exception as e:
            print(f"Warning: Could not interpolate {col}: {e}")
            continue
    
    # Drop the temporary index column
    df_filled = df_filled.drop('index', axis=1)
    
    return df_filled


def create_est_ip_with_extrapolation(df):
    """
    Create Est_IP column by copying Actual_IP values and extrapolating future values
    based on the assumption that the average of three forecasts equals the 3-month 
    moving average of Est_IP values.
    
    Parameters:
    df (pd.DataFrame): DataFrame with Date, Actual_IP, and forecast columns
    
    Returns:
    pd.DataFrame: DataFrame with Est_IP column added and extrapolated
    """
    # Create a copy to avoid modifying the original
    df_est = df.copy()
    
    # Create Est_IP column by copying Actual_IP values
    df_est['Est_IP'] = df_est['Actual_IP'].copy()
    
    # Find the last non-null Actual_IP value
    last_actual_idx = df_est['Actual_IP'].last_valid_index()
    
    if last_actual_idx is None:
        print("Warning: No actual IP data found. Cannot extrapolate.")
        return df_est
    
    # Get forecast columns
    forecast_columns = [col for col in df_est.columns if col.endswith('Forecast')]
    
    # For each future period, calculate Est_IP based on forecast average
    for i in range(last_actual_idx + 1, len(df_est)):
        # Calculate average of available forecasts for this period
        forecast_values = []
        for col in forecast_columns:
            if pd.notna(df_est.loc[i, col]):
                forecast_values.append(df_est.loc[i, col])
        
        if len(forecast_values) == 0:
            # If no forecasts available, use the last known Est_IP value
            df_est.loc[i, 'Est_IP'] = df_est.loc[i-1, 'Est_IP']
        else:
            # Use average of available forecasts
            avg_forecast = np.mean(forecast_values)
            df_est.loc[i, 'Est_IP'] = avg_forecast
    
    return df_est


# Main function to run the script
if __name__ == "__main__":
    # Load raw data
    data = read_raw_data()
    print("Original data shape:", data.shape)
    print("Missing values in forecast columns:")
    forecast_cols = [col for col in data.columns if col.endswith('Forecast')]
    print(data[forecast_cols].isnull().sum())
    
    # Fill missing forecast values with 3-month moving average
    filled_data_ma = fill_forecast_missing_values(data)
    print("\nAfter filling with 3-month moving average:")
    print("Missing values in forecast columns:")
    print(filled_data_ma[forecast_cols].isnull().sum())
    
    # Fill remaining missing values with cubic spline interpolation
    filled_data_spline = fill_forecast_with_spline(filled_data_ma)
    print("\nAfter filling remaining values with cubic spline interpolation:")
    print("Missing values in forecast columns:")
    print(filled_data_spline[forecast_cols].isnull().sum())
    
    # Create Est_IP column with extrapolation
    final_data = create_est_ip_with_extrapolation(filled_data_spline)
    print("\nAfter creating Est_IP column with extrapolation:")
    print("Est_IP missing values:", final_data['Est_IP'].isnull().sum())
    
    # Show sample of final data
    print("\nSample of final data (first 10 rows):")
    print(final_data.head(10))
    print("\nSample of final data (last 10 rows):")
    print(final_data.tail(10))
    
    # Plot forecast columns and Est_IP starting from last year to next two years
    today = datetime.now()
    start_date = today - pd.DateOffset(years=1)
    end_date = today + pd.DateOffset(years=2)
    
    # Filter data for plotting
    plot_data = final_data[(final_data['Date'] >= start_date) & (final_data['Date'] <= end_date)]
    
    plt.figure(figsize=(12, 8))
    plt.plot(plot_data['Date'], plot_data['BlueChip_Forecast'], label='BlueChip Forecast', alpha=0.7)
    plt.plot(plot_data['Date'], plot_data['Oxford_Forecast'], label='Oxford Forecast', alpha=0.7)
    plt.plot(plot_data['Date'], plot_data['MA4CAST_Forecast'], label='MA4CAST Forecast', alpha=0.7)
    plt.plot(plot_data['Date'], plot_data['Est_IP'], label='Est_IP (Extrapolated)', linewidth=2, color='red')
    plt.plot(plot_data['Date'], plot_data['Actual_IP'], label='Actual_IP', linewidth=2, color='black')
    
    plt.title('US Industrial Production: Forecasts vs Est_IP vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Industrial Production Index')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot to PNG
    plot_file = 'outputs/ip_forecasts_vs_actual.png'
    plt.savefig(plot_file)
    print(f"\nPlot saved to: {plot_file}")

    # Save Actual_IP and Est_IP columns to CSV
    output_data = final_data[['Date', 'Actual_IP', 'Est_IP']].copy()
    output_file = 'outputs/ip_actual_vs_estimated.csv'
    output_data.to_csv(output_file, index=False)
    print(f"\nActual_IP and Est_IP data saved to: {output_file}")
    print(f"Output data shape: {output_data.shape}")
    print(f"Output data columns: {list(output_data.columns)}")