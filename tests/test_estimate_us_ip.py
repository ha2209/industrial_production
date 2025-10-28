#!/usr/bin/env python3
"""
Unit tests for estimate_us_ip.py module
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
from unittest.mock import patch, MagicMock

# Add src directory to path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from estimate_us_ip import (
    read_raw_data,
    fill_forecast_missing_values,
    fill_forecast_with_spline,
    create_est_ip_with_extrapolation
)


class TestReadRawData(unittest.TestCase):
    """Test cases for read_raw_data function"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_data = {
            'Date': ['JAN 2020', 'FEB 2020', 'MAR 2020', 'APR 2020'],
            'Actual_IP': ['100.0', '101.0', '---', '103.0'],
            'BlueChip_Forecast': ['---', '---', '102.0', '104.0'],
            'Oxford_Forecast': ['---', '---', '101.5', '103.5'],
            'MA4CAST_Forecast': ['---', '---', '102.5', '104.5']
        }
        self.test_df = pd.DataFrame(self.sample_data)
    
    @patch('pandas.read_csv')
    def test_read_raw_data_basic(self, mock_read_csv):
        """Test basic functionality of read_raw_data"""
        # Mock the CSV reading
        mock_read_csv.return_value = self.test_df
        
        result = read_raw_data()
        
        # Check that read_csv was called with correct parameters
        mock_read_csv.assert_called_once_with('inputs/ip_raw.csv', skiprows=4)
        
        # Check column names
        expected_columns = ['Date', 'Actual_IP', 'BlueChip_Forecast', 'Oxford_Forecast', 'MA4CAST_Forecast']
        self.assertEqual(list(result.columns), expected_columns)
        
        # Check that dates are converted to month-end
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result['Date']))
        
        # Check that '---' values are converted to NaN
        self.assertTrue(pd.isna(result['Actual_IP'].iloc[2]))  # '---' should become NaN
        
        # Check that numeric values are properly converted
        self.assertEqual(result['Actual_IP'].iloc[0], 100.0)
        self.assertEqual(result['Actual_IP'].iloc[1], 101.0)
    
    @patch('pandas.read_csv')
    def test_read_raw_data_date_conversion(self, mock_read_csv):
        """Test date conversion to month-end"""
        mock_read_csv.return_value = self.test_df
        
        result = read_raw_data()
        
        # Check that dates are month-end
        expected_dates = [
            pd.Timestamp('2020-01-31'),
            pd.Timestamp('2020-02-29'),
            pd.Timestamp('2020-03-31'),
            pd.Timestamp('2020-04-30')
        ]
        
        for i, expected_date in enumerate(expected_dates):
            self.assertEqual(result['Date'].iloc[i], expected_date)
    
    @patch('pandas.read_csv')
    def test_read_raw_data_sorting(self, mock_read_csv):
        """Test that data is sorted by date"""
        # Create unsorted data
        unsorted_data = {
            'Date': ['MAR 2020', 'JAN 2020', 'APR 2020', 'FEB 2020'],
            'Actual_IP': ['102.0', '100.0', '103.0', '101.0'],
            'BlueChip_Forecast': ['---', '---', '---', '---'],
            'Oxford_Forecast': ['---', '---', '---', '---'],
            'MA4CAST_Forecast': ['---', '---', '---', '---']
        }
        mock_read_csv.return_value = pd.DataFrame(unsorted_data)
        
        result = read_raw_data()
        
        # Check that data is sorted by date
        self.assertTrue(result['Date'].is_monotonic_increasing)


class TestFillForecastMissingValues(unittest.TestCase):
    """Test cases for fill_forecast_missing_values function"""
    
    def setUp(self):
        """Set up test data"""
        self.test_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-31', periods=6, freq='ME'),
            'Actual_IP': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            'BlueChip_Forecast': [np.nan, np.nan, 102.5, np.nan, np.nan, np.nan],
            'Oxford_Forecast': [np.nan, np.nan, 101.5, np.nan, np.nan, np.nan],
            'MA4CAST_Forecast': [np.nan, np.nan, 102.8, np.nan, np.nan, np.nan]
        })
    
    def test_fill_forecast_missing_values_basic(self):
        """Test basic functionality of fill_forecast_missing_values"""
        result = fill_forecast_missing_values(self.test_data)
        
        # Check that forecast columns are filled
        forecast_cols = ['BlueChip_Forecast', 'Oxford_Forecast', 'MA4CAST_Forecast']
        
        for col in forecast_cols:
            # First two values should be NaN (not enough data for 3-month MA)
            self.assertTrue(pd.isna(result[col].iloc[0]))
            self.assertTrue(pd.isna(result[col].iloc[1]))
            
            # Third value should be filled with 3-month MA
            expected_ma = (100.0 + 101.0 + 102.0) / 3  # 101.0
            self.assertAlmostEqual(result[col].iloc[2], expected_ma, places=5)
            
            # Fourth value should be filled with 3-month MA
            expected_ma = (101.0 + 102.0 + 103.0) / 3  # 102.0
            self.assertAlmostEqual(result[col].iloc[3], expected_ma, places=5)
    
    def test_fill_forecast_missing_values_no_modification(self):
        """Test that original data is not modified"""
        original_data = self.test_data.copy()
        fill_forecast_missing_values(self.test_data)
        
        # Check that original data is unchanged
        pd.testing.assert_frame_equal(original_data, self.test_data)
    
    def test_fill_forecast_missing_values_insufficient_data(self):
        """Test behavior with insufficient data for moving average"""
        # Create data with only 2 rows
        short_data = self.test_data.head(2).copy()
        result = fill_forecast_missing_values(short_data)
        
        # All forecast values should remain NaN
        forecast_cols = ['BlueChip_Forecast', 'Oxford_Forecast', 'MA4CAST_Forecast']
        for col in forecast_cols:
            self.assertTrue(pd.isna(result[col].iloc[0]))
            self.assertTrue(pd.isna(result[col].iloc[1]))


class TestFillForecastWithSpline(unittest.TestCase):
    """Test cases for fill_forecast_with_spline function"""
    
    def setUp(self):
        """Set up test data"""
        self.test_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-31', periods=6, freq='ME'),
            'Actual_IP': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            'BlueChip_Forecast': [100.5, np.nan, 102.5, np.nan, 104.5, np.nan],
            'Oxford_Forecast': [100.2, np.nan, 101.8, np.nan, 104.2, np.nan],
            'MA4CAST_Forecast': [100.8, np.nan, 102.8, np.nan, 104.8, np.nan]
        })
    
    def test_fill_forecast_with_spline_basic(self):
        """Test basic functionality of fill_forecast_with_spline"""
        result = fill_forecast_with_spline(self.test_data)
        
        # Check that missing values are filled
        forecast_cols = ['BlueChip_Forecast', 'Oxford_Forecast', 'MA4CAST_Forecast']
        
        for col in forecast_cols:
            # Should have fewer NaN values after spline interpolation
            original_nans = self.test_data[col].isna().sum()
            result_nans = result[col].isna().sum()
            self.assertLessEqual(result_nans, original_nans)
    
    def test_fill_forecast_with_spline_no_extrapolation(self):
        """Test that spline interpolation doesn't extrapolate beyond data range"""
        result = fill_forecast_with_spline(self.test_data)
        
        # Check that values at the end remain NaN (no extrapolation)
        forecast_cols = ['BlueChip_Forecast', 'Oxford_Forecast', 'MA4CAST_Forecast']
        
        for col in forecast_cols:
            # Last value should remain NaN (no extrapolation)
            self.assertTrue(pd.isna(result[col].iloc[-1]))
    
    def test_fill_forecast_with_spline_insufficient_data(self):
        """Test behavior with insufficient data for interpolation"""
        # Create data with only 1 non-null value
        insufficient_data = self.test_data.copy()
        insufficient_data['BlueChip_Forecast'] = [100.5, np.nan, np.nan, np.nan, np.nan, np.nan]
        
        result = fill_forecast_with_spline(insufficient_data)
        
        # Should skip interpolation and leave values as NaN
        self.assertTrue(pd.isna(result['BlueChip_Forecast'].iloc[1]))
    
    def test_fill_forecast_with_spline_no_modification(self):
        """Test that original data is not modified"""
        original_data = self.test_data.copy()
        fill_forecast_with_spline(self.test_data)
        
        # Check that original data is unchanged
        pd.testing.assert_frame_equal(original_data, self.test_data)


class TestCreateEstIpWithExtrapolation(unittest.TestCase):
    """Test cases for create_est_ip_with_extrapolation function"""
    
    def setUp(self):
        """Set up test data"""
        self.test_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-31', periods=6, freq='ME'),
            'Actual_IP': [100.0, 101.0, 102.0, np.nan, np.nan, np.nan],
            'BlueChip_Forecast': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
            'Oxford_Forecast': [100.2, 101.2, 102.2, 103.2, 104.2, 105.2],
            'MA4CAST_Forecast': [100.8, 101.8, 102.8, 103.8, 104.8, 105.8]
        })
    
    def test_create_est_ip_with_extrapolation_basic(self):
        """Test basic functionality of create_est_ip_with_extrapolation"""
        result = create_est_ip_with_extrapolation(self.test_data)
        
        # Check that Est_IP column is created
        self.assertIn('Est_IP', result.columns)
        
        # Check that historical values match Actual_IP
        self.assertEqual(result['Est_IP'].iloc[0], 100.0)
        self.assertEqual(result['Est_IP'].iloc[1], 101.0)
        self.assertEqual(result['Est_IP'].iloc[2], 102.0)
        
        # Check that future values are filled with forecast averages
        for i in range(3, 6):
            expected_avg = np.mean([
                result['BlueChip_Forecast'].iloc[i],
                result['Oxford_Forecast'].iloc[i],
                result['MA4CAST_Forecast'].iloc[i]
            ])
            self.assertAlmostEqual(result['Est_IP'].iloc[i], expected_avg, places=5)
    
    def test_create_est_ip_with_extrapolation_no_actual_data(self):
        """Test behavior when no actual data is available"""
        no_actual_data = self.test_data.copy()
        no_actual_data['Actual_IP'] = np.nan
        
        result = create_est_ip_with_extrapolation(no_actual_data)
        
        # Should return data unchanged
        self.assertIn('Est_IP', result.columns)
        self.assertTrue(pd.isna(result['Est_IP'].iloc[0]))
    
    def test_create_est_ip_with_extrapolation_missing_forecasts(self):
        """Test behavior when some forecasts are missing"""
        missing_forecast_data = self.test_data.copy()
        missing_forecast_data['BlueChip_Forecast'] = [100.5, 101.5, 102.5, np.nan, np.nan, np.nan]
        
        result = create_est_ip_with_extrapolation(missing_forecast_data)
        
        # Should use available forecasts for averaging
        for i in range(3, 6):
            expected_avg = np.mean([
                result['Oxford_Forecast'].iloc[i],
                result['MA4CAST_Forecast'].iloc[i]
            ])
            self.assertAlmostEqual(result['Est_IP'].iloc[i], expected_avg, places=5)
    
    def test_create_est_ip_with_extrapolation_no_forecasts(self):
        """Test behavior when no forecasts are available"""
        no_forecast_data = self.test_data.copy()
        no_forecast_data['BlueChip_Forecast'] = np.nan
        no_forecast_data['Oxford_Forecast'] = np.nan
        no_forecast_data['MA4CAST_Forecast'] = np.nan
        
        result = create_est_ip_with_extrapolation(no_forecast_data)
        
        # Should use last known Est_IP value
        for i in range(3, 6):
            self.assertEqual(result['Est_IP'].iloc[i], 102.0)  # Last actual value
    
    def test_create_est_ip_with_extrapolation_no_modification(self):
        """Test that original data is not modified"""
        original_data = self.test_data.copy()
        create_est_ip_with_extrapolation(self.test_data)
        
        # Check that original data is unchanged
        pd.testing.assert_frame_equal(original_data, self.test_data)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-31', periods=10, freq='ME'),
            'Actual_IP': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            'BlueChip_Forecast': [np.nan, np.nan, np.nan, np.nan, np.nan, 105.5, 106.5, 107.5, 108.5, 109.5],
            'Oxford_Forecast': [np.nan, np.nan, np.nan, np.nan, np.nan, 105.2, 106.2, 107.2, 108.2, 109.2],
            'MA4CAST_Forecast': [np.nan, np.nan, np.nan, np.nan, np.nan, 105.8, 106.8, 107.8, 108.8, 109.8]
        })
    
    def test_complete_workflow(self):
        """Test the complete workflow from raw data to final output"""
        # Step 1: Fill with moving average
        filled_ma = fill_forecast_missing_values(self.sample_data)
        
        # Step 2: Fill with spline interpolation
        filled_spline = fill_forecast_with_spline(filled_ma)
        
        # Step 3: Create Est_IP with extrapolation
        final_result = create_est_ip_with_extrapolation(filled_spline)
        
        # Check that Est_IP column exists
        self.assertIn('Est_IP', final_result.columns)
        
        # Check that historical values match Actual_IP
        for i in range(5):  # First 5 values should match
            self.assertEqual(final_result['Est_IP'].iloc[i], final_result['Actual_IP'].iloc[i])
        
        # Check that future values are filled
        for i in range(5, 10):  # Last 5 values should be filled
            self.assertFalse(pd.isna(final_result['Est_IP'].iloc[i]))
    
    def test_data_integrity(self):
        """Test that data integrity is maintained throughout the workflow"""
        original_shape = self.sample_data.shape
        
        # Run complete workflow
        filled_ma = fill_forecast_missing_values(self.sample_data)
        filled_spline = fill_forecast_with_spline(filled_ma)
        final_result = create_est_ip_with_extrapolation(filled_spline)
        
        # Check that shape is maintained
        self.assertEqual(final_result.shape[0], original_shape[0])
        self.assertEqual(final_result.shape[1], original_shape[1] + 1)  # +1 for Est_IP column
        
        # Check that original columns are preserved
        original_columns = list(self.sample_data.columns)
        for col in original_columns:
            self.assertIn(col, final_result.columns)


if __name__ == '__main__':
    # Create test suite using TestLoader
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTests(loader.loadTestsFromTestCase(TestReadRawData))
    test_suite.addTests(loader.loadTestsFromTestCase(TestFillForecastMissingValues))
    test_suite.addTests(loader.loadTestsFromTestCase(TestFillForecastWithSpline))
    test_suite.addTests(loader.loadTestsFromTestCase(TestCreateEstIpWithExtrapolation))
    test_suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
