# US Industrial Production Estimation System

A comprehensive system for estimating missing US Industrial Production data using multiple professional forecast sources and advanced statistical techniques.

## Project Overview

This project provides an advanced multi-stage estimation system for handling missing US Industrial Production data using professional forecasts and sophisticated interpolation techniques.

## Quick Start

### Prerequisites
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

### Running the System
```bash
python src/estimate_us_ip.py
```

### Dependencies
The project uses the following Python packages (see `requirements.txt` for versions):
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scipy** - Scientific computing (cubic spline interpolation)
- **matplotlib** - Data visualization
- **Additional dependencies** - Supporting packages for the above

## Project Structure

```
industrial_production/
├── src/
│   └── estimate_us_ip.py          # Main estimation system
├── tests/
│   └── test_estimate_us_ip.py     # Unit tests for estimation system
├── inputs/
│   └── ip_raw.csv                 # Raw industrial production data
├── outputs/
│   ├── ip_actual_vs_estimated.csv # Output from estimation system
│   ├── ip_filled.csv              # Additional output file
│   └── ip_forecasts_vs_actual.png # Visualization
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
├── .venv/                         # Virtual environment
└── README.md                      # This file
```

## Estimation System (`estimate_us_ip.py`)

### Method
- **Multi-stage estimation with interpolation**
- **Approach**: 
  - 3-month moving average filling
  - Cubic spline interpolation
  - Forecast-based extrapolation
- **Use Case**: Complete time series with smooth transitions
- **Output**: `outputs/ip_actual_vs_estimated.csv`

## Data Sources

The system utilizes three professional forecast sources:
- **BlueChip Forecast** (Consensus)
- **Oxford Forecast** (Professional Economic Analysis)  
- **MA4CAST Forecast** (Model-based)

## Key Features

### Data Processing
- Month-end date conversion
- String-to-numeric conversion
- Missing value handling
- Data validation

### Estimation Techniques
- Moving averages
- Cubic spline interpolation
- Forecast combination
- Extrapolation methods

### Output Formats
- CSV file for analysis (`ip_actual_vs_estimated.csv`)
- PNG visualization (`ip_forecasts_vs_actual.png`)
- Comprehensive reporting

## Usage Examples

### Basic Analysis
```python
from src.estimate_us_ip import read_raw_data, create_est_ip_with_extrapolation

# Load and process data
data = read_raw_data()
final_data = create_est_ip_with_extrapolation(data)

# Access the complete time series
est_ip_series = final_data['Est_IP']
```

### Custom Processing
```python
from src.estimate_us_ip import fill_forecast_missing_values, fill_forecast_with_spline

# Step-by-step processing
data = read_raw_data()
filled_ma = fill_forecast_missing_values(data)
filled_spline = fill_forecast_with_spline(filled_ma)
```

## Testing

The system includes comprehensive unit tests to ensure data quality and processing reliability.

### Running Tests
```bash
# Run all unit tests
python tests/test_estimate_us_ip.py

# Run specific test class
python -m unittest tests.test_estimate_us_ip.TestReadRawData

# Run specific test method
python -m unittest tests.test_estimate_us_ip.TestReadRawData.test_read_raw_data_basic
```

### Test Coverage
- **17 test cases** covering all functions
- **5 test classes** for different components
- **Integration tests** for complete workflow
- **Edge case handling** for robust operation

## Documentation

- **README.md** - This comprehensive overview file

## Output Files

### Estimation System Output
- **File**: `outputs/ip_actual_vs_estimated.csv`
- **Content**: Date, Actual_IP, Est_IP columns
- **Method**: Multi-stage estimation with interpolation

### Visualizations
- **File**: `outputs/ip_forecasts_vs_actual.png`
- **Content**: All series plotted together
- **Purpose**: Visual validation and comparison

## Performance Metrics

### Data Coverage
- **Historical Period**: 100% coverage (1921-2024)
- **Future Projections**: Complete through 2050
- **Missing Values**: 0 in final Est_IP column

### Forecast Filling
- **Oxford Forecast**: 99.8% filled
- **MA4CAST Forecast**: 79% filled  
- **BlueChip Forecast**: 81% filled

## Applications

### Economic Research
- Time series analysis
- Trend identification
- Historical comparison

### Financial Modeling
- Economic scenario planning
- Risk assessment
- Portfolio optimization

### Policy Analysis
- Economic indicator tracking
- Policy impact assessment
- Forecasting support

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is provided as-is for research and educational purposes.

## Contact

For questions or support, please refer to the documentation or create an issue in the project repository.

---

*This project provides robust, well-documented tools for handling missing industrial production data using professional forecasts and advanced statistical techniques.*
