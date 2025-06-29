Collecting workspace information# Journey Performance Analysis Dashboard

## Overview

This project provides a comprehensive analytics dashboard for analyzing marketing campaign performance across different channels and customer journeys. The dashboard processes campaign data from CSV files and presents interactive visualizations and metrics to help optimize marketing strategies.

## Features

### Core Analytics
- **ROI Performance Analysis**: Calculate and visualize return on investment for each marketing journey
- **Channel Performance Comparison**: Compare effectiveness across Email, Push, SMS, and WhatsApp channels
- **Conversion Funnel Analysis**: Track user progression from message sent to conversion
- **Cost Efficiency Analysis**: Analyze cost per conversion and identify optimal campaigns
- **Financial Metrics**: Calculate GTV (Gross Transaction Value), costs, and profitability

### Interactive Dashboard
- **Real-time Filtering**: Filter data by journey, channel, status, user type, and date range
- **Dynamic KPI Cards**: Live updating metrics based on applied filters
- **Collapsible Sections**: Organized interface with expandable content areas
- **Data Export**: Export filtered results to Excel format
- **Responsive Charts**: Interactive Plotly visualizations with hover details

### Data Processing
- **Automated Calculations**: Delivery rates, click-through rates, conversion rates
- **User Segmentation**: Categorize users by type (ALL, NU, RU) and channel (ALL, OM, OA, ixigo)
- **Cost Attribution**: Calculate costs based on channel-specific pricing
- **Performance Benchmarking**: Identify top and bottom performers

## Technology Stack

### Backend
- **Python 3.12+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations

### Visualization
- **Plotly**: Interactive charts and graphs
- **Dash**: Web application framework
- **Dash Bootstrap Components**: UI components and styling

### Data Processing
- **CSV Processing**: Load and clean campaign data
- **Real-time Calculations**: Dynamic metric computation
- **Data Validation**: Handle missing values and data inconsistencies

## File Structure

```
JourneyPerformanceAnalysis/
├── test.py                 # Main dashboard application
├── reportjun27.csv         # Sample campaign data
├── Codes/
│   ├── auto.ipynb         # Jupyter notebook prototypes
│   └── report_generator.ipynb
```

## Installation and Setup

### Prerequisites
- Python 3.12 or higher
- pip package manager

### Required Dependencies
```bash
pip install pandas numpy plotly dash dash-bootstrap-components
```

### Configuration
1. Update the file path in test.py:
   ```python
   file_path1 = "your_data_file.csv"
   ```

2. Adjust configuration variables as needed:
   ```python
   AOV = 1500  # Average Order Value
   TAKE_RATE = 0.08  # Platform take rate
   CHANNEL_COSTS = {
       'Email': 0.02,
       'Push': 0.01, 
       'SMS': 0.11,
       'WhatsApp': 0.11
   }
   ```

## Usage

### Running the Dashboard
```bash
python test.py
```

### Access Points
- **Local Access**: http://127.0.0.1:8058
- **Network Access**: http://[your-ip]:8058 (for sharing across network)

### Data Requirements

Your CSV file should contain these columns:
- `Campaign ID`: Unique campaign identifier
- `Journey Name`: Marketing journey name
- `Channel`: Communication channel (Email/Push/SMS/WhatsApp)
- `Status`: Campaign status
- `Sent`: Number of messages sent
- `Delivered`: Number of messages delivered
- `Unique Impressions`: Unique message impressions
- `Unique Clicks`: Unique click events
- `Unique Click-Through Conversions`: Conversion events

## Dashboard Sections

### 1. Filter Controls
- Journey selection with bulk actions
- Channel filtering
- Status and date range filters
- User type and user channel filters
- Quick filter buttons for common scenarios

### 2. KPI Cards
- Total messages sent and delivered
- Click and conversion metrics
- Cost and revenue summaries
- Overall ROI calculation

### 3. Summary Table
- Comprehensive journey performance metrics
- Sortable and filterable data table
- Export functionality
- Calculated fields for rates and ratios

### 4. Performance Charts
- **ROI Performance**: Top performing journeys visualization
- **Cost vs Revenue**: Scatter plot with break-even analysis
- **Channel ROI Trend**: Line chart comparing channel performance
- **Conversion Funnel**: Volume drop-off analysis
- **Efficiency Curve**: Cost per conversion optimization
- **Quadrant Analysis**: Investment vs returns mapping

### 5. Channel Breakdown
- Channel-wise performance breakdown
- Cross-tabulation of journeys and channels
- Aggregated metrics by communication channel

### 6. Campaign Details
- Individual campaign performance
- Detailed metrics per campaign
- Conversion rate calculations

## Key Metrics Calculated

### Performance Metrics
- **Delivery Rate**: (Delivered / Sent) × 100
- **Click-Through Rate**: (Unique Clicks / Delivered) × 100
- **Conversion Rate**: (Conversions / Clicks) × 100
- **Overall Conversion Rate**: (Conversions / Sent) × 100

### Financial Metrics
- **Cost**: Based on channel costs and message volume
- **GTV**: Conversions × Average Order Value
- **ROI**: GTV / Cost
- **ROI with Take Rate**: (GTV × (1 - Take Rate)) / Cost
- **Cost per Conversion**: Cost / Conversions

### Efficiency Metrics
- **Cost per Message**: Total Cost / Messages Sent
- **Revenue per Conversion**: GTV / Conversions
- **Profit Margin**: (GTV - Cost) / GTV × 100

## User Segmentation

### User Types
- **ALL**: General user campaigns
- **NU**: New user focused campaigns
- **RU**: Returning user campaigns

### User Channels
- **ALL**: Multi-channel campaigns
- **OM**: Online Mobile campaigns
- **OA**: Online Apparel campaigns
- **ixigo**: Partner-specific campaigns

## Output and Results

### Performance Insights
- Identify highest ROI campaigns
- Compare channel effectiveness
- Analyze conversion funnel bottlenecks
- Optimize cost efficiency

### Business Intelligence
- Campaign profitability analysis
- Channel allocation recommendations
- User segment performance comparison
- Cost optimization opportunities

### Reporting
- Exportable data tables
- Interactive visualizations
- Real-time metric updates
- Filterable performance views

## Troubleshooting

### Common Issues
1. **Data Loading Errors**: Verify CSV format and column names
2. **Missing Values**: Check for empty cells in required columns
3. **Port Conflicts**: Change port number if 8058 is in use
4. **Network Access**: Ensure firewall allows connections on specified port

### Performance Optimization
- Filter large datasets before analysis
- Use date ranges to limit data scope
- Export filtered results for detailed analysis
- Close unused browser tabs to improve performance

## Development Notes

The application uses a modular architecture with separate classes for data analysis and dashboard presentation. The main components include:

- `CampaignAnalyzer`: Data processing and metric calculations
- Dashboard layout and callbacks for interactivity
- Chart generation using Plotly
- Real-time filtering and data updates

For development and customization, refer to the Jupyter notebooks in the Codes directory which contain prototype implementations and testing scenarios.