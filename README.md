# Journey Performance Analysis Dashboard

## Overview

This project provides a comprehensive analytics dashboard for analyzing marketing campaign performance across different channels and customer journeys. Built with a dark modern UI theme featuring glassmorphism effects and gradient accents, the dashboard processes campaign data from CSV files and presents interactive visualizations, metrics, and downloadable reports.

## Features

### Core Analytics
- **ROI Performance Analysis**: Calculate and visualize return on investment for each marketing journey
- **Channel Performance Comparison**: Compare effectiveness across Email, Push, SMS, and WhatsApp channels
- **Conversion Funnel Analysis**: Track user progression from message sent to conversion
- **Cost Efficiency Analysis**: Analyze cost per conversion and identify optimal campaigns
- **Financial Metrics**: Calculate GTV (Gross Transaction Value), costs, and profitability

### Interactive Dashboard
- **Real-time Filtering**: Filter data by journey, channel, status, user type, and date range
- **Dynamic KPI Cards**: Glassmorphism-styled cards with accent-colored borders and live metrics
- **Collapsible Filter Panel**: Toggle filter controls via a dedicated button
- **Data Export**: Export filtered results to Excel format from any table
- **Dark Theme Charts**: All Plotly visualizations use a consistent dark theme with data labels
- **Responsive Design**: Inter font, smooth hover animations, gradient accents

### Report Generation
- **HTML Reports**: Standalone dark-themed HTML reports with embedded CSS, KPI grids, tables, and print-friendly styles
- **PDF Reports**: PDF generation via WeasyPrint (optional dependency)
- **Configurable Sections**: Choose which sections to include (Executive Summary, Channel Performance, Journey Table, Top/Bottom Performers, Volume Analysis)
- **Custom Titles**: Set a custom report title before downloading

### Data Processing
- **Automated Calculations**: Delivery rates, click-through rates, conversion rates
- **User Segmentation**: Categorize users by type (ALL, NU, RU) and channel (ALL, OM, OA, ixigo)
- **Cost Attribution**: Calculate costs based on channel-specific pricing
- **Performance Benchmarking**: Color-coded percentile rankings in summary tables

## Technology Stack

### Backend
- **Python 3.12+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations

### Visualization
- **Plotly**: Interactive charts with dark theme (`plotly_dark` template)
- **Dash**: Web application framework
- **Dash Bootstrap Components**: UI components and layout grid

### Styling
- **Custom CSS**: Dark modern theme injected via `app.index_string`
- **Inter Font**: Google Fonts integration
- **Font Awesome 6**: Icons throughout the UI
- **Glassmorphism**: Translucent card backgrounds with backdrop blur

### Report Generation
- **WeasyPrint** (optional): HTML-to-PDF conversion
- **Standalone HTML**: Reports with embedded CSS that open in any browser

## File Structure

```
JourneyPerformanceAnalysis/
├── test.py                    # Main dashboard application
├── combined_reports.csv       # Campaign data source
├── combiner.py                # CSV report combiner utility
├── campaign.py                # Campaign-level analysis script
├── reference.py               # Reference/utility script
├── Reports/                   # Generated report outputs
├── Codes/
│   ├── auto.ipynb             # Jupyter notebook prototypes
│   └── report_generator.ipynb # Report generation notebook
└── README.md
```

## Installation and Setup

### Prerequisites
- Python 3.12 or higher
- pip package manager

### Required Dependencies
```bash
pip install pandas numpy plotly dash dash-bootstrap-components
```

### Optional Dependencies
```bash
pip install weasyprint    # For PDF report generation
```

### Configuration
1. Update the file path in `test.py`:
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
- `Day`: Date of the campaign
- `Sent`: Number of messages sent
- `Delivered`: Number of messages delivered
- `Unique Impressions`: Unique message impressions
- `Unique Clicks`: Unique click events
- `Unique Click-Through Conversions`: Conversion events
- `Revenue (INR)`: Revenue generated

## Dashboard Tabs

### 1. Performance Overview
- KPI summary cards (Sent, Delivered, Clicks, Orders, Cost, ROI)
- ROI by Channel bar chart with break-even line
- Campaign Conversion Funnel
- Cost vs Revenue scatter plot (top 15 journeys)

### 2. Channel Analysis
- Journey performance summary table with totals
- Delivery rate, CTR, conversion rate, ROI, and cost per order per journey

### 3. Weekly Analysis
- Per-journey daily and weekly breakdown
- Selectable metrics: Volume, Performance, Financial
- Dual y-axis charts for better scaling
- Daily and weekly data tables with export

### 4. Trend Analysis
- Daily volume trends with data labels (dual y-axis for Sent/Delivered vs Clicks/Orders)
- Daily performance rates (Delivery Rate, CTR, Conversion Rate)
- Daily ROI trend with break-even reference line

### 5. Volume Analysis
- Bar charts by channel: Sent, Delivered, Clicks, Orders
- Data labels showing exact volume numbers

### 6. Summary Table
- All journeys with comprehensive metrics
- Color-coded percentile rankings (green = top performers, red = bottom)
- Advanced sorting buttons (by Performance, ROI, Volume)
- Native filtering and Excel export

### 7. Campaign Details
- Individual campaign-level data
- Sortable, filterable table with all metrics
- User type and user channel classification

### 8. Report Generator
- Filter summary showing current date range, journeys, channels
- Custom report title input
- Section checklist (Executive Summary, Channel Performance, Journey Table, Top/Bottom Performers, Volume Analysis)
- Download HTML report (standalone, dark-themed, print-friendly)
- Download PDF report (requires WeasyPrint)

## Filter Controls

- **Journey Selection**: Checklist with Select All, Clear All, Top 10 ROI buttons
- **Channel Filtering**: Email, Push, SMS, WhatsApp, InApp, etc.
- **Status Filter**: Filter by campaign status
- **User Type Filter**: ALL, NU, RU, Unknown
- **Date Range Picker**: Start and end date selection
- **Performance Filter**: All Campaigns, High ROI, High CTR, High Conversion, Low Performers

## Key Metrics Calculated

### Performance Metrics
- **Delivery Rate**: (Delivered / Sent) x 100
- **Click-Through Rate (CTR)**: (Unique Clicks / Delivered) x 100
- **Conversion Rate**: (Conversions / Clicks) x 100
- **Order per Sent**: (Conversions / Sent) x 100

### Financial Metrics
- **Cost**: Channel cost per message x messages sent
- **GTV**: Conversions x Average Order Value
- **ROI**: GTV / Cost
- **ROI with Take Rate**: (GTV x (1 - Take Rate)) / Cost
- **Cost per Order**: Cost / Conversions

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

## UI Theme

The dashboard uses a custom dark modern theme:
- **Background**: `#0f1419` (dark navy)
- **Card surfaces**: `#1a1f2e` with glassmorphism (backdrop blur + translucent)
- **Accent colors**: Cyan `#00d4ff`, Green `#10b981`, Amber `#f59e0b`, Red `#ef4444`, Purple `#7c3aed`
- **Text**: `#e2e8f0` primary, `#94a3b8` muted, `#64748b` subtle
- **Effects**: Hover glow, gradient title, smooth transitions, custom scrollbars

## Troubleshooting

### Common Issues
1. **Data Loading Errors**: Verify CSV format and column names match expected schema
2. **Missing Values**: The app fills NaN with 0; check for unexpected data formats
3. **Port Conflicts**: Change the `port` variable in `__main__` if 8058 is in use
4. **Network Access**: Ensure firewall allows connections on the specified port
5. **PDF Generation**: Install WeasyPrint (`pip install weasyprint`) for PDF downloads; HTML download works without it

### Performance Tips
- Use date range filters to limit data scope on large datasets
- Select specific journeys rather than all 30+ for faster chart rendering
- Export filtered results for offline analysis
