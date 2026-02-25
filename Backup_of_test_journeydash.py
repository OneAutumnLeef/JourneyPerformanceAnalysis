import pandas as pd
import numpy as np
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import socket
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Configuration Variablesfile_path1 = "reportoct.csv"  # Path to the CSV file
file_path1 = "combined_reports.csv"  # Path to the CSV file

# file_path1 = "OverallReportAug11.csv"


AOV = 1500  
TAKE_RATE = 0.08
CHANNEL_COSTS = {
    'Email': 0.02,
    'Push': 0.01, 
    'SMS': 0.11,
    'WhatsApp': 0.11
}

# Enhanced User Type and Channel Mappings based on CSV data
JOURNEY_USER_TYPE_MAPPING = {
    'Product Viewed No Plan Selected (Pan Not Given)': 'ALL',
    'Wishlist Done': 'ALL',
    'NU ALL Signup but No Product View': 'NU',
    'NU ALL Signup but No Product View ': 'NU',  # Handle trailing space
    'NU ALL CC but no DP': 'NU',
    'NU ALL App Download but No Signup': 'NU',
    'Online Mobile_NLA_OC2 to OC=<3': 'RU',
    'Online Mobile_NLA_OC2 to OC=&lt;3': 'RU',  # HTML encoded version
    'Online Mobile_NLA_OC2 to OC<=3': 'RU',  # Alternative encoding
    'Online mobile_Voucher Eligible_5th June': 'RU',
    'Online Apparel_AIA Eligible_28th May': 'RU',
    'Online Mobile_PL Eligible_27th May': 'RU',
    'Online Mobile_NLA_OC1 to OC2_5th_Dec_2024(V2)': 'RU',
    'OM_CC_But_No_DP_EMI_(Updated)': 'RU',
    'Online Apparel_PL Eligible_27th May(v2)': 'RU',
    'OA CC but no DP': 'RU',
    'Online Apparel_Voucher Eligible_8th June': 'RU',
    'Online Apparel - NLA 1 - Order 1 --> 2': 'RU',
    'Online Mobile_AIA Eligible_22nd_April(Updated)': 'RU',
    'Online Apparel - NLA 1 - Order 2 --> 3': 'RU',
    'Ixigo app download': 'RU',
    'AC (CK2MP)': 'RU'
}

JOURNEY_USER_CHANNEL_MAPPING = {
    'Product Viewed No Plan Selected (Pan Not Given)': 'ALL',
    'Wishlist Done': 'ALL',
    'NU ALL Signup but No Product View': 'ALL',
    'NU ALL Signup but No Product View ': 'ALL',  # Handle trailing space
    'NU ALL CC but no DP': 'ALL',
    'NU ALL App Download but No Signup': 'ALL',
    'Online Mobile_NLA_OC2 to OC=<3': 'OM',
    'Online Mobile_NLA_OC2 to OC=&lt;3': 'OM',  # HTML encoded version
    'Online Mobile_NLA_OC2 to OC<=3': 'OM',  # Alternative encoding
    'Online mobile_Voucher Eligible_5th June': 'OM',
    'Online Apparel_AIA Eligible_28th May': 'OA',
    'Online Mobile_PL Eligible_27th May': 'OM',
    'Online Mobile_NLA_OC1 to OC2_5th_Dec_2024(V2)': 'OM',
    'OM_CC_But_No_DP_EMI_(Updated)': 'OM',
    'Online Apparel_PL Eligible_27th May(v2)': 'OA',
    'OA CC but no DP': 'OA',
    'Online Apparel_Voucher Eligible_8th June': 'OA',
    'Online Apparel - NLA 1 - Order 1 --> 2': 'OA',
    'Online Mobile_AIA Eligible_22nd_April(Updated)': 'OM',
    'Online Apparel - NLA 1 - Order 2 --> 3': 'OA',
    'Ixigo app download': 'ixigo',
    'AC (CK2MP)': 'OA'
}



class CampaignAnalyzer:
    def __init__(self, file_path, aov=AOV, channel_costs=CHANNEL_COSTS, take_rate=TAKE_RATE):
        self.file_path = file_path
        self.aov = aov
        self.channel_costs = channel_costs
        self.take_rate = take_rate
        self.raw_data = None
        
    def load_and_process_data(self):
        """Load and process campaign data with enhanced analytics"""
        try:
            df = pd.read_csv(self.file_path)
            df = df.fillna(0)
            
            print("Available columns:", df.columns.tolist())
            
            # Map correct column names based on CSV structure
            numeric_columns = ['Sent', 'Delivered', 'Unique Clicks', 'Unique Click-Through Conversions', 
                              'Unique Impressions', 'Unique Conversions', 'Revenue (INR)']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(',', '').str.replace('nan', '0'), 
                        errors='coerce'
                    ).fillna(0)
            
            # Process string columns
            string_columns = ['Journey Name', 'Channel', 'Status', 'Campaign Name', 'Day']
            for col in string_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()
            
            # Add enhanced analytics columns
            df['User_Type'] = df['Journey Name'].map(JOURNEY_USER_TYPE_MAPPING).fillna('Unknown')
            df['User_Channel'] = df['Journey Name'].map(JOURNEY_USER_CHANNEL_MAPPING).fillna('Unknown')
            
            # Calculate performance metrics
            df = self._calculate_performance_metrics(df)
            
            print(f"\nData processed successfully: {len(df)} records")
            print(f"Date range: {df['Day'].min()} to {df['Day'].max()}")
            print(f"Unique Journeys: {df['Journey Name'].nunique()}")
            print(f"Unique Channels: {df['Channel'].nunique()}")
            
            self.raw_data = df
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def _calculate_performance_metrics(self, df):
        """Calculate comprehensive performance metrics"""
        try:
            # Delivery Rate
            df['Delivery_Rate'] = np.where(df['Sent'] > 0, 
                                         (df['Delivered'] / df['Sent'] * 100).round(2), 0)
            
            # Click-Through Rate (CTR)
            df['CTR'] = np.where(df['Delivered'] > 0,
                               (df['Unique Clicks'] / df['Delivered'] * 100).round(4), 0)
            
            # Conversion Rate (CR)
            df['Conversion_Rate'] = np.where(df['Unique Clicks'] > 0,
                                           (df['Unique Click-Through Conversions'] / df['Unique Clicks'] * 100).round(4), 0)
            
            # Orders per Sent
            df['Orders_per_Sent'] = np.where(df['Sent'] > 0,
                                           (df['Unique Click-Through Conversions'] / df['Sent'] * 100).round(4), 0)
            
            # Calculate costs based on channel
            df['Cost'] = df.apply(lambda row: row['Sent'] * self.channel_costs.get(row['Channel'], 0), axis=1)
            
            # Calculate GTV
            df['GTV'] = df['Unique Click-Through Conversions'] * self.aov
            
            # Calculate ROI
            df['ROI'] = np.where(df['Cost'] > 0, (df['GTV'] / df['Cost']).round(2), 0)
            
            # Calculate ROI with Take Rate
            df['ROI_with_Take_Rate'] = np.where(df['Cost'] > 0, 
                                              ((df['GTV'] * (1 - self.take_rate)) / df['Cost']).round(2), 0)
            
            # Cost per Order
            df['Cost_per_Order'] = np.where(df['Unique Click-Through Conversions'] > 0,
                                          (df['Cost'] / df['Unique Click-Through Conversions']).round(2), 0)
            
            return df
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return df

    def calculate_summary_metrics(self, filtered_df=None):
        """Calculate comprehensive summary metrics"""
        df = filtered_df if filtered_df is not None else self.raw_data
        
        if df is None or df.empty:
            return pd.DataFrame()
            
        try:
            # Group by Journey Name and aggregate
            grouped = df.groupby('Journey Name').agg({
                'Campaign ID': 'count',
                'Sent': 'sum',
                'Delivered': 'sum', 
                'Unique Impressions': 'sum',
                'Unique Clicks': 'sum',
                'Unique Conversions': 'sum',
                'Unique Click-Through Conversions': 'sum',
                'Cost': 'sum',
                'GTV': 'sum',
                'Revenue (INR)': 'sum'
            }).reset_index()
            
            grouped.rename(columns={'Campaign ID': 'Count of Campaign ID'}, inplace=True)
            
            # Calculate performance metrics
            grouped['Delivery Rate'] = np.where(grouped['Sent'] > 0, 
                                              (grouped['Delivered'] / grouped['Sent'] * 100).round(1), 0)
            
            grouped['CTR'] = np.where(grouped['Delivered'] > 0,
                                    (grouped['Unique Clicks'] / grouped['Delivered'] * 100).round(2), 0)
            
            grouped['Conversion Rate'] = np.where(grouped['Unique Clicks'] > 0,
                                                (grouped['Unique Click-Through Conversions'] / grouped['Unique Clicks'] * 100).round(2), 0)
            
            grouped['Order per Sent'] = np.where(grouped['Sent'] > 0,
                                               (grouped['Unique Click-Through Conversions'] / grouped['Sent'] * 100).round(3), 0)
            
            # Calculate ROI
            grouped['ROI'] = np.where(grouped['Cost'] > 0, (grouped['GTV'] / grouped['Cost']).round(2), 0)
            grouped['ROI (With Take Rate)'] = np.where(grouped['Cost'] > 0, 
                                                     ((grouped['GTV'] * (1 - self.take_rate)) / grouped['Cost']).round(2), 0)
            
            # Cost per Order
            grouped['Cost per Order'] = np.where(grouped['Unique Click-Through Conversions'] > 0,
                                               (grouped['Cost'] / grouped['Unique Click-Through Conversions']).round(2), 0)
            
            return grouped
            
        except Exception as e:
            print(f"Error in calculate_summary_metrics: {e}")
            return pd.DataFrame()

    def get_channel_breakdown(self, filtered_df=None):
        """Enhanced channel breakdown with all metrics"""
        df = filtered_df if filtered_df is not None else self.raw_data
        
        if df is None or df.empty:
            return {}
        
        try:
            # Group by Journey and Channel for detailed breakdown
            channel_summary = df.groupby(['Journey Name', 'Channel']).agg({
                'Sent': 'sum',
                'Delivered': 'sum',
                'Unique Clicks': 'sum',
                'Unique Click-Through Conversions': 'sum',
                'Cost': 'sum',
                'GTV': 'sum'
            }).reset_index()
            
            # Create pivot tables for each metric
            metrics = ['Sent', 'Delivered', 'Unique Clicks', 'Unique Click-Through Conversions', 'Cost', 'GTV']
            pivots = {}
            
            for metric in metrics:
                pivot = channel_summary.pivot_table(
                    values=metric, 
                    index='Journey Name', 
                    columns='Channel', 
                    aggfunc='sum', 
                    fill_value=0
                )
                
                # Add total column
                pivot['Total'] = pivot.sum(axis=1)
                pivots[metric] = pivot
            
            return pivots
            
        except Exception as e:
            print(f"Error in get_channel_breakdown: {e}")
            return {}

    def get_channel_performance_metrics(self, filtered_df=None):
        """Calculate channel-wise performance metrics"""
        df = filtered_df if filtered_df is not None else self.raw_data
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        try:
            channel_metrics = df.groupby('Channel').agg({
                'Sent': 'sum',
                'Delivered': 'sum',
                'Unique Clicks': 'sum',
                'Unique Click-Through Conversions': 'sum',
                'Cost': 'sum',
                'GTV': 'sum'
            }).reset_index()
            
            # Calculate performance metrics
            channel_metrics['Delivery_Rate'] = np.where(channel_metrics['Sent'] > 0,
                                                       (channel_metrics['Delivered'] / channel_metrics['Sent'] * 100).round(2), 0)
            
            channel_metrics['CTR'] = np.where(channel_metrics['Delivered'] > 0,
                                            (channel_metrics['Unique Clicks'] / channel_metrics['Delivered'] * 100).round(4), 0)
            
            channel_metrics['Conversion_Rate'] = np.where(channel_metrics['Unique Clicks'] > 0,
                                                        (channel_metrics['Unique Click-Through Conversions'] / channel_metrics['Unique Clicks'] * 100).round(4), 0)
            
            channel_metrics['ROI'] = np.where(channel_metrics['Cost'] > 0,
                                            (channel_metrics['GTV'] / channel_metrics['Cost']).round(2), 0)
            
            channel_metrics['Cost_per_Order'] = np.where(channel_metrics['Unique Click-Through Conversions'] > 0,
                                                        (channel_metrics['Cost'] / channel_metrics['Unique Click-Through Conversions']).round(2), 0)
            
            return channel_metrics
            
        except Exception as e:
            print(f"Error calculating channel metrics: {e}")
            return pd.DataFrame()

    def get_daily_trends(self, filtered_df=None):
        """Calculate daily trend analysis"""
        df = filtered_df if filtered_df is not None else self.raw_data
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        try:
            # Convert Day to datetime if it's not already
            df['Day'] = pd.to_datetime(df['Day'])
            
            daily_trends = df.groupby('Day').agg({
                'Sent': 'sum',
                'Delivered': 'sum',
                'Unique Clicks': 'sum',
                'Unique Click-Through Conversions': 'sum',
                'Cost': 'sum',
                'GTV': 'sum'
            }).reset_index()
            
            # Calculate daily metrics
            daily_trends['Delivery_Rate'] = np.where(daily_trends['Sent'] > 0,
                                                    (daily_trends['Delivered'] / daily_trends['Sent'] * 100).round(2), 0)
            
            daily_trends['CTR'] = np.where(daily_trends['Delivered'] > 0,
                                         (daily_trends['Unique Clicks'] / daily_trends['Delivered'] * 100).round(4), 0)
            
            daily_trends['Conversion_Rate'] = np.where(daily_trends['Unique Clicks'] > 0,
                                                     (daily_trends['Unique Click-Through Conversions'] / daily_trends['Unique Clicks'] * 100).round(4), 0)
            
            daily_trends['ROI'] = np.where(daily_trends['Cost'] > 0,
                                         (daily_trends['GTV'] / daily_trends['Cost']).round(2), 0)
            
            return daily_trends.sort_values('Day')
            
        except Exception as e:
            print(f"Error calculating daily trends: {e}")
            return pd.DataFrame()

# Initialize the analyzer
try:
    analyzer = CampaignAnalyzer(file_path1)
    data_loaded = analyzer.load_and_process_data()
    
    if data_loaded is not None:
        summary_df = analyzer.calculate_summary_metrics()
        
        print(f"Data loaded successfully: {len(data_loaded)} records")
        print(f"Summary table: {len(summary_df)} journeys")
        
        # Display first few rows of summary for verification
        if not summary_df.empty:
            print("\nFirst 5 summary rows:")
            print(summary_df[['Journey Name', 'Sent', 'Delivered', 'Unique Click-Through Conversions', 'Cost', 'GTV', 'ROI', 'ROI (With Take Rate)']].head())
        
        unique_journeys = sorted([str(j) for j in data_loaded['Journey Name'].unique() if pd.notna(j)])
        unique_channels = sorted([str(c) for c in data_loaded['Channel'].unique() if pd.notna(c)])
        unique_statuses = sorted([str(s) for s in data_loaded['Status'].unique() if pd.notna(s)]) if 'Status' in data_loaded.columns else []
        unique_user_types = sorted([str(u) for u in data_loaded['User_Type'].unique() if pd.notna(u)]) if 'User_Type' in data_loaded.columns else []
        unique_user_channels = sorted([str(uc) for uc in data_loaded['User_Channel'].unique() if pd.notna(uc)]) if 'User_Channel' in data_loaded.columns else []
        
        # Date range
        if 'Day' in data_loaded.columns:
            data_loaded['Day'] = pd.to_datetime(data_loaded['Day'])
            valid_dates = data_loaded['Day'].dropna()
            if not valid_dates.empty:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()
            else:
                min_date = max_date = None
        else:
            min_date = max_date = None
            
    else:
        print("Failed to load data")
        summary_df = pd.DataFrame()
        unique_journeys = []
        unique_channels = []
        unique_statuses = []
        unique_user_types = []
        unique_user_channels = []
        min_date = max_date = None
        
except Exception as e:
    print(f"Error initializing analyzer: {e}")
    summary_df = pd.DataFrame()
    unique_journeys = []
    unique_channels = []
    unique_statuses = []
    unique_user_types = []
    unique_user_channels = []
    min_date = max_date = None

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def get_local_ip():
    """Get the local IP address of the machine"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"

# Enhanced layout with tabs for different analytics views
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Journey Analytics Dashboard", 
                   className="text-center mb-4 text-primary"),
            html.Hr()
        ])
    ]),
    
    # Filter Controls Section
    dbc.Card([
        dbc.CardHeader([
            html.H4("Filter Controls", className="mb-0")
        ]),
        dbc.CardBody([
            dbc.Row([
                # Journey Selection
                dbc.Col([
                    html.Label("Select Journeys:", className="fw-bold"),
                    html.Div([
                        dbc.Button("Select All", id="select-all-journeys", size="sm", color="outline-primary", className="me-2 mb-2"),
                        dbc.Button("Clear All", id="clear-all-journeys", size="sm", color="outline-secondary", className="me-2 mb-2"),
                        dbc.Button("Top 10 ROI", id="top-roi-journeys", size="sm", color="outline-success", className="mb-2"),
                    ]),
                    dcc.Checklist(
                        id='journey-checklist',
                        options=[{'label': journey, 'value': journey} for journey in unique_journeys],
                        value=unique_journeys[:10] if len(unique_journeys) > 10 else unique_journeys,
                        style={'maxHeight': '200px', 'overflowY': 'scroll', 'border': '1px solid #ccc', 'padding': '10px'},
                        className="mt-2"
                    )
                ], width=4),
                
                # Channel Selection
                dbc.Col([
                    html.Label("Select Channels:", className="fw-bold"),
                    html.Div([
                        dbc.Button("Select All", id="select-all-channels", size="sm", color="outline-primary", className="me-2 mb-2"),
                        dbc.Button("Clear All", id="clear-all-channels", size="sm", color="outline-secondary", className="mb-2"),
                    ]),
                    dcc.Checklist(
                        id='channel-checklist',
                        options=[{'label': channel, 'value': channel} for channel in unique_channels],
                        value=unique_channels,
                        className="mt-2"
                    )
                ], width=2),
                
                # Status Selection
                dbc.Col([
                    html.Label("Campaign Status:", className="fw-bold"),
                    dcc.Checklist(
                        id='status-checklist',
                        options=[{'label': status, 'value': status} for status in unique_statuses],
                        value=unique_statuses,
                        className="mt-2"
                    )
                ], width=2),
                
                # User Type Selection
                dbc.Col([
                    html.Label("User Type:", className="fw-bold"),
                    dcc.Checklist(
                        id='user-type-checklist',
                        options=[{'label': user_type, 'value': user_type} for user_type in unique_user_types],
                        value=unique_user_types,
                        className="mt-2"
                    )
                ], width=2),
                
                # Date Range and Performance Filters
                dbc.Col([
                    html.Label("Date Range:", className="fw-bold"),
                    dcc.DatePickerRange(
                        id='date-picker-range',
                        start_date=min_date,
                        end_date=max_date,
                        display_format='YYYY-MM-DD',
                        className="mb-3"
                    ) if min_date and max_date else html.P("No date data available"),
                    
                    html.Label("Performance Filter:", className="fw-bold"),
                    dcc.Dropdown(
                        id='performance-filter',
                        options=[
                            {'label': 'All Campaigns', 'value': 'all'},
                            {'label': 'High ROI (>10x)', 'value': 'high_roi'},
                            {'label': 'High CTR (>2%)', 'value': 'high_ctr'},
                            {'label': 'High Conversion (>1%)', 'value': 'high_conv'},
                            {'label': 'Low Performers', 'value': 'low_perf'}
                        ],
                        value='all',
                        className="mt-2"
                    )
                ], width=2),
            ])
        ])
    ], className="mb-4"),
    
    # KPI Dashboard
    html.Div(id="kpi-dashboard", className="mb-4"),
    
    # Main Content Tabs
    dbc.Tabs([
        dbc.Tab(label="Performance Overview", tab_id="overview"),
        dbc.Tab(label="Channel Analysis", tab_id="channel-analysis"),
        dbc.Tab(label="Weekly Analysis", tab_id="weekly-analysis"),
        dbc.Tab(label="Trend Analysis", tab_id="trends"),
        dbc.Tab(label="Volume Analysis", tab_id="volume-analysis"),  # Add this new tab
        dbc.Tab(label="Summary Table", tab_id="summary"),
        dbc.Tab(label="Campaign Details", tab_id="details"),
    ], id="main-tabs", active_tab="overview"),
    
    # Tab Content
    html.Div(id="tab-content", className="mt-4")
    
], fluid=True)

# Remove the incomplete collapse callbacks and replace with proper callback implementations

# Filter callbacks - Complete implementation
@app.callback(
    [Output('journey-checklist', 'value'),
     Output('channel-checklist', 'value')],
    [Input('select-all-journeys', 'n_clicks'),
     Input('clear-all-journeys', 'n_clicks'),
     Input('top-roi-journeys', 'n_clicks'),
     Input('select-all-channels', 'n_clicks'),
     Input('clear-all-channels', 'n_clicks')],
    [State('journey-checklist', 'value'),
     State('channel-checklist', 'value')],
    prevent_initial_call=True
)
def update_filters(select_all_j, clear_all_j, top_roi_j, select_all_c, clear_all_c,
                  current_journeys, current_channels):
    
    try:
        ctx = dash.callback_context
        if not ctx.triggered:
            return current_journeys or [], current_channels or []
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == 'select-all-journeys':
            return unique_journeys, current_channels or []
        elif button_id == 'clear-all-journeys':
            return [], current_channels or []
        elif button_id == 'top-roi-journeys':
            if not summary_df.empty and 'ROI' in summary_df.columns:
                try:
                    top_journeys = summary_df.nlargest(10, 'ROI')['Journey Name'].tolist()
                    return top_journeys, current_channels or []
                except:
                    pass
            return current_journeys or [], current_channels or []
        elif button_id == 'select-all-channels':
            return current_journeys or [], unique_channels
        elif button_id == 'clear-all-channels':
            return current_journeys or [], []
        
        return current_journeys or [], current_channels or []
        
    except Exception as e:
        print(f"Error in update_filters: {e}")
        return current_journeys or [], current_channels or []

# KPI cards callback - Complete implementation
@app.callback(
    Output("kpi-dashboard", "children"),
    [Input('journey-checklist', 'value'),
     Input('channel-checklist', 'value'),
     Input('status-checklist', 'value'),
     Input('user-type-checklist', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('performance-filter', 'value')]
)
def update_kpi_cards(selected_journeys, selected_channels, selected_statuses, selected_user_types, start_date, end_date, perf_filter):
    try:
        if not selected_journeys or not selected_channels:
            return dbc.Alert("Please select journeys and channels using the Filter Controls section", 
                           color="info", className="text-center")
        
        # Apply filters
        filtered_data = data_loaded.copy()
        
        if selected_journeys:
            filtered_data = filtered_data[filtered_data['Journey Name'].isin(selected_journeys)]
        if selected_channels:
            filtered_data = filtered_data[filtered_data['Channel'].isin(selected_channels)]
        if selected_statuses:
            filtered_data = filtered_data[filtered_data['Status'].isin(selected_statuses)]
        if selected_user_types and selected_user_types:
            filtered_data = filtered_data[filtered_data['User_Type'].isin(selected_user_types)]
        if start_date and end_date and 'Day' in filtered_data.columns:
            filtered_data = filtered_data[(filtered_data['Day'] >= start_date) & (filtered_data['Day'] <= end_date)]
        
        if filtered_data.empty:
            return dbc.Alert("No data matches your selection", color="warning")
        
        # Calculate totals
        total_sent = int(filtered_data['Sent'].sum())
        total_delivered = int(filtered_data['Delivered'].sum())
        total_clicks = int(filtered_data['Unique Clicks'].sum())
        total_conversions = int(filtered_data['Unique Click-Through Conversions'].sum())
        
        # Calculate costs
        total_cost = 0
        for _, row in filtered_data.iterrows():
            if row['Channel'] in CHANNEL_COSTS:
                total_cost += row['Sent'] * CHANNEL_COSTS[row['Channel']]
        
        total_gtv = total_conversions * AOV
        overall_roi = total_gtv / total_cost if total_cost > 0 else 0
        
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{total_sent:,}", className="text-primary text-center"),
                        html.P("Messages Sent", className="text-center mb-0"),
                        html.Small(f"{len(set(filtered_data['Journey Name']))} journeys", className="text-muted text-center d-block")
                    ])
                ])
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{total_delivered:,}", className="text-success text-center"),
                        html.P("Delivered", className="text-center mb-0"),
                        html.Small(f"{(total_delivered/total_sent*100 if total_sent > 0 else 0):.0f}%", 
                                 className="text-muted text-center d-block")
                    ])
                ])
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{total_clicks:,}", className="text-info text-center"),
                        html.P("Clicks", className="text-center mb-0"),
                        html.Small(f"{(total_clicks/total_delivered*100 if total_delivered > 0 else 0):.1f}%", 
                                 className="text-muted text-center d-block")
                    ])
                ])
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{total_conversions:,}", className="text-warning text-center"),
                        html.P("Conversions", className="text-center mb-0"),
                        html.Small(f"{(total_conversions/total_clicks*100 if total_clicks > 0 else 0):.1f}%", 
                                 className="text-muted text-center d-block")
                    ])
                ])
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"₹{total_cost:,.0f}", className="text-danger text-center"),
                        html.P("Total Cost", className="text-center mb-0")
                    ])
                ])
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{overall_roi:.1f}x", className="text-dark text-center"),
                        html.P("ROI", className="text-center mb-0"),
                        html.Small(f"₹{total_gtv:,.0f} GTV", 
                                 className="text-muted text-center d-block")
                    ])
                ])
            ], width=2),
        ])
        
    except Exception as e:
        return dbc.Alert(f"Error calculating metrics: {str(e)}", color="danger")

# Tab content callback - Complete implementation
@app.callback(
    Output("tab-content", "children"),
    [Input("main-tabs", "active_tab"),
     Input('journey-checklist', 'value'),
     Input('channel-checklist', 'value'),
     Input('status-checklist', 'value'),
     Input('user-type-checklist', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('performance-filter', 'value')]
)
def update_tab_content(active_tab, selected_journeys, selected_channels, selected_statuses, selected_user_types, start_date, end_date, perf_filter):
    """Update tab content based on selection"""
    try:
        if not selected_journeys or not selected_channels:
            return html.Div("Please select journeys and channels to view analytics", className="text-center text-muted p-4")
        
        # Apply filters
        filtered_data = data_loaded.copy()
        
        if selected_journeys:
            filtered_data = filtered_data[filtered_data['Journey Name'].isin(selected_journeys)]
        if selected_channels:
            filtered_data = filtered_data[filtered_data['Channel'].isin(selected_channels)]
        if selected_statuses:
            filtered_data = filtered_data[filtered_data['Status'].isin(selected_statuses)]
        if selected_user_types:
            filtered_data = filtered_data[filtered_data['User_Type'].isin(selected_user_types)]
        if start_date and end_date:
            filtered_data = filtered_data[(filtered_data['Day'] >= start_date) & (filtered_data['Day'] <= end_date)]
        
        if filtered_data.empty:
            return html.Div("No data matches your selection", className="text-center text-warning p-4")
        
        if active_tab == "overview":
            return create_overview_tab(filtered_data)
        elif active_tab == "channel-analysis":
            return create_channel_analysis_tab(filtered_data)
        elif active_tab == "weekly-analysis":
            return create_weekly_analysis_tab(filtered_data)
        elif active_tab == "trends":
            return create_trends_tab(filtered_data)
        elif active_tab == "volume-analysis":
            return create_volume_analysis_tab(filtered_data)
        elif active_tab == "summary":
            return create_summary_tab(filtered_data)
        elif active_tab == "details":
            return create_details_tab(filtered_data)
        else:
            return html.Div("Select a tab to view content")
            
    except Exception as e:
        return dbc.Alert(f"Error loading content: {str(e)}", color="danger")

# Complete the weekly analysis callback
@app.callback(
    Output("weekly-analysis-content", "children"),
    [Input("weekly-journey-dropdown", "value"),
     Input("weekly-metrics-checklist", "value"),
     Input('journey-checklist', 'value'),
     Input('channel-checklist', 'value'),
     Input('status-checklist', 'value'),
     Input('user-type-checklist', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_weekly_analysis(selected_journey, selected_metrics, journey_filter, channel_filter, status_filter, user_type_filter, start_date, end_date):
    """Update weekly analysis content with IMPROVED SCALING for charts"""
    try:
        if not selected_journey:
            return html.Div("Please select a journey to analyze", className="text-center text-muted p-4")
        
        # Apply filters
        filtered_data = data_loaded.copy()
        
        if journey_filter:
            filtered_data = filtered_data[filtered_data['Journey Name'].isin(journey_filter)]
        if channel_filter:
            filtered_data = filtered_data[filtered_data['Channel'].isin(channel_filter)]
        if status_filter:
            filtered_data = filtered_data[filtered_data['Status'].isin(status_filter)]
        if user_type_filter:
            filtered_data = filtered_data[filtered_data['User_Type'].isin(user_type_filter)]
        if start_date and end_date:
            filtered_data = filtered_data[(filtered_data['Day'] >= start_date) & (filtered_data['Day'] <= end_date)]
        
        # Filter for selected journey
        journey_data = filtered_data[filtered_data['Journey Name'] == selected_journey].copy()
        
        if journey_data.empty:
            return html.Div("No data available for selected journey", className="text-center text-warning p-4")
        
        # Prepare DAILY data for accurate charts
        journey_data['Day'] = pd.to_datetime(journey_data['Day'])
        
        # Aggregate by DAY for accurate daily charts
        daily_summary = journey_data.groupby('Day').agg({
            'Sent': 'sum',
            'Delivered': 'sum',
            'Unique Clicks': 'sum',
            'Unique Click-Through Conversions': 'sum',
            'Cost': 'sum',
            'GTV': 'sum'
        }).reset_index()
        
        # Calculate daily performance metrics
        daily_summary['Delivery_Rate'] = np.where(daily_summary['Sent'] > 0,
                                                  (daily_summary['Delivered'] / daily_summary['Sent'] * 100).round(2), 0)
        daily_summary['CTR'] = np.where(daily_summary['Delivered'] > 0,
                                        (daily_summary['Unique Clicks'] / daily_summary['Delivered'] * 100).round(2), 0)
        daily_summary['Conversion_Rate'] = np.where(daily_summary['Unique Clicks'] > 0,
                                                    (daily_summary['Unique Click-Through Conversions'] / daily_summary['Unique Clicks'] * 100).round(2), 0)
        daily_summary['ROI'] = np.where(daily_summary['Cost'] > 0,
                                        (daily_summary['GTV'] / daily_summary['Cost']).round(2), 0)
        daily_summary['Cost_per_Order'] = np.where(daily_summary['Unique Click-Through Conversions'] > 0,
                                                   (daily_summary['Cost'] / daily_summary['Unique Click-Through Conversions']).round(2), 0)
        
        # Sort by day
        daily_summary = daily_summary.sort_values('Day')
        
        # Create visualizations with IMPROVED SCALING
        charts = []
        
        if 'volume' in selected_metrics:
            # Use dual y-axis for better scaling of volume metrics
            fig_volume = go.Figure()
            
            # Primary y-axis for high volume metrics (Sent, Delivered)
            fig_volume.add_trace(go.Scatter(
                x=daily_summary['Day'], 
                y=daily_summary['Sent'],
                mode='lines+markers',
                name='Messages Sent',
                line=dict(color='#3498db', width=3),
                yaxis='y'
            ))
            
            fig_volume.add_trace(go.Scatter(
                x=daily_summary['Day'], 
                y=daily_summary['Delivered'],
                mode='lines+markers',
                name='Delivered',
                line=dict(color='#2ecc71', width=3),
                yaxis='y'
            ))
            
            # Secondary y-axis for lower volume metrics (Clicks, Orders)
            fig_volume.add_trace(go.Scatter(
                x=daily_summary['Day'], 
                y=daily_summary['Unique Clicks'],
                mode='lines+markers',
                name='Clicks',
                line=dict(color='#f39c12', width=3),
                yaxis='y2'
            ))
            
            fig_volume.add_trace(go.Scatter(
                x=daily_summary['Day'], 
                y=daily_summary['Unique Click-Through Conversions'],
                mode='lines+markers',
                name='Orders',
                line=dict(color='#e74c3c', width=3),
                yaxis='y2'
            ))
            
            fig_volume.update_layout(
                title=f"Daily Volume Trends - {selected_journey}",
                height=500,
                xaxis_title="Date",
                yaxis=dict(
                    title="Messages/Delivered Count",
                    side="left"
                ),
                yaxis2=dict(
                    title="Clicks/Orders Count", 
                    side="right",
                    overlaying="y"
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                font=dict(size=12),
                title_font_size=16,
                margin=dict(t=80, b=60, l=60, r=60)
            )
            
            charts.append(dbc.Row([dbc.Col([dcc.Graph(figure=fig_volume)], width=12)], className="mb-4"))
        
        if 'performance' in selected_metrics:
            # Performance rates chart with improved styling
            fig_performance = go.Figure()
            
            fig_performance.add_trace(go.Scatter(
                x=daily_summary['Day'], 
                y=daily_summary['Delivery_Rate'],
                mode='lines+markers',
                name='Delivery Rate (%)',
                line=dict(color='#27ae60', width=3),
                marker=dict(size=8)
            ))
            
            fig_performance.add_trace(go.Scatter(
                x=daily_summary['Day'], 
                y=daily_summary['CTR'],
                mode='lines+markers',
                name='Click Through Rate (%)',
                line=dict(color='#3498db', width=3),
                marker=dict(size=8)
            ))
            
            fig_performance.add_trace(go.Scatter(
                x=daily_summary['Day'], 
                y=daily_summary['Conversion_Rate'],
                mode='lines+markers',
                name='Conversion Rate (%)',
                line=dict(color='#e67e22', width=3),
                marker=dict(size=8)
            ))
            
            fig_performance.update_layout(
                title=f"Daily Performance Rates - {selected_journey}",
                height=500,
                xaxis_title="Date",
                yaxis_title="Rate (%)",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                font=dict(size=12),
                title_font_size=16,
                margin=dict(t=80, b=60, l=60, r=60),
                hovermode='x unified'
            )
            
            charts.append(dbc.Row([dbc.Col([dcc.Graph(figure=fig_performance)], width=12)], className="mb-4"))
        
        if 'financial' in selected_metrics:
            # Financial metrics with dual y-axis for better scaling
            fig_financial = go.Figure()
            
            # Primary y-axis for monetary values
            fig_financial.add_trace(go.Scatter(
                x=daily_summary['Day'], 
                y=daily_summary['Cost'],
                mode='lines+markers',
                name='Daily Cost (₹)',
                line=dict(color='#e74c3c', width=3),
                yaxis='y',
                marker=dict(size=8)
            ))
            
            fig_financial.add_trace(go.Scatter(
                x=daily_summary['Day'], 
                y=daily_summary['GTV'],
                mode='lines+markers',
                name='Daily GTV (₹)',
                line=dict(color='#2ecc71', width=3),
                yaxis='y',
                marker=dict(size=8)
            ))
            
            # Secondary y-axis for ROI (ratio)
            fig_financial.add_trace(go.Scatter(
                x=daily_summary['Day'], 
                y=daily_summary['ROI'],
                mode='lines+markers',
                name='ROI (x)',
                line=dict(color='#9b59b6', width=3),
                yaxis='y2',
                marker=dict(size=8)
            ))
            
            # Add break-even line for ROI
            fig_financial.add_hline(y=1, line_dash="dash", line_color="red", 
                                  annotation_text="Break-even ROI", 
                                  annotation_position="top right",
                                  yref="y2")
            
            fig_financial.update_layout(
                title=f"Daily Financial Performance - {selected_journey}",
                height=500,
                xaxis_title="Date",
                yaxis=dict(
                    title="Amount (₹)",
                    side="left"
                ),
                yaxis2=dict(
                    title="ROI (x)", 
                    side="right",
                    overlaying="y"
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                font=dict(size=12),
                title_font_size=16,
                margin=dict(t=80, b=60, l=60, r=60),
                hovermode='x unified'
            )
            
            charts.append(dbc.Row([dbc.Col([dcc.Graph(figure=fig_financial)], width=12)], className="mb-4"))
        
        # Also create weekly summary for table view
        journey_data['Week'] = journey_data['Day'].dt.strftime('%Y-W%U')
        journey_data['Week_Start'] = journey_data['Day'].dt.to_period('W').dt.start_time
        
        weekly_summary = journey_data.groupby(['Week', 'Week_Start']).agg({
            'Sent': 'sum',
            'Delivered': 'sum',
            'Unique Clicks': 'sum',
            'Unique Click-Through Conversions': 'sum',
            'Cost': 'sum',
            'GTV': 'sum'
        }).reset_index()
        
        # Calculate weekly performance metrics
        weekly_summary['Delivery_Rate'] = np.where(weekly_summary['Sent'] > 0,
                                                  (weekly_summary['Delivered'] / weekly_summary['Sent'] * 100).round(2), 0)
        weekly_summary['CTR'] = np.where(weekly_summary['Delivered'] > 0,
                                        (weekly_summary['Unique Clicks'] / weekly_summary['Delivered'] * 100).round(2), 0)
        weekly_summary['Conversion_Rate'] = np.where(weekly_summary['Unique Clicks'] > 0,
                                                    (weekly_summary['Unique Click-Through Conversions'] / weekly_summary['Unique Clicks'] * 100).round(2), 0)
        weekly_summary['ROI'] = np.where(weekly_summary['Cost'] > 0,
                                        (weekly_summary['GTV'] / weekly_summary['Cost']).round(2), 0)
        weekly_summary['Cost_per_Order'] = np.where(weekly_summary['Unique Click-Through Conversions'] > 0,
                                                   (weekly_summary['Cost'] / weekly_summary['Unique Click-Through Conversions']).round(2), 0)
        
        weekly_summary = weekly_summary.sort_values('Week_Start')
        
        # Create DAILY summary table
        display_daily_df = daily_summary.copy()
        display_daily_df['Day'] = display_daily_df['Day'].dt.strftime('%Y-%m-%d')
        display_daily_df['Sent'] = display_daily_df['Sent'].apply(lambda x: f"{int(x):,}")
        display_daily_df['Delivered'] = display_daily_df['Delivered'].apply(lambda x: f"{int(x):,}")
        display_daily_df['Unique Clicks'] = display_daily_df['Unique Clicks'].apply(lambda x: f"{int(x):,}")
        display_daily_df['Unique Click-Through Conversions'] = display_daily_df['Unique Click-Through Conversions'].apply(lambda x: f"{int(x):,}")
        display_daily_df['Cost'] = display_daily_df['Cost'].apply(lambda x: f"₹{x:,.0f}")
        display_daily_df['GTV'] = display_daily_df['GTV'].apply(lambda x: f"₹{x:,.0f}")
        display_daily_df['Delivery_Rate'] = display_daily_df['Delivery_Rate'].apply(lambda x: f"{x:.1f}%")
        display_daily_df['CTR'] = display_daily_df['CTR'].apply(lambda x: f"{x:.2f}%")
        display_daily_df['Conversion_Rate'] = display_daily_df['Conversion_Rate'].apply(lambda x: f"{x:.2f}%")
        display_daily_df['ROI'] = display_daily_df['ROI'].apply(lambda x: f"{x:.2f}x")
        display_daily_df['Cost_per_Order'] = display_daily_df['Cost_per_Order'].apply(lambda x: f"₹{x:.0f}")
        
        daily_table = dash_table.DataTable(
            data=display_daily_df.to_dict('records'),
            columns=[{"name": col.replace('_', ' '), "id": col} for col in display_daily_df.columns],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'center',
                'padding': '8px',
                'fontSize': '12px'
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            sort_action="native",
            export_format="xlsx",
            page_size=15
        )
        
        # Create WEEKLY summary table
        display_weekly_df = weekly_summary.copy()
        display_weekly_df['Week_Start'] = display_weekly_df['Week_Start'].dt.strftime('%Y-%m-%d')
        display_weekly_df['Sent'] = display_weekly_df['Sent'].apply(lambda x: f"{int(x):,}")
        display_weekly_df['Delivered'] = display_weekly_df['Delivered'].apply(lambda x: f"{int(x):,}")
        display_weekly_df['Unique Clicks'] = display_weekly_df['Unique Clicks'].apply(lambda x: f"{int(x):,}")
        display_weekly_df['Unique Click-Through Conversions'] = display_weekly_df['Unique Click-Through Conversions'].apply(lambda x: f"{int(x):,}")
        display_weekly_df['Cost'] = display_weekly_df['Cost'].apply(lambda x: f"₹{x:,.0f}")
        display_weekly_df['GTV'] = display_weekly_df['GTV'].apply(lambda x: f"₹{x:,.0f}")
        display_weekly_df['Delivery_Rate'] = display_weekly_df['Delivery_Rate'].apply(lambda x: f"{x:.1f}%")
        display_weekly_df['CTR'] = display_weekly_df['CTR'].apply(lambda x: f"{x:.2f}%")
        display_weekly_df['Conversion_Rate'] = display_weekly_df['Conversion_Rate'].apply(lambda x: f"{x:.2f}%")
        display_weekly_df['ROI'] = display_weekly_df['ROI'].apply(lambda x: f"{x:.2f}x")
        display_weekly_df['Cost_per_Order'] = display_weekly_df['Cost_per_Order'].apply(lambda x: f"₹{x:.0f}")
        
        weekly_table = dash_table.DataTable(
            data=display_weekly_df.to_dict('records'),
            columns=[{"name": col.replace('_', ' '), "id": col} for col in display_weekly_df.columns],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'center',
                'padding': '8px',
                'fontSize': '12px'
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            export_format="xlsx"
        )
        
        return html.Div([
            html.H5(f"Daily & Weekly Analysis: {selected_journey}", className="mb-3"),
            
            # Charts with improved scaling
            *charts,
            
            # Daily Data Table
            html.H6("Daily Performance Data", className="mt-4 mb-3"),
            daily_table,
            
            # Weekly Summary Table  
            html.H6("Weekly Summary Data", className="mt-4 mb-3"),
            weekly_table
        ])
        
    except Exception as e:
        return dbc.Alert(f"Error in weekly analysis: {str(e)}", color="danger")

def create_overview_tab(filtered_data):
    """Create comprehensive overview tab with improved chart scaling"""
    try:
        # Calculate overall metrics
        total_sent = filtered_data['Sent'].sum()
        total_delivered = filtered_data['Delivered'].sum()
        total_clicks = filtered_data['Unique Clicks'].sum()
        total_orders = filtered_data['Unique Click-Through Conversions'].sum()
        total_cost = filtered_data['Cost'].sum()
        total_gtv = filtered_data['GTV'].sum()
        overall_roi = total_gtv / total_cost if total_cost > 0 else 0
        
        # Performance metrics by channel
        channel_metrics = analyzer.get_channel_performance_metrics(filtered_data)
        
        # Create ROI by Channel chart with better scaling
        if not channel_metrics.empty:
            fig1 = px.bar(channel_metrics, 
                         x='Channel', y='ROI',
                         title="ROI by Channel",
                         labels={'ROI': 'Return on Investment (x)', 'Channel': 'Communication Channel'},
                         color='ROI',
                         color_continuous_scale='Viridis',
                         text='ROI')
            
            # Improve chart formatting
            fig1.update_traces(
                texttemplate='%{text:.1f}x', 
                textposition='outside',
                textfont_size=12
            )
            fig1.update_layout(
                height=500,
                showlegend=False,
                xaxis_title="Communication Channel",
                yaxis_title="ROI (Return on Investment)",
                font=dict(size=12),
                title_font_size=16,
                xaxis=dict(tickangle=0),
                margin=dict(t=60, b=100, l=60, r=60)
            )
            
            # Add horizontal line at ROI = 1 (break-even)
            fig1.add_hline(y=1, line_dash="dash", line_color="red", 
                          annotation_text="Break-even (1x ROI)", 
                          annotation_position="top right")
        else:
            fig1 = px.bar(title="No channel data available")
        
        # Create improved funnel analysis
        funnel_data = [
            ("Messages Sent", total_sent),
            ("Messages Delivered", total_delivered), 
            ("Clicks Generated", total_clicks),
            ("Orders Converted", total_orders)
        ]
        
        fig2 = go.Figure(go.Funnel(
            y=[item[0] for item in funnel_data],
            x=[item[1] for item in funnel_data],
            textposition="inside",
            textinfo="value+percent initial+percent previous",
            marker=dict(
                color=["#3498db", "#2ecc71", "#f39c12", "#e74c3c"],
                line=dict(width=2, color="white")
            )
        ))
        
        fig2.update_layout(
            title="Campaign Conversion Funnel",
            height=500,
            font=dict(size=12),
            title_font_size=16,
            margin=dict(t=60, b=60, l=60, r=60)
        )
        
        # Create additional performance chart - Journey ROI scatter
        summary_metrics = analyzer.calculate_summary_metrics(filtered_data)
        if not summary_metrics.empty and len(summary_metrics) > 1:
            # Get top 15 journeys by ROI for readability
            top_journeys = summary_metrics.nlargest(15, 'ROI')
            
            fig3 = px.scatter(top_journeys, 
                             x='Cost', y='GTV',
                             size='Unique Click-Through Conversions',
                             color='ROI',
                             hover_name='Journey Name',
                             title="Cost vs Revenue Analysis (Top 15 Journeys)",
                             labels={
                                 'Cost': 'Total Cost (₹)',
                                 'GTV': 'Gross Transaction Value (₹)',
                                 'ROI': 'ROI (x)',
                                 'Unique Click-Through Conversions': 'Orders'
                             },
                             color_continuous_scale='RdYlGn',
                             size_max=50)
            
            # Add break-even line (where GTV = Cost)
            max_cost = top_journeys['Cost'].max()
            fig3.add_trace(go.Scatter(
                x=[0, max_cost],
                y=[0, max_cost],
                mode='lines',
                line=dict(dash='dash', color='red'),
                name='Break-even Line',
                showlegend=True
            ))
            
            fig3.update_layout(
                height=500,
                font=dict(size=12),
                title_font_size=16,
                xaxis_title="Total Cost (₹)",
                yaxis_title="Gross Transaction Value (₹)",
                margin=dict(t=60, b=60, l=60, r=60)
            )
        else:
            fig3 = px.scatter(title="Insufficient data for cost-revenue analysis")
        
        return html.Div([
            # Enhanced KPI Cards with better formatting
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3(f"{total_sent:,}", className="text-primary text-center mb-1"),
                            html.P("Messages Sent", className="text-center mb-1 fw-bold"),
                            html.Small(f"{len(set(filtered_data['Journey Name']))} journeys", 
                                     className="text-muted text-center d-block")
                        ])
                    ], className="h-100")
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3(f"{total_delivered:,}", className="text-success text-center mb-1"),
                            html.P("Delivered", className="text-center mb-1 fw-bold"),
                            html.Small(f"{(total_delivered/total_sent*100 if total_sent > 0 else 0):.1f}% delivery rate", 
                                     className="text-muted text-center d-block")
                        ])
                    ], className="h-100")
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3(f"{total_clicks:,}", className="text-info text-center mb-1"),
                            html.P("Clicks", className="text-center mb-1 fw-bold"),
                            html.Small(f"{(total_clicks/total_delivered*100 if total_delivered > 0 else 0):.2f}% CTR", 
                                     className="text-muted text-center d-block")
                        ])
                    ], className="h-100")
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3(f"{total_orders:,}", className="text-warning text-center mb-1"),
                            html.P("Orders", className="text-center mb-1 fw-bold"),
                            html.Small(f"{(total_orders/total_clicks*100 if total_clicks > 0 else 0):.2f}% conversion", 
                                     className="text-muted text-center d-block")
                        ])
                    ], className="h-100")
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3(f"₹{total_cost:,.0f}", className="text-danger text-center mb-1"),
                            html.P("Total Cost", className="text-center mb-1 fw-bold"),
                            html.Small(f"₹{(total_cost/total_orders if total_orders > 0 else 0):.0f} per order", 
                                     className="text-muted text-center d-block")
                        ])
                    ], className="h-100")
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3(f"{overall_roi:.1f}x", className="text-dark text-center mb-1"),
                            html.P("Overall ROI", className="text-center mb-1 fw-bold"),
                            html.Small(f"₹{total_gtv:,.0f} GTV", 
                                     className="text-muted text-center d-block")
                        ])
                    ], className="h-100")
                ], width=2),
            ], className="mb-4"),
            
            # Charts with improved layout
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(figure=fig1)
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(figure=fig2)
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Additional analysis chart
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(figure=fig3)
                        ])
                    ])
                ], width=12)
            ])
        ])
        
    except Exception as e:
        return dbc.Alert(f"Error creating overview: {str(e)}", color="danger")

# Also improve the trends tab for better readability
def create_trends_tab(filtered_data):
    """Create trend analysis tab with improved scaling and data labels"""
    try:
        daily_trends = analyzer.get_daily_trends(filtered_data)
        
        if daily_trends.empty:
            return html.Div("No trend data available", className="text-center text-warning p-4")
        
        # Create volume trends chart with secondary y-axis and data labels
        fig1 = go.Figure()
        
        # Add volume metrics on primary y-axis with data labels
        fig1.add_trace(go.Scatter(
            x=daily_trends['Day'], 
            y=daily_trends['Sent'],
            mode='lines+markers+text',
            name='Messages Sent',
            line=dict(color='#3498db', width=3),
            marker=dict(size=8),
            text=[f"{int(val):,}" for val in daily_trends['Sent']],
            textposition='top center',
            textfont=dict(size=10, color='#2c3e50'),
            yaxis='y'
        ))
        
        fig1.add_trace(go.Scatter(
            x=daily_trends['Day'], 
            y=daily_trends['Delivered'],
            mode='lines+markers+text',
            name='Delivered',
            line=dict(color='#2ecc71', width=3),
            marker=dict(size=8),
            text=[f"{int(val):,}" for val in daily_trends['Delivered']],
            textposition='bottom center',
            textfont=dict(size=10, color='#2c3e50'),
            yaxis='y'
        ))
        
        fig1.add_trace(go.Scatter(
            x=daily_trends['Day'], 
            y=daily_trends['Unique Clicks'],
            mode='lines+markers+text',
            name='Clicks',
            line=dict(color='#f39c12', width=3),
            marker=dict(size=8),
            text=[f"{int(val):,}" for val in daily_trends['Unique Clicks']],
            textposition='top center',
            textfont=dict(size=10, color='#2c3e50'),
            yaxis='y'
        ))
        
        # Add conversions on secondary y-axis for better scaling with data labels
        fig1.add_trace(go.Scatter(
            x=daily_trends['Day'], 
            y=daily_trends['Unique Click-Through Conversions'],
            mode='lines+markers+text',
            name='Orders',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=8),
            text=[f"{int(val):,}" for val in daily_trends['Unique Click-Through Conversions']],
            textposition='top center',
            textfont=dict(size=10, color='#c62828', family='Arial Black'),
            yaxis='y2'
        ))
        
        fig1.update_layout(
            title="Daily Performance Trends (with Data Labels)",
            height=650,  # Increased height to accommodate labels
            xaxis_title="Date",
            yaxis=dict(
                title="Messages/Clicks Count",
                side="left"
            ),
            yaxis2=dict(
                title="Orders Count", 
                side="right",
                overlaying="y"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            font=dict(size=12),
            title_font_size=16,
            margin=dict(t=100, b=60, l=60, r=60)  # Increased top margin for labels
        )
        
        # Create performance rates chart with data labels
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=daily_trends['Day'], 
            y=daily_trends['Delivery_Rate'],
            mode='lines+markers+text',
            name='Delivery Rate (%)',
            line=dict(color='#27ae60', width=3),
            marker=dict(size=8),
            text=[f"{val:.1f}%" for val in daily_trends['Delivery_Rate']],
            textposition='top center',
            textfont=dict(size=10, color='#1e7145')
        ))
        
        fig2.add_trace(go.Scatter(
            x=daily_trends['Day'], 
            y=daily_trends['CTR'],
            mode='lines+markers+text',
            name='Click Rate (%)',
            line=dict(color='#3498db', width=3),
            marker=dict(size=8),
            text=[f"{val:.2f}%" for val in daily_trends['CTR']],
            textposition='middle right',
            textfont=dict(size=10, color='#2980b9')
        ))
        
        fig2.add_trace(go.Scatter(
            x=daily_trends['Day'], 
            y=daily_trends['Conversion_Rate'],
            mode='lines+markers+text',
            name='Conversion Rate (%)',
            line=dict(color='#e67e22', width=3),
            marker=dict(size=8),
            text=[f"{val:.2f}%" for val in daily_trends['Conversion_Rate']],
            textposition='bottom center',
            textfont=dict(size=10, color='#d35400')
        ))
        
        fig2.update_layout(
            title="Daily Performance Rates (with Data Labels)",
            height=550,  # Increased height for labels
            xaxis_title="Date",
            yaxis_title="Rate (%)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            font=dict(size=12),
            title_font_size=16,
            margin=dict(t=100, b=60, l=60, r=60)  # Increased margins for labels
        )
        
        # ROI trends chart with data labels
        fig3 = go.Figure()
        
        fig3.add_trace(go.Scatter(
            x=daily_trends['Day'], 
            y=daily_trends['ROI'],
            mode='lines+markers+text',
            name='ROI',
            line=dict(color='#9b59b6', width=4),
            marker=dict(size=10),
            text=[f"{val:.1f}x" for val in daily_trends['ROI']],
            textposition='top center',
            textfont=dict(size=11, color='#8e44ad', family='Arial Black')
        ))
        
        # Add break-even line
        fig3.add_hline(y=1, line_dash="dash", line_color="red", line_width=2,
                      annotation_text="Break-even (1x ROI)", 
                      annotation_position="top right",
                      annotation_font=dict(size=12, color="red"))
        
        fig3.update_layout(
            title="Daily ROI Trends (with Data Labels)",
            height=450,
            xaxis_title="Date",
            yaxis_title="Return on Investment (x)",
            font=dict(size=12),
            title_font_size=16,
            margin=dict(t=80, b=60, l=60, r=60),
            showlegend=False
        )
        
        return html.Div([
            # Enhanced header with data labels info
            dbc.Alert([
                html.I(className="fas fa-info-circle me-2"),
                "📊 Data labels are displayed on all trend charts for easy reading. Hover over points for detailed information."
            ], color="info", className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(figure=fig1)
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(figure=fig2)
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(figure=fig3)
                        ])
                    ])
                ], width=4)
            ])
        ])
        
    except Exception as e:
        return dbc.Alert(f"Error creating trends: {str(e)}", color="danger")

def create_channel_analysis_tab(filtered_data):
    """Create enhanced channel analysis tab with unified table - ONLY TOTALS"""
    try:
        # Get channel breakdown with all metrics
        df = filtered_data.copy()
        
        if df is None or df.empty:
            return html.Div("No channel data available", className="text-center text-warning p-4")
        
        # Create comprehensive channel breakdown table - ONLY TOTALS
        journeys = df['Journey Name'].unique()
        
        breakdown_data = []
        
        for journey in journeys:
            journey_data = df[df['Journey Name'] == journey]
            
            # Calculate ONLY totals (no channel-wise breakdown)
            total_sent = journey_data['Sent'].sum()
            total_delivered = journey_data['Delivered'].sum()
            total_clicks = journey_data['Unique Clicks'].sum()
            total_conversions = journey_data['Unique Click-Through Conversions'].sum()
            total_cost = journey_data['Cost'].sum()
            total_gtv = journey_data['GTV'].sum()
            
            # Calculate rates
            delivery_rate = (total_delivered / total_sent * 100) if total_sent > 0 else 0
            ctr = (total_clicks / total_delivered * 100) if total_delivered > 0 else 0
            conversion_rate = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
            roi = (total_gtv / total_cost) if total_cost > 0 else 0
            cost_per_order = (total_cost / total_conversions) if total_conversions > 0 else 0
            
            row_data = {
                'Journey Name': journey,
                'Sent': f"{int(total_sent):,}",
                'Delivered': f"{int(total_delivered):,}",
                'Clicks': f"{int(total_clicks):,}",
                'Orders': f"{int(total_conversions):,}",
                'Cost': f"₹{total_cost:,.0f}",
                'GTV': f"₹{total_gtv:,.0f}",
                'Delivery Rate': f"{delivery_rate:.1f}%",
                'CTR': f"{ctr:.2f}%",
                'Conversion Rate': f"{conversion_rate:.2f}%",
                'ROI': f"{roi:.2f}x",
                'Cost per Order': f"₹{cost_per_order:.0f}"
            }
            
            breakdown_data.append(row_data)
        
        breakdown_df = pd.DataFrame(breakdown_data)
        
        return html.Div([
            html.H5(f"Journey Performance Summary - {len(breakdown_df)} journeys", className="mb-3"),
            dash_table.DataTable(
                data=breakdown_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in breakdown_df.columns],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'center',
                    'padding': '8px',
                    'fontSize': '12px',
                    'whiteSpace': 'nowrap'
                },
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold',
                    'textAlign': 'center'
                },
                style_data_conditional=[
                    {
                        'if': {'column_id': 'Journey Name'},
                        'textAlign': 'left',
                        'fontWeight': 'bold'
                    }
                ],
                sort_action="native",
                sort_mode="multi",
                export_format="xlsx",
                page_size=20
            )
        ])
        
    except Exception as e:
        return dbc.Alert(f"Error creating channel analysis: {str(e)}", color="danger")

def create_volume_analysis_tab(filtered_data):
    """Create volume analysis tab with bar graphs similar to campaign.py"""
    try:
        if filtered_data.empty or 'Channel' not in filtered_data.columns:
            return html.Div("No data available for volume analysis", className="text-center text-warning p-4")
        
        # Group by Channel for volume analysis
        volume_metrics = filtered_data.groupby('Channel').agg({
            'Sent': 'sum',
            'Delivered': 'sum',
            'Unique Clicks': 'sum',
            'Unique Click-Through Conversions': 'sum'
        }).reset_index()
        
        # Create bar graphs with enhanced labels
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=volume_metrics['Channel'],
            y=volume_metrics['Sent'],
            name='Messages Sent',
            marker_color='lightblue',
            text=[f"{int(val):,}" for val in volume_metrics['Sent']],
            textposition='outside',
            textfont=dict(size=12, color='black'),
            hovertemplate='<b>%{x}</b><br>Messages Sent: %{y:,.0f}<extra></extra>'
        ))
        fig1.update_layout(
            title="Total Sent Volume by Channel",
            xaxis_title="Channel",
            yaxis_title="Messages Sent",
            height=500,
            template="plotly_white",
            showlegend=False,
            font=dict(size=12)
        )
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=volume_metrics['Channel'],
            y=volume_metrics['Delivered'],
            name='Messages Delivered',
            marker_color='lightgreen',
            text=[f"{int(val):,}" for val in volume_metrics['Delivered']],
            textposition='outside',
            textfont=dict(size=12, color='black'),
            hovertemplate='<b>%{x}</b><br>Messages Delivered: %{y:,.0f}<extra></extra>'
        ))
        fig2.update_layout(
            title="Total Delivered Volume by Channel",
            xaxis_title="Channel",
            yaxis_title="Messages Delivered",
            height=500,
            template="plotly_white",
            showlegend=False,
            font=dict(size=12)
        )
        
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=volume_metrics['Channel'],
            y=volume_metrics['Unique Clicks'],
            name='Total Clicks',
            marker_color='orange',
            text=[f"{int(val):,}" for val in volume_metrics['Unique Clicks']],
            textposition='outside',
            textfont=dict(size=12, color='black'),
            hovertemplate='<b>%{x}</b><br>Total Clicks: %{y:,.0f}<extra></extra>'
        ))
        fig3.update_layout(
            title="Total Clicks Volume by Channel",
            xaxis_title="Channel",
            yaxis_title="Total Clicks",
            height=500,
            template="plotly_white",
            showlegend=False,
            font=dict(size=12)
        )
        
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=volume_metrics['Channel'],
            y=volume_metrics['Unique Click-Through Conversions'],
            name='Order Success',
            marker_color='lightcoral',
            text=[f"{int(val):,}" for val in volume_metrics['Unique Click-Through Conversions']],
            textposition='outside',
            textfont=dict(size=12, color='black'),
            hovertemplate='<b>%{x}</b><br>Orders: %{y:,.0f}<extra></extra>'
        ))
        fig4.update_layout(
            title="Total Orders Volume by Channel",
            xaxis_title="Channel",
            yaxis_title="Orders",
            height=500,
            template="plotly_white",
            showlegend=False,
            font=dict(size=12)
        )
        
        return html.Div([
            html.H4("Volume Analysis by Channel", className="text-center mb-4"),
            html.P("Note: Data labels show exact volume numbers for each channel.", 
                   className="text-muted text-center mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(figure=fig1)
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(figure=fig2)
                        ])
                    ])
                ], width=6),
            ], className="mb-4"),
    
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(figure=fig3)
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(figure=fig4)
                        ])
                    ])
                ], width=6),
            ])
        ])
        
    except Exception as e:
        return dbc.Alert(f"Error creating volume analysis: {str(e)}", color="danger")

# And add the missing create_weekly_analysis_tab function:

def create_weekly_analysis_tab(filtered_data):
    """Create weekly comparison analysis tab"""
    try:
        if filtered_data.empty:
            return html.Div("No data available for weekly analysis", className="text-center text-warning p-4")
        
        # Get unique journeys
        unique_journeys = sorted(filtered_data['Journey Name'].unique())
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("Select Journey for Weekly Analysis:", className="fw-bold"),
                    dcc.Dropdown(
                        id='weekly-journey-dropdown',
                        options=[{'label': journey, 'value': journey} for journey in unique_journeys],
                        value=unique_journeys[0] if unique_journeys else None,
                        placeholder="Select a journey"
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Metrics to Display:", className="fw-bold"),
                    dcc.Checklist(
                        id='weekly-metrics-checklist',
                        options=[
                            {'label': 'Volume Metrics', 'value': 'volume'},
                            {'label': 'Performance Metrics', 'value': 'performance'},
                            {'label': 'Financial Metrics', 'value': 'financial'}
                        ],
                        value=['volume', 'performance', 'financial'],
                        inline=True
                    )
                ], width=6)
            ], className="mb-4"),
            
            html.Div(id="weekly-analysis-content")
        ])
        
    except Exception as e:
        return dbc.Alert(f"Error creating weekly analysis: {str(e)}", color="danger")

# Add these missing functions to your test.py file

def create_summary_tab(filtered_data):
    """Create enhanced summary table with color coding and advanced sorting"""
    try:
        summary_metrics = analyzer.calculate_summary_metrics(filtered_data)
        
        if summary_metrics.empty:
            return html.Div("No summary data available", className="text-center text-warning p-4")
        
        # Format the display
        display_df = summary_metrics.copy()
        
        # Store original numeric values for sorting and color coding
        numeric_df = summary_metrics.copy()
        
        # Format numeric columns for display
        for col in ['Sent', 'Delivered', 'Unique Clicks', 'Unique Click-Through Conversions', 'Unique Impressions', 'Unique Conversions']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{int(x):,}")
        
        # Format percentage columns
        percentage_cols = ['Delivery Rate', 'CTR', 'Conversion Rate', 'Order per Sent']
        for col in percentage_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
        
        # Format financial columns
        if 'Cost' in display_df.columns:
            display_df['Cost'] = display_df['Cost'].apply(lambda x: f"₹{x:,.2f}")
        if 'GTV' in display_df.columns:
            display_df['GTV'] = display_df['GTV'].apply(lambda x: f"₹{x:,.0f}")
        if 'Revenue (INR)' in display_df.columns:
            display_df['Revenue (INR)'] = display_df['Revenue (INR)'].apply(lambda x: f"₹{x:,.0f}")
        
        # Format ROI columns
        roi_cols = ['ROI', 'ROI (With Take Rate)']
        for col in roi_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}x")
        
        if 'Cost per Order' in display_df.columns:
            display_df['Cost per Order'] = display_df['Cost per Order'].apply(lambda x: f"₹{x:.2f}")
        
        # Create color coding based on performance percentiles
        def get_color_conditions(df, numeric_df):
            conditions = []
            
            # Performance-based color coding
            performance_cols = ['ROI', 'CTR', 'Conversion Rate', 'Order per Sent', 'Delivery Rate']
            
            for col in performance_cols:
                if col in numeric_df.columns:
                    values = numeric_df[col]
                    p90 = values.quantile(0.9)
                    p75 = values.quantile(0.75)
                    p25 = values.quantile(0.25)
                    p10 = values.quantile(0.1)
                    
                    # Excellent performance (top 10%) - Dark Green
                    conditions.append({
                        'if': {
                            'filter_query': f'{{{col}}} >= {p90}',
                            'column_id': col
                        },
                        'backgroundColor': '#1e7145',
                        'color': 'white',

                        'fontWeight': 'bold'
                    })
                    
                    # Good performance (75th-90th percentile) - Light Green
                    conditions.append({
                        'if': {
                            'filter_query': f'{{{col}}} >= {p75} && {{{col}}} < {p90}',
                            'column_id': col
                        },
                        'backgroundColor': '#4caf50',
                        'color': 'white'
                    })
                    
                    # Poor performance (10th-25th percentile) - Light Red
                    conditions.append({
                        'if': {
                            'filter_query': f'{{{col}}} > {p10} && {{{col}}} <= {p25}',
                            'column_id': col
                        },
                        'backgroundColor': '#ef5350',
                        'color': 'white'
                    })
                    
                    # Very poor performance (bottom 10%) - Dark Red
                    conditions.append({
                        'if': {
                            'filter_query': f'{{{col}}} <= {p10}',
                            'column_id': col
                        },
                        'backgroundColor': '#c62828',
                        'color': 'white',
                        'fontWeight': 'bold'
                    })
            
            # Volume-based color coding for high volume campaigns
            volume_cols = ['Sent', 'Delivered', 'Unique Clicks', 'Unique Click-Through Conversions']
            for col in volume_cols:
                if col in numeric_df.columns:
                    values = numeric_df[col]
                    p95 = values.quantile(0.95)
                    
                    # High volume campaigns - Blue highlight
                    conditions.append({
                        'if': {
                            'filter_query': f'{{{col}}} >= {p95}',
                            'column_id': col
                        },
                        'backgroundColor': '#1976d2',
                        'color': 'white',
                        'fontWeight': 'bold'
                    })
            
            return conditions
        
        color_conditions = get_color_conditions(display_df, numeric_df)
        
        return html.Div([
            # Sorting Controls
            dbc.Card([
                dbc.CardHeader([
                    html.H6("Advanced Sorting & Filtering", className="mb-0")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "🎯 Sort by Performance (Order per Sent → CTR)",
                                id="sort-performance-btn",
                                color="success",
                                size="sm",
                                className="me-2 mb-2"
                            ),
                            dbc.Button(
                                "💰 Sort by ROI",
                                id="sort-roi-btn", 
                                color="primary",
                                size="sm",
                                className="me-2 mb-2"
                            ),
                            dbc.Button(
                                "📊 Sort by Volume",
                                id="sort-volume-btn",
                                color="info", 
                                size="sm",
                                className="me-2 mb-2"
                            ),
                            dbc.Button(
                                "🔄 Reset Sort",
                                id="reset-sort-btn",
                                color="secondary",
                                size="sm",
                                className="mb-2"
                            )
                        ], width=8),
                        dbc.Col([
                            html.Div([
                                html.Small("🟢 Top Performers", className="me-3"),
                                html.Small("🔴 Poor Performers", className="me-3"),
                                html.Small("🔵 High Volume")
                            ], className="text-end")
                        ], width=4)
                    ])
                ])
            ], className="mb-3"),
            
            html.H5(f"Summary Metrics - {len(display_df)} journeys", className="mb-3"),
            
            # Enhanced Data Table with color coding
            dash_table.DataTable(
                id="summary-data-table",
                data=display_df.to_dict('records'),
                columns=[{"name": col, "id": col, "type": "numeric" if col in ['ROI', 'CTR', 'Conversion Rate', 'Order per Sent', 'Delivery Rate'] else "text"} for col in display_df.columns],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '8px',
                    'fontSize': '12px',
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'fontFamily': 'Arial, sans-serif'
                },
                style_header={
                    'backgroundColor': '#2c3e50',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                    'fontSize': '13px'
                },
                style_data_conditional=[
                    {
                        'if': {'column_id': 'Journey Name'},
                        'textAlign': 'left',
                        'fontWeight': 'bold',
                        'backgroundColor': '#f8f9fa'
                    },
                    # Alternating row colors
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ] + color_conditions,
                sort_action="custom",
                sort_mode="multi",
                sort_by=[],
                filter_action="native",
                page_size=25,
                export_format="xlsx",
                export_headers="display",
                tooltip_data=[
                    {
                        column: {
                            'value': f"Percentile rank for {column}",
                            'type': 'markdown'
                        }
                        for column in display_df.columns
                    } for row in display_df.to_dict('records')
                ],
                tooltip_duration=None
            )
        ])
        
    except Exception as e:
        return dbc.Alert(f"Error creating summary: {str(e)}", color="danger")

# Add these new callbacks for sorting functionality

@app.callback(
    Output("summary-data-table", "sort_by"),
    [Input("sort-performance-btn", "n_clicks"),
     Input("sort-roi-btn", "n_clicks"), 
     Input("sort-volume-btn", "n_clicks"),
     Input("reset-sort-btn", "n_clicks")],
    prevent_initial_call=True
)
def update_table_sorting(perf_clicks, roi_clicks, volume_clicks, reset_clicks):
    """Handle advanced sorting for summary table"""
    ctx = dash.callback_context
    if not ctx.triggered:
        return []
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'sort-performance-btn':
        # Sort by Order per Sent (descending), then CTR (descending)
        return [
            {"column_id": "Order per Sent", "direction": "desc"},
            {"column_id": "CTR", "direction": "desc"}
        ]
    elif button_id == 'sort-roi-btn':
        # Sort by ROI (descending)
        return [{"column_id": "ROI", "direction": "desc"}]
    elif button_id == 'sort-volume-btn':
        # Sort by Sent volume (descending), then Orders (descending)
        return [
            {"column_id": "Sent", "direction": "desc"},
            {"column_id": "Unique Click-Through Conversions", "direction": "desc"}
        ]
    elif button_id == 'reset-sort-btn':
        # Reset to default (no sorting)
        return []
    
    return []

@app.callback(
    Output("summary-data-table", "data"),
    [Input("summary-data-table", "sort_by"),
     Input("main-tabs", "active_tab")],
    [State('journey-checklist', 'value'),
     State('channel-checklist', 'value'),
     State('status-checklist', 'value'),
     State('user-type-checklist', 'value'),
     State('date-picker-range', 'start_date'),
     State('date-picker-range', 'end_date')]
)
def sort_summary_table(sort_by, active_tab, selected_journeys, selected_channels, selected_statuses, selected_user_types, start_date, end_date):
    """Handle custom sorting for summary table"""
    if active_tab != "summary" or not selected_journeys or not selected_channels:
        return []
    
    try:
        # Apply filters to get current data
        filtered_data = data_loaded.copy()
        
        if selected_journeys:
            filtered_data = filtered_data[filtered_data['Journey Name'].isin(selected_journeys)]
        if selected_channels:
            filtered_data = filtered_data[filtered_data['Channel'].isin(selected_channels)]
        if selected_statuses:
            filtered_data = filtered_data[filtered_data['Status'].isin(selected_statuses)]
        if selected_user_types:
            filtered_data = filtered_data[filtered_data['User_Type'].isin(selected_user_types)]
        if start_date and end_date:
            filtered_data = filtered_data[(filtered_data['Day'] >= start_date) & (filtered_data['Day'] <= end_date)]
        
        if filtered_data.empty:
            return []
        
        # Get summary metrics
        summary_metrics = analyzer.calculate_summary_metrics(filtered_data)
        if summary_metrics.empty:
            return []
        
        # Apply custom sorting
        if sort_by:
            # Convert percentage strings back to numeric for sorting
            df_for_sorting = summary_metrics.copy()
            
            # Sort the dataframe
            sort_columns = []
            ascending_list = []
            
            for sort_item in sort_by:
                column_id = sort_item['column_id']
                direction = sort_item['direction']
                
                sort_columns.append(column_id)
                ascending_list.append(direction == 'asc')
            
            if sort_columns:
                df_for_sorting = df_for_sorting.sort_values(
                    by=sort_columns,
                    ascending=ascending_list
                )
        else:
            df_for_sorting = summary_metrics.copy()
        
        # Format for display
        display_df = df_for_sorting.copy()
        
        # Format numeric columns
        for col in ['Sent', 'Delivered', 'Unique Clicks', 'Unique Click-Through Conversions', 'Unique Impressions', 'Unique Conversions']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{int(x):,}")
        
        # Format percentage columns
        percentage_cols = ['Delivery Rate', 'CTR', 'Conversion Rate', 'Order per Sent']
        for col in percentage_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
        
        # Format financial columns
        if 'Cost' in display_df.columns:
            display_df['Cost'] = display_df['Cost'].apply(lambda x: f"₹{x:,.2f}")
        if 'GTV' in display_df.columns:
            display_df['GTV'] = display_df['GTV'].apply(lambda x: f"₹{x:,.0f}")
        if 'Revenue (INR)' in display_df.columns:
            display_df['Revenue (INR)'] = display_df['Revenue (INR)'].apply(lambda x: f"₹{x:,.0f}")
        
        # Format ROI columns
        roi_cols = ['ROI', 'ROI (With Take Rate)']
        for col in roi_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}x")
        
        if 'Cost per Order' in display_df.columns:
            display_df['Cost per Order'] = display_df['Cost per Order'].apply(lambda x: f"₹{x:.2f}")
        
        return display_df.to_dict('records')
        
    except Exception as e:
        print(f"Error in sort_summary_table: {e}")
        return []

# Add this create_details_tab function to your test.py file

def create_details_tab(filtered_data):
    """Create detailed campaign view"""
    try:
        # Select relevant columns for campaign details
        detail_cols = ['Day', 'Campaign Name', 'Journey Name', 'Channel', 'Status', 
                      'Sent', 'Delivered', 'Unique Clicks', 'Unique Click-Through Conversions',
                      'Cost', 'GTV', 'ROI', 'User_Type', 'User_Channel']
        
        available_cols = [col for col in detail_cols if col in filtered_data.columns]
        campaign_details = filtered_data[available_cols].copy()
        
        # Format numeric columns
        for col in ['Sent', 'Delivered', 'Unique Clicks', 'Unique Click-Through Conversions']:
            if col in campaign_details.columns:
                campaign_details[col] = campaign_details[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "0")
        
        # Format financial columns
        if 'Cost' in campaign_details.columns:
            campaign_details['Cost'] = campaign_details['Cost'].apply(lambda x: f"₹{x:.2f}")
        if 'GTV' in campaign_details.columns:
            campaign_details['GTV'] = campaign_details['GTV'].apply(lambda x: f"₹{x:,.0f}")
        if 'ROI' in campaign_details.columns:
            campaign_details['ROI'] = campaign_details['ROI'].apply(lambda x: f"{x:.2f}x")
        
        return html.Div([
            html.H5(f"Campaign Details - {len(campaign_details)} campaigns", className="mb-3"),
            dash_table.DataTable(
                data=campaign_details.to_dict('records'),
                columns=[{"name": i, "id": i} for i in campaign_details.columns],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '8px',
                    'fontSize': '11px',
                    'whiteSpace': 'normal',
                    'height': 'auto'
                },
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold',
                    'textAlign': 'center'
                },
                sort_action="native",
                sort_mode="multi",
                filter_action="native",
                page_size=25,
                export_format="xlsx"
            )
        ])
        
    except Exception as e:
        return dbc.Alert(f"Error creating details: {str(e)}", color="danger")

# Add this at the very end of your test.py file to run the app

if __name__ == '__main__':
    try:
        local_ip = get_local_ip()
        port = 8058
        
        print(f"\n🚀 Starting Journey Analytics Dashboard...")
        print(f"📊 Data loaded: {len(data_loaded) if data_loaded is not None else 0} records")
        print(f"🎯 Journeys available: {len(unique_journeys)}")
        print(f"📡 Channels available: {len(unique_channels)}")
        print(f"\n🌐 Dashboard will be available at:")
        print(f"   Local:    http://127.0.0.1:{port}")
        print(f"   Network:  http://{local_ip}:{port}")
        print(f"\n💡 Tip: Use the network URL to access from other devices on the same network")
        print(f"🔄 Press Ctrl+C to stop the server\n")
        
        app.run(
            debug=False,  # Set to False for production
            host='0.0.0.0',  # Allow access from any IP
            port=port,
            dev_tools_hot_reload=False
        )
        
    except Exception as e:
        print(f"❌ Error starting the app: {e}")
        print("🔧 Try changing the port number if 8050 is already in use")
        
        # Try alternative port
        try:
            port = 8051
            print(f"🔄 Trying alternative port {port}...")
            app.run(
                debug=False,
                host='0.0.0.0',
                port=port,
                dev_tools_hot_reload=False
            )
        except Exception as e2:
            print(f"❌ Error on alternative port: {e2}")
            print("💡 Please check if Python has network permissions or try a different port")