import pandas as pd
import numpy as np
import warnings
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import socket

warnings.filterwarnings('ignore')

# Configuration Variables
file_path1 = "reportjun27.csv"  # Path to the CSV file
AOV = 1500  
TAKE_RATE = 0.08  # Add this line
CHANNEL_COSTS = {
    'Email': 0.02,
    'Push': 0.01, 
    'SMS': 0.11,
    'WhatsApp': 0.11
}

# Update the JOURNEY_USER_TYPE_MAPPING to handle variations
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
        self.take_rate = take_rate  # Add this line
        self.raw_data = None
        
    # Update the load_and_process_data method to use these mappings
    def load_and_process_data(self):
        """Load and process campaign data"""
        try:
            df = pd.read_csv(self.file_path)
            df = df.fillna(0)
            
            # Print column names to debug
            print("Available columns:", df.columns.tolist())
            
            # Map the correct column names based on your CSV structure
            numeric_columns = ['Sent', 'Delivered', 'Unique Impressions', 'Unique Clicks', 
                              'Unique Conversions', 'Unique Click-Through Conversions']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(',', '').str.replace('nan', '0'), 
                        errors='coerce'
                    ).fillna(0)
            
            # Ensure string columns are properly formatted and strip whitespace
            string_columns = ['Journey Name', 'Channel', 'Status', 'Campaign Name']
            for col in string_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()  # Add .str.strip() to remove whitespace
            
            # Debug: Print unique journey names to see exact format
            print("\nUnique Journey Names in CSV:")
            unique_journeys_csv = df['Journey Name'].unique()
            for journey in sorted(unique_journeys_csv):
                print(f"'{journey}'")
            
            # Add User_Type and User_Channel columns based on Journey Name mapping
            df['User_Type'] = df['Journey Name'].map(JOURNEY_USER_TYPE_MAPPING).fillna('Unknown')
            df['User_Channel'] = df['Journey Name'].map(JOURNEY_USER_CHANNEL_MAPPING).fillna('Unknown')
            
            # Debug: Show mapping results
            print("\nMapping Results:")
            mapping_debug = df[['Journey Name', 'User_Type', 'User_Channel']].drop_duplicates()
            for _, row in mapping_debug.iterrows():
                if 'OC2 to OC' in row['Journey Name']:
                    print(f"'{row['Journey Name']}' -> User_Type: '{row['User_Type']}', User_Channel: '{row['User_Channel']}'")
            
            print(f"\nUser_Type values: {sorted(df['User_Type'].unique())}")
            print(f"User_Channel values: {sorted(df['User_Channel'].unique())}")
        
            self.raw_data = df
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def calculate_summary_metrics(self, filtered_df=None):
        """Calculate main summary table metrics matching your reference data"""
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
                'Unique Click-Through Conversions': 'sum'
            }).reset_index()
            
            grouped.rename(columns={'Campaign ID': 'Count of Campaign ID'}, inplace=True)
            
            # Calculate performance metrics exactly as in your reference
            grouped['Delivery Rate'] = np.where(grouped['Sent'] > 0, 
                                      (grouped['Delivered'] / grouped['Sent'] * 100).round(0), 0)
            
            grouped['CTR'] = np.where(grouped['Delivered'] > 0,
                                       (grouped['Unique Clicks'] / grouped['Delivered'] * 100).round(1), 0)
            
            grouped['Conversion Rate'] = np.where(grouped['Unique Clicks'] > 0,
                                      (grouped['Unique Click-Through Conversions'] / grouped['Unique Clicks'] * 100).round(1), 0)
            
            grouped['Order per Sent'] = np.where(grouped['Sent'] > 0,
                                      (grouped['Unique Click-Through Conversions'] / grouped['Sent'] * 100).round(2), 0)
            
            # Calculate financial metrics
            grouped['Cost'] = 0
            grouped['GTV'] = grouped['Unique Click-Through Conversions'] * self.aov
            grouped['ROI'] = 0
            grouped['ROI (With Take Rate)'] = 0  # Add this line
            
            # Calculate cost per journey based on channel breakdown
            for idx, row in grouped.iterrows():
                try:
                    journey_data = df[df['Journey Name'] == row['Journey Name']]
                    total_cost = 0
                    
                    # Group by channel for this journey
                    channel_breakdown = journey_data.groupby('Channel')['Sent'].sum()
                    
                    for channel, sent_count in channel_breakdown.items():
                        if channel in self.channel_costs:
                            total_cost += sent_count * self.channel_costs[channel]
                    
                    grouped.at[idx, 'Cost'] = total_cost
                    
                    # Calculate ROI
                    if total_cost > 0:
                        roi_value = (row['GTV'] / total_cost)
                        grouped.at[idx, 'ROI'] = roi_value
                        grouped.at[idx, 'ROI (With Take Rate)'] = self.take_rate * roi_value  # Add this line
                    else:
                        grouped.at[idx, 'ROI'] = 0
                        grouped.at[idx, 'ROI (With Take Rate)'] = 0  # Add this line
                        
                except Exception as e:
                    print(f"Error calculating cost for {row['Journey Name']}: {e}")
                    grouped.at[idx, 'Cost'] = 0
                    grouped.at[idx, 'ROI'] = 0
                    grouped.at[idx, 'ROI (With Take Rate)'] = 0  # Add this line
            
            return grouped
            
        except Exception as e:
            print(f"Error in calculate_summary_metrics: {e}")
            return pd.DataFrame()

    def get_channel_breakdown(self, filtered_df=None):
        """Get channel-wise breakdown matching your pivot table format"""
        df = filtered_df if filtered_df is not None else self.raw_data
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        try:
            # Create pivot tables for each metric
            sent_pivot = df.pivot_table(
                values='Sent', 
                index='Journey Name', 
                columns='Channel', 
                aggfunc='sum', 
                fill_value=0
            )
            
            delivered_pivot = df.pivot_table(
                values='Delivered', 
                index='Journey Name', 
                columns='Channel', 
                aggfunc='sum', 
                fill_value=0
            )
            
            conversions_pivot = df.pivot_table(
                values='Unique Click-Through Conversions', 
                index='Journey Name', 
                columns='Channel', 
                aggfunc='sum', 
                fill_value=0
            )

            channel_name_mapping = {
            'Push': 'Push Sent',
            'SMS': 'SMS Sent', 
            'WhatsApp': 'WhatsApp Sent',
            'Email': 'Email Sent'
        }
            
        
            
            # Calculate totals
            sent_pivot['Total'] = sent_pivot.sum(axis=1)
            sent_pivot = sent_pivot.rename(columns=channel_name_mapping)
            delivered_pivot['Total'] = delivered_pivot.sum(axis=1)
            conversions_pivot['Total'] = conversions_pivot.sum(axis=1)
            
            return {
                'sent': sent_pivot,
                'delivered': delivered_pivot,
                'conversions': conversions_pivot
            }
            
        except Exception as e:
            print(f"Error in get_channel_breakdown: {e}")
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
            valid_dates = data_loaded['Day'].dropna()
            if not valid_dates.empty:
                min_date = valid_dates.min()
                max_date = valid_dates.max()
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

# Define the layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Journeys Performance Analytics Dashboard", 
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
                
                # User Channel Selection
                dbc.Col([
                    html.Label("User Channel:", className="fw-bold"),
                    dcc.Checklist(
                        id='user-channel-checklist',
                        options=[{'label': user_channel, 'value': user_channel} for user_channel in unique_user_channels],
                        value=unique_user_channels,
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
            ]),
            
            html.Hr(),
            
            # Quick Filter Buttons
            dbc.Row([
                dbc.Col([
                    html.Label("Quick Filters:", className="fw-bold"),
                    html.Div([
                        dbc.Button("Top Performers", id="quick-top-performers", size="sm", color="success", className="me-2 mb-2"),
                        dbc.Button("Email Only", id="quick-email-only", size="sm", color="outline-primary", className="me-2 mb-2"),
                        dbc.Button("Push Only", id="quick-push-only", size="sm", color="outline-warning", className="me-2 mb-2"),
                        dbc.Button("High Value", id="quick-high-value", size="sm", color="primary", className="me-2 mb-2"),
                        dbc.Button("Reset All", id="reset-filters", size="sm", color="secondary", className="mb-2"),
                    ])
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Always visible KPI Cards
    html.Div(id="kpi-cards", className="mb-4"),
    
    # Collapsible Content Sections
    html.Div([
        
        # Summary Table Section
        dbc.Card([
            dbc.CardHeader([
                dbc.Button(
                    "Summary Table",
                    id="collapse-table-btn",
                    color="primary",
                    className="btn-block text-left w-100",
                    style={"border": "none", "background": "none", "color": "inherit", "font-weight": "bold", "text-align": "left"}
                )
            ]),
            dbc.Collapse([
                dbc.CardBody([
                    html.Div(id="summary-table")
                ])
            ], id="collapse-table", is_open=True)
        ], className="mb-3"),
        
        # Charts Section
        dbc.Card([
            dbc.CardHeader([
                dbc.Button(
                    "Performance Charts",
                    id="collapse-charts-btn",
                    color="success",
                    className="btn-block text-left w-100",
                    style={"border": "none", "background": "none", "color": "inherit", "font-weight": "bold", "text-align": "left"}
                )
            ]),
            dbc.Collapse([
                dbc.CardBody([
                    html.Div(id="charts-content")
                ])
            ], id="collapse-charts", is_open=False)
        ], className="mb-3"),
        
        # Channel Breakdown Section
        dbc.Card([
            dbc.CardHeader([
                dbc.Button(
                    "Channel Breakdown",
                    id="collapse-channels-btn",
                    color="info",
                    className="btn-block text-left w-100",
                    style={"border": "none", "background": "none", "color": "inherit", "font-weight": "bold", "text-align": "left"}
                )
            ]),
            dbc.Collapse([
                dbc.CardBody([
                    html.Div(id="channel-breakdown")
                ])
            ], id="collapse-channels", is_open=False)
        ], className="mb-3"),
        
        # Campaign Details Section
        dbc.Card([
            dbc.CardHeader([
                dbc.Button(
                    "Campaign Details",
                    id="collapse-details-btn",
                    color="warning",
                    className="btn-block text-left w-100",
                    style={"border": "none", "background": "none", "color": "inherit", "font-weight": "bold", "text-align": "left"}
                )
            ]),
            dbc.Collapse([
                dbc.CardBody([
                    html.Div(id="campaign-details")
                ])
            ], id="collapse-details", is_open=False)
        ], className="mb-3"),
        
    ])
    
], fluid=True)

# Collapse button callbacks
@app.callback(
    Output("collapse-table", "is_open"),
    [Input("collapse-table-btn", "n_clicks")],
    [State("collapse-table", "is_open")],
)
def toggle_table_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("collapse-charts", "is_open"),
    [Input("collapse-charts-btn", "n_clicks")],
    [State("collapse-charts", "is_open")],
)
def toggle_charts_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("collapse-channels", "is_open"),
    [Input("collapse-channels-btn", "n_clicks")],
    [State("collapse-channels", "is_open")],
)
def toggle_channels_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("collapse-details", "is_open"),
    [Input("collapse-details-btn", "n_clicks")],
    [State("collapse-details", "is_open")],
)
def toggle_details_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# Filter callbacks
@app.callback(
    [Output('journey-checklist', 'value'),
     Output('channel-checklist', 'value')],
    [Input('select-all-journeys', 'n_clicks'),
     Input('clear-all-journeys', 'n_clicks'),
     Input('top-roi-journeys', 'n_clicks'),
     Input('select-all-channels', 'n_clicks'),
     Input('clear-all-channels', 'n_clicks'),
     Input('quick-top-performers', 'n_clicks'),
     Input('quick-email-only', 'n_clicks'),
     Input('quick-push-only', 'n_clicks'),
     Input('quick-high-value', 'n_clicks'),
     Input('reset-filters', 'n_clicks')],
    [State('journey-checklist', 'value'),
     State('channel-checklist', 'value')],
    prevent_initial_call=True
)
def update_filters(select_all_j, clear_all_j, top_roi_j, select_all_c, clear_all_c,
                  quick_top, quick_email, quick_push, quick_value, reset_all,
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
        elif button_id == 'quick-top-performers':
            if not summary_df.empty and 'ROI' in summary_df.columns:
                try:
                    top_journeys = summary_df.nlargest(5, 'ROI')['Journey Name'].tolist()
                    return top_journeys, current_channels or []
                except:
                    pass
            return current_journeys or [], current_channels or []
        elif button_id == 'quick-email-only':
            return current_journeys or [], ['Email']
        elif button_id == 'quick-push-only':
            return current_journeys or [], ['Push']
        elif button_id == 'quick-high-value':
            if not summary_df.empty and 'GTV' in summary_df.columns:
                try:
                    median_gtv = summary_df['GTV'].median()
                    high_value = summary_df[summary_df['GTV'] > median_gtv]['Journey Name'].tolist()
                    return high_value, current_channels or []
                except:
                    pass
            return current_journeys or [], current_channels or []
        elif button_id == 'reset-filters':
            return unique_journeys, unique_channels
        
        return current_journeys or [], current_channels or []
        
    except Exception as e:
        print(f"Error in update_filters: {e}")
        return current_journeys or [], current_channels or []

# KPI Cards callback
@app.callback(
    Output("kpi-cards", "children"),
    [Input('journey-checklist', 'value'),
     Input('channel-checklist', 'value'),
     Input('status-checklist', 'value'),
     Input('user-type-checklist', 'value'),
     Input('user-channel-checklist', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('performance-filter', 'value')]
)
def update_kpi_cards(selected_journeys, selected_channels, selected_statuses, selected_user_types, selected_user_channels, start_date, end_date, perf_filter):
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
        if selected_user_channels and selected_user_channels:
            filtered_data = filtered_data[filtered_data['User_Channel'].isin(selected_user_channels)]
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

# Summary table callback
@app.callback(
    Output("summary-table", "children"),
    [Input('journey-checklist', 'value'),
     Input('channel-checklist', 'value'),
     Input('status-checklist', 'value'),
     Input('user-type-checklist', 'value'),
     Input('user-channel-checklist', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('performance-filter', 'value')]
)
def update_summary_table(selected_journeys, selected_channels, selected_statuses, selected_user_types, selected_user_channels, start_date, end_date, perf_filter):
    try:
        if not selected_journeys or not selected_channels:
            return html.Div("Select filters to view summary table", className="text-center text-muted p-4")
        
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
        if selected_user_channels and selected_user_channels:
            filtered_data = filtered_data[filtered_data['User_Channel'].isin(selected_user_channels)]
        if start_date and end_date and 'Day' in filtered_data.columns:
            filtered_data = filtered_data[(filtered_data['Day'] >= start_date) & (filtered_data['Day'] <= end_date)]
        
        if filtered_data.empty:
            return html.Div("No data matches your selection", className="text-center text-warning p-4")
        
        # Get summary for selected data
        filtered_summary = analyzer.calculate_summary_metrics(filtered_data)
        
        # Apply performance filter
        if perf_filter == 'high_roi' and 'ROI' in filtered_summary.columns:
            filtered_summary = filtered_summary[filtered_summary['ROI'] > 10]
        elif perf_filter == 'high_ctr' and 'CTR' in filtered_summary.columns:
            filtered_summary = filtered_summary[filtered_summary['CTR'] > 2]
        elif perf_filter == 'high_conv' and 'Conversion Rate' in filtered_summary.columns:
            filtered_summary = filtered_summary[filtered_summary['Conversion Rate'] > 1]
        elif perf_filter == 'low_perf' and 'ROI' in filtered_summary.columns:
            filtered_summary = filtered_summary[filtered_summary['ROI'] < 5]
        
        if filtered_summary.empty:
            return html.Div("No summary data available for current filters", className="text-center text-warning p-4")
        
        # Define columns list (changed from set to list)
        summary_columns = [
            'Journey Name', 'Count of Campaign ID', 'Sent', 'Delivered', 
            'Unique Clicks', 'Unique Click-Through Conversions', 
            'Delivery Rate', 'CTR', 'Conversion Rate', 'Order per Sent', 'Cost', 'GTV', 'ROI', 'ROI (With Take Rate)'
        ]
        
        # Keep numeric values for conditional formatting
        numeric_summary = filtered_summary[summary_columns].copy()
        
        # Format display columns to match your reference format
        display_df = filtered_summary[summary_columns].copy()
        
        # Format numbers to match your reference
        for col in ['Sent', 'Delivered', 'Unique Clicks', 'Unique Click-Through Conversions']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{int(x):,}")
        
        display_df['Delivery Rate'] = display_df['Delivery Rate'].apply(lambda x: f"{x:.0f}%")
        display_df['CTR'] = display_df['CTR'].apply(lambda x: f"{x:.1f}%")
        display_df['Conversion Rate'] = display_df['Conversion Rate'].apply(lambda x: f"{x:.1f}%")
        display_df['Order per Sent'] = display_df['Order per Sent'].apply(lambda x: f"{x:.2f}%")
        
        display_df['Cost'] = display_df['Cost'].apply(lambda x: f"₹{x:,.2f}")
        display_df['GTV'] = display_df['GTV'].apply(lambda x: f"₹{x:,.0f}")
        display_df['ROI'] = display_df['ROI'].apply(lambda x: f"₹{x:.2f}")
        display_df['ROI (With Take Rate)'] = display_df['ROI (With Take Rate)'].apply(lambda x: f"₹{x:.2f}")  # Add this line
        
        # Define conditional formatting for important columns
        def get_color_scale(values, column_name):
            if len(values) == 0:
                return []
            
            min_val = values.min()
            max_val = values.max()
            
            colors = []
            for val in values:
                if column_name in ['ROI', 'Order per Sent', 'Conversion Rate', 'CTR']:
                    # Higher is better - green scale
                    if max_val == min_val:
                        intensity = 0.5
                    else:
                        intensity = (val - min_val) / (max_val - min_val)
                    
                    if intensity >= 0.8:
                        color = '#1f7a1f'  # Dark green
                    elif intensity >= 0.6:
                        color = '#28a745'  # Green
                    elif intensity >= 0.4:
                        color = '#6db36d'  # Light green
                    elif intensity >= 0.2:
                        color = '#ffc107'  # Yellow
                    else:
                        color = '#dc3545'  # Red
                        
                elif column_name == 'Cost':
                    # Lower is better - reverse scale
                    if max_val == min_val:
                        intensity = 0.5
                    else:
                        intensity = 1 - (val - min_val) / (max_val - min_val)
                    
                    if intensity >= 0.8:
                        color = '#1f7a1f'  # Dark green
                    elif intensity >= 0.6:
                        color = '#28a745'  # Green
                    elif intensity >= 0.4:
                        color = '#6db36d'  # Light green
                    elif intensity >= 0.2:
                        color = '#ffc107'  # Yellow
                    else:
                        color = '#dc3545'  # Red
                else:
                    color = '#ffffff'  # White for other columns
                    
                colors.append(color)
            return colors
        
        # Prepare style_data_conditional for color coding
        style_data_conditional = []
        
        # Color code important columns
        important_columns = ['ROI', 'ROI (With Take Rate)', 'Order per Sent', 'Conversion Rate', 'CTR', 'Cost']
        
        for col in important_columns:
            if col in numeric_summary.columns:
                values = numeric_summary[col]
                colors = get_color_scale(values, col)
                
                for idx, color in enumerate(colors):
                    style_data_conditional.append({
                        'if': {'row_index': idx, 'column_id': col},
                        'backgroundColor': color,
                        'color': 'white' if color in ['#1f7a1f', '#28a745', '#dc3545'] else 'black'
                    })
        
        return [
            html.H5(f"Summary for {len(display_df)} filtered journeys", className="mb-3"),
            dash_table.DataTable(
                data=display_df.to_dict('records'),
                columns=[{"name": col, "id": col} for col in display_df.columns],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left', 
                    'padding': '10px', 
                    'fontSize': '12px',
                    'whiteSpace': 'normal',
                    'height': 'auto'
                },
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)', 
                    'fontWeight': 'bold',
                    'textAlign': 'center'
                },
                style_data_conditional=style_data_conditional,
                sort_action="native",
                sort_mode="multi",
                page_size=15,
                export_format="xlsx"
            )
        ]
        
    except Exception as e:
        return dbc.Alert(f"Error creating summary table: {str(e)}", color="danger")

# Channel breakdown callback - Enhanced with sorting
@app.callback(
    Output("channel-breakdown", "children"),
    [Input('journey-checklist', 'value'),
     Input('channel-checklist', 'value'),
     Input('status-checklist', 'value'),
     Input('user-type-checklist', 'value'),
     Input('user-channel-checklist', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_channel_breakdown(selected_journeys, selected_channels, selected_statuses, selected_user_types, selected_user_channels, start_date, end_date):
    try:
        if not selected_journeys or not selected_channels:
            return html.Div("Select filters to view channel breakdown", className="text-center text-muted p-4")
        
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
        if selected_user_channels and selected_user_channels:
            filtered_data = filtered_data[filtered_data['User_Channel'].isin(selected_user_channels)]
        if start_date and end_date and 'Day' in filtered_data.columns:
            filtered_data = filtered_data[(filtered_data['Day'] >= start_date) & (filtered_data['Day'] <= end_date)]
        
        # Create channel breakdown similar to your reference pivot table
        breakdown_data = analyzer.get_channel_breakdown(filtered_data)
        
        if not breakdown_data:
            return html.Div("No channel breakdown data available", className="text-center text-warning p-4")
        
        # Convert to display format
        sent_pivot = breakdown_data['sent']
        
        # Reset index to make Journey Name a column
        sent_display = sent_pivot.reset_index()
        
        # Format numeric columns with commas
        numeric_cols = [col for col in sent_display.columns if col != 'Journey Name']
        for col in numeric_cols:
            sent_display[col] = sent_display[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) and x != 0 else "0")
        
        return [
            html.H5(f"Channel breakdown for {len(set(filtered_data['Journey Name']))} filtered journeys", className="mb-3"),
            dash_table.DataTable(
                data=sent_display.to_dict('records'),
                columns=[{"name": i, "id": i} for i in sent_display.columns],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left', 
                    'padding': '8px', 
                    'fontSize': '12px',
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
                page_size=20,
                export_format="xlsx"
            )
        ]
        
    except Exception as e:
        return dbc.Alert(f"Error creating channel breakdown: {str(e)}", color="danger")

# Campaign details callback - Enhanced with sorting
@app.callback(
    Output("campaign-details", "children"),
    [Input('journey-checklist', 'value'),
     Input('channel-checklist', 'value'),
     Input('status-checklist', 'value'),
     Input('user-type-checklist', 'value'),
     Input('user-channel-checklist', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_campaign_details(selected_journeys, selected_channels, selected_statuses, selected_user_types, selected_user_channels, start_date, end_date):
    try:
        if not selected_journeys or not selected_channels:
            return html.Div("Select filters to view campaign details", className="text-center text-muted p-4")
        
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
        if selected_user_channels and selected_user_channels:
            filtered_data = filtered_data[filtered_data['User_Channel'].isin(selected_user_channels)]
        if start_date and end_date and 'Day' in filtered_data.columns:
            filtered_data = filtered_data[(filtered_data['Day'] >= start_date) & (filtered_data['Day'] <= end_date)]
        
        # Select relevant columns for campaign details
        detail_cols = ['Campaign Name', 'Journey Name', 'Channel', 'Status', 'Sent', 'Delivered', 'Unique Click-Through Conversions']
        available_detail_cols = [col for col in detail_cols if col in filtered_data.columns]
        
        campaign_details = filtered_data[available_detail_cols].copy()
        
        # Add the new conversion rate column
        campaign_details['Conv Rate from Delivered'] = np.where(
            campaign_details['Delivered'] > 0,
            (campaign_details['Unique Click-Through Conversions'] / campaign_details['Delivered'] * 100).round(2),
            0
        )
        
        # Format numeric columns
        for col in ['Sent', 'Delivered', 'Unique Click-Through Conversions']:
            if col in campaign_details.columns:
                campaign_details[col] = campaign_details[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "0")
        
        # Format the conversion rate column
        campaign_details['Conv Rate from Delivered'] = campaign_details['Conv Rate from Delivered'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "0.00%")
        
        return [
            html.H5(f"Campaign Details - {len(campaign_details)} campaigns", className="mb-3"),
            dash_table.DataTable(
                data=campaign_details.to_dict('records'),
                columns=[{"name": i, "id": i} for i in campaign_details.columns],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left', 
                    'padding': '8px', 
                    'fontSize': '12px',
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
        ]
        
    except Exception as e:
        return dbc.Alert(f"Error creating campaign details: {str(e)}", color="danger")

# Charts callback - Complete implementation
@app.callback(
    Output("charts-content", "children"),
    [Input('journey-checklist', 'value'),
     Input('channel-checklist', 'value'),
     Input('status-checklist', 'value'),
     Input('user-type-checklist', 'value'),
     Input('user-channel-checklist', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('performance-filter', 'value')]
)
def update_charts(selected_journeys, selected_channels, selected_statuses, selected_user_types, selected_user_channels, start_date, end_date, perf_filter):
    try:
        if not selected_journeys or not selected_channels:
            return html.Div("Select filters to view charts", className="text-center text-muted p-4")
        
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
        if selected_user_channels and selected_user_channels:
            filtered_data = filtered_data[filtered_data['User_Channel'].isin(selected_user_channels)]
        if start_date and end_date and 'Day' in filtered_data.columns:
            filtered_data = filtered_data[(filtered_data['Day'] >= start_date) & (filtered_data['Day'] <= end_date)]
        
        filtered_summary = analyzer.calculate_summary_metrics(filtered_data)
        
        # Apply performance filter
        if perf_filter == 'high_roi' and 'ROI' in filtered_summary.columns:
            filtered_summary = filtered_summary[filtered_summary['ROI'] > 10]
        elif perf_filter == 'high_ctr' and 'CTR' in filtered_summary.columns:
            filtered_summary = filtered_summary[filtered_summary['CTR'] > 2]
        elif perf_filter == 'high_conv' and 'Conversion Rate' in filtered_summary.columns:
            filtered_summary = filtered_summary[filtered_summary['Conversion Rate'] > 1]
        elif perf_filter == 'low_perf' and 'ROI' in filtered_summary.columns:
            filtered_summary = filtered_summary[filtered_summary['ROI'] < 5]
        
        if filtered_summary.empty:
            return html.Div("No data for charts", className="text-center text-warning p-4")
        
        # 1. ROI Performance Analysis - Top & Bottom Performers
        top_performers = filtered_summary.nlargest(8, 'ROI')
        bottom_performers = filtered_summary.nsmallest(5, 'ROI')[filtered_summary['ROI'] > 0]
        
        fig1 = px.bar(top_performers, 
                     x='Journey Name', y='ROI', 
                     title="Top Performing Journeys by ROI",
                     labels={'ROI': 'Return on Investment (x)', 'Journey Name': 'Journey'},
                     color='ROI',
                     color_continuous_scale='Greens')
        fig1.update_layout(height=500, xaxis_tickangle=45, showlegend=False,
                          title_font_size=16, font_size=12)
        fig1.update_xaxes(title_font_size=14)
        fig1.update_yaxes(title_font_size=14)
        
        # 2. Cost vs Revenue Analysis
        cost_revenue_data = filtered_summary[filtered_summary['Cost'] > 0].copy()
        if not cost_revenue_data.empty:
            cost_revenue_data['Profit'] = cost_revenue_data['GTV'] - cost_revenue_data['Cost']
            cost_revenue_data['Margin %'] = (cost_revenue_data['Profit'] / cost_revenue_data['GTV'] * 100).round(1)
            
            fig2 = px.scatter(cost_revenue_data, 
                             x='Cost', y='GTV',
                             size='Unique Click-Through Conversions',
                             hover_name='Journey Name',
                             hover_data={'Margin %': True, 'ROI': ':.1f'},
                             title="Cost vs Revenue Analysis (Bubble size = Conversions)",
                             labels={'Cost': 'Total Cost (₹)', 'GTV': 'Gross Transaction Value (₹)'},
                             color='ROI',
                             color_continuous_scale='RdYlGn')
            
            # Add break-even line
            max_val = max(cost_revenue_data['Cost'].max(), cost_revenue_data['GTV'].max())
            fig2.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                          line=dict(color="red", width=2, dash="dash"))
            fig2.add_annotation(x=max_val*0.7, y=max_val*0.8, text="Break-even line",
                               showarrow=False, font=dict(color="red"))
            
            fig2.update_layout(height=500, title_font_size=16, font_size=12)
        else:
            fig2 = px.bar(title="No cost data available for analysis")
            fig2.update_layout(height=500)
        
        # 3. Channel Performance Comparison - Convert to line graph
        channel_performance = filtered_data.groupby('Channel').agg({
            'Sent': 'sum',
            'Delivered': 'sum',
            'Unique Clicks': 'sum',
            'Unique Click-Through Conversions': 'sum'
        }).reset_index()
        
        # Calculate channel metrics
        channel_performance['Delivery Rate'] = (channel_performance['Delivered'] / channel_performance['Sent'] * 100).round(1)
        channel_performance['CTR'] = (channel_performance['Unique Clicks'] / channel_performance['Delivered'] * 100).round(2)
        channel_performance['Conversion Rate'] = (channel_performance['Unique Click-Through Conversions'] / channel_performance['Unique Clicks'] * 100).round(2)
        
        # Calculate channel costs and ROI
        channel_performance['Cost'] = 0
        channel_performance['GTV'] = channel_performance['Unique Click-Through Conversions'] * AOV
        
        for idx, row in channel_performance.iterrows():
            if row['Channel'] in CHANNEL_COSTS:
                channel_performance.at[idx, 'Cost'] = row['Sent'] * CHANNEL_COSTS[row['Channel']]
        
        channel_performance['ROI'] = np.where(channel_performance['Cost'] > 0,
                                            channel_performance['GTV'] / channel_performance['Cost'], 0)
        
        # Convert to line graph with markers
        fig3 = px.line(channel_performance, 
                      x='Channel', y='ROI',
                      title="Channel ROI Performance Trend",
                      labels={'ROI': 'Return on Investment (x)', 'Channel': 'Marketing Channel'},
                      markers=True,
                      text='ROI')
        fig3.update_traces(texttemplate='%{text:.1f}x', textposition='top center', 
                          line=dict(width=3), marker=dict(size=10))
        fig3.update_layout(height=400, title_font_size=16, font_size=12, showlegend=False)
        
        # 4. Conversion Funnel Analysis - Improved for better insights
        funnel_data = filtered_summary.nlargest(10, 'Sent')[['Journey Name', 'Sent', 'Delivered', 'Unique Clicks', 'Unique Click-Through Conversions']].copy()
        
        # Create a more insightful funnel visualization
        funnel_metrics = []
        for _, row in funnel_data.iterrows():
            journey_name = row['Journey Name'][:20] + "..." if len(row['Journey Name']) > 20 else row['Journey Name']
            
            # Calculate absolute drop-offs at each stage
            sent = row['Sent']
            delivered = row['Delivered'] 
            clicks = row['Unique Clicks']
            conversions = row['Unique Click-Through Conversions']
            
            funnel_metrics.append({'Journey': journey_name, 'Stage': 'Sent', 'Count': sent})
            funnel_metrics.append({'Journey': journey_name, 'Stage': 'Delivered', 'Count': delivered})
            funnel_metrics.append({'Journey': journey_name, 'Stage': 'Clicked', 'Count': clicks})
            funnel_metrics.append({'Journey': journey_name, 'Stage': 'Converted', 'Count': conversions})
        
        funnel_df = pd.DataFrame(funnel_metrics)
        
        # Create grouped bar chart showing funnel progression
        fig4 = px.bar(funnel_df, 
                     x='Journey', y='Count', color='Stage',
                     title="Conversion Funnel - Volume Drop-off Analysis (Top 10 by Volume)",
                     labels={'Count': 'Number of Users', 'Journey': 'Journey Name'},
                     color_discrete_map={
                         'Sent': '#1f77b4',
                         'Delivered': '#ff7f0e', 
                         'Clicked': '#2ca02c',
                         'Converted': '#d62728'
                     },
                     barmode='group')
        fig4.update_layout(height=500, xaxis_tickangle=45, title_font_size=16, font_size=12)
        fig4.update_xaxes(title_font_size=14)
        fig4.update_yaxes(title_font_size=14)
        
        # Add conversion rate annotations
        for i, (_, row) in enumerate(funnel_data.iterrows()):
            conversion_rate = (row['Unique Click-Through Conversions'] / row['Sent'] * 100) if row['Sent'] > 0 else 0
            fig4.add_annotation(
                x=i,
                y=row['Sent'] * 1.1,
                text=f"{conversion_rate:.2f}%",
                showarrow=False,
                font=dict(size=10, color="black"),
                bgcolor="yellow",
                opacity=0.7
            )
        
        # 5. Spend Efficiency Analysis - Convert to line graph
        spend_data = filtered_summary[filtered_summary['Cost'] > 0].copy()
        if not spend_data.empty:
            # Sort by cost per conversion for line graph
            spend_data['Cost per Conversion'] = (spend_data['Cost'] / spend_data['Unique Click-Through Conversions']).round(2)
            spend_data = spend_data[spend_data['Cost per Conversion'] < spend_data['Cost per Conversion'].quantile(0.95)]  # Remove outliers
            spend_data_sorted = spend_data.sort_values('Cost per Conversion').reset_index(drop=True)
            spend_data_sorted['Journey_Index'] = range(len(spend_data_sorted))
            
            fig5 = px.line(spend_data_sorted, 
                          x='Journey_Index', y='Cost per Conversion',
                          title="Cost per Conversion Efficiency Curve",
                          labels={'Cost per Conversion': 'Cost per Conversion (₹)', 'Journey_Index': 'Journey Rank (by efficiency)'},
                          markers=True,
                          hover_data=['Journey Name'])
            
            # Add median line
            median_cost = spend_data['Cost per Conversion'].median()
            fig5.add_hline(y=median_cost, 
                          line_dash="dash", line_color="red",
                          annotation_text=f"Median: ₹{median_cost:.0f}")
            
            # Add efficiency zones
            percentile_25 = spend_data['Cost per Conversion'].quantile(0.25)
            percentile_75 = spend_data['Cost per Conversion'].quantile(0.75)
            
            fig5.add_hrect(y0=0, y1=percentile_25, fillcolor="green", opacity=0.1, 
                          annotation_text="High Efficiency", annotation_position="top left")
            fig5.add_hrect(y0=percentile_75, y1=spend_data['Cost per Conversion'].max(), 
                          fillcolor="red", opacity=0.1, 
                          annotation_text="Low Efficiency", annotation_position="top right")
            
            fig5.update_traces(line=dict(width=2), marker=dict(size=6))
            fig5.update_layout(height=400, title_font_size=16, font_size=12)
        else:
            fig5 = px.bar(title="No cost data available for spend analysis")
            fig5.update_layout(height=400)
        
        # 6. Performance vs Investment Quadrant
        if not cost_revenue_data.empty:
            # Create quadrant analysis
            median_cost = cost_revenue_data['Cost'].median()
            median_roi_take_rate = cost_revenue_data['ROI (With Take Rate)'].median()
            
            fig6 = px.scatter(cost_revenue_data, 
                             x='Cost', y='ROI (With Take Rate)',
                             size='GTV',
                             hover_name='Journey Name',
                             title="Performance vs Investment Quadrant Analysis (With Take Rate)",
                             labels={'Cost': 'Total Investment (₹)', 'ROI (With Take Rate)': 'ROI with Take Rate (x)'},
                             color='Conversion Rate',
                             color_continuous_scale='Viridis')
            
            # Add quadrant lines
            fig6.add_hline(y=median_roi_take_rate, line_dash="dash", line_color="gray", opacity=0.5)
            fig6.add_vline(x=median_cost, line_dash="dash", line_color="gray", opacity=0.5)
            
            # Add quadrant labels
            max_cost = cost_revenue_data['Cost'].max()
            max_roi_take_rate = cost_revenue_data['ROI (With Take Rate)'].max()
            
            fig6.add_annotation(x=max_cost*0.75, y=max_roi_take_rate*0.9, text="High Investment<br>High Returns", 
                               showarrow=False, bgcolor="lightgreen", opacity=0.7)
            fig6.add_annotation(x=max_cost*0.25, y=max_roi_take_rate*0.9, text="Low Investment<br>High Returns", 
                               showarrow=False, bgcolor="gold", opacity=0.7)
            fig6.add_annotation(x=max_cost*0.75, y=max_roi_take_rate*0.1, text="High Investment<br>Low Returns", 
                               showarrow=False, bgcolor="lightcoral", opacity=0.7)
            fig6.add_annotation(x=max_cost*0.25, y=max_roi_take_rate*0.1, text="Low Investment<br>Low Returns", 
                               showarrow=False, bgcolor="lightblue", opacity=0.7)
            
            fig6.update_layout(height=500, title_font_size=16, font_size=12)
        else:
            fig6 = px.bar(title="Insufficient data for quadrant analysis")
            fig6.update_layout(height=500)
        
        # Return complete layout with business summary cards
        return html.Div([
            # Business Summary Cards
            dbc.Card([
                dbc.CardHeader(html.H5("Business Insights Summary", className="mb-0")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Best Performer", className="text-success"),
                            html.P(f"{top_performers.iloc[0]['Journey Name'][:30]}..." if len(top_performers.iloc[0]['Journey Name']) > 30 else top_performers.iloc[0]['Journey Name'], className="mb-1"),
                            html.Small(f"ROI: {top_performers.iloc[0]['ROI']:.1f}x", className="text-muted")
                        ], width=3),
                        dbc.Col([
                            html.H6("Total Investment", className="text-primary"),
                            html.P(f"₹{cost_revenue_data['Cost'].sum():,.0f}" if not cost_revenue_data.empty else "₹0", className="mb-1"),
                            html.Small(f"{len(cost_revenue_data)} campaigns" if not cost_revenue_data.empty else "No cost data", className="text-muted")
                        ], width=3),
                        dbc.Col([
                            html.H6("Total Revenue", className="text-success"),
                            html.P(f"₹{cost_revenue_data['GTV'].sum():,.0f}" if not cost_revenue_data.empty else "₹0", className="mb-1"),
                            html.Small(f"Overall ROI: {(cost_revenue_data['GTV'].sum() / cost_revenue_data['Cost'].sum()):.1f}x" if not cost_revenue_data.empty and cost_revenue_data['Cost'].sum() > 0 else "N/A", className="text-muted")
                        ], width=3),
                        dbc.Col([
                            html.H6("Avg. Efficiency", className="text-info"),
                            html.P(f"₹{spend_data['Cost per Conversion'].median():.0f}" if not spend_data.empty else "N/A", className="mb-1"),
                            html.Small("Cost per conversion", className="text-muted")
                        ], width=3),
                    ])
                ])
            ], className="mb-4"),
            
            # Charts in vertical layout with improved spacing
            dbc.Row([dbc.Col([dcc.Graph(figure=fig1)], width=12)], className="mb-4"),
            dbc.Row([dbc.Col([dcc.Graph(figure=fig2)], width=12)], className="mb-4"),
            dbc.Row([
                dbc.Col([dcc.Graph(figure=fig3)], width=6),
                dbc.Col([dcc.Graph(figure=fig5)], width=6)
            ], className="mb-4"),
            dbc.Row([dbc.Col([dcc.Graph(figure=fig4)], width=12)], className="mb-4"),
            dbc.Row([dbc.Col([dcc.Graph(figure=fig6)], width=12)], className="mb-4")
        ])
        
    except Exception as e:
        return dbc.Alert(f"Error creating charts: {str(e)}", color="danger")

# Main execution block
if __name__ == "__main__":
    local_ip = get_local_ip()
    
    print("Starting Campaign Analytics Dashboard...")
    print(f"Local URL: http://127.0.0.1:8058")
    print(f"Network URL: http://{local_ip}:8058")
    print("=" * 50)
    print("Features:")
    print("- Professional dashboard interface")
    print("- Accurate calculations matching reference data")
    print("- Collapsible content sections")
    print("- Real-time filtering and analysis")
    print("=" * 50)
    
    try:
        app.run(
            debug=True,
            host='0.0.0.0',
            port=8058,
            dev_tools_hot_reload=True,
            dev_tools_ui=True
        )
    except Exception as e:
        print(f"Error: {e}")
        print("Trying localhost only...")
        app.run(debug=True, host='127.0.0.1', port=8058)