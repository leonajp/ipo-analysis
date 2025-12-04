"""
IPO Operation Analysis Dashboard
Streamlit app to explore IPOs by underwriter with multi-timeframe charts

Features:
- Filter by underwriter
- View all IPOs for selected underwriter
- 1-minute chart (Day 1)
- Daily chart (First Month)
- Daily chart (First Year)

Requirements:
    pip install streamlit pandas plotly requests polygon-api-client

Usage:
    streamlit run ipo_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="IPO Operation Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data
def load_ipo_data():
    """Load the IPO analysis data (fully split-adjusted)."""
    # Paths to check - update these for your setup
    possible_paths = [
        'small_ipo_fully_adjusted.csv',  # Same folder as dashboard
        'data/small_ipo_fully_adjusted.csv',
        r'C:\Users\msui\Documents\Coding Projects\IPO Analysis\small_ipo_fully_adjusted.csv',
        r'P:\Hamren\Other\small_ipo_fully_adjusted.csv',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['date'] = pd.to_datetime(df['date'])
            df['ticker_clean'] = df['Ticker'].str.replace(' US Equity', '')
            
            # Add risk score if missing
            if 'operation_risk_score' not in df.columns:
                df['operation_risk_score'] = 0
                df.loc[df['IPO Sh Px'] <= 5, 'operation_risk_score'] += 15
                df.loc[df['IPO Sh Offered'] <= 1_500_000, 'operation_risk_score'] += 15
            
            st.sidebar.success(f"Loaded: {os.path.basename(path)}")
            return df
    
    st.error("""
    Could not find small_ipo_fully_adjusted.csv
    
    Please either:
    1. Run ipo_full_adjustment.py first to generate the file
    2. Place small_ipo_fully_adjusted.csv in the same folder as this dashboard
    """)
    return pd.DataFrame()


# ============================================================================
# DATA FETCHING FUNCTIONS (Customize for your data source)
# ============================================================================

# Option 1: Polygon API
def fetch_polygon_bars(ticker: str, start_date: str, end_date: str, 
                       timeframe: str = "day", multiplier: int = 1) -> pd.DataFrame:
    """
    Fetch bars from Polygon.io API
    
    Args:
        ticker: Stock ticker
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD
        timeframe: minute, hour, day, week, month
        multiplier: Timeframe multiplier (1 for 1-minute, 1 for 1-day, etc.)
    """
    try:
        from polygon import RESTClient
        
        # Get API key from environment or Streamlit secrets
        api_key = os.environ.get('POLYGON_API_KEY') or st.secrets.get('POLYGON_API_KEY', '')
        
        if not api_key:
            return pd.DataFrame()
        
        client = RESTClient(api_key)
        
        bars = []
        for bar in client.list_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timeframe,
            from_=start_date,
            to=end_date,
            limit=50000
        ):
            bars.append({
                'timestamp': pd.to_datetime(bar.timestamp, unit='ms'),
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            })
        
        if bars:
            return pd.DataFrame(bars)
        return pd.DataFrame()
        
    except Exception as e:
        st.warning(f"Polygon fetch error: {e}")
        return pd.DataFrame()


# Option 2: ClickHouse (if you have data stored there)
def fetch_clickhouse_bars(ticker: str, start_date: str, end_date: str,
                          timeframe: str = "daily") -> pd.DataFrame:
    """
    Fetch bars from ClickHouse database
    
    Customize the connection and table names for your setup.
    """
    try:
        import clickhouse_connect
        
        # Customize these for your setup
        client = clickhouse_connect.get_client(
            host='localhost',
            port=8123,
            database='market_data'
        )
        
        if timeframe == "minute":
            table = "minute_bars"
        else:
            table = "daily_bars"
        
        query = f"""
        SELECT 
            timestamp,
            open, high, low, close, volume
        FROM {table}
        WHERE ticker = '{ticker}'
        AND timestamp >= '{start_date}'
        AND timestamp <= '{end_date}'
        ORDER BY timestamp
        """
        
        result = client.query(query)
        df = pd.DataFrame(result.result_rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
        
    except Exception as e:
        st.warning(f"ClickHouse fetch error: {e}")
        return pd.DataFrame()


# Option 3: Generate sample data for demo
def generate_sample_bars(ticker: str, start_date: str, end_date: str,
                         timeframe: str = "day", ipo_price: float = 4.0) -> pd.DataFrame:
    """
    Generate sample OHLCV data for demonstration.
    Replace this with real data fetching in production.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    if timeframe == "minute":
        # Generate minute bars for one trading day (9:30 AM - 4:00 PM)
        times = pd.date_range(start=start.replace(hour=9, minute=30),
                              end=start.replace(hour=16, minute=0),
                              freq='1min')
    else:
        # Generate daily bars
        times = pd.date_range(start=start, end=end, freq='B')  # Business days
    
    if len(times) == 0:
        return pd.DataFrame()
    
    # Simulate price movement with some randomness
    np.random.seed(hash(ticker) % 2**32)
    
    # Random walk with drift
    returns = np.random.normal(0.001, 0.03, len(times))
    
    # Add some "operation" characteristics - drift up then crash
    if len(times) > 50:
        # Drift up period
        returns[10:40] += 0.02
        # Crash
        if len(times) > 45:
            returns[40:45] -= 0.15
    
    prices = ipo_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC from prices
    data = []
    for i, (t, p) in enumerate(zip(times, prices)):
        noise = np.random.uniform(0.98, 1.02, 4)
        o = p * noise[0]
        h = p * max(noise)
        l = p * min(noise)
        c = p * noise[3]
        v = int(np.random.uniform(10000, 500000))
        
        data.append({
            'timestamp': t,
            'open': round(o, 2),
            'high': round(h, 2),
            'low': round(l, 2),
            'close': round(c, 2),
            'volume': v
        })
    
    return pd.DataFrame(data)


def fetch_bars(ticker: str, start_date: str, end_date: str, 
               timeframe: str = "day", ipo_price: float = 4.0,
               data_source: str = "sample") -> pd.DataFrame:
    """
    Main function to fetch bars - routes to appropriate data source.
    """
    if data_source == "polygon":
        multiplier = 1
        tf = "minute" if timeframe == "minute" else "day"
        return fetch_polygon_bars(ticker, start_date, end_date, tf, multiplier)
    elif data_source == "clickhouse":
        return fetch_clickhouse_bars(ticker, start_date, end_date, timeframe)
    else:
        return generate_sample_bars(ticker, start_date, end_date, timeframe, ipo_price)


# ============================================================================
# CHART FUNCTIONS
# ============================================================================

def create_candlestick_chart(df: pd.DataFrame, title: str, height: int = 400) -> go.Figure:
    """Create a candlestick chart with volume."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font_size=20)
        fig.update_layout(height=height, title=title)
        return fig
    
    # Create figure with secondary y-axis for volume
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.7, 0.3])
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Volume bars
    colors = ['#26a69a' if row['close'] >= row['open'] else '#ef5350' 
              for _, row in df.iterrows()]
    
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Add price annotations
    max_price = df['high'].max()
    min_price = df['low'].min()
    last_price = df['close'].iloc[-1]
    first_price = df['open'].iloc[0]
    pct_change = (last_price / first_price - 1) * 100
    
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>High: ${max_price:.2f} | Low: ${min_price:.2f} | Change: {pct_change:+.1f}%</sup>",
            x=0.5,
            font_size=14
        ),
        height=height,
        xaxis_rangeslider_visible=False,
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=50),
        template='plotly_white'
    )
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig


def create_summary_chart(df: pd.DataFrame, title: str) -> go.Figure:
    """Create a simple line chart for quick overview."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['close'],
        mode='lines',
        name='Close',
        line=dict(color='#1976d2', width=2)
    ))
    
    fig.update_layout(
        title=title,
        height=200,
        margin=dict(l=40, r=40, t=40, b=30),
        template='plotly_white',
        showlegend=False
    )
    
    return fig


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("📈 IPO Operation Analysis Dashboard")
    st.markdown("*Explore small IPOs by underwriter with multi-timeframe charts*")
    
    # Load data
    df = load_ipo_data()
    
    if df.empty:
        st.stop()
    
    # ========================================================================
    # OVERALL STATS (Collapsible)
    # ========================================================================
    with st.expander("📊 Overall Dataset Statistics", expanded=False):
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.metric("Total Small IPOs", len(df))
            st.metric("Date Range", f"{df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
        
        with stat_col2:
            st.metric("Median Bounce", f"{df['lifetime_hi_vs_ipo'].median():.0f}%")
            bounce_100 = (df['lifetime_hi_vs_ipo'] > 100).mean() * 100
            st.metric("Bounce >100%", f"{bounce_100:.0f}%")
        
        with stat_col3:
            if 'is_tax_haven' in df.columns:
                th_pct = df['is_tax_haven'].mean() * 100
                us_pct = df['is_us'].mean() * 100 if 'is_us' in df.columns else 0
                st.metric("Tax Haven %", f"{th_pct:.0f}%")
                st.metric("US %", f"{us_pct:.0f}%")
            else:
                st.metric("Tax Haven %", "N/A")
                st.metric("US %", "N/A")
        
        with stat_col4:
            if 'vc_backed' in df.columns:
                vc_pct = df['vc_backed'].mean() * 100
                st.metric("VC Backed %", f"{vc_pct:.0f}%")
            if 'has_operation_underwriter' in df.columns:
                op_uw_pct = df['has_operation_underwriter'].mean() * 100
                st.metric("Operation UW %", f"{op_uw_pct:.0f}%")
        
        # Bounce by country type
        st.markdown("---")
        st.markdown("**Bounce Rate by Country Type**")
        
        if 'is_tax_haven' in df.columns:
            bounce_data = []
            
            th_df = df[df['is_tax_haven']]
            if len(th_df) > 0:
                bounce_data.append({
                    'Type': '🏝️ Tax Haven',
                    'Count': len(th_df),
                    'Median Bounce': f"{th_df['lifetime_hi_vs_ipo'].median():.0f}%",
                    'Bounce >100%': f"{(th_df['lifetime_hi_vs_ipo'] > 100).mean()*100:.0f}%",
                    'Avg Risk': f"{th_df['operation_risk_score'].mean():.0f}"
                })
            
            if 'is_us' in df.columns:
                us_df = df[df['is_us']]
                if len(us_df) > 0:
                    bounce_data.append({
                        'Type': '🇺🇸 US',
                        'Count': len(us_df),
                        'Median Bounce': f"{us_df['lifetime_hi_vs_ipo'].median():.0f}%",
                        'Bounce >100%': f"{(us_df['lifetime_hi_vs_ipo'] > 100).mean()*100:.0f}%",
                        'Avg Risk': f"{us_df['operation_risk_score'].mean():.0f}"
                    })
                
                other_df = df[~df['is_tax_haven'] & ~df['is_us']]
                if len(other_df) > 0:
                    bounce_data.append({
                        'Type': '🌍 Other',
                        'Count': len(other_df),
                        'Median Bounce': f"{other_df['lifetime_hi_vs_ipo'].median():.0f}%",
                        'Bounce >100%': f"{(other_df['lifetime_hi_vs_ipo'] > 100).mean()*100:.0f}%",
                        'Avg Risk': f"{other_df['operation_risk_score'].mean():.0f}"
                    })
            
            if bounce_data:
                st.dataframe(pd.DataFrame(bounce_data), use_container_width=True, hide_index=True)
    
    # ========================================================================
    # SIDEBAR - FILTERS
    # ========================================================================
    st.sidebar.header("🔍 Filters")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Data Source",
        options=["sample", "polygon", "clickhouse"],
        index=0,
        help="Select your market data source. 'sample' generates demo data."
    )
    
    if data_source == "polygon":
        api_key = st.sidebar.text_input("Polygon API Key", type="password")
        if api_key:
            os.environ['POLYGON_API_KEY'] = api_key
    
    st.sidebar.markdown("---")
    
    # Get underwriters with deal counts
    uw_counts = df.groupby('underwriter').agg({
        'Ticker': 'count',
        'lifetime_hi_vs_ipo': 'median',
        'operation_risk_score': 'mean'
    }).reset_index()
    uw_counts.columns = ['underwriter', 'num_deals', 'median_bounce', 'avg_risk']
    uw_counts = uw_counts.sort_values('num_deals', ascending=False)
    
    # Create display labels
    uw_options = [f"{row['underwriter'][:40]} ({row['num_deals']} deals)" 
                  for _, row in uw_counts.iterrows()]
    uw_map = dict(zip(uw_options, uw_counts['underwriter']))
    
    selected_uw_label = st.sidebar.selectbox(
        "Select Underwriter",
        options=uw_options,
        index=0
    )
    
    selected_uw = uw_map[selected_uw_label]
    
    # Filter options
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Risk Filters")
    
    min_risk = st.sidebar.slider("Min Risk Score", 0, 100, 0)
    
    date_range = st.sidebar.date_input(
        "IPO Date Range",
        value=(df['date'].min().date(), df['date'].max().date()),
        min_value=df['date'].min().date(),
        max_value=df['date'].max().date()
    )
    
    # New heuristic filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Heuristic Filters")
    
    # Country filter
    country_filter = st.sidebar.radio(
        "Country Type",
        options=["All", "Tax Haven Only", "US Only", "Other"],
        index=0
    )
    
    # Checkboxes for additional filters
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        show_operation_uw = st.sidebar.checkbox("Operation UW", value=False, 
                                                 help="Show only IPOs with operation underwriters")
    with col_b:
        show_no_vc = st.sidebar.checkbox("No VC", value=False,
                                          help="Show only non-VC backed IPOs")
    
    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    
    # Build filter mask
    mask = (
        (df['underwriter'] == selected_uw) &
        (df['operation_risk_score'] >= min_risk) &
        (df['date'].dt.date >= date_range[0]) &
        (df['date'].dt.date <= date_range[1])
    )
    
    # Apply country filter
    if country_filter == "Tax Haven Only" and 'is_tax_haven' in df.columns:
        mask &= df['is_tax_haven']
    elif country_filter == "US Only" and 'is_us' in df.columns:
        mask &= df['is_us']
    elif country_filter == "Other" and 'is_tax_haven' in df.columns and 'is_us' in df.columns:
        mask &= ~df['is_tax_haven'] & ~df['is_us']
    
    # Apply heuristic filters
    if show_operation_uw and 'has_operation_underwriter' in df.columns:
        mask &= df['has_operation_underwriter']
    if show_no_vc and 'vc_backed' in df.columns:
        mask &= ~df['vc_backed']
    
    filtered_df = df[mask].sort_values('date', ascending=False)
    
    # Underwriter stats
    st.header(f"📊 {selected_uw[:50]}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total IPOs", len(filtered_df))
    with col2:
        st.metric("Avg Risk Score", f"{filtered_df['operation_risk_score'].mean():.0f}" if len(filtered_df) > 0 else "N/A")
    with col3:
        st.metric("Median Bounce", f"{filtered_df['lifetime_hi_vs_ipo'].median():.0f}%" if len(filtered_df) > 0 else "N/A")
    with col4:
        major_bounce_pct = (filtered_df['lifetime_hi_vs_ipo'] > 100).mean() * 100 if len(filtered_df) > 0 else 0
        st.metric("Bounce >100%", f"{major_bounce_pct:.0f}%")
    with col5:
        if 'is_tax_haven' in filtered_df.columns:
            tax_haven_pct = filtered_df['is_tax_haven'].mean() * 100 if len(filtered_df) > 0 else 0
            st.metric("Tax Haven %", f"{tax_haven_pct:.0f}%")
        else:
            st.metric("Tax Haven %", "N/A")
    
    # Risk factor breakdown
    if len(filtered_df) > 0:
        with st.expander("📊 Risk Factor Breakdown", expanded=False):
            rf_col1, rf_col2, rf_col3 = st.columns(3)
            
            with rf_col1:
                st.markdown("**Country Type**")
                if 'is_tax_haven' in filtered_df.columns:
                    tax_haven_n = filtered_df['is_tax_haven'].sum()
                    us_n = filtered_df['is_us'].sum() if 'is_us' in filtered_df.columns else 0
                    other_n = len(filtered_df) - tax_haven_n - us_n
                    st.write(f"🏝️ Tax Haven: {tax_haven_n}")
                    st.write(f"🇺🇸 US: {us_n}")
                    st.write(f"🌍 Other: {other_n}")
            
            with rf_col2:
                st.markdown("**Underwriter Type**")
                if 'has_operation_underwriter' in filtered_df.columns:
                    op_uw = filtered_df['has_operation_underwriter'].sum()
                    legit_uw = filtered_df['has_legit_underwriter'].sum() if 'has_legit_underwriter' in filtered_df.columns else 0
                    st.write(f"⚠️ Operation UW: {op_uw}")
                    st.write(f"✅ Legit UW: {legit_uw}")
            
            with rf_col3:
                st.markdown("**VC Status**")
                if 'vc_backed' in filtered_df.columns:
                    vc_n = filtered_df['vc_backed'].sum()
                    no_vc_n = len(filtered_df) - vc_n
                    st.write(f"💰 VC Backed: {vc_n}")
                    st.write(f"❌ No VC: {no_vc_n}")
    
    st.markdown("---")
    
    # IPO Table
    st.subheader("📋 IPO List")
    
    # Build display columns dynamically based on what's available
    base_cols = ['ticker_clean', 'Name', 'date', 'IPO Sh Px', 
                 'operation_risk_score', 'lifetime_hi_vs_ipo', 'ret_d1']
    
    heuristic_cols = []
    if 'is_tax_haven' in filtered_df.columns:
        heuristic_cols.append('is_tax_haven')
    if 'vc_backed' in filtered_df.columns:
        heuristic_cols.append('vc_backed')
    if 'Cntry Terrtry Of Inc' in filtered_df.columns:
        heuristic_cols.append('Cntry Terrtry Of Inc')
    
    display_cols = base_cols + heuristic_cols
    display_cols = [c for c in display_cols if c in filtered_df.columns]
    
    display_df = filtered_df[display_cols].copy()
    
    # Rename columns for display
    col_rename = {
        'ticker_clean': 'Ticker',
        'Name': 'Name', 
        'date': 'IPO Date',
        'IPO Sh Px': 'Price',
        'operation_risk_score': 'Risk',
        'lifetime_hi_vs_ipo': 'Bounce %',
        'ret_d1': 'D1 %',
        'is_tax_haven': '🏝️',
        'vc_backed': '💰',
        'Cntry Terrtry Of Inc': 'Country'
    }
    display_df = display_df.rename(columns={k:v for k,v in col_rename.items() if k in display_df.columns})
    
    # Format columns
    if 'Bounce %' in display_df.columns:
        display_df['Bounce %'] = display_df['Bounce %'].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "N/A")
    if 'D1 %' in display_df.columns:
        display_df['D1 %'] = display_df['D1 %'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
    if 'Price' in display_df.columns:
        display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.0f}" if pd.notna(x) else "N/A")
    if 'Risk' in display_df.columns:
        display_df['Risk'] = display_df['Risk'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")
    if '🏝️' in display_df.columns:
        display_df['🏝️'] = display_df['🏝️'].apply(lambda x: "✓" if x else "")
    if '💰' in display_df.columns:
        display_df['💰'] = display_df['💰'].apply(lambda x: "✓" if x else "")
    if 'Name' in display_df.columns:
        display_df['Name'] = display_df['Name'].apply(lambda x: str(x)[:25] if pd.notna(x) else "")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # ========================================================================
    # INDIVIDUAL IPO CHARTS
    # ========================================================================
    st.subheader("📈 IPO Charts")
    
    # Select IPO to view
    ticker_options = filtered_df['ticker_clean'].tolist()
    
    if not ticker_options:
        st.warning("No IPOs match the current filters.")
        st.stop()
    
    selected_ticker = st.selectbox("Select IPO to View", options=ticker_options)
    
    # Get IPO info
    ipo_info = filtered_df[filtered_df['ticker_clean'] == selected_ticker].iloc[0]
    ipo_date = ipo_info['date']
    ipo_price = ipo_info['IPO Sh Px'] if pd.notna(ipo_info['IPO Sh Px']) else 4.0
    
    # Display IPO info
    vc_status = "✅ VC Backed" if ipo_info.get('vc_backed', False) else "❌ No VC"
    tax_status = "🏝️ Tax Haven" if ipo_info.get('is_tax_haven', False) else ""
    country = ipo_info.get('Cntry Terrtry Of Inc', 'N/A')
    
    st.markdown(f"""
    **{selected_ticker}** - {str(ipo_info['Name'])[:50]}
    - 📅 IPO Date: {ipo_date.strftime('%Y-%m-%d')}
    - 💵 IPO Price: ${ipo_price:.2f}
    - ⚠️ Risk Score: **{ipo_info['operation_risk_score']:.0f}**
    - 📈 Lifetime High: **{ipo_info['lifetime_hi_vs_ipo']:.0f}%**
    - 🌍 Country: {country} {tax_status}
    - {vc_status}
    """)
    
    # Fetch and display charts
    with st.spinner("Loading chart data..."):
        
        # Chart 1: Day 1 - 1 Minute Bars
        st.markdown("### 🕐 Day 1 (1-Minute Bars)")
        
        d1_start = ipo_date.strftime('%Y-%m-%d')
        d1_end = d1_start
        
        d1_bars = fetch_bars(selected_ticker, d1_start, d1_end, 
                             timeframe="minute", ipo_price=ipo_price,
                             data_source=data_source)
        
        fig1 = create_candlestick_chart(d1_bars, f"{selected_ticker} - Day 1 (1-Min)", height=450)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Chart 2: First Month - Daily Bars
        st.markdown("### 📅 First Month (Daily Bars)")
        
        m1_start = ipo_date.strftime('%Y-%m-%d')
        m1_end = (ipo_date + timedelta(days=30)).strftime('%Y-%m-%d')
        
        m1_bars = fetch_bars(selected_ticker, m1_start, m1_end,
                             timeframe="day", ipo_price=ipo_price,
                             data_source=data_source)
        
        fig2 = create_candlestick_chart(m1_bars, f"{selected_ticker} - First Month (Daily)", height=450)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Chart 3: First Year - Daily Bars
        st.markdown("### 📆 First Year (Daily Bars)")
        
        y1_start = ipo_date.strftime('%Y-%m-%d')
        y1_end = min(
            (ipo_date + timedelta(days=365)).strftime('%Y-%m-%d'),
            datetime.now().strftime('%Y-%m-%d')
        )
        
        y1_bars = fetch_bars(selected_ticker, y1_start, y1_end,
                             timeframe="day", ipo_price=ipo_price,
                             data_source=data_source)
        
        fig3 = create_candlestick_chart(y1_bars, f"{selected_ticker} - First Year (Daily)", height=450)
        st.plotly_chart(fig3, use_container_width=True)
    
    # ========================================================================
    # COMPARISON VIEW
    # ========================================================================
    st.markdown("---")
    st.subheader("🔄 Quick Comparison - All IPOs from This Underwriter")
    
    # Show mini charts for all IPOs
    cols_per_row = 3
    
    for i in range(0, min(len(filtered_df), 12), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(filtered_df):
                break
            
            row = filtered_df.iloc[idx]
            ticker = row['ticker_clean']
            ipo_dt = row['date']
            price = row['IPO Sh Px'] if pd.notna(row['IPO Sh Px']) else 4.0
            
            with col:
                # Fetch 30-day data
                start = ipo_dt.strftime('%Y-%m-%d')
                end = (ipo_dt + timedelta(days=30)).strftime('%Y-%m-%d')
                
                bars = fetch_bars(ticker, start, end, "day", price, data_source)
                
                title = f"{ticker} ({ipo_dt.strftime('%Y-%m')})"
                if pd.notna(row['lifetime_hi_vs_ipo']):
                    title += f" | +{row['lifetime_hi_vs_ipo']:.0f}%"
                
                fig = create_summary_chart(bars, title)
                st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    IPO Operation Analysis Dashboard | Data: Bloomberg EQS + Polygon<br>
    Risk scores based on underwriter, price, float, and volume characteristics
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()