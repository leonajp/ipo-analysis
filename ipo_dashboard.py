"""
IPO Operation Analysis Dashboard
Streamlit app to explore IPOs by underwriter with multi-timeframe charts

Features:
- Filter by underwriter (partial match)
- View all IPOs for selected underwriter
- 1-minute chart (Day 1)
- Daily chart (First Month)
- Daily chart (First Year)
- Grid view of all IPO charts
- Direct ClickHouse database connection

Requirements:
    pip install streamlit pandas plotly requests polygon-api-client clickhouse-connect

Usage:
    streamlit run ipo_dashboard.py
"""

# VERSION - Update when making changes to verify user has latest code
DASHBOARD_VERSION = "2.3.0-operations"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import re
import json

# Optional imports for live IPO fetching
try:
    import requests
    from bs4 import BeautifulSoup
    HAS_SCRAPING = True
except ImportError:
    HAS_SCRAPING = False

# Optional imports for ClickHouse
try:
    import clickhouse_connect
    HAS_CLICKHOUSE = True
except ImportError:
    HAS_CLICKHOUSE = False

# ============================================================================
# CONFIG / API KEY PERSISTENCE
# ============================================================================
CONFIG_FILE = os.path.expanduser("~/.ipo_dashboard_config.json")

# ClickHouse Configuration (can be overridden by environment variables or config file)
CLICKHOUSE_CONFIG = {
    'host': os.environ.get('CLICKHOUSE_HOST', 'i35q8zrtq4.us-east-2.aws.clickhouse.cloud'),
    'port': int(os.environ.get('CLICKHOUSE_PORT', '443')),
    'user': os.environ.get('CLICKHOUSE_USER', 'default'),
    'password': os.environ.get('CLICKHOUSE_PASSWORD', ''),
    'database': os.environ.get('CLICKHOUSE_DATABASE', 'ipo'),
    'table': os.environ.get('CLICKHOUSE_TABLE', 'ipo_master'),
    'secure': True,
}

# ============================================================================
# OPERATION UNDERWRITERS & LEGIT UNDERWRITERS (for risk scoring)
# ============================================================================
OPERATION_UNDERWRITERS = {
    'D BORAL', 'KINGSWOOD', 'US TIGER', 'PRIME NUMBER', 'NETWORK 1',
    'EF HUTTON', 'BANCROFT', 'CATHAY', 'RVRS', 'VIEWTRADE',
    'JOSEPH STONE', 'BOUSTEAD', 'MAXIM', 'DAWSON', 'REVERE',
    'DOMINARI', 'CRAFT CAPITAL', 'THINKEQUITY', 'AEGIS', 'EDDID',
    'SPARTAN', 'AC SUNSHINE', 'WTF SECURITIES'  # WTF's underwriters pattern
}

LEGIT_UNDERWRITERS = {
    'GOLDMAN', 'MORGAN STANLEY', 'JPMORGAN', 'JP MORGAN', 'CITI',
    'BOFA', 'JEFFERIES', 'CREDIT SUISSE', 'UBS', 'BARCLAYS', 'DEUTSCHE',
    'WELLS FARGO', 'RBC', 'PIPER', 'STIFEL', 'WILLIAM BLAIR'
}

def load_config() -> dict:
    """Load saved configuration including API keys (local file)."""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_config(config: dict):
    """Save configuration to local file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
    except Exception as e:
        pass  # Silently fail on cloud

def inject_local_storage_js():
    """Inject JavaScript for browser localStorage access."""
    st.markdown("""
    <script>
    // Save API key to localStorage
    window.saveApiKey = function(key) {
        localStorage.setItem('polygon_api_key', key);
    }
    
    // Load API key from localStorage
    window.loadApiKey = function() {
        return localStorage.getItem('polygon_api_key') || '';
    }
    
    // Clear API key from localStorage  
    window.clearApiKey = function() {
        localStorage.removeItem('polygon_api_key');
    }
    
    // On page load, send saved key to Streamlit via query params
    (function() {
        const savedKey = localStorage.getItem('polygon_api_key');
        if (savedKey && !window.location.search.includes('polygon_key=')) {
            // Check if we need to reload with the key
            const urlParams = new URLSearchParams(window.location.search);
            if (!urlParams.has('polygon_key')) {
                urlParams.set('polygon_key', savedKey);
                // Use a hidden input to communicate with Streamlit
                const event = new CustomEvent('localStorageKey', { detail: savedKey });
                window.dispatchEvent(event);
            }
        }
    })();
    </script>
    """, unsafe_allow_html=True)

def get_polygon_api_key() -> str:
    """Get Polygon API key from various sources (priority order)."""
    # 1. Check session state (user just entered it)
    if st.session_state.get('polygon_api_key'):
        return st.session_state.polygon_api_key
    
    # 2. Check query params (from localStorage redirect)
    try:
        params = st.query_params
        if 'polygon_key' in params:
            key = params['polygon_key']
            if key:
                st.session_state.polygon_api_key = key
                return key
    except Exception:
        pass
    
    # 3. Check Streamlit secrets (for cloud deployment - shared key)
    try:
        if hasattr(st, 'secrets') and 'POLYGON_API_KEY' in st.secrets:
            return st.secrets['POLYGON_API_KEY']
    except Exception:
        pass
    
    # 4. Check environment variable
    if os.environ.get('POLYGON_API_KEY'):
        return os.environ['POLYGON_API_KEY']
    
    # 5. Check saved config file (local persistence)
    config = load_config()
    if config.get('polygon_api_key'):
        return config['polygon_api_key']
    
    return ''

def save_polygon_api_key(key: str):
    """Save Polygon API key to multiple storage locations."""
    if key:
        # Save to session state
        st.session_state.polygon_api_key = key
        os.environ['POLYGON_API_KEY'] = key
        
        # Save to local config file (works locally)
        try:
            config = load_config()
            config['polygon_api_key'] = key
            save_config(config)
        except Exception:
            pass
        
        # Save to browser localStorage via JavaScript
        st.markdown(f"""
        <script>
        localStorage.setItem('polygon_api_key', '{key}');
        </script>
        """, unsafe_allow_html=True)

def clear_polygon_api_key():
    """Clear API key from all storage locations."""
    # Clear session state
    st.session_state.polygon_api_key = ''
    if 'POLYGON_API_KEY' in os.environ:
        del os.environ['POLYGON_API_KEY']
    
    # Clear local config
    try:
        config = load_config()
        config.pop('polygon_api_key', None)
        save_config(config)
    except Exception:
        pass
    
    # Clear browser localStorage
    st.markdown("""
    <script>
    localStorage.removeItem('polygon_api_key');
    </script>
    """, unsafe_allow_html=True)

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
# CSS OVERRIDES
# ============================================================================
def apply_custom_css():
    """Apply custom CSS for better UX - larger dropdowns, etc."""
    st.markdown(
        """
        <style>
        /* Expand selectbox dropdown height (default is 300px) */
        [data-testid="stSelectboxVirtualDropdown"] > div {
            max-height: 480px !important;
        }
        
        /* Make selectbox text slightly larger */
        [data-testid="stSelectbox"] {
            font-size: 0.95rem !important;
        }
        
        /* Wider sidebar */
        [data-testid="stSidebar"] {
            min-width: 320px !important;
        }
        
        /* Better table styling */
        [data-testid="stDataFrame"] {
            font-size: 0.85rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_custom_css()

# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_from_clickhouse(password: str = None) -> pd.DataFrame:
    """Load IPO data directly from ClickHouse Cloud."""
    if not HAS_CLICKHOUSE:
        return None
    
    # Get password from parameter, env var, or config
    ch_password = password or CLICKHOUSE_CONFIG['password']
    if not ch_password:
        config = load_config()
        ch_password = config.get('clickhouse_password', '')
    
    if not ch_password:
        return None
    
    try:
        client = clickhouse_connect.get_client(
            host=CLICKHOUSE_CONFIG['host'],
            port=CLICKHOUSE_CONFIG['port'],
            user=CLICKHOUSE_CONFIG['user'],
            password=ch_password,
            secure=CLICKHOUSE_CONFIG['secure'],
        )
        
        # First, check what columns exist in the table
        try:
            cols_result = client.query(f"DESCRIBE {CLICKHOUSE_CONFIG['database']}.{CLICKHOUSE_CONFIG['table']}")
            existing_cols = {row[0] for row in cols_result.result_rows}
        except:
            existing_cols = set()
        
        # Define desired columns with fallbacks
        desired_cols = [
            'polygon_ticker', 'ticker', 'ticker_bbg', 'Name',
            'ipo_date', 'ipo_price', 'ipo_shares_offered', 'offer_size_m',
            'exchange', 'country', 'underwriter',
            'all_underwriters', 'underwriter_tier',
            'is_operation_uw', 'operation_risk_score', 'is_tax_haven', 'vc_backed',
            'top_shareholders', 'vc_investors',
            'vc_ownership_pct', 'founder_ownership_pct', 'institutional_ownership_pct',
            'd1_open', 'd1_high', 'd1_low', 'd1_close', 'd1_volume', 'd1_open_adj',
            'd2_open', 'd2_high', 'd2_low', 'd2_close', 'd2_volume',
            'd3_open', 'd3_high', 'd3_low', 'd3_close', 'd3_volume',
            'd4_open', 'd4_high', 'd4_low', 'd4_close', 'd4_volume',
            'd5_open', 'd5_high', 'd5_low', 'd5_close', 'd5_volume',
            'ret_d1', 'ret_d5', 'ret_d30', 'lifetime_high', 'lifetime_high_unadj', 'lifetime_hi_vs_ipo',
            'open_px', 'first_5m_high', 'first_5m_low',
            'halted_d1', 'num_halts',
            'sec_filing_type', 'sec_filing_date',
            'updated_at',
            # Operation detection columns
            'operation_count', 'has_operation', 'first_operation_date', 'max_operation_gain'
        ]
        
        # Only query columns that exist
        if existing_cols:
            query_cols = [c for c in desired_cols if c in existing_cols]
        else:
            query_cols = desired_cols  # Try all if we couldn't get column list
        
        query = f"""
        SELECT {', '.join(query_cols)}
        FROM {CLICKHOUSE_CONFIG['database']}.{CLICKHOUSE_CONFIG['table']}
        ORDER BY ipo_date DESC
        """
        
        result = client.query(query)
        df = pd.DataFrame(result.result_rows, columns=result.column_names)
        
        return df
        
    except Exception as e:
        st.sidebar.warning(f"ClickHouse error: {str(e)[:50]}...")
        return None


@st.cache_data(ttl=300, show_spinner="Loading IPO data...")
def load_ipo_data(use_clickhouse: bool = True, ch_password: str = None, _cache_key: str = None):
    """Load the IPO analysis data (from ClickHouse or CSV fallback).
    
    Note: _cache_key is used to force cache refresh when data source changes.
    """
    
    # Try ClickHouse first if enabled
    if use_clickhouse and HAS_CLICKHOUSE:
        df = load_from_clickhouse(ch_password)
        if df is not None and len(df) > 0:
            # Map ClickHouse columns to expected dashboard columns
            df['date'] = pd.to_datetime(df['ipo_date'], errors='coerce')
            df['ticker_clean'] = df['polygon_ticker'].fillna(df['ticker']).fillna('')
            df['Ticker'] = df['ticker_bbg'].fillna(df['ticker_clean'] + ' US Equity')
            df['IPO Sh Px'] = df['ipo_price']
            df['IPO Sh Offered'] = df['ipo_shares_offered']
            df['OfferSizeM'] = df['offer_size_m']
            df['Prim Exch Nm'] = df['exchange']
            df['Cntry Terrtry Of Inc'] = df['country']
            df['IPO Lead'] = df['underwriter']
            df['Lifetime High'] = df['lifetime_high']
            
            # Convert numeric columns
            df['d1_open'] = pd.to_numeric(df.get('d1_open', 0), errors='coerce').fillna(0)
            df['d1_close'] = pd.to_numeric(df.get('d1_close', 0), errors='coerce').fillna(0)
            df['ipo_price'] = pd.to_numeric(df['ipo_price'], errors='coerce').fillna(0)
            df['lifetime_high'] = pd.to_numeric(df['lifetime_high'], errors='coerce').fillna(0)
            
            # Calculate ratios for data quality checks
            df['d1_to_ipo_ratio'] = np.where(
                (df['ipo_price'] > 0) & (df['d1_open'] > 0),
                df['d1_open'] / df['ipo_price'],
                1.0
            )
            df['lt_to_ipo_ratio'] = np.where(
                df['ipo_price'] > 0,
                df['lifetime_high'] / df['ipo_price'],
                0
            )
            
            # Flag potentially bad data for review
            df['price_adjusted'] = (df['d1_to_ipo_ratio'] > 2.0) | (df['d1_to_ipo_ratio'] < 0.5)
            
            # === BOUNCE CALCULATIONS (two methods) ===
            # Method 1: vs D1 Open (adjusted) - day trading perspective
            # Method 2: vs IPO Price - investment perspective
            
            # Both will be calculated and user can toggle between them
            
            # BOUNCE vs D1 Open (adjusted)
            db_bounce_col = 'lifetime_hi_vs_ipo'
            if db_bounce_col in df.columns and df[db_bounce_col].notna().any() and (df[db_bounce_col] != 0).any():
                df['bounce_vs_d1open'] = pd.to_numeric(df[db_bounce_col], errors='coerce').fillna(0)
                df['bounce_vs_d1open'] = np.clip(df['bounce_vs_d1open'].values, -100, 5000)
            elif 'd1_open_adj' in df.columns:
                d1_open_adj = pd.to_numeric(df['d1_open_adj'], errors='coerce').fillna(0)
                df['bounce_vs_d1open'] = np.where(
                    (d1_open_adj > 0) & (df['lifetime_high'] > 0),
                    ((df['lifetime_high'] / d1_open_adj) - 1) * 100,
                    0
                )
                df['bounce_vs_d1open'] = np.clip(df['bounce_vs_d1open'].values, -100, 5000)
            elif 'd1_open' in df.columns:
                df['bounce_vs_d1open'] = np.where(
                    (df['d1_open'] > 0) & (df['lifetime_high'] > 0),
                    ((df['lifetime_high'] / df['d1_open']) - 1) * 100,
                    0
                )
                df['bounce_vs_d1open'] = np.clip(df['bounce_vs_d1open'].values, -100, 1000)
            else:
                df['bounce_vs_d1open'] = 0
            
            # BOUNCE vs IPO Price
            df['bounce_vs_ipo'] = np.where(
                (df['ipo_price'] > 0) & (df['lifetime_high'] > 0),
                ((df['lifetime_high'] / df['ipo_price']) - 1) * 100,
                0
            )
            df['bounce_vs_ipo'] = np.clip(df['bounce_vs_ipo'].values, -100, 10000)
            
            # Legacy column - default to bounce_vs_d1open
            df['lifetime_hi_vs_ipo'] = df['bounce_vs_d1open']
            
            # ============================================================
            # FINAL UNCONDITIONAL CAP - catches ANY edge cases
            # ============================================================
            df['bounce_vs_d1open'] = pd.to_numeric(df['bounce_vs_d1open'], errors='coerce').fillna(0)
            df['bounce_vs_ipo'] = pd.to_numeric(df['bounce_vs_ipo'], errors='coerce').fillna(0)
            df['lifetime_hi_vs_ipo'] = pd.to_numeric(df['lifetime_hi_vs_ipo'], errors='coerce').fillna(0)
            
            # Debug: Check for astronomical values BEFORE cap
            astronomical_mask = df['bounce_vs_d1open'] > 10000
            if astronomical_mask.any():
                bad_tickers = df.loc[astronomical_mask, 'polygon_ticker'].tolist()[:5]
                st.sidebar.warning(f"⚠️ Found {astronomical_mask.sum()} astronomical bounce values (>{10000}%): {bad_tickers}")
            
            # Apply the cap to all bounce columns
            df['bounce_vs_d1open'] = np.clip(df['bounce_vs_d1open'].values, -100, 10000)
            df['bounce_vs_ipo'] = np.clip(df['bounce_vs_ipo'].values, -100, 10000)
            df['lifetime_hi_vs_ipo'] = np.clip(df['lifetime_hi_vs_ipo'].values, -100, 10000)
            
            # Also filter out IPOs with suspiciously low prices (likely bad data)
            bad_data_mask = (df['ipo_price'] < 0.50) & (df['bounce_vs_d1open'] > 1000)
            n_bad = bad_data_mask.sum()
            if n_bad > 0:
                st.sidebar.info(f"ℹ️ Set bounce=0 for {n_bad} low-price IPOs")
            df.loc[bad_data_mask, 'bounce_vs_d1open'] = 0
            df.loc[bad_data_mask, 'bounce_vs_ipo'] = 0
            df.loc[bad_data_mask, 'lifetime_hi_vs_ipo'] = 0
            
            # Add missing columns with defaults if they don't exist
            if 'is_tax_haven' not in df.columns:
                # Detect tax haven countries
                TAX_HAVEN_COUNTRIES = ['CAYMAN ISLANDS', 'BRITISH VIRGIN ISLANDS', 'BERMUDA', 
                                       'MARSHALL ISLANDS', 'CYPRUS', 'JERSEY', 'GUERNSEY',
                                       'ISLE OF MAN', 'BAHAMAS', 'LUXEMBOURG', 'MALTA',
                                       'VIRGIN ISLANDS', 'BVI', 'PANAMA', 'SEYCHELLES']
                if 'country' in df.columns:
                    df['is_tax_haven'] = df['country'].str.upper().isin(TAX_HAVEN_COUNTRIES).astype(int)
                else:
                    df['is_tax_haven'] = 0
            
            if 'is_us' not in df.columns:
                if 'country' in df.columns:
                    df['is_us'] = df['country'].str.upper().isin(['US', 'USA', 'UNITED STATES', 'U.S.']).astype(int)
                else:
                    df['is_us'] = 0
            
            if 'operation_risk_score' not in df.columns:
                df['operation_risk_score'] = 0
                # Add basic risk factors
                if 'ipo_price' in df.columns:
                    df.loc[df['ipo_price'] <= 5, 'operation_risk_score'] += 15
                if 'ipo_shares_offered' in df.columns:
                    df.loc[df['ipo_shares_offered'] <= 1_500_000, 'operation_risk_score'] += 15
                if 'is_tax_haven' in df.columns:
                    df.loc[df['is_tax_haven'] == 1, 'operation_risk_score'] += 10
                if 'is_operation_uw' in df.columns:
                    df.loc[df['is_operation_uw'] == 1, 'operation_risk_score'] += 15
            
            # Count underwriter coverage
            has_uw = df['underwriter'].notna() & (df['underwriter'] != '')
            uw_count = has_uw.sum()
            uw_pct = uw_count / len(df) * 100 if len(df) > 0 else 0
            
            # Diagnostic: Show bounce data source
            bounce_source = "unknown"
            if 'lifetime_hi_vs_ipo' in df.columns and (df['lifetime_hi_vs_ipo'] != 0).any():
                bounce_source = "DB (pre-calculated)"
            elif 'd1_open_adj' in df.columns and (df['d1_open_adj'] > 0).any():
                bounce_source = "Calculated (LT_High_Adj / D1_Open_Adj)"
            else:
                bounce_source = "⚠️ Fallback (may need fix_bounce_source.py)"
            
            st.sidebar.success(f"✅ ClickHouse: {len(df):,} IPOs")
            st.sidebar.caption(f"📊 Underwriters: {uw_count:,} ({uw_pct:.0f}%)")
            st.sidebar.caption(f"📈 Bounce source: {bounce_source}")
            return df
    
    # Fallback to CSV files
    possible_paths = [
        # Priority 1: eqsipo files (most complete with WTF etc.)
        r'C:\Users\msui\Documents\Coding Projects\IPO Analysis\eqsipo_final.csv',
        r'C:\Users\msui\Documents\Coding Projects\IPO Analysis\eqsipo_complete.csv',
        'eqsipo_final.csv',
        'eqsipo_complete.csv',
        # Priority 2: adjusted files
        'small_ipo_fully_adjusted.csv',
        'data/small_ipo_fully_adjusted.csv',
        r'C:\Users\msui\Documents\Coding Projects\IPO Analysis\small_ipo_fully_adjusted.csv',
        r'P:\Hamren\Other\small_ipo_fully_adjusted.csv',
        # Priority 3: unadjusted
        'eqsipo_unadj.csv',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            
            # Handle different date column names
            if 'date' not in df.columns and 'IPO Dt' in df.columns:
                df['date'] = pd.to_datetime(df['IPO Dt'], errors='coerce')
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Clean ticker - handle both "WTF US Equity" and "WTF" formats
            if 'ticker_clean' not in df.columns:
                if 'Ticker' in df.columns:
                    df['ticker_clean'] = df['Ticker'].astype(str).str.replace(' US Equity', '', regex=False).str.strip()
                elif 'ticker' in df.columns:
                    df['ticker_clean'] = df['ticker'].astype(str).str.replace(' US Equity', '', regex=False).str.strip()
                else:
                    df['ticker_clean'] = ''
            
            # Ensure underwriter column exists
            if 'underwriter' not in df.columns:
                if 'IPO Lead' in df.columns:
                    df['underwriter'] = df['IPO Lead'].fillna('')
                else:
                    df['underwriter'] = ''
            
            # Patch known missing underwriter data
            KNOWN_UNDERWRITERS = {
                'WTF': 'CATHAY SECURITIES,DOMINARI',
                'WATO': 'CATHAY SECURITIES',
            }
            for ticker, uw in KNOWN_UNDERWRITERS.items():
                mask = df['ticker_clean'] == ticker
                if mask.any():
                    # Only patch if underwriter is empty/NaN
                    empty_mask = mask & (df['underwriter'].isna() | (df['underwriter'] == ''))
                    df.loc[empty_mask, 'underwriter'] = uw
            
            # Add risk score if missing
            if 'operation_risk_score' not in df.columns:
                df['operation_risk_score'] = 0
                if 'IPO Sh Px' in df.columns:
                    df.loc[df['IPO Sh Px'] <= 5, 'operation_risk_score'] += 15
                if 'IPO Sh Offered' in df.columns:
                    df.loc[df['IPO Sh Offered'] <= 1_500_000, 'operation_risk_score'] += 15
                
                # Add operation underwriter risk
                if 'underwriter' in df.columns:
                    for op_uw in OPERATION_UNDERWRITERS:
                        mask = df['underwriter'].str.contains(op_uw, case=False, na=False)
                        df.loc[mask, 'operation_risk_score'] += 10
            
            # Always recalculate/validate lifetime_hi_vs_ipo with improved split adjustment
            if 'Lifetime High' in df.columns and 'IPO Sh Px' in df.columns:
                # Convert to numeric
                df['IPO Sh Px'] = pd.to_numeric(df['IPO Sh Px'], errors='coerce').fillna(0)
                df['Lifetime High'] = pd.to_numeric(df['Lifetime High'], errors='coerce').fillna(0)
                
                # Check for d1_open and d1_close columns
                d1_open_col = None
                d1_close_col = None
                for col in ['d1_open', 'D1 Open', 'open_px', 'Open']:
                    if col in df.columns:
                        d1_open_col = col
                        break
                for col in ['d1_close', 'D1 Close', 'close_px', 'Close']:
                    if col in df.columns:
                        d1_close_col = col
                        break
                
                if d1_open_col:
                    df['d1_open_price'] = pd.to_numeric(df[d1_open_col], errors='coerce').fillna(0)
                else:
                    df['d1_open_price'] = 0
                    
                if d1_close_col:
                    df['d1_close_price'] = pd.to_numeric(df[d1_close_col], errors='coerce').fillna(0)
                else:
                    df['d1_close_price'] = 0
                
                # Calculate ratios
                df['d1_to_ipo_ratio'] = np.where(
                    (df['IPO Sh Px'] > 0) & (df['d1_open_price'] > 0),
                    df['d1_open_price'] / df['IPO Sh Px'],
                    1.0
                )
                df['lt_to_ipo_ratio'] = np.where(
                    df['IPO Sh Px'] > 0,
                    df['Lifetime High'] / df['IPO Sh Px'],
                    0
                )
                
                # Determine reference price based on scenario
                df['reference_price'] = np.where(
                    # Scenario 1: D1 open differs > 2x from IPO price
                    (df['d1_open_price'] > 0.10) & ((df['d1_to_ipo_ratio'] > 2.0) | (df['d1_to_ipo_ratio'] < 0.5)),
                    df['d1_open_price'],
                    np.where(
                        # Scenario 2: Both prices similar but lifetime_high >> both (50x+)
                        (df['d1_to_ipo_ratio'] >= 0.5) & (df['d1_to_ipo_ratio'] <= 2.0) & (df['lt_to_ipo_ratio'] > 50),
                        np.where(df['d1_close_price'] > 0.10, df['d1_close_price'], df['d1_open_price']),
                        # Scenario 3: Normal - use IPO price or D1 open
                        np.where(df['IPO Sh Px'] >= 0.50, df['IPO Sh Px'], df['d1_open_price'])
                    )
                )
                
                # Calculate bounce using reference price
                df['lifetime_hi_vs_ipo'] = np.where(
                    (df['reference_price'] >= 0.10) & (df['Lifetime High'] > 0),
                    ((df['Lifetime High'] / df['reference_price']) - 1) * 100,
                    0
                )
                
                # Cap based on scenario - aggressive cap for suspected unadjusted data
                df['lifetime_hi_vs_ipo'] = np.where(
                    (df['d1_to_ipo_ratio'] >= 0.5) & (df['d1_to_ipo_ratio'] <= 2.0) & (df['lt_to_ipo_ratio'] > 50),
                    np.clip(df['lifetime_hi_vs_ipo'].values, -100, 1000),  # Aggressive cap
                    np.clip(df['lifetime_hi_vs_ipo'].values, -100, 5000)   # Normal cap
                )
                
                # ============================================================
                # FINAL UNCONDITIONAL CAP - catches ANY edge cases
                # ============================================================
                df['lifetime_hi_vs_ipo'] = np.clip(df['lifetime_hi_vs_ipo'].values, -100, 10000)
                
                # Filter out bad data: IPO price < $0.50 AND bounce > 1000%
                bad_data_mask = (df['IPO Sh Px'] < 0.50) & (df['lifetime_hi_vs_ipo'] > 1000)
                df.loc[bad_data_mask, 'lifetime_hi_vs_ipo'] = 0
                
            elif 'lifetime_hi_vs_ipo' in df.columns:
                # If column exists but source columns don't, just cap existing values
                df['lifetime_hi_vs_ipo'] = pd.to_numeric(df['lifetime_hi_vs_ipo'], errors='coerce').fillna(0)
                df['lifetime_hi_vs_ipo'] = np.clip(df['lifetime_hi_vs_ipo'].values, -100, 10000)
            else:
                df['lifetime_hi_vs_ipo'] = 0
            
            # Count underwriter coverage
            uw_col = 'underwriter' if 'underwriter' in df.columns else 'IPO Lead' if 'IPO Lead' in df.columns else None
            if uw_col:
                has_uw = df[uw_col].notna() & (df[uw_col] != '') & (df[uw_col] != 'Unknown')
                uw_count = has_uw.sum()
                uw_pct = uw_count / len(df) * 100 if len(df) > 0 else 0
                st.sidebar.success(f"✅ CSV: {len(df):,} IPOs")
                st.sidebar.caption(f"📊 Underwriters: {uw_count:,} ({uw_pct:.0f}%)")
            else:
                st.sidebar.success(f"✅ CSV: {len(df):,} IPOs")
            
            return df
    
    st.error("""
    Could not find IPO data file.
    
    Please place one of these files in the same folder as this dashboard:
    - small_ipo_fully_adjusted.csv
    - eqsipo_complete.csv
    - eqsipo_final.csv
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
            # Convert UTC to EST
            ts = pd.to_datetime(bar.timestamp, unit='ms', utc=True)
            ts_est = ts.tz_convert('America/New_York').tz_localize(None)
            bars.append({
                'timestamp': ts_est,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            })
        
        if bars:
            df = pd.DataFrame(bars)
            # Filter to regular trading hours only (9:30 AM - 4:00 PM EST) for minute data
            if timeframe == "minute" and len(df) > 0:
                df['hour'] = df['timestamp'].dt.hour
                df['minute'] = df['timestamp'].dt.minute
                df['time_decimal'] = df['hour'] + df['minute'] / 60
                # 9:30 AM = 9.5, 8:00 PM = 20.0 (includes after-hours)
                df = df[(df['time_decimal'] >= 9.5) & (df['time_decimal'] <= 20.0)]
                df = df.drop(columns=['hour', 'minute', 'time_decimal'])
            return df
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
                              end=start.replace(hour=20, minute=0),
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
    """Create a candlestick chart with volume and scrollable x-axis."""
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
    
    # Set x-axis range for minute data (9:30 AM - 4:00 PM)
    x_range = None
    if ('minute' in title.lower() or '1-min' in title.lower()) and len(df) > 0:
        chart_date = df['timestamp'].iloc[0].date()
        x_start = pd.Timestamp(chart_date).replace(hour=9, minute=30)
        x_end = pd.Timestamp(chart_date).replace(hour=20, minute=0)
        x_range = [x_start, x_end]
    
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>High: ${max_price:.2f} | Low: ${min_price:.2f} | Change: {pct_change:+.1f}%</sup>",
            x=0.5,
            font_size=14
        ),
        height=height,
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=50),
        template='plotly_white',
        
        # Enable range slider for scrolling
        xaxis=dict(
            rangeslider=dict(visible=False),  # Disable on price chart (use bottom one)
            type='date',
            tickformat='%Y-%m-%d %H:%M' if 'minute' in title.lower() or '1-min' in title.lower() else '%Y-%m-%d',
            range=x_range,
        ),
        xaxis2=dict(
            rangeslider=dict(visible=True, thickness=0.05),  # Scrollable range slider
            type='date',
            tickformat='%Y-%m-%d',
        ),
        
        # Enable zooming and panning
        dragmode='zoom',
    )
    
    # Add range selector buttons
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1D", step="day", stepmode="backward"),
                dict(count=7, label="1W", step="day", stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            bgcolor='#f0f0f0',
            font=dict(size=10),
        ),
        row=1, col=1
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
        showlegend=False,
        xaxis=dict(
            tickformat='%m/%d',
            showgrid=True,
        ),
        yaxis=dict(
            showgrid=True,
        ),
    )
    
    return fig


# ============================================================================
# AUTO-SYNC ON STARTUP
# ============================================================================

def check_and_sync_on_startup():
    """Check if CSV has newer data than ClickHouse and sync if needed."""
    if not HAS_CLICKHOUSE:
        return

    try:
        config = load_config()
        ch_password = config.get('clickhouse_password', '') or os.environ.get('CLICKHOUSE_PASSWORD', '')
        if not ch_password:
            return

        client = clickhouse_connect.get_client(
            host=CLICKHOUSE_CONFIG['host'],
            port=8443,
            username=CLICKHOUSE_CONFIG['user'],
            password=ch_password,
            secure=True,
            database='market_data'
        )

        # Get ClickHouse row count
        ch_count = client.query("SELECT count(*) FROM ipo_master").result_rows[0][0]

        # Get CSV row count
        csv_path = os.path.join(os.path.dirname(__file__), 'eqsipo_final.csv')
        if os.path.exists(csv_path):
            csv_df = pd.read_csv(csv_path, usecols=[0], nrows=None)
            csv_count = len(csv_df)

            if csv_count > ch_count:
                st.toast(f"📊 CSV has {csv_count - ch_count} more IPOs than ClickHouse. Consider syncing.", icon="ℹ️")
    except Exception:
        pass  # Silent fail - don't block dashboard startup


# ============================================================================
# UNDERWRITER OPPORTUNITIES PAGE
# ============================================================================

def underwriter_opportunities_page():
    """Page showing underwriter-based buying opportunities for low-dollar IPOs."""
    st.header("💰 Underwriter Opportunities")
    st.markdown("*Find low-dollar IPOs from high-success underwriters that haven't doubled yet*")

    # Load data
    use_ch = st.session_state.get('use_clickhouse', False)
    ch_pw = st.session_state.get('ch_password', None)
    cache_key = f"ch_{use_ch}_{bool(ch_pw)}"
    df = load_ipo_data(use_clickhouse=use_ch, ch_password=ch_pw, _cache_key=cache_key)

    if df.empty:
        st.warning("No data loaded. Please configure data source in the Analysis tab.")
        return

    # Prepare data for analysis
    df_analysis = df.copy()

    # Ensure we have required columns
    if 'IPO Sh Px' not in df_analysis.columns:
        st.error("Missing IPO price column")
        return

    # Clean up columns
    df_analysis['ipo_price'] = pd.to_numeric(df_analysis.get('IPO Sh Px', df_analysis.get('ipo_price', 0)), errors='coerce')
    df_analysis['lifetime_high_raw'] = pd.to_numeric(df_analysis.get('Lifetime High', df_analysis.get('lifetime_high', 0)), errors='coerce')
    df_analysis['current_price'] = pd.to_numeric(df_analysis.get('last_px_adj', df_analysis.get('current_price', 0)), errors='coerce')

    # Get split factor for adjustment (default to 1 if not available)
    df_analysis['split_factor'] = pd.to_numeric(df_analysis.get('cum_split_factor_since_base', 1), errors='coerce').fillna(1)
    df_analysis['split_factor'] = df_analysis['split_factor'].replace(0, 1)  # Avoid division by zero

    # Adjust lifetime high for splits (divide by split factor to get adjusted price)
    # This ensures we compare apples to apples with the IPO price
    df_analysis['lifetime_high'] = df_analysis['lifetime_high_raw'] / df_analysis['split_factor']

    # Cap lifetime high at reasonable multiple of IPO price to handle bad data
    # (some tickers have incorrect split factors or unadjusted lifetime highs)
    # A 50x gain (5000%) is extremely rare but possible; anything beyond is likely data error
    MAX_REASONABLE_MULTIPLE = 50
    df_analysis['lifetime_high'] = np.minimum(
        df_analysis['lifetime_high'],
        df_analysis['ipo_price'] * MAX_REASONABLE_MULTIPLE
    )

    # Get underwriter column
    uw_col = 'underwriter' if 'underwriter' in df_analysis.columns else 'IPO Lead'
    if uw_col in df_analysis.columns:
        df_analysis['underwriter_clean'] = df_analysis[uw_col].astype(str).str.upper().str.strip()
    else:
        st.error("Missing underwriter column")
        return

    # Parse dates
    date_col = 'ipo_date' if 'ipo_date' in df_analysis.columns else 'date'
    if date_col in df_analysis.columns:
        df_analysis['ipo_date_parsed'] = pd.to_datetime(df_analysis[date_col], errors='coerce')
    else:
        st.error("Missing IPO date column")
        return

    # Get ticker column
    ticker_col = 'ticker_clean' if 'ticker_clean' in df_analysis.columns else 'polygon_ticker'
    if ticker_col not in df_analysis.columns:
        df_analysis['ticker_clean'] = df_analysis['Ticker'].astype(str).str.replace(' US Equity', '').str.strip() if 'Ticker' in df_analysis.columns else 'N/A'
        ticker_col = 'ticker_clean'

    # ========================================================================
    # FILTERS (with saved defaults)
    # ========================================================================
    st.subheader("🔧 Filters")

    # Load saved filter defaults
    config = load_config()
    saved_filters = config.get('opportunity_filters', {})

    # Default values (used if no saved config)
    default_filters = {
        'low_dollar_min': 1.0,
        'low_dollar_max': 10.0,
        'lookback_months': 24,
        'min_uw_ipos': 2,
        'min_double_rate': 30,
        'min_close_to_target': 80,
        'max_lifetime_gain_idx': 0,  # Index for "Below IPO price" - hasn't pumped at all
        'min_uw_rate_idx': 1,  # Index for "30%"
        'operation_filter_idx': 0,  # Index for "No operations" - hasn't been operated
    }

    # Merge saved with defaults
    for key, default_val in default_filters.items():
        if key not in saved_filters:
            saved_filters[key] = default_val

    col1, col2, col3 = st.columns(3)

    with col1:
        low_dollar_min = st.number_input(
            "Min IPO Price ($)",
            value=float(saved_filters['low_dollar_min']),
            min_value=0.0, max_value=50.0, step=0.5,
            key="opp_low_dollar_min"
        )
    with col2:
        low_dollar_max = st.number_input(
            "Max IPO Price ($)",
            value=float(saved_filters['low_dollar_max']),
            min_value=0.0, max_value=100.0, step=1.0,
            key="opp_low_dollar_max"
        )
    with col3:
        lookback_months = st.slider(
            "Lookback Period (months)",
            6, 24,
            value=int(saved_filters['lookback_months']),
            key="opp_lookback"
        )

    col4, col5 = st.columns(2)
    with col4:
        min_uw_ipos = st.slider(
            "Min IPOs per Underwriter",
            2, 10,
            value=int(saved_filters['min_uw_ipos']),
            key="opp_min_uw_ipos"
        )
    with col5:
        min_double_rate = st.slider(
            "Min Double Rate (%)",
            0, 100,
            value=int(saved_filters['min_double_rate']),
            key="opp_min_double_rate"
        )

    # Opportunity filters
    st.markdown("---")
    st.markdown("**🎯 Opportunity Criteria** - Find IPOs that haven't been 'operated' (pumped) yet")

    col6, col7, col8 = st.columns(3)

    close_to_target_options = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    max_lifetime_options = ["Below IPO price", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%", "150%", "200%", "Any"]
    min_uw_rate_options = ["Any", "30%", "50%", "75%", "100%"]
    operation_filter_options = ["No operations", "Any"]

    with col6:
        min_close_to_target = st.select_slider(
            "Min 'Close to Target' %",
            options=close_to_target_options,
            value=int(saved_filters['min_close_to_target']),
            help="Minimum % progress toward the target (e.g., 50% = halfway to 2x)",
            key="opp_min_close"
        )
    with col7:
        # Handle legacy saved index that may be out of range
        saved_max_idx = int(saved_filters.get('max_lifetime_gain_idx', 9))
        if saved_max_idx >= len(max_lifetime_options):
            saved_max_idx = 9  # Default to 100%
        max_lifetime_gain = st.selectbox(
            "Max Lifetime Gain (no pump)",
            options=max_lifetime_options,
            index=saved_max_idx,
            help="Filter out IPOs that already pumped. Lower = hasn't run yet (e.g., 50% = never gained more than 50%)",
            key="opp_max_lifetime"
        )
    with col8:
        min_uw_rate_filter = st.selectbox(
            "Min Underwriter Success Rate",
            options=min_uw_rate_options,
            index=int(saved_filters['min_uw_rate_idx']),
            help="Only show IPOs from underwriters with this historical double rate",
            key="opp_min_uw_rate"
        )

    # Operation filter row
    col9, col10, col11 = st.columns(3)
    with col9:
        operation_filter = st.selectbox(
            "Operation Status",
            options=operation_filter_options,
            index=int(saved_filters.get('operation_filter_idx', 0)),
            help="Filter by whether IPO has had a volume+price spike (operation). 'No operations' = hasn't been pumped yet.",
            key="opp_operation_filter"
        )

    # Save as default button
    st.markdown("---")
    col_save, col_reset = st.columns([1, 1])
    with col_save:
        if st.button("💾 Save as Default", key="save_opp_filters", help="Save current filters as your default"):
            new_filters = {
                'low_dollar_min': low_dollar_min,
                'low_dollar_max': low_dollar_max,
                'lookback_months': lookback_months,
                'min_uw_ipos': min_uw_ipos,
                'min_double_rate': min_double_rate,
                'min_close_to_target': min_close_to_target,
                'max_lifetime_gain_idx': max_lifetime_options.index(max_lifetime_gain),
                'min_uw_rate_idx': min_uw_rate_options.index(min_uw_rate_filter),
                'operation_filter_idx': operation_filter_options.index(operation_filter),
            }
            config['opportunity_filters'] = new_filters
            save_config(config)
            st.success("✅ Filters saved as default!")
    with col_reset:
        if st.button("🔄 Reset to Default", key="reset_opp_filters", help="Reset filters to saved defaults"):
            st.rerun()

    # Parse filter values
    filter_below_ipo = False  # Special flag for "below IPO price" filter
    if max_lifetime_gain == "Any":
        max_lifetime_pct = 10000  # Effectively no limit
    elif max_lifetime_gain == "Below IPO price":
        filter_below_ipo = True  # Will filter for lifetime_high < ipo_price directly
        max_lifetime_pct = None  # Not used when filter_below_ipo is True
    else:
        # Extract number from string like "50%" or "100%"
        max_lifetime_pct = int(max_lifetime_gain.replace("%", ""))

    if min_uw_rate_filter == "Any":
        min_uw_rate_value = 0
    else:
        # Extract number from string like "30%" or "50%"
        min_uw_rate_value = int(min_uw_rate_filter.replace("%", ""))

    # ========================================================================
    # CALCULATE METRICS
    # ========================================================================

    # Filter for low dollar IPOs
    low_dollar = df_analysis[
        (df_analysis['ipo_price'] >= low_dollar_min) &
        (df_analysis['ipo_price'] <= low_dollar_max) &
        (df_analysis['ipo_price'] > 0)
    ].copy()

    if len(low_dollar) == 0:
        st.warning(f"No IPOs found in the ${low_dollar_min}-${low_dollar_max} price range")
        return

    # Calculate metrics
    low_dollar['hit_double'] = low_dollar['lifetime_high'] >= (low_dollar['ipo_price'] * 2)
    low_dollar['max_gain_pct'] = np.where(
        low_dollar['ipo_price'] > 0,
        ((low_dollar['lifetime_high'] - low_dollar['ipo_price']) / low_dollar['ipo_price']) * 100,
        0
    )
    low_dollar['current_return_pct'] = np.where(
        (low_dollar['ipo_price'] > 0) & (low_dollar['current_price'] > 0),
        ((low_dollar['current_price'] - low_dollar['ipo_price']) / low_dollar['ipo_price']) * 100,
        np.nan
    )

    # ========================================================================
    # CALCULATE UNDERWRITER STATISTICS (needed for opportunities)
    # ========================================================================
    uw_stats = low_dollar.groupby('underwriter_clean').agg(
        total_ipos=(ticker_col, 'count'),
        doubled_count=('hit_double', 'sum'),
        avg_max_gain=('max_gain_pct', 'mean'),
        median_max_gain=('max_gain_pct', 'median'),
    ).reset_index()

    uw_stats['double_rate'] = (uw_stats['doubled_count'] / uw_stats['total_ipos']) * 100
    uw_stats = uw_stats[uw_stats['total_ipos'] >= min_uw_ipos].sort_values('double_rate', ascending=False)

    # ========================================================================
    # OPPORTUNITIES TABLE (displayed first)
    # ========================================================================
    st.subheader("🎯 Buying Opportunities")
    st.markdown(f"*Recent IPOs from high-success underwriters matching your criteria*")

    # Filter for recent IPOs that haven't doubled
    cutoff_date = datetime.now() - timedelta(days=lookback_months * 30)

    # Get high success underwriters (using the stricter of min_double_rate and min_uw_rate_value)
    effective_min_uw_rate = max(min_double_rate, min_uw_rate_value)
    high_success_uw = uw_stats[uw_stats['double_rate'] >= effective_min_uw_rate]['underwriter_clean'].tolist()

    # Calculate lifetime gain % for filtering
    low_dollar['lifetime_gain_pct'] = np.where(
        low_dollar['ipo_price'] > 0,
        ((low_dollar['lifetime_high'] - low_dollar['ipo_price']) / low_dollar['ipo_price']) * 100,
        0
    )

    # Calculate close to target % (using 2x as target)
    low_dollar['close_to_target_pct'] = np.where(
        low_dollar['ipo_price'] > 0,
        (low_dollar['lifetime_high'] / (low_dollar['ipo_price'] * 2)) * 100,
        0
    )

    # Apply all filters
    # Build lifetime high filter based on user selection
    if filter_below_ipo:
        # Direct comparison: lifetime high must be below IPO price
        lifetime_filter = low_dollar['lifetime_high'] < low_dollar['ipo_price']
    else:
        # Percentage gain filter
        lifetime_filter = low_dollar['lifetime_gain_pct'] < max_lifetime_pct

    # Build operation filter based on user selection
    if operation_filter == "No operations":
        # Only show IPOs that haven't had any detected operations (pumps)
        if 'has_operation' in low_dollar.columns:
            operation_status_filter = (low_dollar['has_operation'] == 0) | (low_dollar['has_operation'].isna())
        else:
            # Fallback if column doesn't exist - use lifetime gain as proxy
            operation_status_filter = low_dollar['lifetime_gain_pct'] < 50
    else:
        # "Any" - no filter
        operation_status_filter = True

    opportunities = low_dollar[
        (low_dollar['underwriter_clean'].isin(high_success_uw)) &
        lifetime_filter &  # Hasn't exceeded max lifetime gain
        operation_status_filter &  # Operation status filter
        (low_dollar['close_to_target_pct'] >= min_close_to_target) &  # Got close enough to target
        (low_dollar['ipo_date_parsed'] >= cutoff_date) &
        (low_dollar['current_price'].notna()) &
        (low_dollar['current_price'] > 0)
    ].copy()

    # Merge with underwriter stats
    opportunities = opportunities.merge(
        uw_stats[['underwriter_clean', 'double_rate', 'total_ipos', 'doubled_count']],
        on='underwriter_clean',
        how='left'
    )

    # Calculate opportunity metrics
    opportunities['close_to_double_pct'] = opportunities['close_to_target_pct']
    opportunities['upside_to_double'] = np.where(
        opportunities['current_price'] > 0,
        ((opportunities['ipo_price'] * 2 - opportunities['current_price']) / opportunities['current_price']) * 100,
        0
    )

    # Sort by opportunity score
    opportunities['opp_score'] = (
        opportunities['double_rate'] * 0.4 +
        opportunities['close_to_double_pct'].clip(0, 100) * 0.3 +
        opportunities['upside_to_double'].clip(0, 300) * 0.3
    )
    opportunities = opportunities.sort_values('opp_score', ascending=False)

    # Show active filter summary
    if filter_below_ipo:
        lifetime_filter_text = "Lifetime high < IPO price"
    else:
        lifetime_filter_text = f"Lifetime gain < {max_lifetime_pct}%"
    operation_filter_text = "No prior operations" if operation_filter == "No operations" else "Any"
    filter_summary = f"Filters: Close to 2x ≥ {min_close_to_target}% | {lifetime_filter_text} | {operation_filter_text} | UW rate ≥ {effective_min_uw_rate}%"
    st.caption(filter_summary)

    if len(opportunities) == 0:
        st.info("No opportunities found matching current filters. Try adjusting the parameters.")
    else:
        st.success(f"Found **{len(opportunities)}** potential opportunities")

        # Display table
        display_opps = opportunities.head(25).copy()

        # Format columns for display
        display_opps['IPO Date'] = display_opps['ipo_date_parsed'].dt.strftime('%Y-%m-%d')
        display_opps['IPO Px'] = display_opps['ipo_price'].apply(lambda x: f"${x:.2f}")
        display_opps['Life Hi'] = display_opps['lifetime_high'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
        display_opps['Current'] = display_opps['current_price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
        display_opps['Return'] = display_opps['current_return_pct'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
        display_opps['Close to 2x'] = display_opps['close_to_double_pct'].apply(lambda x: f"{x:.0f}%")
        display_opps['Upside'] = display_opps['upside_to_double'].apply(lambda x: f"{x:.0f}%")
        display_opps['UW Rate'] = display_opps['double_rate'].apply(lambda x: f"{x:.0f}%")

        # Add operation count if available
        if 'operation_count' in display_opps.columns:
            display_opps['Ops'] = display_opps['operation_count'].fillna(0).astype(int)
        else:
            display_opps['Ops'] = 0

        # Get name column
        name_col = 'Name' if 'Name' in display_opps.columns else 'name'
        if name_col in display_opps.columns:
            display_opps['Company'] = display_opps[name_col].astype(str).str[:25]
        else:
            display_opps['Company'] = ''

        # Build column list - include Ops if we have operation data
        display_cols = [ticker_col, 'Company', 'underwriter_clean', 'IPO Date', 'IPO Px', 'Life Hi', 'Current', 'Return', 'Close to 2x', 'Upside', 'Ops', 'UW Rate']

        st.dataframe(
            display_opps[display_cols].rename(columns={
                ticker_col: 'Ticker',
                'underwriter_clean': 'Underwriter'
            }),
            width="stretch",
            hide_index=True
        )

        # Legend
        st.markdown("""
        **Legend:**
        - **Life Hi**: Lifetime high price (adjusted for splits, capped at 50x IPO)
        - **Close to 2x**: How close lifetime high got to 2x IPO price (100% = hit double)
        - **Upside**: Potential gain if stock reaches 2x IPO price from current price
        - **Ops**: Number of detected operations (volume+price spikes)
        - **UW Rate**: Historical % of underwriter's low-dollar IPOs that doubled
        """)

    # ========================================================================
    # UNDERWRITER STATISTICS (displayed after opportunities)
    # ========================================================================
    st.subheader("📊 Underwriter Performance (Low Dollar IPOs)")
    st.markdown(f"**Top Underwriters by Double Rate** (min {min_uw_ipos} IPOs)")

    display_uw = uw_stats.head(15).copy()
    display_uw['Double Rate'] = display_uw['double_rate'].apply(lambda x: f"{x:.1f}%")
    display_uw['Median Max Gain'] = display_uw['median_max_gain'].apply(lambda x: f"{x:.0f}%")
    display_uw = display_uw.rename(columns={
        'underwriter_clean': 'Underwriter',
        'total_ipos': 'Total IPOs',
        'doubled_count': 'Doubled'
    })

    st.dataframe(
        display_uw[['Underwriter', 'Total IPOs', 'Doubled', 'Double Rate', 'Median Max Gain']],
        width="stretch",
        hide_index=True
    )

    # ========================================================================
    # KNOWN OPERATION UNDERWRITERS
    # ========================================================================
    with st.expander("📋 Known 'Operation' Underwriters Reference"):
        st.markdown("""
        These underwriters are associated with small-cap/low-dollar IPOs that historically show "pump" patterns:

        **High Success Rate (>70%):**
        - Prime Number Capital, Viewtrade Securities, Univest Securities
        - Spartan Capital, Dawson James, EF Hutton
        - Network 1 Financial, Boustead Securities, Aegis Capital

        **Moderate Success Rate (50-70%):**
        - Maxim Group, ThinkEquity, Joseph Stone Capital
        - RF Lafferty, Kingswood Capital, D Boral Capital
        - US Tiger Securities, Cathay Securities

        ⚠️ **Warning**: High double rates come with high volatility and crash risk. These patterns are associated with pump-and-dump activity.
        """)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("📈 IPO Operation Analysis Dashboard")

    # Auto-sync check on startup (only once per session)
    if 'startup_sync_done' not in st.session_state:
        st.session_state.startup_sync_done = True
        # Check if CSV has newer data than ClickHouse and offer to sync
        check_and_sync_on_startup()

    # Create tabs for different views (Opportunities first)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💰 Opportunities", "📊 Analysis", "🆕 Upcoming IPOs", "📈 Statistics & Prediction", "ℹ️ About"])

    with tab1:
        underwriter_opportunities_page()

    with tab2:
        analysis_page()

    with tab3:
        upcoming_ipos_page()

    with tab4:
        statistics_page()

    with tab5:
        about_page()


def statistics_page():
    """Statistical analysis page with correlation, regression, and predictions."""
    st.header("📈 Statistics & Prediction")
    
    # Load data (use session state for ClickHouse settings if available)
    use_ch = st.session_state.get('use_clickhouse', False)
    ch_pw = st.session_state.get('ch_password', None)
    cache_key = f"ch_{use_ch}_{bool(ch_pw)}"
    df = load_ipo_data(use_clickhouse=use_ch, ch_password=ch_pw, _cache_key=cache_key)
    if df.empty:
        st.warning("No data loaded")
        return
    
    # Prepare numeric columns for analysis
    numeric_cols = ['IPO Sh Px', 'IPO Sh Offered', 'lifetime_hi_vs_ipo', 
                    'ret_d1', 'operation_risk_score']
    
    # Add derived columns if not present
    if 'market_cap_mm' not in df.columns and 'IPO Sh Px' in df.columns and 'IPO Sh Offered' in df.columns:
        df['market_cap_mm'] = (df['IPO Sh Px'] * df['IPO Sh Offered']) / 1_000_000
    
    if 'log_shares' not in df.columns and 'IPO Sh Offered' in df.columns:
        df['log_shares'] = np.log1p(df['IPO Sh Offered'])
    
    # Sub-tabs for different analyses
    stat_tab1, stat_tab2, stat_tab3, stat_tab4 = st.tabs([
        "📊 Correlation", "📈 Regression", "🎯 Prediction", "📋 Factor Analysis"
    ])
    
    # ========================================================================
    # TAB 1: CORRELATION ANALYSIS
    # ========================================================================
    with stat_tab1:
        st.subheader("Correlation Matrix")
        
        # Select columns for correlation
        available_numeric = [col for col in df.columns if df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
        
        default_cols = ['IPO Sh Px', 'IPO Sh Offered', 'lifetime_hi_vs_ipo', 'ret_d1', 'operation_risk_score']
        default_cols = [c for c in default_cols if c in available_numeric]
        
        selected_cols = st.multiselect(
            "Select variables for correlation",
            options=available_numeric,
            default=default_cols[:6]
        )
        
        if len(selected_cols) >= 2:
            corr_df = df[selected_cols].corr()
            
            # Heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns,
                y=corr_df.columns,
                colorscale='RdBu_r',
                zmid=0,
                text=np.round(corr_df.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Correlation Heatmap',
                height=500,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, width="stretch")
            
            # Key correlations with bounce
            if 'lifetime_hi_vs_ipo' in selected_cols:
                st.markdown("### Key Correlations with Lifetime Bounce")
                bounce_corr = corr_df['lifetime_hi_vs_ipo'].drop('lifetime_hi_vs_ipo').sort_values(key=abs, ascending=False)
                
                for var, corr in bounce_corr.items():
                    direction = "📈" if corr > 0 else "📉"
                    strength = "Strong" if abs(corr) > 0.3 else "Moderate" if abs(corr) > 0.15 else "Weak"
                    st.write(f"{direction} **{var}**: {corr:.3f} ({strength})")
    
    # ========================================================================
    # TAB 2: REGRESSION ANALYSIS
    # ========================================================================
    with stat_tab2:
        st.subheader("Regression Analysis: Predicting Bounce")
        
        try:
            from scipy import stats
            
            # Simple OLS regression
            st.markdown("### Simple Linear Regression")
            
            # Select predictor
            predictors = ['IPO Sh Px', 'IPO Sh Offered', 'operation_risk_score', 'market_cap_mm']
            predictors = [p for p in predictors if p in df.columns]
            
            if predictors and 'lifetime_hi_vs_ipo' in df.columns:
                x_var = st.selectbox("Select Predictor (X)", predictors)
                
                # Clean data
                reg_df = df[[x_var, 'lifetime_hi_vs_ipo']].dropna()
                reg_df = reg_df[reg_df['lifetime_hi_vs_ipo'] < 5000]  # Remove extreme outliers
                
                if len(reg_df) > 10:
                    X = reg_df[x_var].values
                    y = reg_df['lifetime_hi_vs_ipo'].values
                    
                    slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
                    
                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("R²", f"{r_value**2:.3f}")
                    col2.metric("Slope", f"{slope:.3f}")
                    col3.metric("P-value", f"{p_value:.4f}")
                    col4.metric("Std Error", f"{std_err:.3f}")
                    
                    # Scatter plot with regression line
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=X, y=y,
                        mode='markers',
                        name='Data',
                        marker=dict(size=5, opacity=0.5)
                    ))
                    
                    x_line = np.linspace(X.min(), X.max(), 100)
                    y_line = slope * x_line + intercept
                    
                    fig.add_trace(go.Scatter(
                        x=x_line, y=y_line,
                        mode='lines',
                        name=f'y = {slope:.2f}x + {intercept:.2f}',
                        line=dict(color='red', width=2)
                    ))
                    
                    fig.update_layout(
                        title=f'{x_var} vs Lifetime Bounce',
                        xaxis_title=x_var,
                        yaxis_title='Lifetime Bounce (%)',
                        height=400
                    )
                    
                    st.plotly_chart(fig, width="stretch")
                    
                    # Interpretation
                    st.markdown("### Interpretation")
                    if p_value < 0.05:
                        st.success(f"✅ Statistically significant (p={p_value:.4f})")
                        if slope > 0:
                            st.write(f"Higher {x_var} → Higher bounce (positive relationship)")
                        else:
                            st.write(f"Higher {x_var} → Lower bounce (negative relationship)")
                    else:
                        st.warning(f"⚠️ Not statistically significant (p={p_value:.4f})")
                    
        except ImportError:
            st.warning("Install scipy for regression analysis: pip install scipy")
    
    # ========================================================================
    # TAB 3: PREDICTION MODEL
    # ========================================================================
    with stat_tab3:
        st.subheader("🎯 Bounce Prediction Model")
        
        st.markdown("""
        This model predicts likely bounce range based on IPO characteristics.
        Uses historical data to estimate probability distributions.
        """)
        
        # Input form for prediction
        st.markdown("### Enter IPO Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pred_price = st.number_input("IPO Price ($)", min_value=1.0, max_value=50.0, value=4.0)
            pred_shares = st.number_input("Shares Offered (M)", min_value=0.5, max_value=10.0, value=1.5)
            pred_country = st.selectbox("Country", ["US", "Tax Haven (KY, VG)", "Other"])
        
        with col2:
            pred_underwriter = st.selectbox("Underwriter Type", ["Operation UW", "Neutral", "Legit UW"])
            pred_vc = st.selectbox("VC Backed?", ["No", "Yes"])
        
        if st.button("🔮 Predict Bounce"):
            # Calculate risk score
            risk_score = 0
            
            if pred_price <= 5:
                risk_score += 15
            if pred_shares <= 1.5:
                risk_score += 15
            elif pred_shares <= 2.5:
                risk_score += 10
            if pred_country == "Tax Haven (KY, VG)":
                risk_score += 15
            elif pred_country == "US":
                risk_score -= 5
            if pred_underwriter == "Operation UW":
                risk_score += 10
            elif pred_underwriter == "Legit UW":
                risk_score -= 10
            if pred_vc == "No":
                risk_score += 10
            
            # Find similar IPOs
            similar_mask = (
                (df['IPO Sh Px'] >= pred_price - 2) & 
                (df['IPO Sh Px'] <= pred_price + 2) &
                (df['operation_risk_score'] >= risk_score - 10) &
                (df['operation_risk_score'] <= risk_score + 10)
            )
            similar_df = df[similar_mask]
            
            if len(similar_df) < 5:
                # Fallback to risk score only
                similar_mask = (
                    (df['operation_risk_score'] >= risk_score - 15) &
                    (df['operation_risk_score'] <= risk_score + 15)
                )
                similar_df = df[similar_mask]
            
            st.markdown("---")
            st.markdown("### Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Risk Score", f"{risk_score}")
                risk_level = "🔴 High" if risk_score >= 40 else "🟡 Medium" if risk_score >= 25 else "🟢 Low"
                st.write(f"Risk Level: {risk_level}")
            
            with col2:
                if len(similar_df) > 0:
                    median_bounce = similar_df['lifetime_hi_vs_ipo'].median()
                    st.metric("Expected Median Bounce", f"{median_bounce:.0f}%")
                else:
                    st.metric("Expected Median Bounce", "N/A")
            
            with col3:
                if len(similar_df) > 0:
                    pct_100 = (similar_df['lifetime_hi_vs_ipo'] > 100).mean() * 100
                    st.metric("P(Bounce > 100%)", f"{pct_100:.0f}%")
                else:
                    st.metric("P(Bounce > 100%)", "N/A")
            
            # Distribution of similar IPOs
            if len(similar_df) > 5:
                st.markdown(f"### Distribution of {len(similar_df)} Similar IPOs")
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=similar_df['lifetime_hi_vs_ipo'].clip(upper=2000),
                    nbinsx=30,
                    name='Bounce Distribution'
                ))
                
                fig.add_vline(x=similar_df['lifetime_hi_vs_ipo'].median(), 
                             line_dash="dash", line_color="red",
                             annotation_text="Median")
                
                fig.update_layout(
                    title='Bounce Distribution for Similar IPOs',
                    xaxis_title='Lifetime Bounce (%)',
                    yaxis_title='Count',
                    height=300
                )
                
                st.plotly_chart(fig, width="stretch")
                
                # Percentiles
                st.markdown("### Bounce Percentiles")
                percentiles = [10, 25, 50, 75, 90]
                pct_values = np.percentile(similar_df['lifetime_hi_vs_ipo'].dropna(), percentiles)
                
                pct_df = pd.DataFrame({
                    'Percentile': [f'{p}th' for p in percentiles],
                    'Bounce (%)': [f'{v:.0f}%' for v in pct_values]
                })
                st.dataframe(pct_df, width="stretch", hide_index=True)
    
    # ========================================================================
    # TAB 4: FACTOR ANALYSIS
    # ========================================================================
    with stat_tab4:
        st.subheader("📋 Factor Analysis by Category")
        
        # By Underwriter
        st.markdown("### Bounce by Underwriter (Top 20)")
        
        uw_stats = df.groupby('underwriter').agg({
            'lifetime_hi_vs_ipo': ['median', 'mean', 'count'],
            'operation_risk_score': 'mean'
        }).reset_index()
        uw_stats.columns = ['Underwriter', 'Median Bounce', 'Mean Bounce', 'Count', 'Avg Risk']
        uw_stats = uw_stats[uw_stats['Count'] >= 3].sort_values('Median Bounce', ascending=False).head(20)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=uw_stats['Underwriter'].str[:30],
            y=uw_stats['Median Bounce'],
            text=uw_stats['Count'].apply(lambda x: f'n={x}'),
            textposition='outside',
            marker_color=uw_stats['Avg Risk'],
            marker_colorscale='RdYlGn_r',
            marker_showscale=True,
            marker_colorbar_title='Risk'
        ))
        
        fig.update_layout(
            title='Median Bounce by Underwriter (color = avg risk score)',
            xaxis_tickangle=-45,
            height=500,
            yaxis_title='Median Bounce (%)'
        )
        
        st.plotly_chart(fig, width="stretch")
        
        # By Country
        if 'Cntry Terrtry Of Inc' in df.columns:
            st.markdown("### Bounce by Country (Top 15)")
            
            country_stats = df.groupby('Cntry Terrtry Of Inc').agg({
                'lifetime_hi_vs_ipo': ['median', 'mean', 'count']
            }).reset_index()
            country_stats.columns = ['Country', 'Median Bounce', 'Mean Bounce', 'Count']
            country_stats = country_stats[country_stats['Count'] >= 5].sort_values('Median Bounce', ascending=False).head(15)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=country_stats['Country'],
                y=country_stats['Median Bounce'],
                text=country_stats['Count'].apply(lambda x: f'n={x}'),
                textposition='outside'
            ))
            
            fig.update_layout(
                title='Median Bounce by Country of Incorporation',
                height=400,
                yaxis_title='Median Bounce (%)'
            )
            
            st.plotly_chart(fig, width="stretch")
        
        # By Price Bucket
        st.markdown("### Bounce by IPO Price Range")
        
        df['price_bucket'] = pd.cut(df['IPO Sh Px'], bins=[0, 4, 5, 8, 10, 50], 
                                     labels=['$0-4', '$4-5', '$5-8', '$8-10', '$10+'])
        
        price_stats = df.groupby('price_bucket').agg({
            'lifetime_hi_vs_ipo': ['median', 'count']
        }).reset_index()
        price_stats.columns = ['Price Range', 'Median Bounce', 'Count']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=price_stats['Price Range'].astype(str),
            y=price_stats['Median Bounce'],
            text=price_stats['Count'].apply(lambda x: f'n={x}'),
            textposition='outside',
            marker_color=['#ef5350', '#ff9800', '#ffeb3b', '#8bc34a', '#4caf50']
        ))
        
        fig.update_layout(
            title='Median Bounce by IPO Price Range',
            height=350,
            yaxis_title='Median Bounce (%)'
        )
        
        st.plotly_chart(fig, width="stretch")
        
        # Summary Table
        st.markdown("### Summary Statistics by Risk Level")
        
        df['risk_level'] = pd.cut(df['operation_risk_score'], 
                                   bins=[-1, 25, 40, 100],
                                   labels=['Low Risk', 'Medium Risk', 'High Risk'])
        
        risk_stats = df.groupby('risk_level').agg({
            'lifetime_hi_vs_ipo': ['median', 'mean', 'std', 'count'],
            'IPO Sh Px': 'mean',
            'IPO Sh Offered': 'mean'
        }).reset_index()
        
        risk_stats.columns = ['Risk Level', 'Median Bounce', 'Mean Bounce', 'Std Dev', 'Count', 'Avg Price', 'Avg Shares']
        
        for col in ['Median Bounce', 'Mean Bounce', 'Std Dev']:
            risk_stats[col] = risk_stats[col].apply(lambda x: f'{x:.0f}%')
        risk_stats['Avg Price'] = risk_stats['Avg Price'].apply(lambda x: f'${x:.2f}')
        risk_stats['Avg Shares'] = risk_stats['Avg Shares'].apply(lambda x: f'{x/1e6:.1f}M')
        
        st.dataframe(risk_stats, width="stretch", hide_index=True)


def about_page():
    """About page with data sources and methodology."""
    st.header("ℹ️ About This Dashboard")
    
    st.markdown("""
    ### Data Sources
    - **Historical IPO Data**: Bloomberg EQS + Polygon.io
    - **Upcoming IPOs**: IPOScoop, Nasdaq, NYSE, Yahoo Finance
    
    ### Risk Score Methodology
    | Factor | Points |
    |--------|--------|
    | IPO Price ≤ $5 | +15 |
    | Shares Offered ≤ 1.5M | +15 |
    | Tax Haven Country (KY, VG, etc.) | +15 |
    | Operation Underwriter | +10 |
    | No VC Backing | +10 |
    | Legit Underwriter (Goldman, etc.) | -10 |
    | US Company | -5 |
    
    ### Tax Haven Countries
    KY (Cayman), VG (BVI), MH (Marshall Islands), CY (Cyprus), MU (Mauritius), PA (Panama), JE (Jersey), GG (Guernsey)
    
    ### Operation Underwriters
    D Boral, Kingswood, US Tiger, Prime Number, Network 1, EF Hutton, Bancroft, Cathay, RVRS, Viewtrade, Joseph Stone, Boustead, Maxim, Dawson, Revere, Dominari, Craft Capital
    """)


def upcoming_ipos_page():
    """Page for upcoming IPOs with data from various sources."""
    st.header("🆕 Upcoming IPOs")
    
    st.markdown("""
    ### IPO Data Sources
    | Source | URL | Notes |
    |--------|-----|-------|
    | IPOScoop | [iposcoop.com](https://www.iposcoop.com/ipo-calendar/) | Best for small IPOs |
    | Nasdaq | [nasdaq.com/market-activity/ipos](https://www.nasdaq.com/market-activity/ipos) | Official Nasdaq listings |
    | NYSE | [nyse.com/ipo-center](https://www.nyse.com/ipo-center/filings) | Official NYSE listings |
    | Yahoo Finance | [finance.yahoo.com/calendar/ipo](https://finance.yahoo.com/calendar/ipo) | Good calendar view |
    | StockAnalysis | [stockanalysis.com/ipos/calendar](https://stockanalysis.com/ipos/calendar/) | Fast updates |
    """)
    
    st.markdown("---")
    
    # Manual refresh button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Try to fetch upcoming IPOs
    st.subheader("📅 This Week's IPOs")
    
    upcoming_df = fetch_upcoming_ipos()
    
    if upcoming_df is not None and len(upcoming_df) > 0:
        st.dataframe(upcoming_df, width="stretch", hide_index=True)
        
        # Alert section
        st.markdown("---")
        st.subheader("🔔 IPO Alerts")
        st.info("To receive alerts for new IPOs, you can:")
        st.markdown("""
        1. **Email alerts**: Set up with IPOScoop Pro
        2. **Bloomberg alerts**: Use `EVTS` function
        3. **Custom script**: Run `ipo_alert.py` (see below)
        """)
    else:
        st.warning("Could not fetch upcoming IPO data. Check the sources above manually.")
        
        # Show sample data structure
        st.markdown("### Expected Data Format")
        sample_data = pd.DataFrame({
            'Date': ['2025-12-10', '2025-12-11', '2025-12-12'],
            'Company': ['Sample Corp', 'Test Holdings', 'Demo Inc'],
            'Ticker': ['SMPL', 'TEST', 'DEMO'],
            'Price Range': ['$4-5', '$8-10', '$5-6'],
            'Shares (M)': [1.5, 2.0, 1.0],
            'Underwriter': ['D Boral', 'Maxim', 'EF Hutton'],
            'Exchange': ['NASDAQ', 'NASDAQ', 'NYSE']
        })
        st.dataframe(sample_data, width="stretch", hide_index=True)
    
    # Instructions for setting up alerts
    st.markdown("---")
    st.subheader("⚡ Setting Up Automated Alerts")
    
    with st.expander("Python Alert Script", expanded=False):
        st.code('''
# ipo_alert.py - Run every 30 min via Task Scheduler or cron
import requests
from bs4 import BeautifulSoup
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

def check_iposcoop():
    """Scrape IPOScoop for today's IPOs."""
    url = "https://www.iposcoop.com/ipo-calendar/"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    # Parse the calendar table
    # ... (implementation depends on their HTML structure)
    
    return today_ipos

def send_alert(ipos):
    """Send email/SMS alert for new IPOs."""
    # Configure your email settings
    msg = MIMEText(f"New IPOs today: {ipos}")
    msg['Subject'] = f"🚨 IPO Alert - {datetime.now().strftime('%Y-%m-%d')}"
    # ... send email
    
if __name__ == "__main__":
    ipos = check_iposcoop()
    if ipos:
        send_alert(ipos)
        print(f"Alert sent for: {ipos}")
''', language='python')
    
    with st.expander("PySide6 Desktop Alert Window", expanded=False):
        st.code('''
# ipo_alert_gui.py - Desktop notification window
from PySide6.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
from PySide6.QtCore import QTimer
import sys

class IPOAlertWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IPO Alerts")
        self.setGeometry(100, 100, 800, 400)
        
        # Create table
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Date", "Ticker", "Company", "Price", "Shares", "Underwriter", "Exchange"
        ])
        
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.table)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        # Auto-refresh every 30 min
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_data)
        self.timer.start(30 * 60 * 1000)  # 30 minutes
        
        self.refresh_data()
    
    def refresh_data(self):
        """Fetch and display upcoming IPOs."""
        # Fetch from IPOScoop, etc.
        # Update table rows
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IPOAlertWindow()
    window.show()
    sys.exit(app.exec())
''', language='python')


@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_upcoming_ipos():
    """Fetch upcoming IPOs from various sources - LIVE."""
    
    if not HAS_SCRAPING:
        st.warning("Install requests and beautifulsoup4 for live IPO data: pip install requests beautifulsoup4")
        return None
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }
    
    all_ipos = []
    
    # Source 1: Nasdaq API
    try:
        resp = requests.get(
            "https://api.nasdaq.com/api/ipo/calendar",
            headers={**HEADERS, 'Accept': 'application/json'},
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            
            # Upcoming IPOs
            if 'data' in data and 'upcoming' in data['data']:
                for ipo in data['data']['upcoming'].get('rows', []):
                    all_ipos.append({
                        'Date': ipo.get('expectedPriceDate', ''),
                        'Company': ipo.get('companyName', ''),
                        'Ticker': ipo.get('proposedTickerSymbol', ''),
                        'Price': ipo.get('proposedSharePrice', ''),
                        'Shares': ipo.get('sharesOffered', ''),
                        'Exchange': ipo.get('proposedExchange', ''),
                        'Status': 'Upcoming',
                        'Source': 'Nasdaq'
                    })
            
            # Priced (recent)
            if 'data' in data and 'priced' in data['data']:
                for ipo in data['data']['priced'].get('rows', [])[:10]:  # Last 10
                    all_ipos.append({
                        'Date': ipo.get('pricedDate', ''),
                        'Company': ipo.get('companyName', ''),
                        'Ticker': ipo.get('proposedTickerSymbol', ''),
                        'Price': ipo.get('proposedSharePrice', ''),
                        'Shares': ipo.get('sharesOffered', ''),
                        'Exchange': ipo.get('proposedExchange', ''),
                        'Status': 'Priced',
                        'Source': 'Nasdaq'
                    })
    except Exception as e:
        st.warning(f"Nasdaq API error: {e}")
    
    # Source 2: IPOScoop (scrape)
    # Columns: Company(0) | Symbol(1) | Lead Managers(2) | Shares(3) | Price Low(4) | Price High(5) | Est $ Vol(6) | Expected to Trade(7)
    try:
        resp = requests.get(
            "https://www.iposcoop.com/ipo-calendar/",
            headers=HEADERS,
            timeout=10
        )
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            table = soup.find('table')
            
            if table:
                for tr in table.find_all('tr')[1:25]:  # Skip header
                    cells = tr.find_all(['td', 'th'])
                    if len(cells) >= 8:
                        company = cells[0].get_text(strip=True)
                        ticker = cells[1].get_text(strip=True)
                        underwriter = cells[2].get_text(strip=True)
                        shares = cells[3].get_text(strip=True)
                        price_low = cells[4].get_text(strip=True)
                        price_high = cells[5].get_text(strip=True)
                        date = cells[7].get_text(strip=True) if len(cells) > 7 else 'TBD'
                        
                        # Combine price range
                        if price_low and price_high and price_low != price_high:
                            price = f"${price_low}-${price_high}"
                        elif price_low:
                            price = f"${price_low}"
                        else:
                            price = ''
                        
                        # Skip header rows
                        if not ticker or ticker.upper() in ['SYMBOL', 'TICKER', 'SYMBOL PROPOSED']:
                            continue
                        
                        # Skip duplicates from Nasdaq
                        if not any(ipo['Ticker'] == ticker for ipo in all_ipos):
                            all_ipos.append({
                                'Date': date,
                                'Company': company,
                                'Ticker': ticker,
                                'Price': price,
                                'Shares': shares,
                                'Underwriter': underwriter,
                                'Exchange': '',
                                'Status': 'Upcoming',
                                'Source': 'IPOScoop'
                            })
    except Exception as e:
        st.warning(f"IPOScoop error: {e}")
    
    if all_ipos:
        return pd.DataFrame(all_ipos)
    return None


def analysis_page():
    """Main analysis page with underwriter filtering and charts."""
    st.markdown("*Explore small IPOs by underwriter with multi-timeframe charts*")
    
    # Data source for chart bars (Polygon for real-time data)
    data_source = "polygon"
    
    # Initialize session state
    if 'detailed_ipo' not in st.session_state:
        st.session_state.detailed_ipo = None
    if 'polygon_api_key' not in st.session_state:
        st.session_state.polygon_api_key = get_polygon_api_key()
    
    # ========================================================================
    # SIDEBAR - Define before data loading (Streamlit renders in order)
    # ========================================================================
    st.sidebar.header("🔍 Filters")
    st.sidebar.caption(f"v{DASHBOARD_VERSION}")
    
    # ========================================================================
    # DATA SOURCE - ClickHouse or CSV fallback
    # ========================================================================
    st.sidebar.subheader("🗄️ Data Source")
    
    use_clickhouse = False
    ch_password = None
    
    if not HAS_CLICKHOUSE:
        st.sidebar.warning("Install: `pip install clickhouse-connect`")
        st.sidebar.caption("📁 Using CSV fallback")
    else:
        config = load_config()
        saved_ch_password = config.get('clickhouse_password', '') or os.environ.get('CLICKHOUSE_PASSWORD', '')
        
        # Default to ClickHouse if password is available
        use_clickhouse = st.sidebar.checkbox(
            "Use ClickHouse",
            value=bool(saved_ch_password),
            help="Load IPO data from ClickHouse Cloud"
        )
        
        if use_clickhouse:
            if saved_ch_password:
                ch_password = saved_ch_password
                
                # Password management in expander
                with st.sidebar.expander("⚙️ ClickHouse Settings"):
                    st.caption(f"Host: {CLICKHOUSE_CONFIG['host'][:30]}...")
                    
                    new_password = st.text_input(
                        "Password",
                        value=saved_ch_password,
                        type="password",
                        key="ch_password_input"
                    )
                    if new_password != saved_ch_password:
                        config['clickhouse_password'] = new_password
                        save_config(config)
                        st.success("✅ Password updated!")
                        st.rerun()
                    
                    if st.button("🗑️ Clear Password", key="clear_ch_btn"):
                        config.pop('clickhouse_password', None)
                        save_config(config)
                        st.rerun()
                
                # Reload button
                if st.sidebar.button("🔄 Reload Data", key="reload_ch_btn"):
                    st.cache_data.clear()
                    st.rerun()
            else:
                st.sidebar.info("Enter ClickHouse password")
                ch_password = st.sidebar.text_input(
                    "Password",
                    value='',
                    type="password",
                    key="ch_password_new"
                )
                if ch_password:
                    config['clickhouse_password'] = ch_password
                    save_config(config)
                    st.sidebar.success("✅ Password saved!")
                    st.rerun()
        else:
            st.sidebar.caption("📁 Using CSV file")
    
    # Polygon API key section
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔑 Your API Key")
    st.sidebar.caption("Each user saves their own key in browser")
    
    saved_key = get_polygon_api_key()
    
    if saved_key:
        st.sidebar.success("✅ Your Polygon key is saved")
        st.sidebar.caption(f"Key: {saved_key[:6]}...{saved_key[-4:]}" if len(saved_key) > 10 else "Key saved")
        
        show_key = st.sidebar.checkbox("Show/Edit Key", value=False, key="show_edit_key")
        if show_key:
            new_key = st.sidebar.text_input(
                "Polygon API Key", 
                value=saved_key,
                type="password",
                key="edit_polygon_key",
                help="Get a free key at polygon.io"
            )
            if new_key and new_key != saved_key:
                save_polygon_api_key(new_key)
                st.sidebar.success("✅ Key updated!")
                st.rerun()
        
        if st.sidebar.button("🗑️ Clear My Saved Key", key="clear_key_btn"):
            clear_polygon_api_key()
            st.rerun()
    else:
        st.sidebar.info("💡 Get a free API key at [polygon.io](https://polygon.io)")
        polygon_key = st.sidebar.text_input(
            "Polygon API Key", 
            value='',
            type="password",
            key="new_polygon_key",
            help="Your key is saved in YOUR browser only - not shared"
        )
        if polygon_key:
            save_polygon_api_key(polygon_key)
            st.sidebar.success("✅ Key saved to your browser!")
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # ========================================================================
    # LOAD DATA (after sidebar settings are defined)
    # ========================================================================
    st.session_state['use_clickhouse'] = use_clickhouse
    st.session_state['ch_password'] = ch_password
    
    # Create cache key based on data source settings
    cache_key = f"ch_{use_clickhouse}_{bool(ch_password)}"
    df = load_ipo_data(use_clickhouse=use_clickhouse, ch_password=ch_password, _cache_key=cache_key)
    
    if df.empty:
        st.warning("No data loaded. Check your data source configuration.")
        st.stop()
    
    # ========================================================================
    # UNDERWRITER SELECTION - Extract individual underwriters from combos
    # ========================================================================
    
    # Extract all unique individual underwriters from combo strings
    def extract_underwriters(uw_series):
        """Extract individual underwriters from comma-separated strings."""
        all_uws = set()
        for uw in uw_series.dropna().unique():
            # Split by comma and clean
            parts = re.split(r'[,/]', str(uw))
            for p in parts:
                cleaned = p.strip().upper()
                if cleaned and len(cleaned) > 2:
                    all_uws.add(cleaned)
        return sorted(all_uws)
    
    # Get underwriter column
    uw_col = 'underwriter' if 'underwriter' in df.columns else 'IPO Lead' if 'IPO Lead' in df.columns else None
    
    if uw_col:
        all_underwriters = extract_underwriters(df[uw_col])
    else:
        all_underwriters = []
    
    # Count IPOs per underwriter
    uw_counts = {}
    if uw_col:
        for _, row in df.iterrows():
            uw = str(row.get(uw_col, ''))
            for part in re.split(r'[,/]', uw):
                cleaned = part.strip().upper()
                if cleaned and len(cleaned) > 2:
                    uw_counts[cleaned] = uw_counts.get(cleaned, 0) + 1
    
    # Create selection with counts
    uw_options = ["All Underwriters"] + [f"{uw} ({uw_counts.get(uw, 0)})" for uw in all_underwriters]
    
    selected_uw_label = st.sidebar.selectbox(
        "Filter by Underwriter",
        options=uw_options,
        index=0,
        key="uw_filter"
    )
    
    # Extract just the underwriter name
    if selected_uw_label == "All Underwriters":
        selected_uw = None
    else:
        selected_uw = selected_uw_label.split(" (")[0]
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔎 Ticker Search")
    
    # Get all tickers
    ticker_col = 'ticker_clean' if 'ticker_clean' in df.columns else 'polygon_ticker' if 'polygon_ticker' in df.columns else None
    if ticker_col:
        all_tickers = sorted(df[ticker_col].dropna().unique().tolist())
    else:
        all_tickers = []
    
    st.sidebar.caption(f"Total tickers in data: {len(all_tickers)}")
    
    # Search input
    ticker_search = st.sidebar.text_input(
        "Search Ticker",
        value="",
        placeholder="e.g., WTF, RDDT",
        key="ticker_search"
    )
    
    # Filter tickers based on search
    if ticker_search:
        search_upper = ticker_search.upper().strip()
        filtered_tickers = [t for t in all_tickers if search_upper in str(t).upper()]
        if filtered_tickers:
            st.sidebar.caption(f"Found {len(filtered_tickers)} match(es)")
            # Auto-select first match if exact match exists
            exact_match = [t for t in filtered_tickers if str(t).upper() == search_upper]
            default_idx = 1 if exact_match else 0  # Select first match, or "All Matching"
        else:
            st.sidebar.warning(f"'{ticker_search}' not found")
            filtered_tickers = []
            default_idx = 0
    else:
        filtered_tickers = all_tickers
        default_idx = 0
    
    # Build dropdown options
    if ticker_search and filtered_tickers:
        ticker_dropdown_options = ["All Matching"] + filtered_tickers[:100]
    else:
        ticker_dropdown_options = ["All Tickers"] + filtered_tickers[:100]
    
    selected_ticker_filter = st.sidebar.selectbox(
        "Select Ticker",
        options=ticker_dropdown_options,
        index=min(default_idx, len(ticker_dropdown_options) - 1),
        key="ticker_select"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Risk Filters")
    
    # Risk score filter
    min_risk = st.sidebar.slider("Min Risk Score", 0, 100, 0, key="min_risk")
    max_risk = st.sidebar.slider("Max Risk Score", 0, 100, 100, key="max_risk")
    
    # Tax haven filter
    tax_haven_filter = st.sidebar.selectbox(
        "Tax Haven",
        options=["All", "Tax Haven Only", "Non-Tax Haven Only"],
        index=0,
        key="tax_haven_filter"
    )
    
    # VC backed filter
    vc_filter = st.sidebar.selectbox(
        "VC Backed",
        options=["All", "VC Backed Only", "Non-VC Only"],
        index=0,
        key="vc_filter"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Bounce Calculation")
    
    # Bounce calculation method toggle
    bounce_method = st.sidebar.radio(
        "Calculate bounce from:",
        options=["D1 Open (Day Trading)", "IPO Price (Investment)"],
        index=0,
        key="bounce_method",
        help="D1 Open: How much it ran from first trade. IPO Price: Total return from offering price."
    )
    
    # Set the active bounce column based on selection
    if bounce_method == "IPO Price (Investment)":
        bounce_col = 'bounce_vs_ipo'
        st.sidebar.caption("📊 Bounce = (Lifetime High / IPO Price - 1) × 100")
    else:
        bounce_col = 'bounce_vs_d1open'
        st.sidebar.caption("📊 Bounce = (Lifetime High / D1 Open - 1) × 100")
    
    # Store in session state for use throughout
    st.session_state['bounce_col'] = bounce_col
    
    # Update lifetime_hi_vs_ipo to use selected method
    if bounce_col in df.columns:
        df['lifetime_hi_vs_ipo'] = df[bounce_col]
    
    st.sidebar.markdown("---")
    
    # ========================================================================
    # DETAILED VIEW POPUP - SHOW AT TOP IF SELECTED
    # ========================================================================
    if st.session_state.detailed_ipo:
        ipo = st.session_state.detailed_ipo
        
        # Very prominent container
        st.markdown("""
        <style>
        .stAlert {border: 3px solid #1f77b4 !important; background-color: #e6f3ff !important;}
        </style>
        """, unsafe_allow_html=True)
        
        st.warning(f"📊 **DETAILED VIEW OPEN** - Scroll down to see chart, or click Close when done")
        
        with st.container():
            st.subheader(f"📊 {ipo['ticker']} - {ipo['name'][:40]}")
            
            # Close button at top - very prominent
            close_col1, close_col2, close_col3 = st.columns([1, 2, 3])
            with close_col1:
                if st.button("❌ CLOSE", key="close_detail_top", type="primary"):
                    st.session_state.detailed_ipo = None
                    st.rerun()
            with close_col2:
                st.markdown(f"**IPO:** {ipo['date'].strftime('%Y-%m-%d')} | **Price:** ${ipo['price']:.2f}")
            with close_col3:
                st.link_button(f"📈 View {ipo['ticker']} on Yahoo Finance", 
                              f"https://finance.yahoo.com/quote/{ipo['ticker']}")
            
            st.markdown("---")
            
            # Data source selection
            st.markdown("#### Select Data Source & Timeframe")
            src_col1, src_col2, src_col3 = st.columns(3)
            
            with src_col1:
                detail_source = st.radio(
                    "Data Source:",
                    ["Sample (Demo)", "Polygon API (Live)"],
                    key="detail_source_top",
                    horizontal=True
                )
            
            with src_col2:
                if "Polygon" in detail_source:
                    saved_key = get_polygon_api_key()
                    if saved_key:
                        st.success("✅ Using your saved Polygon key")
                    else:
                        st.error("⚠️ Add Polygon API key in sidebar first!")
            
            with src_col3:
                timeframe = st.selectbox(
                    "Timeframe:",
                    ["1 Minute", "5 Minute", "15 Minute", "1 Hour", "1 Day"],
                    key="detail_timeframe_top"
                )
            
            # Date range
            date_col1, date_col2, date_col3 = st.columns(3)
            with date_col1:
                start_date = st.date_input(
                    "Start Date",
                    value=ipo['date'].date(),
                    key="detail_start_top"
                )
            with date_col2:
                default_end = min(ipo['date'] + timedelta(days=30), datetime.now())
                end_date = st.date_input(
                    "End Date", 
                    value=default_end.date(),
                    key="detail_end_top"
                )
            with date_col3:
                load_btn = st.button("🔄 LOAD CHART", key="load_chart_top", type="primary")
            
            # Load and display chart
            if load_btn:
                with st.spinner("Fetching data..."):
                    tf_map = {
                        "1 Minute": ("minute", 1),
                        "5 Minute": ("minute", 5),
                        "15 Minute": ("minute", 15),
                        "1 Hour": ("hour", 1),
                        "1 Day": ("day", 1)
                    }
                    tf, mult = tf_map.get(timeframe, ("day", 1))
                    
                    polygon_key = get_polygon_api_key()
                    
                    if "Polygon" in detail_source and polygon_key:
                        # Fetch from Polygon
                        try:
                            url = f"https://api.polygon.io/v2/aggs/ticker/{ipo['ticker']}/range/{mult}/{tf}/{start_date}/{end_date}"
                            params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": polygon_key}
                            resp = requests.get(url, params=params, timeout=30)
                            
                            if resp.status_code == 200:
                                data = resp.json()
                                if 'results' in data and data['results']:
                                    bars = pd.DataFrame(data['results'])
                                    # Convert UTC to EST
                                    bars['timestamp'] = pd.to_datetime(bars['t'], unit='ms', utc=True)
                                    bars['timestamp'] = bars['timestamp'].dt.tz_convert('America/New_York').dt.tz_localize(None)
                                    bars = bars.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
                                    bars = bars[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                                    # Filter to regular trading hours (9:30 AM - 4:00 PM EST) for minute data
                                    if tf == 'minute' and len(bars) > 0:
                                        bars['hour'] = bars['timestamp'].dt.hour
                                        bars['minute'] = bars['timestamp'].dt.minute
                                        bars['time_decimal'] = bars['hour'] + bars['minute'] / 60
                                        bars = bars[(bars['time_decimal'] >= 9.5) & (bars['time_decimal'] <= 20.0)]
                                        bars = bars.drop(columns=['hour', 'minute', 'time_decimal'])
                                else:
                                    st.error("No data returned from Polygon")
                                    bars = pd.DataFrame()
                            else:
                                st.error(f"Polygon API error: {resp.status_code}")
                                bars = pd.DataFrame()
                        except Exception as e:
                            st.error(f"Error: {e}")
                            bars = pd.DataFrame()
                    else:
                        # Generate sample data
                        bars = fetch_bars(ipo['ticker'], str(start_date), str(end_date), tf, ipo['price'], "sample")
                    
                    if bars is not None and not bars.empty:
                        # Create candlestick chart
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                           vertical_spacing=0.03, row_heights=[0.7, 0.3])
                        
                        fig.add_trace(go.Candlestick(
                            x=bars['timestamp'], open=bars['open'], high=bars['high'],
                            low=bars['low'], close=bars['close'], name='Price',
                            increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
                        ), row=1, col=1)
                        
                        # IPO price line
                        fig.add_hline(y=ipo['price'], line_dash="dash", line_color="blue",
                                     annotation_text=f"IPO ${ipo['price']:.2f}", row=1, col=1)
                        
                        # Volume
                        colors = ['#26a69a' if bars['close'].iloc[i] >= bars['open'].iloc[i] else '#ef5350' 
                                  for i in range(len(bars))]
                        fig.add_trace(go.Bar(x=bars['timestamp'], y=bars['volume'], 
                                            marker_color=colors, opacity=0.7, name='Volume'), row=2, col=1)
                        
                        # Stats
                        max_price = bars['high'].max()
                        min_price = bars['low'].min()
                        pct_from_ipo = (max_price / ipo['price'] - 1) * 100
                        
                        # Set x-axis range for minute data (9:30 AM - 4:00 PM)
                        x_range = None
                        if tf == 'minute' and len(bars) > 0:
                            chart_date = bars['timestamp'].iloc[0].date()
                            x_start = pd.Timestamp(chart_date).replace(hour=9, minute=30)
                            x_end = pd.Timestamp(chart_date).replace(hour=20, minute=0)
                            x_range = [x_start, x_end]
                        
                        fig.update_layout(
                            title=f"{ipo['ticker']} - {timeframe} | High: ${max_price:.2f} (+{pct_from_ipo:.0f}%) | Low: ${min_price:.2f}",
                            height=500, showlegend=False, template='plotly_white',
                            xaxis=dict(range=x_range) if x_range else {},
                            xaxis2=dict(rangeslider=dict(visible=True, thickness=0.05))
                        )
                        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                        fig.update_yaxes(title_text="Volume", row=2, col=1)
                        
                        st.plotly_chart(fig, width="stretch")
                        
                        # Stats row
                        st_col1, st_col2, st_col3, st_col4 = st.columns(4)
                        with st_col1:
                            st.metric("Data Points", len(bars))
                        with st_col2:
                            st.metric("High", f"${max_price:.2f}")
                        with st_col3:
                            st.metric("Low", f"${min_price:.2f}")
                        with st_col4:
                            pct = (bars['close'].iloc[-1] / bars['open'].iloc[0] - 1) * 100
                            st.metric("Period Change", f"{pct:+.1f}%")
                    else:
                        st.error("No data available. Try different dates or check API key.")
            
            # Another close button at bottom
            st.markdown("---")
            if st.button("❌ Close Detailed View", key="close_detail_bottom"):
                st.session_state.detailed_ipo = None
                st.rerun()
        
        st.markdown("---")
        st.markdown("---")
    
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
            
            # Convert to boolean for filtering (column contains 0/1 integers)
            th_df = df[df['is_tax_haven'] == 1]
            if len(th_df) > 0:
                bounce_data.append({
                    'Type': '🏝️ Tax Haven',
                    'Count': len(th_df),
                    'Median Bounce': f"{th_df['lifetime_hi_vs_ipo'].median():.0f}%",
                    'Bounce >100%': f"{(th_df['lifetime_hi_vs_ipo'] > 100).mean()*100:.0f}%",
                    'Avg Risk': f"{th_df['operation_risk_score'].mean():.0f}" if 'operation_risk_score' in th_df.columns else "N/A"
                })
            
            if 'is_us' in df.columns:
                us_df = df[df['is_us'] == 1]
                if len(us_df) > 0:
                    bounce_data.append({
                        'Type': '🇺🇸 US',
                        'Count': len(us_df),
                        'Median Bounce': f"{us_df['lifetime_hi_vs_ipo'].median():.0f}%",
                        'Bounce >100%': f"{(us_df['lifetime_hi_vs_ipo'] > 100).mean()*100:.0f}%",
                        'Avg Risk': f"{us_df['operation_risk_score'].mean():.0f}" if 'operation_risk_score' in us_df.columns else "N/A"
                    })
                
                other_df = df[(df['is_tax_haven'] == 0) & (df['is_us'] == 0)]
                if len(other_df) > 0:
                    bounce_data.append({
                        'Type': '🌍 Other',
                        'Count': len(other_df),
                        'Median Bounce': f"{other_df['lifetime_hi_vs_ipo'].median():.0f}%",
                        'Bounce >100%': f"{(other_df['lifetime_hi_vs_ipo'] > 100).mean()*100:.0f}%",
                        'Avg Risk': f"{other_df['operation_risk_score'].mean():.0f}" if 'operation_risk_score' in other_df.columns else "N/A"
                    })
            
            if bounce_data:
                st.dataframe(pd.DataFrame(bounce_data), width="stretch", hide_index=True)
    
    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    
    # Build filter mask
    mask = pd.Series([True] * len(df), index=df.index)
    
    # Apply risk score filter
    if 'operation_risk_score' in df.columns:
        mask &= (df['operation_risk_score'] >= min_risk) & (df['operation_risk_score'] <= max_risk)
    
    # Apply underwriter filter (partial match)
    if selected_uw is not None and uw_col:
        mask &= df[uw_col].str.contains(selected_uw, case=False, na=False, regex=False)
    
    # Apply ticker filter
    if ticker_search:
        search_upper = ticker_search.upper().strip()
        if selected_ticker_filter == "All Matching":
            # Filter to all matching tickers
            mask &= df[ticker_col].str.upper().str.contains(search_upper, na=False)
        elif selected_ticker_filter not in ["All Tickers", "All Matching", "No matches found"]:
            # Specific ticker selected
            mask &= df[ticker_col] == selected_ticker_filter
        else:
            # Search text entered but "All Tickers" still selected - apply search anyway
            mask &= df[ticker_col].str.upper().str.contains(search_upper, na=False)
    elif selected_ticker_filter not in ["All Tickers", "All Matching", "No matches found"] and ticker_col:
        mask &= df[ticker_col] == selected_ticker_filter
    
    # Apply tax haven filter
    if tax_haven_filter == "Tax Haven Only" and 'is_tax_haven' in df.columns:
        mask &= df['is_tax_haven'] == 1
    elif tax_haven_filter == "Non-Tax Haven Only" and 'is_tax_haven' in df.columns:
        mask &= df['is_tax_haven'] == 0
    
    # Apply VC backed filter
    if vc_filter == "VC Backed Only" and 'vc_backed' in df.columns:
        mask &= df['vc_backed'] == 1
    elif vc_filter == "Non-VC Only" and 'vc_backed' in df.columns:
        mask &= df['vc_backed'] == 0
    
    filtered_df = df[mask].sort_values('date', ascending=False)
    
    # Header
    header_text = f"📊 {selected_uw}" if selected_uw else "📊 All Underwriters"
    st.header(header_text)
    
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
    base_cols = ['ticker_clean', 'Name', 'date', 'IPO Sh Px', 'underwriter',
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
        'underwriter': 'Underwriter',
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
    if 'Underwriter' in display_df.columns:
        display_df['Underwriter'] = display_df['Underwriter'].apply(lambda x: str(x)[:20] if pd.notna(x) and str(x).strip() else "Unknown")
    
    st.dataframe(display_df, width="stretch", hide_index=True)
    
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
    
    # Get underwriter
    underwriter = ipo_info.get('underwriter', '')
    if pd.isna(underwriter) or str(underwriter).strip() == '':
        underwriter = ipo_info.get('IPO Lead', 'Unknown')
    if pd.isna(underwriter) or str(underwriter).strip() == '':
        underwriter = 'Unknown'
    
    # Check if operation underwriter
    uw_upper = str(underwriter).upper()
    is_operation_uw = any(op in uw_upper for op in ['CATHAY', 'D BORAL', 'KINGSWOOD', 'US TIGER', 'PRIME NUMBER', 
                                                      'NETWORK 1', 'EF HUTTON', 'BANCROFT', 'RVRS', 'VIEWTRADE',
                                                      'JOSEPH STONE', 'BOUSTEAD', 'MAXIM', 'DAWSON', 'REVERE',
                                                      'DOMINARI', 'CRAFT CAPITAL', 'THINKEQUITY', 'AEGIS'])
    is_legit_uw = any(leg in uw_upper for leg in ['GOLDMAN', 'MORGAN STANLEY', 'JPMORGAN', 'JP MORGAN', 'CITI',
                                                    'BOFA', 'JEFFERIES', 'CREDIT SUISSE', 'UBS', 'BARCLAYS'])
    
    # Format underwriter with color
    if is_operation_uw:
        uw_display = f"⚠️ **{underwriter}** (Operation UW)"
    elif is_legit_uw:
        uw_display = f"✅ **{underwriter}** (Legit)"
    else:
        uw_display = f"**{underwriter}**"
    
    st.markdown(f"""
    **{selected_ticker}** - {str(ipo_info['Name'])[:50]}
    - 📅 IPO Date: {ipo_date.strftime('%Y-%m-%d')}
    - 💵 IPO Price: ${ipo_price:.2f}
    - 🏦 Underwriter: {uw_display}
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
        st.plotly_chart(fig1, width="stretch")
        
        # Chart 2: First Month - Daily Bars
        st.markdown("### 📅 First Month (Daily Bars)")
        
        m1_start = ipo_date.strftime('%Y-%m-%d')
        m1_end = (ipo_date + timedelta(days=30)).strftime('%Y-%m-%d')
        
        m1_bars = fetch_bars(selected_ticker, m1_start, m1_end,
                             timeframe="day", ipo_price=ipo_price,
                             data_source=data_source)
        
        fig2 = create_candlestick_chart(m1_bars, f"{selected_ticker} - First Month (Daily)", height=450)
        st.plotly_chart(fig2, width="stretch")
        
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
        st.plotly_chart(fig3, width="stretch")
    
    # ========================================================================
    # DETAILED CHART HELPER FUNCTIONS
    # ========================================================================
    
    def show_detailed_chart(ticker: str, ipo_date: datetime, ipo_price: float, name: str):
        """Set session state to show detailed chart."""
        st.session_state.detailed_ipo = {
            'ticker': ticker,
            'date': ipo_date,
            'price': ipo_price,
            'name': name
        }
    
    def close_detailed_view():
        st.session_state.detailed_ipo = None
    
    # Helper function for Polygon API
    def fetch_polygon_bars_detailed(ticker: str, start: str, end: str, 
                                     timeframe: str, multiplier: int, api_key: str) -> pd.DataFrame:
        """Fetch detailed bars from Polygon.io API."""
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timeframe}/{start}/{end}"
            params = {
                "adjusted": "true",
                "sort": "asc",
                "limit": 50000,
                "apiKey": api_key
            }
            
            resp = requests.get(url, params=params, timeout=30)
            
            if resp.status_code == 200:
                data = resp.json()
                if 'results' in data and data['results']:
                    df = pd.DataFrame(data['results'])
                    # Convert UTC to EST
                    df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True)
                    df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York').dt.tz_localize(None)
                    df = df.rename(columns={
                        'o': 'open', 'h': 'high', 'l': 'low', 
                        'c': 'close', 'v': 'volume'
                    })
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    # Filter to regular trading hours (9:30 AM - 4:00 PM EST) for minute data
                    if timeframe == 'minute' and len(df) > 0:
                        df['hour'] = df['timestamp'].dt.hour
                        df['minute'] = df['timestamp'].dt.minute
                        df['time_decimal'] = df['hour'] + df['minute'] / 60
                        df = df[(df['time_decimal'] >= 9.5) & (df['time_decimal'] <= 20.0)]
                        df = df.drop(columns=['hour', 'minute', 'time_decimal'])
                    return df
            else:
                st.warning(f"Polygon API error: {resp.status_code} - {resp.text[:100]}")
                
        except Exception as e:
            st.error(f"Error fetching from Polygon: {e}")
        
        return pd.DataFrame()
    
    # Detailed chart function with more features
    def create_detailed_chart(df: pd.DataFrame, title: str, ipo_price: float) -> go.Figure:
        """Create a detailed candlestick chart with volume and IPO price line."""
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)
            return fig
        
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.03, row_heights=[0.7, 0.3]
        )
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'], high=df['high'],
            low=df['low'], close=df['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ), row=1, col=1)
        
        # IPO Price line
        fig.add_hline(
            y=ipo_price, line_dash="dash", line_color="blue",
            annotation_text=f"IPO ${ipo_price:.2f}",
            row=1, col=1
        )
        
        # Volume bars
        colors = ['#26a69a' if row['close'] >= row['open'] else '#ef5350' 
                  for _, row in df.iterrows()]
        fig.add_trace(go.Bar(
            x=df['timestamp'], y=df['volume'],
            name='Volume', marker_color=colors, opacity=0.7
        ), row=2, col=1)
        
        # Calculate stats for title
        max_price = df['high'].max()
        min_price = df['low'].min()
        pct_from_ipo = (max_price / ipo_price - 1) * 100
        
        fig.update_layout(
            title=f"{title}<br><sup>High: ${max_price:.2f} (+{pct_from_ipo:.0f}% from IPO) | Low: ${min_price:.2f}</sup>",
            height=600,
            showlegend=False,
            template='plotly_white',
            xaxis2=dict(
                rangeslider=dict(visible=True, thickness=0.05),
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1H", step="hour", stepmode="backward"),
                        dict(count=1, label="1D", step="day", stepmode="backward"),
                        dict(count=7, label="1W", step="day", stepmode="backward"),
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                )
            )
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    # ========================================================================
    # CHART GRID WITH CLICKABLE BUTTONS
    # ========================================================================
    st.markdown("---")
    st.subheader("🔄 Chart Grid - All IPOs (Day 1 | Month | Lifetime)")
    
    # Quick select dropdown for detailed view
    st.markdown("##### 🔍 Quick View - Select IPO for Detailed Chart")
    
    if len(filtered_df) > 0:
        # Create options for dropdown
        quick_select_options = ["-- Select an IPO --"] + [
            f"{row['ticker_clean']} - {str(row.get('Name', ''))[:25]} | {row['date'].strftime('%Y-%m-%d')} | ${row['IPO Sh Px']:.0f}" 
            if pd.notna(row.get('IPO Sh Px')) else 
            f"{row['ticker_clean']} - {str(row.get('Name', ''))[:25]} | {row['date'].strftime('%Y-%m-%d')}"
            for _, row in filtered_df.head(100).iterrows()
        ]
        
        quick_col1, quick_col2 = st.columns([3, 1])
        with quick_col1:
            selected_quick = st.selectbox(
                "Choose IPO to view detailed chart:",
                options=quick_select_options,
                index=0,
                key="quick_select_ipo",
                label_visibility="collapsed"
            )
        
        with quick_col2:
            if selected_quick != "-- Select an IPO --":
                # Extract ticker from selection
                quick_ticker = selected_quick.split(" - ")[0]
                quick_row = filtered_df[filtered_df['ticker_clean'] == quick_ticker].iloc[0]
                
                if st.button("📊 Open Detailed View", key="quick_open_btn", type="primary"):
                    show_detailed_chart(
                        quick_ticker,
                        quick_row['date'],
                        quick_row['IPO Sh Px'] if pd.notna(quick_row['IPO Sh Px']) else 4.0,
                        str(quick_row.get('Name', ''))[:40]
                    )
                    st.rerun()
    
    st.markdown("---")
    
    # Limit for performance
    max_ipos = st.slider("Max IPOs to show", 5, 50, 15, help="More IPOs = slower loading")
    
    if len(filtered_df) == 0:
        st.warning("No IPOs match the current filters.")
    else:
        st.info(f"Showing {min(len(filtered_df), max_ipos)} of {len(filtered_df)} IPOs | 💡 **Click 📊 Expand button** on any IPO to see detailed chart with live Polygon data")
        
        # Create a compact mini-chart function
        def create_mini_chart(df: pd.DataFrame, title: str, height: int = 200) -> go.Figure:
            """Create a compact line chart for grid view with visible date axis."""
            if df.empty or len(df) < 2:
                fig = go.Figure()
                fig.add_annotation(text="No data", xref="paper", yref="paper",
                                  x=0.5, y=0.5, showarrow=False, font_size=10)
                fig.update_layout(height=height, margin=dict(l=5, r=5, t=25, b=25))
                return fig
            
            # Calculate color based on performance
            start_price = df['close'].iloc[0]
            end_price = df['close'].iloc[-1]
            color = '#26a69a' if end_price >= start_price else '#ef5350'
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['close'],
                mode='lines',
                line=dict(color=color, width=1.5),
                fill='tozeroy',
                fillcolor=f'rgba{tuple(list(bytes.fromhex(color[1:])) + [0.1])}'
            ))
            
            # Add high/low annotations
            max_price = df['high'].max()
            min_price = df['low'].min()
            pct_change = (end_price / start_price - 1) * 100
            
            # Determine date format based on title/timeframe
            if '1min' in title.lower() or 'minute' in title.lower():
                tick_format = '%H:%M'
                dtick = None  # Auto
                # Set x-axis range from 9:30 AM to 4:00 PM
                chart_date = df['timestamp'].iloc[0].date()
                x_start = pd.Timestamp(chart_date).replace(hour=9, minute=30)
                x_end = pd.Timestamp(chart_date).replace(hour=20, minute=0)
            elif 'month' in title.lower():
                tick_format = '%m/%d'
                dtick = 'D7'  # Weekly ticks
                x_start = None
                x_end = None
            else:  # Lifetime
                tick_format = '%Y-%m'
                dtick = 'M3'  # Quarterly ticks
                x_start = None
                x_end = None
            
            fig.update_layout(
                title=dict(text=f"{title} ({pct_change:+.0f}%)", font_size=10, x=0.5),
                height=height,
                margin=dict(l=40, r=10, t=30, b=40),
                xaxis=dict(
                    showticklabels=True, 
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.2)',
                    tickformat=tick_format,
                    tickfont=dict(size=8),
                    tickangle=-45,
                    nticks=5,  # Limit number of ticks
                    range=[x_start, x_end] if x_start else None,
                ),
                yaxis=dict(
                    showticklabels=True, 
                    showgrid=True, 
                    gridcolor='rgba(128,128,128,0.2)',
                    tickfont=dict(size=8),
                    tickprefix='$',
                ),
                showlegend=False,
                template='plotly_white'
            )
            
            return fig
        
        # Display grid - each IPO gets a row with 3 charts + detail button
        for i, (_, row) in enumerate(filtered_df.head(max_ipos).iterrows()):
            ticker = row['ticker_clean']
            ipo_dt = row['date']
            price = row['IPO Sh Px'] if pd.notna(row['IPO Sh Px']) else 4.0
            bounce = row['lifetime_hi_vs_ipo'] if pd.notna(row['lifetime_hi_vs_ipo']) else 0
            bounce_d1 = row.get('bounce_vs_d1open', bounce) if pd.notna(row.get('bounce_vs_d1open', 0)) else 0
            bounce_ipo = row.get('bounce_vs_ipo', bounce) if pd.notna(row.get('bounce_vs_ipo', 0)) else 0
            name = str(row.get('Name', ''))[:40]
            
            # Row container with border
            with st.container():
                # Row header with detail button
                header_col1, header_col2, header_col3 = st.columns([4, 1, 1])
                with header_col1:
                    st.markdown(f"**{i+1}. {ticker}** - {name}")
                    st.caption(f"IPO: {ipo_dt.strftime('%Y-%m-%d')} | ${price:.2f} | D1: {bounce_d1:.0f}% | IPO: {bounce_ipo:.0f}%")
                with header_col2:
                    if st.button("📊 Expand", key=f"detail_btn_{ticker}_{i}", 
                                help="Click to view detailed chart with Polygon data",
                                type="primary", width="stretch"):
                        show_detailed_chart(ticker, ipo_dt, price, name)
                        st.rerun()
                with header_col3:
                    # Link to external chart
                    st.link_button("📈 Yahoo", f"https://finance.yahoo.com/quote/{ticker}", 
                                   width="stretch")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Day 1 - 1 minute (or daily if minute not available)
                d1_start = ipo_dt.strftime('%Y-%m-%d')
                d1_bars = fetch_bars(ticker, d1_start, d1_start, "minute", price, data_source)
                fig1 = create_mini_chart(d1_bars, "Day 1 (1min)")
                st.plotly_chart(fig1, width="stretch", key=f"d1_{ticker}_{i}")
            
            with col2:
                # First month - daily
                m1_start = ipo_dt.strftime('%Y-%m-%d')
                m1_end = (ipo_dt + timedelta(days=30)).strftime('%Y-%m-%d')
                m1_bars = fetch_bars(ticker, m1_start, m1_end, "day", price, data_source)
                fig2 = create_mini_chart(m1_bars, "Month (daily)")
                st.plotly_chart(fig2, width="stretch", key=f"m1_{ticker}_{i}")
            
            with col3:
                # Lifetime - daily
                lt_start = ipo_dt.strftime('%Y-%m-%d')
                lt_end = datetime.now().strftime('%Y-%m-%d')
                lt_bars = fetch_bars(ticker, lt_start, lt_end, "day", price, data_source)
                fig3 = create_mini_chart(lt_bars, "Lifetime (daily)")
                st.plotly_chart(fig3, width="stretch", key=f"lt_{ticker}_{i}")
            
            st.markdown("---")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    IPO Operation Analysis Dashboard | Data: Bloomberg EQS + Polygon<br>
    Risk scores based on underwriter, price, float, country, and VC heuristics
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()