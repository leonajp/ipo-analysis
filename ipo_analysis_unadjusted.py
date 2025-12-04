"""
IPO Operation Analysis - Using Unadjusted Prices
Source: eqsipo_unadj.csv (prices already comparable to IPO price)

This script is SIMPLER because:
- No split adjustment needed
- lifetime_high_vs_ipo_pct is pre-calculated
- All d1-d5 prices are in IPO-day terms
"""

import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR SETUP
# ============================================================================
INPUT_FILE = r'P:\Hamren\Other\eqsipo_unadj.csv'
OUTPUT_FILE = 'small_ipo_fully_adjusted.csv'  # Will save in current directory

# Small IPO criteria
MIN_IPO_PRICE = 3
MAX_IPO_PRICE = 10
MIN_SHARES_OFFERED = 500_000
MAX_SHARES_OFFERED = 5_000_000

# ============================================================================
# LOAD DATA
# ============================================================================
print("="*80)
print("IPO OPERATION ANALYSIS - Using Unadjusted Prices")
print("="*80)

df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} rows from {INPUT_FILE}")

# ============================================================================
# STEP 1: Convert columns to numeric
# ============================================================================
print("\n1. Converting columns to numeric...")

numeric_cols = [
    'IPO Sh Px', 'IPO Sh Offered', 'float_shares',
    'Lifetime High Unadj', 'lifetime_high_vs_ipo_pct', 'lifetime_high_vs_d1open_pct',
    'd1_open', 'd1_high', 'd1_low', 'd1_close',
    'd2_open', 'd2_high', 'd2_low', 'd2_close',
    'd3_open', 'd3_high', 'd3_low', 'd3_close',
    'd4_open', 'd4_high', 'd4_low', 'd4_close',
    'd5_open', 'd5_high', 'd5_low', 'd5_close',
    'first5d_hi', 'first5d_lo', '30d_hi', '30d_lo', '30d_cls',
    'd1_volume', 'd2_volume', 'd3_volume', 'd4_volume', 'd5_volume',
    'open_premium_pct', 'ret_d1', 'ret_d5', 'ret_d30',
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ============================================================================
# STEP 2: Calculate lifetime high % (prices are already unadjusted)
# ============================================================================
print("\n2. Processing lifetime high...")

# In unadjusted file, Lifetime High is already comparable to IPO price
if 'Lifetime High' in df.columns:
    df['lifetime_high'] = pd.to_numeric(df['Lifetime High'], errors='coerce')
    df['lifetime_hi_vs_ipo'] = ((df['lifetime_high'] / df['IPO Sh Px']) - 1) * 100
    print("   Calculated lifetime_hi_vs_ipo from Lifetime High")
else:
    print("   ERROR: No Lifetime High column found!")

# Sanity check - cap extreme values (likely bad data)
extreme_mask = (df['lifetime_hi_vs_ipo'] > 10000) | (df['lifetime_hi_vs_ipo'].isna())
extreme_count = extreme_mask.sum()
print(f"   Extreme/missing values: {extreme_count}")

# For extreme values, use 30d_hi if available
if extreme_count > 0 and '30d_hi' in df.columns:
    df['30d_hi'] = pd.to_numeric(df['30d_hi'], errors='coerce')
    df.loc[extreme_mask, 'lifetime_high'] = df.loc[extreme_mask, '30d_hi']
    df.loc[extreme_mask, 'lifetime_hi_vs_ipo'] = ((df.loc[extreme_mask, '30d_hi'] / df.loc[extreme_mask, 'IPO Sh Px']) - 1) * 100
    print(f"   Replaced {extreme_count} extreme values with 30d_hi")

# ============================================================================
# STEP 3: Calculate additional metrics
# ============================================================================
print("\n3. Calculating metrics...")

# Convert to numeric first
for col in ['first5d_hi', 'first5d_lo', '30d_hi', '30d_lo', '30d_cls', 'float_shares',
            'd1_volume', 'd2_volume', 'd3_volume', 'd4_volume', 'd5_volume']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# First 5 day high vs IPO
if 'first5d_hi' in df.columns:
    df['first5d_hi_vs_ipo'] = ((df['first5d_hi'] / df['IPO Sh Px']) - 1) * 100

# 30 day high vs IPO  
if '30d_hi' in df.columns:
    df['d30_hi_vs_ipo'] = ((df['30d_hi'] / df['IPO Sh Px']) - 1) * 100

# Float turnover
if 'float_shares' in df.columns and 'd1_volume' in df.columns:
    df['d1_float_turnover'] = (df['d1_volume'] / df['float_shares']) * 100
    total_vol_5d = df[['d1_volume', 'd2_volume', 'd3_volume', 'd4_volume', 'd5_volume']].sum(axis=1)
    df['float_turnover_5d'] = (total_vol_5d / df['float_shares']) * 100

# ============================================================================
# STEP 4: Filter to small IPOs
# ============================================================================
print("\n4. Filtering to small IPOs...")

# Parse date
if 'IPO Dt' in df.columns:
    df['date'] = pd.to_datetime(df['IPO Dt'], errors='coerce')
elif 'ipo_date' in df.columns:
    df['date'] = pd.to_datetime(df['ipo_date'], errors='coerce')

# Underwriter
df['underwriter'] = df['IPO Lead'].fillna('Unknown') if 'IPO Lead' in df.columns else 'Unknown'

# VC backed - use heuristic based on country + other signals
# Small foreign IPOs (Cayman, BVI, etc.) rarely have real VC backing
print("   Applying VC heuristic...")

tax_haven_countries = ['KY', 'VG', 'MH', 'D8', 'CY', 'MU', 'PA', 'JE', 'GG']

df['is_tax_haven'] = df['Cntry Terrtry Of Inc'].isin(tax_haven_countries)
df['is_us'] = df['Cntry Terrtry Of Inc'] == 'US'
df['is_tiny_offering'] = df['IPO Sh Offered'] <= 2_000_000
df['is_low_price'] = df['IPO Sh Px'] <= 5

# High-risk underwriters (associated with "operations")
operation_underwriters = [
    'D Boral', 'Kingswood', 'US Tiger', 'Prime Number', 'Network 1', 
    'EF Hutton', 'Bancroft', 'Cathay', 'RVRS', 'Viewtrade', 'Joseph Stone', 
    'Boustead', 'Maxim', 'Dawson', 'Revere', 'Dominari', 'Craft Capital'
]
df['has_operation_underwriter'] = df['IPO Lead'].apply(
    lambda x: any(uw.lower() in str(x).lower() for uw in operation_underwriters)
)

# Heuristic: likely NOT VC-backed if:
# - Tax haven country + tiny offering + operation underwriter
# Override any existing vc_backed flag with heuristic for suspicious cases
df['vc_backed_raw'] = df.get('vc_backed', False)
if 'vc_backed' in df.columns:
    df['vc_backed_raw'] = df['vc_backed'].fillna(False)
    if df['vc_backed_raw'].dtype == object:
        df['vc_backed_raw'] = df['vc_backed_raw'].astype(str).str.lower().isin(['true', '1', 'yes'])

# Apply heuristic: if tax haven + operation underwriter, override to False
df['vc_backed'] = df['vc_backed_raw']
suspicious_mask = df['is_tax_haven'] & df['has_operation_underwriter']
df.loc[suspicious_mask, 'vc_backed'] = False

# Also: US companies with legit underwriters are more likely VC-backed
legit_underwriters = ['Goldman', 'Morgan Stanley', 'JPMorgan', 'Citi', 'BofA', 
                      'Jefferies', 'Credit Suisse', 'UBS', 'Barclays', 'Deutsche']
df['has_legit_underwriter'] = df['IPO Lead'].apply(
    lambda x: any(uw.lower() in str(x).lower() for uw in legit_underwriters)
)

print(f"   Tax haven countries: {df['is_tax_haven'].sum()}")
print(f"   US companies: {df['is_us'].sum()}")
print(f"   Operation underwriters: {df['has_operation_underwriter'].sum()}")

# Small IPO filter
small_mask = (
    (df['IPO Sh Px'] >= MIN_IPO_PRICE) & 
    (df['IPO Sh Px'] <= MAX_IPO_PRICE) &
    (df['IPO Sh Offered'] >= MIN_SHARES_OFFERED) &
    (df['IPO Sh Offered'] <= MAX_SHARES_OFFERED) &
    (~df['Name'].str.contains('ACQUISITION|BLANK CHECK|SPAC', case=False, na=False)) &
    (df['lifetime_hi_vs_ipo'].notna()) &
    (df['lifetime_hi_vs_ipo'].between(-100, 10000))  # Reasonable range
)

small_ipos = df[small_mask].copy()
print(f"   Small IPOs: {len(small_ipos)} (from {len(df)} total)")

# ============================================================================
# STEP 5: Calculate risk score
# ============================================================================
print("\n5. Calculating risk scores...")

small_ipos['operation_risk_score'] = 0

# Low IPO price ($3-5)
small_ipos.loc[small_ipos['IPO Sh Px'] <= 5, 'operation_risk_score'] += 15

# Very small offering (<1.5M shares)
small_ipos.loc[small_ipos['IPO Sh Offered'] <= 1_500_000, 'operation_risk_score'] += 15
small_ipos.loc[(small_ipos['IPO Sh Offered'] > 1_500_000) & 
               (small_ipos['IPO Sh Offered'] <= 2_500_000), 'operation_risk_score'] += 10

# High d1 float turnover (>50%)
if 'd1_float_turnover' in small_ipos.columns:
    small_ipos.loc[small_ipos['d1_float_turnover'] > 50, 'operation_risk_score'] += 10

# Tax haven country (+15)
small_ipos.loc[small_ipos['is_tax_haven'], 'operation_risk_score'] += 15

# Operation underwriter (+10)
small_ipos.loc[small_ipos['has_operation_underwriter'], 'operation_risk_score'] += 10

# Legit underwriter (-10, reduces risk)
small_ipos.loc[small_ipos['has_legit_underwriter'], 'operation_risk_score'] -= 10

# No VC backing = higher risk (+10)
small_ipos.loc[~small_ipos['vc_backed'], 'operation_risk_score'] += 10

# US company = lower risk (-5)
small_ipos.loc[small_ipos['is_us'], 'operation_risk_score'] -= 5

# Ensure score doesn't go negative
small_ipos['operation_risk_score'] = small_ipos['operation_risk_score'].clip(lower=0)

# Bounce categories
small_ipos['had_major_bounce'] = (small_ipos['lifetime_hi_vs_ipo'] > 100).astype(int)

# ============================================================================
# STEP 6: Summary statistics
# ============================================================================
print("\n" + "="*80)
print("STATISTICS")
print("="*80)

print(f"\nLifetime High vs IPO:")
print(f"  Count:  {small_ipos['lifetime_hi_vs_ipo'].count()}")
print(f"  Mean:   {small_ipos['lifetime_hi_vs_ipo'].mean():.1f}%")
print(f"  Median: {small_ipos['lifetime_hi_vs_ipo'].median():.1f}%")
print(f"  Std:    {small_ipos['lifetime_hi_vs_ipo'].std():.1f}%")
print(f"  Max:    {small_ipos['lifetime_hi_vs_ipo'].max():.1f}%")
print(f"  Min:    {small_ipos['lifetime_hi_vs_ipo'].min():.1f}%")

print(f"\n% with >100% bounce: {small_ipos['had_major_bounce'].mean()*100:.1f}%")
print(f"% with >200% bounce: {(small_ipos['lifetime_hi_vs_ipo'] > 200).mean()*100:.1f}%")
print(f"% with >500% bounce: {(small_ipos['lifetime_hi_vs_ipo'] > 500).mean()*100:.1f}%")

print(f"\nVC Backed (after heuristic): {small_ipos['vc_backed'].sum()} ({small_ipos['vc_backed'].mean()*100:.1f}%)")
print(f"Tax Haven Countries: {small_ipos['is_tax_haven'].sum()} ({small_ipos['is_tax_haven'].mean()*100:.1f}%)")
print(f"US Companies: {small_ipos['is_us'].sum()} ({small_ipos['is_us'].mean()*100:.1f}%)")
print(f"Operation Underwriters: {small_ipos['has_operation_underwriter'].sum()} ({small_ipos['has_operation_underwriter'].mean()*100:.1f}%)")

# Bounce rate by country type
print(f"\nBounce >100% by Country Type:")
tax_haven_bounce = small_ipos[small_ipos['is_tax_haven']]['had_major_bounce'].mean() * 100
us_bounce = small_ipos[small_ipos['is_us']]['had_major_bounce'].mean() * 100
other_bounce = small_ipos[~small_ipos['is_tax_haven'] & ~small_ipos['is_us']]['had_major_bounce'].mean() * 100
print(f"  Tax Haven: {tax_haven_bounce:.1f}%")
print(f"  US: {us_bounce:.1f}%")
print(f"  Other: {other_bounce:.1f}%")

# ============================================================================
# STEP 7: Top bounces
# ============================================================================
print("\n" + "="*80)
print("TOP BOUNCES (2024+)")
print("="*80)

recent = small_ipos[small_ipos['date'] >= '2024-01-01'].sort_values('lifetime_hi_vs_ipo', ascending=False)

print("\nTop 30:")
for i, (_, row) in enumerate(recent.head(30).iterrows()):
    ticker = str(row['Ticker']).replace(' US Equity', '')
    date = row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else 'N/A'
    uw = str(row['underwriter'])[:20]
    vc = "VC" if row['vc_backed'] else ""
    lh = row['lifetime_high'] if pd.notna(row['lifetime_high']) else 0
    print(f"{i+1:2}. {ticker:8} | {date} | ${row['IPO Sh Px']:.0f} -> ${lh:.2f} | "
          f"{row['lifetime_hi_vs_ipo']:6.0f}% | {vc:3} | {uw}")

# ============================================================================
# STEP 8: Export
# ============================================================================
print("\n" + "="*80)
print("EXPORTING")
print("="*80)

export_cols = [
    'Ticker', 'Name', 'date', 'underwriter', 'vc_backed',
    'IPO Sh Px', 'IPO Sh Offered', 'float_shares',
    # Prices (already unadjusted)
    'd1_open', 'd1_high', 'd1_low', 'd1_close',
    'first5d_hi', 'first5d_lo',
    '30d_hi', '30d_lo', '30d_cls',
    'lifetime_high',
    # Metrics
    'lifetime_hi_vs_ipo', 'first5d_hi_vs_ipo', 'd30_hi_vs_ipo',
    'ret_d1', 'ret_d5', 'ret_d30',
    'd1_float_turnover', 'float_turnover_5d',
    'operation_risk_score', 'had_major_bounce',
    # Heuristic flags
    'is_tax_haven', 'is_us', 'has_operation_underwriter', 'has_legit_underwriter',
    # Volumes
    'd1_volume', 'd2_volume', 'd3_volume', 'd4_volume', 'd5_volume',
    # Extra
    'GICS Sector', 'Cntry Terrtry Of Inc',
]

# Only keep columns that exist
export_cols = [c for c in export_cols if c in small_ipos.columns]

output_df = small_ipos[export_cols].copy()
output_df.to_csv(OUTPUT_FILE, index=False)
print(f"Exported {len(output_df)} rows to: {OUTPUT_FILE}")

# ============================================================================
# VERIFICATION
# ============================================================================
print("\n" + "="*80)
print("VERIFICATION")
print("="*80)

for ticker_check in ['BMNR', 'MGN', 'FOFO', 'NNE']:
    check = output_df[output_df['Ticker'].str.contains(ticker_check, na=False)]
    if len(check) > 0:
        row = check.iloc[0]
        lh = row['lifetime_high'] if 'lifetime_high' in row and pd.notna(row['lifetime_high']) else 'N/A'
        print(f"\n{ticker_check}:")
        print(f"  IPO Price: ${row['IPO Sh Px']:.2f}")
        print(f"  Lifetime High: ${lh}")
        print(f"  Bounce: {row['lifetime_hi_vs_ipo']:.1f}%")
        if 'vc_backed' in row:
            print(f"  VC Backed: {row['vc_backed']}")

print("\n" + "="*80)
print("✓ Done!")
print("="*80)