#!/usr/bin/env python3
"""
Compare adjusted prices between database (Polygon/ThetaData) and Bloomberg.

Uses xbbg to fetch Bloomberg historical adjusted prices and compares with
daily_prepost (Polygon) and pq_daily (ThetaData) tables.

Requirements:
- Bloomberg Terminal running
- pip install xbbg clickhouse-connect

Usage:
    python compare_bbg_prices.py --ticker SLE
    python compare_bbg_prices.py --ticker SLE --start 2020-01-01 --end 2025-12-31
    python compare_bbg_prices.py --ticker TNXP,SLE,DCTH  # Multiple tickers
"""

import argparse
import os
from datetime import datetime, timedelta

import pandas as pd
import clickhouse_connect

# ClickHouse Cloud connection
CH_HOST = os.environ.get('CLICKHOUSE_HOST', 'i35q8zrtq4.us-east-2.aws.clickhouse.cloud')
CH_PASSWORD = os.environ.get('CLICKHOUSE_PASSWORD', '~AiDc7hJ7m1Bv')


def get_client():
    return clickhouse_connect.get_client(
        host=CH_HOST, port=8443, user='default', password=CH_PASSWORD,
        secure=True, database='market_data'
    )


def fetch_bloomberg_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch adjusted historical prices from Bloomberg via xbbg."""
    try:
        from xbbg import blp
    except ImportError:
        print("ERROR: xbbg not installed. pip install xbbg")
        return pd.DataFrame()

    bbg_ticker = f"{ticker} US Equity"

    print(f"\nFetching Bloomberg data for {bbg_ticker}...")
    print(f"  Date range: {start_date} to {end_date}")

    try:
        # Fetch adjusted OHLCV from Bloomberg
        # PX_LAST is the adjusted close by default
        df = blp.bdh(
            bbg_ticker,
            ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "VOLUME"],
            start_date,
            end_date,
        )

        if df is None or df.empty:
            print(f"  No Bloomberg data returned for {ticker}")
            return pd.DataFrame()

        # Flatten multi-index columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[1] if isinstance(col, tuple) else col for col in df.columns]

        df = df.reset_index()
        df.columns = ["trade_dt", "bbg_open", "bbg_high", "bbg_low", "bbg_close", "bbg_volume"]
        df["trade_dt"] = pd.to_datetime(df["trade_dt"]).dt.date

        print(f"  Retrieved {len(df)} days from Bloomberg")
        return df

    except Exception as e:
        print(f"  Bloomberg error: {e}")
        return pd.DataFrame()


def fetch_db_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch prices from daily_prepost (Polygon) and pq_daily (ThetaData)."""
    client = get_client()

    print(f"\nFetching database prices for {ticker}...")

    # Get data from both tables - cast to Float64 to avoid Decimal issues
    result = client.query(f"""
        SELECT
            dp.trade_dt,
            -- Polygon raw
            toFloat64(dp.rawO) as polygon_raw_open,
            toFloat64(dp.rawH) as polygon_raw_high,
            toFloat64(dp.rawL) as polygon_raw_low,
            toFloat64(dp.rawC) as polygon_raw_close,
            -- Polygon adjusted
            toFloat64(dp.adjO) as polygon_adj_open,
            toFloat64(dp.adjH) as polygon_adj_high,
            toFloat64(dp.adjL) as polygon_adj_low,
            toFloat64(dp.adjC) as polygon_adj_close,
            -- ThetaData raw (pq_daily)
            toFloat64(pq.open) as thd_raw_open,
            toFloat64(pq.high) as thd_raw_high,
            toFloat64(pq.low) as thd_raw_low,
            toFloat64(pq.close) as thd_raw_close,
            -- ThetaData adjusted (pq_daily)
            toFloat64(pq.adjO) as thd_adj_open,
            toFloat64(pq.adjH) as thd_adj_high,
            toFloat64(pq.adjL) as thd_adj_low,
            toFloat64(pq.adjC) as thd_adj_close
        FROM daily_prepost dp
        LEFT JOIN pq_daily pq ON dp.symbol = pq.symbol AND dp.trade_dt = pq.trade_dt
        WHERE dp.symbol = '{ticker}'
          AND dp.trade_dt >= '{start_date}'
          AND dp.trade_dt <= '{end_date}'
        ORDER BY dp.trade_dt
    """)

    if not result.result_rows:
        print(f"  No database data for {ticker}")
        return pd.DataFrame()

    columns = [
        "trade_dt",
        "polygon_raw_open", "polygon_raw_high", "polygon_raw_low", "polygon_raw_close",
        "polygon_adj_open", "polygon_adj_high", "polygon_adj_low", "polygon_adj_close",
        "thd_raw_open", "thd_raw_high", "thd_raw_low", "thd_raw_close",
        "thd_adj_open", "thd_adj_high", "thd_adj_low", "thd_adj_close",
    ]

    df = pd.DataFrame(result.result_rows, columns=columns)
    print(f"  Retrieved {len(df)} days from database")

    return df


def compare_prices(ticker: str, start_date: str, end_date: str) -> None:
    """Compare prices between Bloomberg and database sources."""

    print("\n" + "=" * 80)
    print(f"PRICE COMPARISON: {ticker}")
    print("=" * 80)

    # Fetch data from all sources
    bbg_df = fetch_bloomberg_prices(ticker, start_date, end_date)
    db_df = fetch_db_prices(ticker, start_date, end_date)

    if bbg_df.empty:
        print("\n[!] No Bloomberg data - cannot compare")
        return

    if db_df.empty:
        print("\n[!] No database data - cannot compare")
        return

    # Merge on trade_dt
    merged = pd.merge(
        db_df,
        bbg_df,
        on="trade_dt",
        how="inner"
    )

    print(f"\n{len(merged)} overlapping days found")

    if merged.empty:
        print("[!] No overlapping dates between Bloomberg and database")
        return

    # Calculate differences
    merged["polygon_adj_vs_bbg"] = merged["polygon_adj_close"] / merged["bbg_close"]
    merged["thd_adj_vs_bbg"] = merged["thd_adj_close"] / merged["bbg_close"]
    merged["polygon_raw_vs_bbg"] = merged["polygon_raw_close"] / merged["bbg_close"]
    merged["thd_raw_vs_bbg"] = merged["thd_raw_close"] / merged["bbg_close"]

    # Summary statistics
    print("\n" + "-" * 80)
    print("ADJUSTED CLOSE COMPARISON (ratio to Bloomberg)")
    print("-" * 80)

    print(f"\nPolygon Adjusted vs Bloomberg:")
    print(f"  Mean ratio:   {merged['polygon_adj_vs_bbg'].mean():.6f}")
    print(f"  Std dev:      {merged['polygon_adj_vs_bbg'].std():.6f}")
    print(f"  Min ratio:    {merged['polygon_adj_vs_bbg'].min():.6f}")
    print(f"  Max ratio:    {merged['polygon_adj_vs_bbg'].max():.6f}")

    print(f"\nThetaData Adjusted vs Bloomberg:")
    print(f"  Mean ratio:   {merged['thd_adj_vs_bbg'].mean():.6f}")
    print(f"  Std dev:      {merged['thd_adj_vs_bbg'].std():.6f}")
    print(f"  Min ratio:    {merged['thd_adj_vs_bbg'].min():.6f}")
    print(f"  Max ratio:    {merged['thd_adj_vs_bbg'].max():.6f}")

    # Check if they match (ratio close to 1.0)
    polygon_matches = abs(merged["polygon_adj_vs_bbg"].mean() - 1.0) < 0.01
    thd_matches = abs(merged["thd_adj_vs_bbg"].mean() - 1.0) < 0.01

    print("\n" + "-" * 80)
    print("VERDICT")
    print("-" * 80)

    if polygon_matches:
        print(f"[OK] Polygon adjusted prices MATCH Bloomberg (ratio ~1.0)")
    else:
        print(f"[!] Polygon adjusted prices DO NOT match Bloomberg")
        print(f"    Average ratio: {merged['polygon_adj_vs_bbg'].mean():.4f}")

    if thd_matches:
        print(f"[OK] ThetaData adjusted prices MATCH Bloomberg (ratio ~1.0)")
    else:
        print(f"[!] ThetaData adjusted prices DO NOT match Bloomberg")
        print(f"    Average ratio: {merged['thd_adj_vs_bbg'].mean():.4f}")

    # Show sample data
    print("\n" + "-" * 80)
    print("SAMPLE DATA (first 10 and last 10 days)")
    print("-" * 80)

    sample_cols = ["trade_dt", "bbg_close", "polygon_adj_close", "thd_adj_close",
                   "polygon_adj_vs_bbg", "thd_adj_vs_bbg"]

    # First 10 days
    print("\nFirst 10 days:")
    print(f"{'Date':<12} {'BBG Close':>12} {'Polygon Adj':>12} {'THD Adj':>12} {'Poly Ratio':>12} {'THD Ratio':>12}")
    print("-" * 76)
    for _, row in merged.head(10).iterrows():
        print(f"{str(row['trade_dt']):<12} "
              f"{row['bbg_close']:>12.4f} "
              f"{row['polygon_adj_close']:>12.4f} "
              f"{row['thd_adj_close']:>12.4f} "
              f"{row['polygon_adj_vs_bbg']:>12.4f} "
              f"{row['thd_adj_vs_bbg']:>12.4f}")

    # Last 10 days
    print("\nLast 10 days:")
    print(f"{'Date':<12} {'BBG Close':>12} {'Polygon Adj':>12} {'THD Adj':>12} {'Poly Ratio':>12} {'THD Ratio':>12}")
    print("-" * 76)
    for _, row in merged.tail(10).iterrows():
        print(f"{str(row['trade_dt']):<12} "
              f"{row['bbg_close']:>12.4f} "
              f"{row['polygon_adj_close']:>12.4f} "
              f"{row['thd_adj_close']:>12.4f} "
              f"{row['polygon_adj_vs_bbg']:>12.4f} "
              f"{row['thd_adj_vs_bbg']:>12.4f}")

    # Check for any dates with large discrepancies
    large_diff = merged[
        (abs(merged["polygon_adj_vs_bbg"] - 1.0) > 0.05) |
        (abs(merged["thd_adj_vs_bbg"] - 1.0) > 0.05)
    ]

    if not large_diff.empty:
        print("\n" + "-" * 80)
        print(f"DATES WITH >5% DISCREPANCY ({len(large_diff)} days)")
        print("-" * 80)

        print(f"{'Date':<12} {'BBG Close':>12} {'Polygon Adj':>12} {'THD Adj':>12} {'Poly Ratio':>12} {'THD Ratio':>12}")
        print("-" * 76)
        for _, row in large_diff.head(20).iterrows():
            print(f"{str(row['trade_dt']):<12} "
                  f"{row['bbg_close']:>12.4f} "
                  f"{row['polygon_adj_close']:>12.4f} "
                  f"{row['thd_adj_close']:>12.4f} "
                  f"{row['polygon_adj_vs_bbg']:>12.4f} "
                  f"{row['thd_adj_vs_bbg']:>12.4f}")


def main():
    parser = argparse.ArgumentParser(description="Compare adjusted prices with Bloomberg")
    parser.add_argument("--ticker", required=True, help="Ticker(s) to compare (comma-separated)")
    parser.add_argument("--start", default="2012-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"), help="End date (YYYY-MM-DD)")

    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.ticker.split(",")]

    for ticker in tickers:
        compare_prices(ticker, args.start, args.end)
        print("\n")


if __name__ == "__main__":
    main()
