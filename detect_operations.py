#!/usr/bin/env python3
"""
Detect "operations" (price pumps with abnormal volume) for IPOs.

An operation is defined as:
1. Volume spike: volume > 5x the 20-day trailing median
2. Price spike: high > 50% above IPO price
3. Multiple spike days within 5 trading days = same operation

This script:
1. Detects all operations for each IPO
2. Stores operation history in ClickHouse
3. Calculates operation metrics (count, max gain, dates)
4. Analyzes underwriter patterns

Usage:
    python detect_operations.py --analyze         # Show operation statistics
    python detect_operations.py --execute         # Detect and store all operations
    python detect_operations.py --ticker RYOJ     # Analyze specific ticker
    python detect_operations.py --underwriters    # Show underwriter operation patterns
"""

import argparse
import os
import clickhouse_connect
from datetime import datetime, timedelta
from tqdm import tqdm
import pandas as pd

# ClickHouse Cloud connection
CH_HOST = os.environ.get('CLICKHOUSE_HOST', 'i35q8zrtq4.us-east-2.aws.clickhouse.cloud')
CH_PASSWORD = os.environ.get('CLICKHOUSE_PASSWORD', '~AiDc7hJ7m1Bv')

# Operation detection parameters
VOL_SPIKE_THRESHOLD = 5.0    # Volume must be 5x median
PRICE_ABOVE_IPO_PCT = 50.0   # Price must be 50% above IPO
CLUSTER_DAYS = 5             # Spikes within 5 days = same operation
MIN_TRADING_DAYS = 20        # Need at least 20 days of data for baseline


def get_client():
    return clickhouse_connect.get_client(
        host=CH_HOST, port=8443, user='default', password=CH_PASSWORD,
        secure=True, database='market_data'
    )


def create_operations_table():
    """Create the ipo_operations table if it doesn't exist."""
    client = get_client()

    client.command("""
        CREATE TABLE IF NOT EXISTS ipo_operations (
            polygon_ticker String,
            operation_id UInt32,
            start_date Date,
            end_date Date,
            peak_date Date,
            peak_price Float64,
            peak_volume UInt64,
            peak_vol_multiple Float64,
            gain_vs_ipo_pct Float64,
            spike_days UInt32,
            ipo_price Float64,
            underwriter String,
            detected_at DateTime DEFAULT now()
        ) ENGINE = ReplacingMergeTree(detected_at)
        ORDER BY (polygon_ticker, operation_id)
    """)
    print("Created/verified ipo_operations table")


def detect_operations_for_ticker(client, ticker: str, ipo_date, ipo_price: float, underwriter: str):
    """Detect all operations for a single ticker."""

    # Get daily data with volume metrics
    result = client.query(f"""
        WITH ranked AS (
            SELECT
                trade_dt,
                high,
                close,
                volume,
                -- Calculate 20-day trailing median volume (excluding current day)
                medianExact(volume) OVER (
                    ORDER BY trade_dt
                    ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                ) as vol_20d_median
            FROM pq_daily
            WHERE symbol = '{ticker}'
              AND trade_dt >= '{ipo_date}'
            ORDER BY trade_dt
        )
        SELECT
            trade_dt,
            high,
            close,
            volume,
            vol_20d_median,
            volume / nullIf(vol_20d_median, 0) as vol_ratio
        FROM ranked
        WHERE vol_20d_median > 0
        ORDER BY trade_dt
    """)

    if not result.result_rows:
        return []

    # Find spike days
    spike_days = []
    for row in result.result_rows:
        trade_dt, high, close, volume, vol_median, vol_ratio = row
        vol_ratio = vol_ratio or 0

        # Check if this is a spike day
        gain_vs_ipo = ((high - ipo_price) / ipo_price) * 100 if ipo_price > 0 else 0

        if vol_ratio >= VOL_SPIKE_THRESHOLD and gain_vs_ipo >= PRICE_ABOVE_IPO_PCT:
            spike_days.append({
                'date': trade_dt,
                'high': high,
                'volume': volume,
                'vol_ratio': vol_ratio,
                'gain_vs_ipo': gain_vs_ipo
            })

    if not spike_days:
        return []

    # Cluster spikes into operations
    operations = []
    current_op = None

    for spike in spike_days:
        if current_op is None:
            # Start new operation
            current_op = {
                'start_date': spike['date'],
                'end_date': spike['date'],
                'peak_date': spike['date'],
                'peak_price': spike['high'],
                'peak_volume': spike['volume'],
                'peak_vol_multiple': spike['vol_ratio'],
                'max_gain': spike['gain_vs_ipo'],
                'spike_count': 1
            }
        else:
            # Check if this spike is within CLUSTER_DAYS of the current operation
            days_diff = (spike['date'] - current_op['end_date']).days

            if days_diff <= CLUSTER_DAYS:
                # Extend current operation
                current_op['end_date'] = spike['date']
                current_op['spike_count'] += 1

                # Update peak if this spike is higher
                if spike['high'] > current_op['peak_price']:
                    current_op['peak_date'] = spike['date']
                    current_op['peak_price'] = spike['high']
                    current_op['peak_volume'] = spike['volume']
                    current_op['peak_vol_multiple'] = spike['vol_ratio']
                    current_op['max_gain'] = spike['gain_vs_ipo']
            else:
                # Save current operation and start new one
                operations.append(current_op)
                current_op = {
                    'start_date': spike['date'],
                    'end_date': spike['date'],
                    'peak_date': spike['date'],
                    'peak_price': spike['high'],
                    'peak_volume': spike['volume'],
                    'peak_vol_multiple': spike['vol_ratio'],
                    'max_gain': spike['gain_vs_ipo'],
                    'spike_count': 1
                }

    # Don't forget the last operation
    if current_op is not None:
        operations.append(current_op)

    # Add metadata to operations
    for i, op in enumerate(operations):
        op['operation_id'] = i + 1
        op['ticker'] = ticker
        op['ipo_price'] = ipo_price
        op['underwriter'] = underwriter or ''

    return operations


def analyze_ticker(ticker: str):
    """Analyze operations for a specific ticker."""
    client = get_client()

    print(f"\n{'=' * 70}")
    print(f"OPERATION ANALYSIS: {ticker}")
    print('=' * 70)

    # Get IPO info
    ipo_info = client.query(f"""
        SELECT ipo_date, ipo_price, underwriter, lifetime_high
        FROM ipo_master
        WHERE polygon_ticker = '{ticker}'
    """)

    if not ipo_info.result_rows:
        print(f"Ticker {ticker} not found in ipo_master")
        return

    ipo_date, ipo_price, underwriter, lifetime_high = ipo_info.result_rows[0]
    print(f"IPO Date: {ipo_date}")
    print(f"IPO Price: ${ipo_price}")
    print(f"Underwriter: {underwriter}")
    print(f"Lifetime High: ${lifetime_high:.2f}")

    # Detect operations
    operations = detect_operations_for_ticker(client, ticker, ipo_date, ipo_price, underwriter)

    if not operations:
        print(f"\nNo operations detected for {ticker}")
        print("(No days with volume >= 5x median AND price >= 50% above IPO)")
        return

    print(f"\nDetected {len(operations)} operation(s):")
    print(f"{'#':<3} {'Start':<12} {'End':<12} {'Peak Date':<12} {'Peak$':>8} {'Gain%':>8} {'VolX':>8} {'Spikes':>6}")
    print('-' * 80)

    for op in operations:
        print(f"{op['operation_id']:<3} {str(op['start_date']):<12} {str(op['end_date']):<12} "
              f"{str(op['peak_date']):<12} ${op['peak_price']:>6.2f} {op['max_gain']:>7.0f}% "
              f"{op['peak_vol_multiple']:>7.1f}x {op['spike_count']:>6}")


def execute(limit=None):
    """Detect operations for all IPOs and store in ClickHouse."""
    client = get_client()

    print("\n" + "=" * 70)
    print("DETECTING OPERATIONS FOR ALL IPOs")
    print("=" * 70)

    # Create table if needed
    create_operations_table()

    # Clear existing data
    client.command("TRUNCATE TABLE ipo_operations")

    # Get all IPOs with sufficient trading history
    limit_clause = f"LIMIT {limit}" if limit else ""

    # First get tickers with enough trading days
    tickers_with_data = client.query(f"""
        SELECT symbol, count() as days
        FROM pq_daily
        GROUP BY symbol
        HAVING days >= {MIN_TRADING_DAYS}
    """)
    valid_tickers = set(row[0] for row in tickers_with_data.result_rows)
    print(f"Found {len(valid_tickers)} tickers with >= {MIN_TRADING_DAYS} trading days")

    ipos = client.query(f"""
        SELECT
            polygon_ticker,
            ipo_date,
            ipo_price,
            underwriter
        FROM ipo_master
        WHERE ipo_date >= '2015-01-01'
          AND ipo_date != toDate('1970-01-01')
          AND ipo_price > 0
        ORDER BY ipo_date DESC
        {limit_clause}
    """).result_rows

    # Filter to only those with enough trading data
    ipos = [ipo for ipo in ipos if ipo[0] in valid_tickers]

    print(f"Processing {len(ipos)} IPOs...")

    total_operations = 0
    ipos_with_ops = 0
    all_operations = []

    for ticker, ipo_date, ipo_price, underwriter in tqdm(ipos, desc="Detecting"):
        try:
            operations = detect_operations_for_ticker(client, ticker, ipo_date, ipo_price, underwriter)

            if operations:
                ipos_with_ops += 1
                total_operations += len(operations)
                all_operations.extend(operations)
        except Exception as e:
            pass  # Skip errors silently

    # Insert operations into ClickHouse
    if all_operations:
        print(f"\nInserting {len(all_operations)} operations into ClickHouse...")

        rows = []
        for op in all_operations:
            rows.append([
                op['ticker'],
                op['operation_id'],
                op['start_date'],
                op['end_date'],
                op['peak_date'],
                op['peak_price'],
                op['peak_volume'],
                op['peak_vol_multiple'],
                op['max_gain'],
                op['spike_count'],
                op['ipo_price'],
                op['underwriter']
            ])

        client.insert('ipo_operations', rows, column_names=[
            'polygon_ticker', 'operation_id', 'start_date', 'end_date',
            'peak_date', 'peak_price', 'peak_volume', 'peak_vol_multiple',
            'gain_vs_ipo_pct', 'spike_days', 'ipo_price', 'underwriter'
        ])

    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"IPOs processed: {len(ipos):,}")
    print(f"IPOs with operations: {ipos_with_ops:,} ({ipos_with_ops/len(ipos)*100:.1f}%)")
    print(f"Total operations detected: {total_operations:,}")
    print(f"Avg operations per IPO (with ops): {total_operations/max(ipos_with_ops,1):.2f}")


def analyze_underwriters():
    """Analyze which underwriters have the most operations."""
    client = get_client()

    print("\n" + "=" * 70)
    print("UNDERWRITER OPERATION ANALYSIS")
    print("=" * 70)

    # Check if operations table exists and has data
    try:
        count = client.query("SELECT count() FROM ipo_operations").result_rows[0][0]
        if count == 0:
            print("No operations data found. Run --execute first.")
            return
    except:
        print("Operations table not found. Run --execute first.")
        return

    # Underwriters with most operations
    result = client.query("""
        SELECT
            underwriter,
            count(DISTINCT polygon_ticker) as ipos_with_ops,
            count() as total_operations,
            count() / count(DISTINCT polygon_ticker) as ops_per_ipo,
            avg(gain_vs_ipo_pct) as avg_gain,
            max(gain_vs_ipo_pct) as max_gain,
            avg(peak_vol_multiple) as avg_vol_multiple
        FROM ipo_operations
        WHERE underwriter != ''
        GROUP BY underwriter
        HAVING ipos_with_ops >= 3
        ORDER BY ops_per_ipo DESC, total_operations DESC
        LIMIT 30
    """)

    print(f"\n{'Underwriter':<35} {'IPOs':>6} {'Ops':>6} {'Ops/IPO':>8} {'AvgGain':>10} {'MaxGain':>10}")
    print('-' * 90)

    for row in result.result_rows:
        uw = str(row[0])[:35]
        print(f"{uw:<35} {row[1]:>6} {row[2]:>6} {row[3]:>8.2f} {row[4]:>9.0f}% {row[5]:>9.0f}%")

    # Multi-operation tickers by underwriter
    print(f"\n\n{'=' * 70}")
    print("IPOs WITH MULTIPLE OPERATIONS (by underwriter)")
    print('=' * 70)

    result = client.query("""
        SELECT
            underwriter,
            polygon_ticker,
            count() as num_ops,
            min(peak_date) as first_op,
            max(peak_date) as last_op,
            max(gain_vs_ipo_pct) as max_gain
        FROM ipo_operations
        WHERE underwriter != ''
        GROUP BY underwriter, polygon_ticker
        HAVING num_ops >= 2
        ORDER BY num_ops DESC, max_gain DESC
        LIMIT 30
    """)

    print(f"{'Underwriter':<30} {'Ticker':<10} {'Ops':>4} {'First Op':<12} {'Last Op':<12} {'MaxGain':>10}")
    print('-' * 90)

    for row in result.result_rows:
        uw = str(row[0])[:30]
        print(f"{uw:<30} {row[1]:<10} {row[2]:>4} {str(row[3]):<12} {str(row[4]):<12} {row[5]:>9.0f}%")


def analyze_summary():
    """Show summary statistics of detected operations."""
    client = get_client()

    print("\n" + "=" * 70)
    print("OPERATION DETECTION SUMMARY")
    print("=" * 70)

    # Check if operations table has data
    try:
        result = client.query("""
            SELECT
                count() as total_ops,
                count(DISTINCT polygon_ticker) as ipos_with_ops,
                avg(gain_vs_ipo_pct) as avg_gain,
                max(gain_vs_ipo_pct) as max_gain,
                avg(peak_vol_multiple) as avg_vol_mult,
                avg(spike_days) as avg_spike_days
            FROM ipo_operations
        """)

        row = result.result_rows[0]
        print(f"\nTotal operations detected: {row[0]:,}")
        print(f"IPOs with operations: {row[1]:,}")
        print(f"Average gain vs IPO: {row[2]:.0f}%")
        print(f"Max gain vs IPO: {row[3]:.0f}%")
        print(f"Average volume multiple: {row[4]:.1f}x")
        print(f"Average spike days per operation: {row[5]:.1f}")

        # Operations by year
        print(f"\n{'=' * 70}")
        print("OPERATIONS BY YEAR")
        print('=' * 70)

        result = client.query("""
            SELECT
                toYear(peak_date) as year,
                count() as ops,
                count(DISTINCT polygon_ticker) as ipos,
                avg(gain_vs_ipo_pct) as avg_gain
            FROM ipo_operations
            GROUP BY year
            ORDER BY year DESC
        """)

        print(f"{'Year':>6} {'Operations':>12} {'IPOs':>8} {'Avg Gain':>12}")
        print('-' * 45)
        for row in result.result_rows:
            print(f"{row[0]:>6} {row[1]:>12} {row[2]:>8} {row[3]:>11.0f}%")

    except Exception as e:
        print(f"Error: {e}")
        print("Run --execute first to detect operations.")


def main():
    parser = argparse.ArgumentParser(description="Detect IPO operations (pumps)")
    parser.add_argument("--analyze", action="store_true", help="Show operation summary")
    parser.add_argument("--execute", action="store_true", help="Detect all operations")
    parser.add_argument("--ticker", type=str, help="Analyze specific ticker")
    parser.add_argument("--underwriters", action="store_true", help="Analyze underwriter patterns")
    parser.add_argument("--limit", type=int, help="Limit to N tickers")

    args = parser.parse_args()

    if args.ticker:
        analyze_ticker(args.ticker)
    elif args.execute:
        execute(limit=args.limit)
        analyze_summary()
    elif args.underwriters:
        analyze_underwriters()
    else:
        analyze_summary()


if __name__ == "__main__":
    main()
