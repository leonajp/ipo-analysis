#!/usr/bin/env python3
"""
Update lifetime_high in ipo_master from actual daily price data (pq_daily).

This script:
1. Gets all IPO tickers and their IPO dates from ipo_master
2. Queries pq_daily to find the actual maximum high since IPO date
3. Updates ipo_master.lifetime_high with the correct value

Usage:
    python update_lifetime_high.py --analyze      # Show discrepancies
    python update_lifetime_high.py --execute      # Update all
    python update_lifetime_high.py --ticker RYOJ  # Check/update specific ticker
"""

import argparse
import os
import clickhouse_connect
from tqdm import tqdm

# ClickHouse Cloud connection
CH_HOST = os.environ.get('CLICKHOUSE_HOST', 'i35q8zrtq4.us-east-2.aws.clickhouse.cloud')
CH_PASSWORD = os.environ.get('CLICKHOUSE_PASSWORD', '~AiDc7hJ7m1Bv')


def get_client():
    return clickhouse_connect.get_client(
        host=CH_HOST, port=8443, user='default', password=CH_PASSWORD,
        secure=True, database='market_data'
    )


def analyze_single(ticker: str):
    """Analyze lifetime_high for a single ticker."""
    client = get_client()

    print(f"\n{'=' * 60}")
    print(f"ANALYZING: {ticker}")
    print('=' * 60)

    # Get current ipo_master data
    result = client.query(f"""
        SELECT polygon_ticker, ipo_date, ipo_price, lifetime_high
        FROM ipo_master
        WHERE polygon_ticker = '{ticker}'
    """)

    if not result.result_rows:
        print(f"Ticker {ticker} not found in ipo_master")
        return

    row = result.result_rows[0]
    print(f"IPO Date: {row[1]}")
    print(f"IPO Price: ${row[2]}")
    print(f"Current lifetime_high in DB: ${row[3]}")

    # Calculate actual lifetime high from pq_daily
    actual = client.query(f"""
        SELECT
            max(high) as actual_lifetime_high,
            argMax(trade_dt, high) as high_date,
            count() as trading_days
        FROM pq_daily
        WHERE symbol = '{ticker}'
          AND trade_dt >= '{row[1]}'
          AND high > 0
    """)

    if actual.result_rows and actual.result_rows[0][0]:
        actual_high = actual.result_rows[0][0]
        high_date = actual.result_rows[0][1]
        trading_days = actual.result_rows[0][2]

        print(f"\nActual lifetime high from pq_daily: ${actual_high:.2f} on {high_date}")
        print(f"Trading days in database: {trading_days}")

        if abs(actual_high - row[3]) > 0.01:
            print(f"\n⚠️  DISCREPANCY FOUND!")
            print(f"   DB value: ${row[3]:.2f}")
            print(f"   Actual:   ${actual_high:.2f}")
            print(f"   Difference: ${actual_high - row[3]:.2f}")
        else:
            print(f"\n✓ Lifetime high is correct")
    else:
        print(f"\nNo daily price data found for {ticker}")


def analyze():
    """Analyze all IPOs for lifetime_high discrepancies."""
    client = get_client()

    print("\n" + "=" * 60)
    print("LIFETIME HIGH DISCREPANCY ANALYSIS")
    print("=" * 60)

    # Find IPOs where lifetime_high doesn't match max(high) from pq_daily
    result = client.query("""
        SELECT
            i.polygon_ticker,
            i.ipo_date,
            i.ipo_price,
            i.lifetime_high as db_lifetime_high,
            p.actual_high,
            p.high_date,
            p.actual_high - i.lifetime_high as diff
        FROM ipo_master i
        JOIN (
            SELECT
                symbol,
                max(high) as actual_high,
                argMax(trade_dt, high) as high_date
            FROM pq_daily
            WHERE high > 0
            GROUP BY symbol
        ) p ON i.polygon_ticker = p.symbol
        WHERE i.ipo_date != toDate('1970-01-01')
          AND abs(p.actual_high - i.lifetime_high) > 0.05
        ORDER BY abs(p.actual_high - i.lifetime_high) DESC
        LIMIT 100
    """)

    print(f"\nFound {len(result.result_rows)} IPOs with lifetime_high discrepancies:")
    print(f"{'Ticker':<12} {'IPO Date':<12} {'DB Value':>10} {'Actual':>10} {'Diff':>10}")
    print("-" * 60)

    for row in result.result_rows[:30]:
        ticker, ipo_date, ipo_price, db_val, actual, high_date, diff = row
        print(f"{ticker:<12} {str(ipo_date):<12} ${db_val:>8.2f} ${actual:>8.2f} ${diff:>+8.2f}")

    if len(result.result_rows) > 30:
        print(f"... and {len(result.result_rows) - 30} more")

    # Summary stats
    summary = client.query("""
        SELECT
            count() as total_ipos,
            countIf(i.lifetime_high = 0) as zero_lifetime_high,
            countIf(abs(p.actual_high - i.lifetime_high) > 0.05) as mismatched
        FROM ipo_master i
        LEFT JOIN (
            SELECT
                symbol,
                max(high) as actual_high
            FROM pq_daily
            WHERE high > 0
            GROUP BY symbol
        ) p ON i.polygon_ticker = p.symbol
        WHERE i.ipo_date != toDate('1970-01-01')
    """)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    row = summary.result_rows[0]
    print(f"Total IPOs with valid date: {row[0]:,}")
    print(f"IPOs with lifetime_high = 0: {row[1]:,}")
    print(f"IPOs with mismatched lifetime_high: {row[2]:,}")


def execute(limit=None, dry_run=False):
    """Update lifetime_high for all IPOs from pq_daily data."""
    client = get_client()

    print("\n" + "=" * 60)
    print("UPDATING LIFETIME HIGH" + (" (DRY RUN)" if dry_run else ""))
    print("=" * 60)

    limit_clause = f"LIMIT {limit}" if limit else ""

    # Get IPOs that need updating (where actual differs from stored)
    query = f"""
        SELECT
            i.polygon_ticker,
            i.lifetime_high as current_val,
            p.actual_high,
            p.actual_high - i.lifetime_high as diff
        FROM ipo_master i
        JOIN (
            SELECT
                symbol,
                max(high) as actual_high
            FROM pq_daily
            WHERE high > 0
            GROUP BY symbol
        ) p ON i.polygon_ticker = p.symbol
        WHERE i.ipo_date != toDate('1970-01-01')
          AND (i.lifetime_high = 0 OR abs(p.actual_high - i.lifetime_high) > 0.01)
        ORDER BY abs(p.actual_high - i.lifetime_high) DESC
        {limit_clause}
    """

    ipos = client.query(query).result_rows
    print(f"\nFound {len(ipos)} IPOs to update...")

    if dry_run:
        print("\nDRY RUN - No changes will be made")
        print(f"{'Ticker':<12} {'Current':>10} {'New':>10} {'Diff':>10}")
        print("-" * 50)
        for row in ipos[:20]:
            ticker, current, new_val, diff = row
            print(f"{ticker:<12} ${current:>8.2f} ${new_val:>8.2f} ${diff:>+8.2f}")
        if len(ipos) > 20:
            print(f"... and {len(ipos) - 20} more")
        return

    # Update in batches
    success = 0
    errors = 0

    for ticker, current_val, actual_high, diff in tqdm(ipos, desc="Updating"):
        try:
            client.command(f"""
                ALTER TABLE ipo_master UPDATE
                    lifetime_high = {actual_high}
                WHERE polygon_ticker = '{ticker}'
            """)
            success += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\nError updating {ticker}: {e}")

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Successfully updated: {success:,}")
    print(f"Errors: {errors:,}")

    if success > 0:
        print("\nOptimizing table...")
        client.command("OPTIMIZE TABLE ipo_master FINAL")
        print("Done.")


def update_single(ticker: str):
    """Update lifetime_high for a single ticker."""
    client = get_client()

    # Get actual lifetime high
    result = client.query(f"""
        SELECT max(high) as actual_high
        FROM pq_daily
        WHERE symbol = '{ticker}'
          AND high > 0
    """)

    if result.result_rows and result.result_rows[0][0]:
        actual_high = result.result_rows[0][0]

        print(f"Updating {ticker} lifetime_high to ${actual_high:.2f}")

        client.command(f"""
            ALTER TABLE ipo_master UPDATE
                lifetime_high = {actual_high}
            WHERE polygon_ticker = '{ticker}'
        """)

        print("Updated successfully!")
    else:
        print(f"No price data found for {ticker}")


def main():
    parser = argparse.ArgumentParser(description="Update lifetime_high from pq_daily data")
    parser.add_argument("--analyze", action="store_true", help="Analyze discrepancies")
    parser.add_argument("--execute", action="store_true", help="Update all IPOs")
    parser.add_argument("--ticker", type=str, help="Analyze/update specific ticker")
    parser.add_argument("--limit", type=int, help="Limit to N tickers")
    parser.add_argument("--dry-run", action="store_true", help="Don't update, just show what would change")
    parser.add_argument("--update", action="store_true", help="Update the ticker (use with --ticker)")

    args = parser.parse_args()

    if args.ticker:
        analyze_single(args.ticker)
        if args.update:
            update_single(args.ticker)
    elif args.analyze or not args.execute:
        analyze()

    if args.execute:
        execute(limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
