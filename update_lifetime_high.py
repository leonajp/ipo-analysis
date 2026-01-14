#!/usr/bin/env python3
"""
Update lifetime_high in ipo_master from actual daily price data.

This script:
1. Gets all IPO tickers and their IPO dates from ipo_master
2. Queries pq_daily for regular session high
3. Queries daily_prepost for pre-market and after-hours high
4. Takes the maximum of all three for the true lifetime high
5. Updates ipo_master.lifetime_high with the correct value

IMPORTANT: Always use the RAW columns from daily_prepost (premarket_high_raw,
afterhours_high_raw) - the adjusted columns (premarket_high, afterhours_high)
have incorrect split adjustments that can be 1000x+ off.

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
    """Analyze lifetime_high for a single ticker including pre/post market."""
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
    ipo_date = row[1]
    print(f"IPO Date: {ipo_date}")
    print(f"IPO Price: ${row[2]}")
    print(f"Current lifetime_high in DB: ${row[3]}")

    # Get regular session high from pq_daily
    regular = client.query(f"""
        SELECT
            max(high) as max_high,
            argMax(trade_dt, high) as high_date,
            count() as trading_days
        FROM pq_daily
        WHERE symbol = '{ticker}'
          AND trade_dt >= '{ipo_date}'
          AND high > 0
    """)

    regular_high = regular.result_rows[0][0] if regular.result_rows and regular.result_rows[0][0] else 0
    regular_date = regular.result_rows[0][1] if regular.result_rows else None
    trading_days = regular.result_rows[0][2] if regular.result_rows else 0

    print(f"\nRegular session high: ${regular_high:.2f}" + (f" on {regular_date}" if regular_date else ""))
    print(f"Trading days in database: {trading_days}")

    # Get pre-market high from daily_prepost (use RAW columns - adjusted columns are broken)
    premarket = client.query(f"""
        SELECT
            max(premarket_high_raw) as max_high,
            argMax(trade_dt, premarket_high_raw) as high_date
        FROM daily_prepost
        WHERE symbol = '{ticker}'
          AND trade_dt >= '{ipo_date}'
          AND premarket_high_raw > 0
    """)

    premarket_high = premarket.result_rows[0][0] if premarket.result_rows and premarket.result_rows[0][0] else 0
    premarket_date = premarket.result_rows[0][1] if premarket.result_rows else None

    print(f"Pre-market high: ${premarket_high:.2f}" + (f" on {premarket_date}" if premarket_date and premarket_high > 0 else ""))

    # Get after-hours high from daily_prepost (use RAW columns)
    afterhours = client.query(f"""
        SELECT
            max(afterhours_high_raw) as max_high,
            argMax(trade_dt, afterhours_high_raw) as high_date
        FROM daily_prepost
        WHERE symbol = '{ticker}'
          AND trade_dt >= '{ipo_date}'
          AND afterhours_high_raw > 0
    """)

    afterhours_high = afterhours.result_rows[0][0] if afterhours.result_rows and afterhours.result_rows[0][0] else 0
    afterhours_date = afterhours.result_rows[0][1] if afterhours.result_rows else None

    print(f"After-hours high: ${afterhours_high:.2f}" + (f" on {afterhours_date}" if afterhours_date and afterhours_high > 0 else ""))

    # Calculate true lifetime high (max of all three)
    all_highs = [
        (regular_high, regular_date, "Regular"),
        (premarket_high, premarket_date, "Pre-market"),
        (afterhours_high, afterhours_date, "After-hours")
    ]
    actual_high, actual_date, session = max(all_highs, key=lambda x: x[0] or 0)

    print(f"\n>>> TRUE LIFETIME HIGH: ${actual_high:.2f} ({session}" + (f" on {actual_date})" if actual_date else ")"))

    if abs(actual_high - row[3]) > 0.01:
        print(f"\n[!] DISCREPANCY FOUND!")
        print(f"   DB value: ${row[3]:.2f}")
        print(f"   Actual:   ${actual_high:.2f}")
        print(f"   Difference: ${actual_high - row[3]:.2f}")
    else:
        print(f"\n[OK] Lifetime high is correct")


def analyze():
    """Analyze all IPOs for lifetime_high discrepancies including pre/post market."""
    client = get_client()

    print("\n" + "=" * 60)
    print("LIFETIME HIGH DISCREPANCY ANALYSIS (incl. pre/post market)")
    print("=" * 60)

    # Find IPOs where lifetime_high doesn't match max across all sessions
    # Combine regular, pre-market, and after-hours highs
    # Filter out obviously bad data (prices over $100,000 are likely errors)
    MAX_SANE_PRICE = 100000.0

    result = client.query(f"""
        SELECT
            i.polygon_ticker,
            i.ipo_date,
            i.ipo_price,
            i.lifetime_high as db_lifetime_high,
            greatest(
                coalesce(p.regular_high, 0),
                if(coalesce(pp.pm_high, 0) < {MAX_SANE_PRICE}, coalesce(pp.pm_high, 0), 0),
                if(coalesce(pp.ah_high, 0) < {MAX_SANE_PRICE}, coalesce(pp.ah_high, 0), 0)
            ) as actual_high,
            greatest(
                coalesce(p.regular_high, 0),
                if(coalesce(pp.pm_high, 0) < {MAX_SANE_PRICE}, coalesce(pp.pm_high, 0), 0),
                if(coalesce(pp.ah_high, 0) < {MAX_SANE_PRICE}, coalesce(pp.ah_high, 0), 0)
            ) - i.lifetime_high as diff
        FROM ipo_master i
        LEFT JOIN (
            SELECT
                symbol,
                max(high) as regular_high
            FROM pq_daily
            WHERE high > 0 AND high < {MAX_SANE_PRICE}
            GROUP BY symbol
        ) p ON i.polygon_ticker = p.symbol
        LEFT JOIN (
            SELECT
                symbol,
                max(premarket_high_raw) as pm_high,
                max(afterhours_high_raw) as ah_high
            FROM daily_prepost
            GROUP BY symbol
        ) pp ON i.polygon_ticker = pp.symbol
        WHERE i.ipo_date != toDate('1970-01-01')
          AND abs(greatest(
                coalesce(p.regular_high, 0),
                if(coalesce(pp.pm_high, 0) < {MAX_SANE_PRICE}, coalesce(pp.pm_high, 0), 0),
                if(coalesce(pp.ah_high, 0) < {MAX_SANE_PRICE}, coalesce(pp.ah_high, 0), 0)
            ) - i.lifetime_high) > 0.05
        ORDER BY abs(greatest(
                coalesce(p.regular_high, 0),
                if(coalesce(pp.pm_high, 0) < {MAX_SANE_PRICE}, coalesce(pp.pm_high, 0), 0),
                if(coalesce(pp.ah_high, 0) < {MAX_SANE_PRICE}, coalesce(pp.ah_high, 0), 0)
            ) - i.lifetime_high) DESC
        LIMIT 100
    """)

    print(f"\nFound {len(result.result_rows)} IPOs with lifetime_high discrepancies:")
    print(f"{'Ticker':<12} {'IPO Date':<12} {'DB Value':>10} {'Actual':>10} {'Diff':>10}")
    print("-" * 60)

    for row in result.result_rows[:30]:
        ticker, ipo_date, ipo_price, db_val, actual, diff = row
        print(f"{ticker:<12} {str(ipo_date):<12} ${db_val:>8.2f} ${actual:>8.2f} ${diff:>+8.2f}")

    if len(result.result_rows) > 30:
        print(f"... and {len(result.result_rows) - 30} more")

    # Summary stats
    summary = client.query(f"""
        SELECT
            count() as total_ipos,
            countIf(i.lifetime_high = 0) as zero_lifetime_high,
            countIf(abs(greatest(
                coalesce(p.regular_high, 0),
                if(coalesce(pp.pm_high, 0) < {MAX_SANE_PRICE}, coalesce(pp.pm_high, 0), 0),
                if(coalesce(pp.ah_high, 0) < {MAX_SANE_PRICE}, coalesce(pp.ah_high, 0), 0)
            ) - i.lifetime_high) > 0.05) as mismatched
        FROM ipo_master i
        LEFT JOIN (
            SELECT symbol, max(high) as regular_high
            FROM pq_daily WHERE high > 0 AND high < {MAX_SANE_PRICE} GROUP BY symbol
        ) p ON i.polygon_ticker = p.symbol
        LEFT JOIN (
            SELECT symbol,
                max(premarket_high_raw) as pm_high,
                max(afterhours_high_raw) as ah_high
            FROM daily_prepost GROUP BY symbol
        ) pp ON i.polygon_ticker = pp.symbol
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
    """Update lifetime_high for all IPOs including pre/post market highs."""
    client = get_client()

    print("\n" + "=" * 60)
    print("UPDATING LIFETIME HIGH (incl. pre/post market)" + (" (DRY RUN)" if dry_run else ""))
    print("=" * 60)

    limit_clause = f"LIMIT {limit}" if limit else ""

    # Filter out obviously bad data (prices over $100,000 are likely errors)
    MAX_SANE_PRICE = 100000.0

    # Get IPOs that need updating - max across regular, premarket, afterhours
    query = f"""
        SELECT
            i.polygon_ticker,
            i.lifetime_high as current_val,
            greatest(
                coalesce(p.regular_high, 0),
                if(coalesce(pp.pm_high, 0) < {MAX_SANE_PRICE}, coalesce(pp.pm_high, 0), 0),
                if(coalesce(pp.ah_high, 0) < {MAX_SANE_PRICE}, coalesce(pp.ah_high, 0), 0)
            ) as actual_high,
            greatest(
                coalesce(p.regular_high, 0),
                if(coalesce(pp.pm_high, 0) < {MAX_SANE_PRICE}, coalesce(pp.pm_high, 0), 0),
                if(coalesce(pp.ah_high, 0) < {MAX_SANE_PRICE}, coalesce(pp.ah_high, 0), 0)
            ) - i.lifetime_high as diff
        FROM ipo_master i
        LEFT JOIN (
            SELECT symbol, max(high) as regular_high
            FROM pq_daily WHERE high > 0 AND high < {MAX_SANE_PRICE} GROUP BY symbol
        ) p ON i.polygon_ticker = p.symbol
        LEFT JOIN (
            SELECT symbol,
                max(premarket_high_raw) as pm_high,
                max(afterhours_high_raw) as ah_high
            FROM daily_prepost GROUP BY symbol
        ) pp ON i.polygon_ticker = pp.symbol
        WHERE i.ipo_date != toDate('1970-01-01')
          AND (i.lifetime_high = 0 OR abs(greatest(
                coalesce(p.regular_high, 0),
                if(coalesce(pp.pm_high, 0) < {MAX_SANE_PRICE}, coalesce(pp.pm_high, 0), 0),
                if(coalesce(pp.ah_high, 0) < {MAX_SANE_PRICE}, coalesce(pp.ah_high, 0), 0)
            ) - i.lifetime_high) > 0.01)
        ORDER BY abs(greatest(
                coalesce(p.regular_high, 0),
                if(coalesce(pp.pm_high, 0) < {MAX_SANE_PRICE}, coalesce(pp.pm_high, 0), 0),
                if(coalesce(pp.ah_high, 0) < {MAX_SANE_PRICE}, coalesce(pp.ah_high, 0), 0)
            ) - i.lifetime_high) DESC
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
    """Update lifetime_high for a single ticker including pre/post market."""
    client = get_client()

    # Get max across regular, premarket, afterhours sessions
    result = client.query(f"""
        SELECT
            greatest(
                coalesce(p.regular_high, 0),
                coalesce(pp.premarket_high, 0),
                coalesce(pp.afterhours_high, 0)
            ) as actual_high
        FROM (SELECT 1) dummy
        LEFT JOIN (
            SELECT max(high) as regular_high
            FROM pq_daily WHERE symbol = '{ticker}' AND high > 0
        ) p ON 1=1
        LEFT JOIN (
            SELECT
                max(premarket_high_raw) as premarket_high,
                max(afterhours_high_raw) as afterhours_high
            FROM daily_prepost WHERE symbol = '{ticker}' AND (premarket_high_raw > 0 OR afterhours_high_raw > 0)
        ) pp ON 1=1
    """)

    if result.result_rows and result.result_rows[0][0]:
        actual_high = result.result_rows[0][0]

        print(f"Updating {ticker} lifetime_high to ${actual_high:.2f} (incl. pre/post market)")

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
