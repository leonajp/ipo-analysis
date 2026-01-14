#!/usr/bin/env python3
"""
Update ipo_master table with trading metrics from pq_daily.

This script populates d1-d20 metrics, lifetime_high, and last_price_adj
from the pq_daily table for IPOs that have trading data but missing metrics.

Usage:
    python update_ipo_master_from_pq.py                    # Update all missing
    python update_ipo_master_from_pq.py --ticker GCDT      # Update specific ticker
    python update_ipo_master_from_pq.py --days 60          # Look back 60 days
    python update_ipo_master_from_pq.py --dry-run          # Preview without writing
"""

import argparse
import os
from datetime import datetime, timedelta

import clickhouse_connect

# ClickHouse connection
CH_HOST = os.environ.get('CLICKHOUSE_HOST', 'i35q8zrtq4.us-east-2.aws.clickhouse.cloud')
CH_PASSWORD = os.environ.get('CLICKHOUSE_PASSWORD', '~AiDc7hJ7m1Bv')


def get_client():
    return clickhouse_connect.get_client(
        host=CH_HOST, port=8443, user='default', password=CH_PASSWORD,
        secure=True, database='market_data'
    )


def get_ipos_needing_update(days_back: int = 60) -> list:
    """Get IPOs that have pq_daily data but missing ipo_master metrics."""
    client = get_client()
    cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

    # Use a JOIN instead of correlated subquery
    result = client.query(f"""
        SELECT
            i.polygon_ticker,
            i.ipo_date
        FROM ipo_master i
        INNER JOIN (
            SELECT DISTINCT symbol FROM pq_daily
        ) p ON i.polygon_ticker = p.symbol
        WHERE i.ipo_date >= '{cutoff_date}'
          AND i.d1_high = 0
        ORDER BY i.ipo_date DESC
    """)

    return [(row[0], row[1]) for row in result.result_rows]


def update_ticker(ticker: str, ipo_date, dry_run: bool = False) -> bool:
    """Update ipo_master with trading data from pq_daily for a ticker."""
    client = get_client()

    # Get trading data from pq_daily, ordered by date
    result = client.query(f"""
        SELECT trade_dt, open, high, low, close, volume
        FROM pq_daily
        WHERE symbol = '{ticker}'
        ORDER BY trade_dt ASC
    """)

    if not result.result_rows:
        print(f"  {ticker}: No pq_daily data")
        return False

    rows = result.result_rows
    print(f"  {ticker} (IPO: {ipo_date}): {len(rows)} trading days")

    # Calculate metrics
    # d1 = first trading day
    d1 = rows[0] if len(rows) >= 1 else None
    d2 = rows[1] if len(rows) >= 2 else None
    d3 = rows[2] if len(rows) >= 3 else None
    d4 = rows[3] if len(rows) >= 4 else None
    d5 = rows[4] if len(rows) >= 5 else None
    d10 = rows[9] if len(rows) >= 10 else None
    d20 = rows[19] if len(rows) >= 20 else None

    # Calculate lifetime high/low from all data
    lifetime_high = max(r[2] for r in rows)  # max of high
    lifetime_low = min(r[3] for r in rows if r[3] > 0)  # min of low where > 0

    # Get most recent close as last_price
    last_close = rows[-1][4]

    # d1-d5 highs for first 5 days
    d5_highs = [r[2] for r in rows[:5]]
    d5_lows = [r[3] for r in rows[:5] if r[3] > 0]
    first_5d_high = max(d5_highs) if d5_highs else 0
    first_5d_low = min(d5_lows) if d5_lows else 0

    # d10, d20 ranges
    d10_highs = [r[2] for r in rows[:10]]
    d10_lows = [r[3] for r in rows[:10] if r[3] > 0]
    d20_highs = [r[2] for r in rows[:20]]
    d20_lows = [r[3] for r in rows[:20] if r[3] > 0]

    d10_high = max(d10_highs) if d10_highs else 0
    d10_low = min(d10_lows) if d10_lows else 0
    d20_high = max(d20_highs) if d20_highs else 0
    d20_low = min(d20_lows) if d20_lows else 0

    # Build update values
    updates = []

    if d1:
        updates.extend([
            f"d1_open = {d1[1]}",
            f"d1_high = {d1[2]}",
            f"d1_low = {d1[3]}",
            f"d1_close = {d1[4]}",
            f"d1_volume = {d1[5]}",
        ])

    if d2:
        updates.extend([
            f"d2_open = {d2[1]}",
            f"d2_high = {d2[2]}",
            f"d2_low = {d2[3]}",
            f"d2_close = {d2[4]}",
            f"d2_volume = {d2[5]}",
        ])

    if d3:
        updates.extend([
            f"d3_open = {d3[1]}",
            f"d3_high = {d3[2]}",
            f"d3_low = {d3[3]}",
            f"d3_close = {d3[4]}",
            f"d3_volume = {d3[5]}",
        ])

    if d4:
        updates.extend([
            f"d4_open = {d4[1]}",
            f"d4_high = {d4[2]}",
            f"d4_low = {d4[3]}",
            f"d4_close = {d4[4]}",
            f"d4_volume = {d4[5]}",
        ])

    if d5:
        updates.extend([
            f"d5_open = {d5[1]}",
            f"d5_high = {d5[2]}",
            f"d5_low = {d5[3]}",
            f"d5_close = {d5[4]}",
            f"d5_volume = {d5[5]}",
        ])

    if d10:
        updates.extend([
            f"d10_open = {d10[1]}",
            f"d10_high = {d10_high}",
            f"d10_low = {d10_low}",
            f"d10_close = {d10[4]}",
            f"d10_volume = {d10[5]}",
        ])

    if d20:
        updates.extend([
            f"d20_open = {d20[1]}",
            f"d20_high = {d20_high}",
            f"d20_low = {d20_low}",
            f"d20_close = {d20[4]}",
            f"d20_volume = {d20[5]}",
        ])

    # Always update these
    updates.extend([
        f"lifetime_high = {lifetime_high}",
        f"lifetime_high_unadj = {lifetime_high}",
        f"last_price_adj = {last_close}",
        f"last_price_unadj = {last_close}",
        f"first_5d_high = {first_5d_high}",
        f"first_5d_low = {first_5d_low}",
        f"updated_at = now()",
    ])

    # Calculate returns if we have IPO price
    ipo_price_result = client.query(f"SELECT ipo_price FROM ipo_master WHERE polygon_ticker = '{ticker}'")
    if ipo_price_result.result_rows and ipo_price_result.result_rows[0][0] > 0:
        ipo_price = ipo_price_result.result_rows[0][0]

        if d1:
            ret_d1 = ((d1[4] - ipo_price) / ipo_price) * 100
            updates.append(f"ret_d1 = {ret_d1}")

        if d5:
            ret_d5 = ((d5[4] - ipo_price) / ipo_price) * 100
            updates.append(f"ret_d5 = {ret_d5}")

        if d10:
            ret_d10 = ((d10[4] - ipo_price) / ipo_price) * 100
            updates.append(f"ret_d10 = {ret_d10}")

        if d20:
            ret_d20 = ((d20[4] - ipo_price) / ipo_price) * 100
            updates.append(f"ret_d20 = {ret_d20}")

        # lifetime_hi_vs_ipo
        lifetime_hi_vs_ipo = ((lifetime_high - ipo_price) / ipo_price) * 100
        updates.append(f"lifetime_hi_vs_ipo = {lifetime_hi_vs_ipo}")

    if dry_run:
        print(f"    Would update: d1_high={d1[2] if d1 else 0}, lifetime_high={lifetime_high}, last_price={last_close}")
        return True

    # Execute update
    update_sql = f"""
        ALTER TABLE ipo_master UPDATE
            {', '.join(updates)}
        WHERE polygon_ticker = '{ticker}'
    """

    try:
        client.command(update_sql)
        print(f"    Updated: d1_high={d1[2] if d1 else 0}, lifetime_high={lifetime_high}, last_price={last_close}")
        return True
    except Exception as e:
        print(f"    Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Update ipo_master from pq_daily")
    parser.add_argument("--ticker", help="Specific ticker to update")
    parser.add_argument("--days", type=int, default=60, help="Look back N days (default: 60)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    if args.ticker:
        # Get IPO date for this ticker
        client = get_client()
        result = client.query(f"""
            SELECT ipo_date FROM ipo_master WHERE polygon_ticker = '{args.ticker.upper()}'
        """)
        if result.result_rows:
            update_ticker(args.ticker.upper(), result.result_rows[0][0], dry_run=args.dry_run)
        else:
            print(f"Ticker {args.ticker} not found in ipo_master")
    else:
        # Find all IPOs needing update
        ipos = get_ipos_needing_update(days_back=args.days)

        if not ipos:
            print(f"No IPOs needing update in the last {args.days} days")
            return

        print(f"Found {len(ipos)} IPOs needing ipo_master update:\n")

        updated = 0
        for ticker, ipo_date in ipos:
            if update_ticker(ticker, ipo_date, dry_run=args.dry_run):
                updated += 1

        print(f"\n{'Would update' if args.dry_run else 'Updated'} {updated}/{len(ipos)} IPOs")


if __name__ == "__main__":
    main()
