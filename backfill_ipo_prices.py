#!/usr/bin/env python3
"""
Backfill OHLCV data for IPO tickers into pq_daily and daily_prepost tables.

Uses Polygon API to fetch historical daily data for tickers missing from the database.

Usage:
    python backfill_ipo_prices.py                    # Backfill all missing Jan 2026 IPOs
    python backfill_ipo_prices.py --ticker GCDT     # Backfill specific ticker
    python backfill_ipo_prices.py --days 30         # Look back 30 days for missing IPOs
    python backfill_ipo_prices.py --dry-run         # Preview without writing
"""

import argparse
import os
import time
from datetime import datetime, timedelta

import requests
import clickhouse_connect

# ClickHouse connection
CH_HOST = os.environ.get('CLICKHOUSE_HOST', 'i35q8zrtq4.us-east-2.aws.clickhouse.cloud')
CH_PASSWORD = os.environ.get('CLICKHOUSE_PASSWORD', '~AiDc7hJ7m1Bv')

# Polygon API
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'KF9TH0GU8NHST9PL')


def get_client():
    return clickhouse_connect.get_client(
        host=CH_HOST, port=8443, user='default', password=CH_PASSWORD,
        secure=True, database='market_data'
    )


def fetch_polygon_daily(ticker: str, start_date: str, end_date: str) -> list:
    """Fetch daily OHLCV data from Polygon API."""
    # Try a wider date range - some IPOs may have been listed before their official IPO date
    # or Polygon may have data from a slightly different start
    from datetime import datetime, timedelta

    # Parse start date and go back 30 days to catch any early trading
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        adjusted_start = (start_dt - timedelta(days=30)).strftime('%Y-%m-%d')
    except:
        adjusted_start = start_date

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{adjusted_start}/{end_date}"
    params = {
        'apiKey': POLYGON_API_KEY,
        'adjusted': 'false',  # Get raw prices
        'sort': 'asc',
        'limit': 50000
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Status can be 'OK', 'DELAYED', or 'SUCCESS'
        if data.get('results'):
            return data['results']
        return []
    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return []


def fetch_polygon_prepost(ticker: str, trade_date: str) -> dict:
    """Fetch pre/post market data from Polygon for a specific date."""
    # Get extended hours data using grouped daily endpoint
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{trade_date}/{trade_date}"
    params = {
        'apiKey': POLYGON_API_KEY,
        'adjusted': 'false',
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get('status') == 'OK' and data.get('results'):
            bar = data['results'][0]
            return {
                'open': bar.get('o', 0),
                'high': bar.get('h', 0),
                'low': bar.get('l', 0),
                'close': bar.get('c', 0),
                'volume': bar.get('v', 0),
                'vwap': bar.get('vw', 0),
                'transactions': bar.get('n', 0),
            }
        return {}
    except Exception as e:
        print(f"  Error fetching prepost for {ticker} on {trade_date}: {e}")
        return {}


def get_missing_ipos(days_back: int = 30) -> list:
    """Get IPO tickers that have no data in pq_daily."""
    client = get_client()

    cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

    result = client.query(f"""
        SELECT
            i.polygon_ticker,
            i.ipo_date
        FROM ipo_master i
        LEFT JOIN (
            SELECT symbol, count(*) as cnt
            FROM pq_daily
            GROUP BY symbol
        ) p ON i.polygon_ticker = p.symbol
        WHERE i.ipo_date >= '{cutoff_date}'
          AND i.polygon_ticker IS NOT NULL
          AND i.polygon_ticker != ''
          AND (p.cnt IS NULL OR p.cnt = 0)
        ORDER BY i.ipo_date DESC
    """)

    return [(row[0], str(row[1])) for row in result.result_rows]


def backfill_ticker(ticker: str, ipo_date: str, dry_run: bool = False) -> dict:
    """Backfill data for a single ticker into pq_daily and daily_prepost."""
    client = get_client()

    # Fetch from IPO date to today
    end_date = datetime.now().strftime('%Y-%m-%d')

    print(f"\nBackfilling {ticker} (IPO: {ipo_date})...")

    # Fetch daily OHLCV from Polygon
    bars = fetch_polygon_daily(ticker, ipo_date, end_date)

    if not bars:
        print(f"  No Polygon data available for {ticker}")
        return {'ticker': ticker, 'pq_daily': 0, 'prepost': 0}

    print(f"  Found {len(bars)} daily bars from Polygon")

    if dry_run:
        print(f"  DRY RUN - Would insert {len(bars)} rows")
        return {'ticker': ticker, 'pq_daily': len(bars), 'prepost': len(bars)}

    # Insert into pq_daily
    pq_rows = []
    for bar in bars:
        trade_dt = datetime.fromtimestamp(bar['t'] / 1000).date()  # Use date object, not string
        pq_rows.append((
            ticker,                    # symbol
            trade_dt,                  # trade_dt
            trade_dt,                  # tradeDate
            float(bar.get('o', 0)),    # open
            float(bar.get('h', 0)),    # high
            float(bar.get('l', 0)),    # low
            float(bar.get('c', 0)),    # close
            int(bar.get('v', 0)),      # volume
            int(bar.get('n', 0)),      # count (transactions)
            # Adjusted = raw for new IPOs (no splits yet)
            float(bar.get('o', 0)),    # adjO
            float(bar.get('h', 0)),    # adjH
            float(bar.get('l', 0)),    # adjL
            float(bar.get('c', 0)),    # adjC
            1.0,                       # adjFactor
        ))

    # Check which columns exist in pq_daily
    desc_result = client.query("DESCRIBE TABLE pq_daily")
    pq_columns = [row[0] for row in desc_result.result_rows]

    # Use available columns
    insert_cols = ['symbol', 'trade_dt', 'tradeDate', 'open', 'high', 'low', 'close',
                   'volume', 'count', 'adjO', 'adjH', 'adjL', 'adjC', 'adjFactor']
    insert_cols = [c for c in insert_cols if c in pq_columns]

    # Trim rows to match columns
    col_indices = {
        'symbol': 0, 'trade_dt': 1, 'tradeDate': 2, 'open': 3, 'high': 4,
        'low': 5, 'close': 6, 'volume': 7, 'count': 8, 'adjO': 9,
        'adjH': 10, 'adjL': 11, 'adjC': 12, 'adjFactor': 13
    }
    trimmed_rows = []
    for row in pq_rows:
        trimmed_rows.append(tuple(row[col_indices[c]] for c in insert_cols))

    try:
        client.insert('pq_daily', trimmed_rows, column_names=insert_cols)
        print(f"  Inserted {len(trimmed_rows)} rows into pq_daily")
    except Exception as e:
        print(f"  Error inserting into pq_daily: {e}")
        return {'ticker': ticker, 'pq_daily': 0, 'prepost': 0}

    # Insert into daily_prepost
    prepost_rows = []
    for bar in bars:
        trade_dt = datetime.fromtimestamp(bar['t'] / 1000).date()  # Use date object
        o, h, l, c = float(bar.get('o', 0)), float(bar.get('h', 0)), float(bar.get('l', 0)), float(bar.get('c', 0))
        v = int(bar.get('v', 0))
        vwap = float(bar.get('vw', 0))
        n = int(bar.get('n', 0))

        prepost_rows.append((
            ticker,      # symbol
            trade_dt,    # trade_dt
            # RTH raw
            o, h, l, c,  # rawO, rawH, rawL, rawC
            # RTH adjusted (same as raw for new IPOs)
            o, h, l, c,  # adjO, adjH, adjL, adjC
            v,           # volume
            vwap,        # vwap
            n,           # transactions
            # Pre/post market - use RTH values as placeholders
            # (actual pre/post would need intraday data)
            0.0, 0.0, 0.0, 0.0,  # premarket_open/high/low/close raw
            0.0, 0.0, 0.0, 0.0,  # premarket adjusted
            0.0, 0.0, 0.0, 0.0,  # afterhours raw
            0.0, 0.0, 0.0, 0.0,  # afterhours adjusted
        ))

    # Check daily_prepost columns
    desc_result = client.query("DESCRIBE TABLE daily_prepost")
    prepost_columns = [row[0] for row in desc_result.result_rows]

    # Basic columns that should exist
    prepost_insert_cols = ['symbol', 'trade_dt', 'rawO', 'rawH', 'rawL', 'rawC',
                           'adjO', 'adjH', 'adjL', 'adjC', 'volume', 'vwap', 'transactions']
    prepost_insert_cols = [c for c in prepost_insert_cols if c in prepost_columns]

    prepost_col_indices = {
        'symbol': 0, 'trade_dt': 1, 'rawO': 2, 'rawH': 3, 'rawL': 4, 'rawC': 5,
        'adjO': 6, 'adjH': 7, 'adjL': 8, 'adjC': 9, 'volume': 10, 'vwap': 11, 'transactions': 12
    }

    trimmed_prepost = []
    for row in prepost_rows:
        trimmed_prepost.append(tuple(row[prepost_col_indices[c]] for c in prepost_insert_cols))

    try:
        client.insert('daily_prepost', trimmed_prepost, column_names=prepost_insert_cols)
        print(f"  Inserted {len(trimmed_prepost)} rows into daily_prepost")
    except Exception as e:
        print(f"  Error inserting into daily_prepost: {e}")

    return {'ticker': ticker, 'pq_daily': len(pq_rows), 'prepost': len(trimmed_prepost)}


def main():
    parser = argparse.ArgumentParser(description="Backfill IPO prices to pq_daily and daily_prepost")
    parser.add_argument("--ticker", help="Specific ticker to backfill")
    parser.add_argument("--days", type=int, default=30, help="Look back N days for missing IPOs (default: 30)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    if args.ticker:
        # Get IPO date for this ticker
        client = get_client()
        result = client.query(f"""
            SELECT ipo_date FROM ipo_master
            WHERE polygon_ticker = '{args.ticker.upper()}'
        """)
        if result.result_rows:
            ipo_date = str(result.result_rows[0][0])
            backfill_ticker(args.ticker.upper(), ipo_date, dry_run=args.dry_run)
        else:
            print(f"Ticker {args.ticker} not found in ipo_master")
    else:
        # Find all missing IPOs
        missing = get_missing_ipos(days_back=args.days)

        if not missing:
            print(f"No IPOs missing from pq_daily in the last {args.days} days")
            return

        print(f"Found {len(missing)} IPOs missing from pq_daily:")
        for ticker, ipo_date in missing:
            print(f"  {ticker} (IPO: {ipo_date})")

        print("\nBackfilling...")

        results = []
        for ticker, ipo_date in missing:
            result = backfill_ticker(ticker, ipo_date, dry_run=args.dry_run)
            results.append(result)
            time.sleep(0.5)  # Rate limiting

        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        total_pq = sum(r['pq_daily'] for r in results)
        total_prepost = sum(r['prepost'] for r in results)
        print(f"Total rows inserted into pq_daily: {total_pq}")
        print(f"Total rows inserted into daily_prepost: {total_prepost}")


if __name__ == "__main__":
    main()
