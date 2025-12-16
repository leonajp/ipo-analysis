#!/usr/bin/env python3
"""
Comprehensive IPO Data Backfill System v4 - PARALLEL THREADED

Same as v4 but with parallel processing for much faster backfill.

Features:
- Configurable thread pool (default: 10 workers)
- Thread-safe rate limiting across all threads
- Batch DataFrame updates for efficiency
- Progress bar with ETA

Speed comparison (5000 tickers):
- Sequential v4: ~3 hours
- Parallel v4 (10 threads): ~20-30 minutes

Usage:
    # Default 10 threads
    python ipo_backfill_v4_parallel.py --input merged.csv --output complete.csv
    
    # More threads (faster but may hit rate limits)
    python ipo_backfill_v4_parallel.py --input merged.csv --workers 20
    
    # Fewer threads (safer for API limits)
    python ipo_backfill_v4_parallel.py --input merged.csv --workers 5
    
    # With all options
    python ipo_backfill_v4_parallel.py --input merged.csv --workers 15 --rate-limit 10 --limit 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from queue import Queue
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "XwKz5sDplukJRPvbdRtSjADlnWtmxedH")


# ============================================================================
# THREAD-SAFE RATE LIMITER
# ============================================================================

class RateLimiter:
    """Thread-safe rate limiter using token bucket algorithm."""
    
    def __init__(self, calls_per_second: float):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0.0
        self.lock = threading.Lock()
    
    def acquire(self):
        """Wait until we can make another call."""
        with self.lock:
            now = time.time()
            time_since_last = now - self.last_call
            
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)
            
            self.last_call = time.time()


# ============================================================================
# THREAD-SAFE STATS
# ============================================================================

class ThreadSafeStats:
    """Thread-safe statistics counter."""
    
    def __init__(self):
        self.lock = threading.Lock()
        self._stats = defaultdict(int)
    
    def increment(self, key: str, value: int = 1):
        with self.lock:
            self._stats[key] += value
    
    def get(self, key: str) -> int:
        with self.lock:
            return self._stats[key]
    
    def to_dict(self) -> dict:
        with self.lock:
            return dict(self._stats)


# ============================================================================
# THREAD-SAFE PROGRESS TRACKER
# ============================================================================

class ProgressTracker:
    """Thread-safe progress tracker with ETA."""
    
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
    
    def update(self, ticker: str = ""):
        with self.lock:
            self.completed += 1
            elapsed = time.time() - self.start_time
            rate = self.completed / elapsed if elapsed > 0 else 0
            remaining = self.total - self.completed
            eta_seconds = remaining / rate if rate > 0 else 0
            eta_minutes = eta_seconds / 60
            
            pct = (self.completed / self.total) * 100
            
            # Print progress
            sys.stdout.write(f"\r[{self.completed}/{self.total}] {pct:5.1f}% "
                           f"| {rate:.1f} tickers/sec | ETA: {eta_minutes:.1f} min "
                           f"| Current: {ticker:10}")
            sys.stdout.flush()
    
    def finish(self):
        elapsed = time.time() - self.start_time
        print(f"\n\nCompleted {self.completed} tickers in {elapsed/60:.1f} minutes")


# ============================================================================
# POLYGON CLIENT - THREAD SAFE
# ============================================================================

class PolygonClient:
    """Thread-safe Polygon.io API client."""
    
    BASE_URL = "https://api.polygon.io"
    
    def __init__(self, api_key: str = None, rate_limiter: RateLimiter = None):
        self.api_key = api_key or POLYGON_API_KEY
        self.rate_limiter = rate_limiter
        self._local = threading.local()
        self._call_count = 0
        self._count_lock = threading.Lock()
    
    @property
    def session(self) -> requests.Session:
        """Thread-local session."""
        if not hasattr(self._local, 'session'):
            self._local.session = requests.Session()
        return self._local.session
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def _request(self, url: str, params: dict = None, timeout: int = 15) -> dict:
        """Make API request with rate limiting."""
        if self.rate_limiter:
            self.rate_limiter.acquire()
        
        params = params or {}
        params["apiKey"] = self.api_key
        
        with self._count_lock:
            self._call_count += 1
        
        try:
            resp = self.session.get(url, params=params, timeout=timeout)
            
            if resp.status_code == 429:
                logger.warning("Rate limited, waiting 60s...")
                time.sleep(60)
                return self._request(url, params, timeout)
            
            if resp.status_code == 200:
                return resp.json()
            
        except Exception as e:
            logger.debug(f"Request error: {e}")
        
        return {}
    
    def get_daily_bars(self, ticker: str, from_date: str, to_date: str) -> List[dict]:
        url = f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
        data = self._request(url, {"adjusted": "true", "sort": "asc", "limit": 50})
        return data.get("results", []) or []
    
    def get_minute_bars(self, ticker: str, date: str) -> List[dict]:
        url = f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/range/1/minute/{date}/{date}"
        data = self._request(url, {"adjusted": "true", "sort": "asc", "limit": 50000})
        return data.get("results", []) or []
    
    def get_ticker_details(self, ticker: str, date: str = None) -> dict:
        url = f"{self.BASE_URL}/v3/reference/tickers/{ticker}"
        params = {"date": date} if date else {}
        data = self._request(url, params)
        return data.get("results", {})
    
    def get_ticker_events(self, ticker: str) -> List[dict]:
        url = f"{self.BASE_URL}/vX/reference/tickers/{ticker}/events"
        data = self._request(url)
        return data.get("results", {}).get("events", []) or []
    
    def get_splits(self, ticker: str) -> List[dict]:
        url = f"{self.BASE_URL}/v3/reference/splits"
        data = self._request(url, {"ticker": ticker, "limit": 100})
        return data.get("results", []) or []
    
    def get_previous_close(self, ticker: str) -> dict:
        url = f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/prev"
        data = self._request(url, {"adjusted": "true"})
        results = data.get("results", [])
        return results[0] if results else {}
    
    @property
    def call_count(self) -> int:
        with self._count_lock:
            return self._call_count


# ============================================================================
# BLOOMBERG CLIENT (Thread-safe)
# ============================================================================

class BloombergClient:
    """Thread-safe Bloomberg client."""
    
    def __init__(self):
        self.available = self._check_available()
        self.lock = threading.Lock()
    
    def _check_available(self) -> bool:
        try:
            from xbbg import blp
            return True
        except ImportError:
            return False
    
    def is_available(self) -> bool:
        return self.available
    
    def get_reference_data(self, ticker: str, fields: List[str]) -> dict:
        if not self.available:
            return {}
        
        from xbbg import blp
        
        if not ticker.endswith("Equity"):
            ticker = f"{ticker} US Equity"
        
        try:
            with self.lock:  # Bloomberg may not be thread-safe
                df = blp.bdp(ticker, fields)
                if not df.empty:
                    return df.iloc[0].to_dict()
        except Exception as e:
            logger.debug(f"Bloomberg error: {e}")
        
        return {}
    
    def get_ticker_changes(self, ticker: str) -> dict:
        if not self.available:
            return {}
        
        fields = ["PREV_TICKER", "TICKER_CHG_DT"]
        return self.get_reference_data(ticker, fields)


# ============================================================================
# SEC CLIENT (Thread-safe)
# ============================================================================

class SECClient:
    """Thread-safe SEC client."""
    
    def __init__(self, rate_limiter: RateLimiter = None):
        self.rate_limiter = rate_limiter
        self._local = threading.local()
        self._ticker_to_cik = None
        self._cik_lock = threading.Lock()
    
    @property
    def session(self) -> requests.Session:
        if not hasattr(self._local, 'session'):
            self._local.session = requests.Session()
            self._local.session.headers.update({
                "User-Agent": "IPOAnalysis research@example.com",
            })
        return self._local.session
    
    def is_available(self) -> bool:
        return True
    
    def get_cik(self, ticker: str) -> Optional[str]:
        with self._cik_lock:
            if self._ticker_to_cik is None:
                try:
                    url = "https://www.sec.gov/files/company_tickers.json"
                    resp = self.session.get(url, timeout=30)
                    if resp.status_code == 200:
                        data = resp.json()
                        self._ticker_to_cik = {
                            v["ticker"].upper(): str(v["cik_str"]).zfill(10)
                            for v in data.values()
                        }
                    else:
                        self._ticker_to_cik = {}
                except:
                    self._ticker_to_cik = {}
        
        return self._ticker_to_cik.get(ticker.upper())
    
    def check_vc_backed(self, ticker: str) -> Optional[bool]:
        if self.rate_limiter:
            self.rate_limiter.acquire()
        
        cik = self.get_cik(ticker)
        if not cik:
            return None
        
        try:
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            resp = self.session.get(url, timeout=15)
            
            if resp.status_code != 200:
                return None
            
            data = resp.json()
            forms = data.get("filings", {}).get("recent", {}).get("form", [])
            has_s1 = any(f in ["S-1", "S-1/A"] for f in forms)
            return has_s1
            
        except:
            return None


# ============================================================================
# BACKFILL WORKER
# ============================================================================

class BackfillWorker:
    """Worker that processes a single ticker."""
    
    OHLCV_COLS = {
        1: ["d1_open", "d1_high", "d1_low", "d1_close", "d1_volume"],
        2: ["d2_open", "d2_high", "d2_low", "d2_close", "d2_volume"],
        3: ["d3_open", "d3_high", "d3_low", "d3_close", "d3_volume"],
        4: ["d4_open", "d4_high", "d4_low", "d4_close", "d4_volume"],
        5: ["d5_open", "d5_high", "d5_low", "d5_close", "d5_volume"],
    }
    
    EXTENDED_COLS = ["first5d_hi", "first5d_lo", "30d_hi", "30d_lo", "30d_cls",
                    "d_until_30dhi", "d_until_30dlo"]
    
    INTRADAY_COLS = ["open_px", "first_trade_time", "1st_5m_hi", "1st_5m_lo", 
                    "1st_5m_cls", "1st_5m_cls_time", "5mClsOrUnhaltPx", "min_to_30mHi"]
    
    HALT_COLS = ["halted_d1", "num_halts", "num_halts_d1", "num_h_1st_30m",
                "time_until_1halt", "first_halt_start", "first_halt_end",
                "1st_unhalt_px", "halt_windows"]
    
    COMPANY_COLS = ["GICS Sector", "GICS Ind Name", "float_shares", 
                   "market_cap_ipo", "IPO Lead"]
    
    RETURN_COLS = ["open_premium_pct", "ret_d1", "ret_d5", "ret_d30",
                  "IPO Offer Px 1st Opn Px % Chg", "pctChg_since_ipo_open"]
    
    PRICE_TRACKING_COLS = ["Lifetime High", "last_px_unadj", "last_px_adj",
                          "cum_split_factor_since_base", "price_change_since_ipo_pct"]
    
    TICKER_CHANGE_COLS = ["prevTkr", "TkrChgDt"]
    
    VC_COLS = ["vc_backed"]
    
    def __init__(self, polygon: PolygonClient, bloomberg: BloombergClient, 
                 sec: SECClient, config: dict):
        self.polygon = polygon
        self.bloomberg = bloomberg
        self.sec = sec
        self.config = config
    
    def process_ticker(self, ticker: str, info: dict, row_data: dict) -> Tuple[str, int, Dict[str, Any]]:
        """
        Process a single ticker and return results.
        
        Returns: (ticker, row_idx, filled_data)
        """
        result = {}
        ipo_date = info["ipo_date"]
        missing_cols = info["missing"]
        row_idx = info["row_idx"]
        
        # Determine what to fill
        need_ohlcv = any(c.startswith("d") and "_" in c for c in missing_cols)
        need_intraday = any(c in self.INTRADAY_COLS for c in missing_cols) and self.config.get("backfill_intraday", True)
        need_halts = any(c in self.HALT_COLS for c in missing_cols) and self.config.get("backfill_halts", True)
        need_company = any(c in self.COMPANY_COLS for c in missing_cols)
        need_returns = any(c in self.RETURN_COLS for c in missing_cols)
        need_price_tracking = any(c in self.PRICE_TRACKING_COLS for c in missing_cols) and self.config.get("backfill_price_tracking", True)
        need_ticker_changes = any(c in self.TICKER_CHANGE_COLS for c in missing_cols) and self.config.get("backfill_ticker_changes", True)
        need_vc = any(c in self.VC_COLS for c in missing_cols) and self.config.get("backfill_vc", True)
        
        # 1. OHLCV
        if need_ohlcv:
            ohlcv = self._backfill_ohlcv(ticker, ipo_date)
            result.update(ohlcv)
        
        # 2. Intraday
        minute_bars = None
        if need_intraday or need_halts:
            minute_bars = self.polygon.get_minute_bars(ticker, ipo_date)
        
        if need_intraday and minute_bars:
            intraday = self._analyze_intraday(minute_bars)
            result.update(intraday)
        
        # 3. Halts
        if need_halts and minute_bars:
            halts = self._analyze_halts(minute_bars)
            result.update(halts)
        
        # 4. Company
        if need_company:
            company = self._backfill_company(ticker)
            result.update(company)
        
        # 5. Price tracking
        if need_price_tracking:
            ipo_price = row_data.get("IPO Sh Px")
            price_data = self._backfill_price_tracking(ticker, ipo_date, ipo_price)
            result.update(price_data)
        
        # 6. Ticker changes
        if need_ticker_changes:
            ticker_changes = self._backfill_ticker_changes(ticker)
            result.update(ticker_changes)
        
        # 7. VC backing
        if need_vc:
            vc_backed = self.sec.check_vc_backed(ticker)
            if vc_backed is not None:
                result["vc_backed"] = vc_backed
        
        # 8. Calculate returns
        if need_returns:
            returns = self._calculate_returns(row_data, result)
            result.update(returns)
        
        return ticker, row_idx, result
    
    def _backfill_ohlcv(self, ticker: str, ipo_date: str) -> dict:
        result = {}
        
        try:
            ipo_dt = datetime.strptime(ipo_date, "%Y-%m-%d")
            end_dt = ipo_dt + timedelta(days=45)
            
            bars = self.polygon.get_daily_bars(ticker, ipo_date, end_dt.strftime("%Y-%m-%d"))
            
            if not bars:
                return {}
            
            # D1-D5
            for i, bar in enumerate(bars[:5]):
                day = i + 1
                result[f"d{day}_open"] = bar.get("o")
                result[f"d{day}_high"] = bar.get("h")
                result[f"d{day}_low"] = bar.get("l")
                result[f"d{day}_close"] = bar.get("c")
                result[f"d{day}_volume"] = bar.get("v")
            
            # First 5d hi/lo
            first5 = bars[:min(5, len(bars))]
            if first5:
                result["first5d_hi"] = max(b.get("h", 0) for b in first5)
                result["first5d_lo"] = min(b.get("l", float("inf")) for b in first5)
            
            # 30d stats
            if len(bars) >= 20:
                first30 = bars[:min(30, len(bars))]
                
                highs = [(i, b.get("h", 0)) for i, b in enumerate(first30)]
                lows = [(i, b.get("l", float("inf"))) for i, b in enumerate(first30)]
                
                max_high = max(highs, key=lambda x: x[1])
                min_low = min(lows, key=lambda x: x[1])
                
                result["30d_hi"] = max_high[1]
                result["30d_lo"] = min_low[1]
                result["30d_cls"] = first30[-1].get("c")
                result["d_until_30dhi"] = max_high[0]
                result["d_until_30dlo"] = min_low[0]
            
        except Exception as e:
            logger.debug(f"OHLCV error for {ticker}: {e}")
        
        return result
    
    def _analyze_intraday(self, bars: List[dict]) -> dict:
        result = {}
        
        if not bars:
            return result
        
        # Convert timestamps
        for bar in bars:
            bar["dt"] = datetime.fromtimestamp(bar["t"] / 1000)
        
        # Filter to market hours
        market_bars = [
            b for b in bars 
            if (b["dt"].hour > 9 or (b["dt"].hour == 9 and b["dt"].minute >= 30))
            and b["dt"].hour < 16
        ]
        
        if not market_bars:
            return result
        
        # First trade
        first_bar = market_bars[0]
        result["open_px"] = first_bar["o"]
        result["first_trade_time"] = first_bar["dt"].strftime("%H:%M:%S")
        
        # First 5 minutes
        first_trade_time = first_bar["dt"]
        five_min_end = first_trade_time + timedelta(minutes=5)
        
        first_5m_bars = [b for b in market_bars if b["dt"] < five_min_end]
        
        if first_5m_bars:
            result["1st_5m_hi"] = max(b["h"] for b in first_5m_bars)
            result["1st_5m_lo"] = min(b["l"] for b in first_5m_bars)
            result["1st_5m_cls"] = first_5m_bars[-1]["c"]
            result["1st_5m_cls_time"] = first_5m_bars[-1]["dt"].strftime("%H:%M:%S")
            result["5mClsOrUnhaltPx"] = first_5m_bars[-1]["c"]
        
        # First 30 minutes high
        thirty_min_end = first_trade_time + timedelta(minutes=30)
        first_30m_bars = [b for b in market_bars if b["dt"] < thirty_min_end]
        
        if first_30m_bars:
            max_high = max(first_30m_bars, key=lambda b: b["h"])
            min_from_open = (max_high["dt"] - first_trade_time).total_seconds() / 60
            result["min_to_30mHi"] = round(min_from_open, 1)
        
        return result
    
    def _analyze_halts(self, bars: List[dict]) -> dict:
        result = {
            "halted_d1": False,
            "num_halts": 0,
            "num_halts_d1": 0,
            "num_h_1st_30m": 0,
        }
        
        if not bars:
            return result
        
        # Convert timestamps
        for bar in bars:
            if "dt" not in bar:
                bar["dt"] = datetime.fromtimestamp(bar["t"] / 1000)
        
        # Filter market hours
        market_bars = [
            b for b in bars 
            if (b["dt"].hour > 9 or (b["dt"].hour == 9 and b["dt"].minute >= 30))
            and b["dt"].hour < 16
        ]
        
        if len(market_bars) < 2:
            return result
        
        # Detect halts
        halts = []
        first_trade_time = market_bars[0]["dt"]
        thirty_min_mark = first_trade_time + timedelta(minutes=30)
        
        for i in range(1, len(market_bars)):
            prev_bar = market_bars[i - 1]
            curr_bar = market_bars[i]
            
            gap_minutes = (curr_bar["dt"] - prev_bar["dt"]).total_seconds() / 60
            
            if gap_minutes > 5:  # Gap > 5 min = likely halt
                halts.append({
                    "start": prev_bar["dt"].strftime("%H:%M:%S"),
                    "end": curr_bar["dt"].strftime("%H:%M:%S"),
                    "duration_min": round(gap_minutes, 1),
                    "unhalt_px": curr_bar["o"],
                    "in_first_30m": prev_bar["dt"] < thirty_min_mark,
                })
        
        if halts:
            result["halted_d1"] = True
            result["num_halts"] = len(halts)
            result["num_halts_d1"] = len(halts)
            result["num_h_1st_30m"] = sum(1 for h in halts if h["in_first_30m"])
            
            first_halt = halts[0]
            result["first_halt_start"] = first_halt["start"]
            result["first_halt_end"] = first_halt["end"]
            result["1st_unhalt_px"] = first_halt["unhalt_px"]
            
            # Time until first halt
            try:
                halt_start = datetime.strptime(first_halt["start"], "%H:%M:%S")
                first_trade = datetime.strptime(result.get("first_trade_time", "09:30:00"), "%H:%M:%S")
                time_to_halt = (halt_start - first_trade).total_seconds() / 60
                result["time_until_1halt"] = round(max(0, time_to_halt), 1)
            except:
                pass
            
            result["halt_windows"] = json.dumps([
                {"start": h["start"], "end": h["end"], "duration": h["duration_min"]}
                for h in halts
            ])
        
        return result
    
    def _backfill_company(self, ticker: str) -> dict:
        result = {}
        
        # Try Bloomberg
        if self.bloomberg.is_available():
            fields = ["GICS_SECTOR_NAME", "GICS_INDUSTRY_NAME", "EQY_FLOAT", 
                     "CUR_MKT_CAP", "IPO_LEAD_MGR"]
            bbg_data = self.bloomberg.get_reference_data(ticker, fields)
            
            if bbg_data:
                result["GICS Sector"] = bbg_data.get("GICS_SECTOR_NAME")
                result["GICS Ind Name"] = bbg_data.get("GICS_INDUSTRY_NAME")
                result["float_shares"] = bbg_data.get("EQY_FLOAT")
                result["market_cap_ipo"] = bbg_data.get("CUR_MKT_CAP")
                result["IPO Lead"] = bbg_data.get("IPO_LEAD_MGR")
        
        # Fallback to Polygon
        if not result.get("GICS Sector"):
            details = self.polygon.get_ticker_details(ticker)
            if details:
                result["GICS Sector"] = details.get("sic_description")
                if details.get("share_class_shares_outstanding"):
                    result["float_shares"] = details.get("share_class_shares_outstanding")
                if details.get("market_cap"):
                    result["market_cap_ipo"] = details.get("market_cap")
        
        return result
    
    def _backfill_price_tracking(self, ticker: str, ipo_date: str, ipo_price: float) -> dict:
        result = {}
        
        # Current price
        prev_close = self.polygon.get_previous_close(ticker)
        if prev_close:
            result["last_px_adj"] = prev_close.get("c")
            result["last_px_unadj"] = prev_close.get("c")
        
        # Splits
        splits = self.polygon.get_splits(ticker)
        cum_split_factor = 1.0
        for split in splits:
            split_date = split.get("execution_date", "")
            if split_date >= ipo_date:
                split_from = split.get("split_from", 1)
                split_to = split.get("split_to", 1)
                if split_from and split_to:
                    cum_split_factor *= (split_to / split_from)
        
        result["cum_split_factor_since_base"] = round(cum_split_factor, 4)
        
        if result.get("last_px_adj") and cum_split_factor != 1.0:
            result["last_px_unadj"] = result["last_px_adj"] * cum_split_factor
        
        # Lifetime high
        today = datetime.now().strftime("%Y-%m-%d")
        bars = self.polygon.get_daily_bars(ticker, ipo_date, today)
        if bars:
            result["Lifetime High"] = max(b.get("h", 0) for b in bars)
        
        # Returns
        if ipo_price and ipo_price > 0 and result.get("last_px_adj"):
            result["price_change_since_ipo_pct"] = round(
                ((result["last_px_adj"] - ipo_price) / ipo_price) * 100, 2
            )
        
        return result
    
    def _backfill_ticker_changes(self, ticker: str) -> dict:
        result = {}
        
        # Try Bloomberg
        if self.bloomberg.is_available():
            bbg_data = self.bloomberg.get_ticker_changes(ticker)
            if bbg_data.get("PREV_TICKER"):
                result["prevTkr"] = bbg_data.get("PREV_TICKER")
                result["TkrChgDt"] = bbg_data.get("TICKER_CHG_DT")
                return result
        
        # Try Polygon
        events = self.polygon.get_ticker_events(ticker)
        for event in events:
            if event.get("type") == "ticker_change":
                result["prevTkr"] = event.get("ticker_change", {}).get("ticker")
                result["TkrChgDt"] = event.get("date")
                break
        
        return result
    
    def _calculate_returns(self, row_data: dict, filled: dict) -> dict:
        result = {}
        
        ipo_price = row_data.get("IPO Sh Px")
        open_px = filled.get("open_px") or filled.get("d1_open") or row_data.get("open_px")
        d1_close = filled.get("d1_close") or row_data.get("d1_close")
        d5_close = filled.get("d5_close") or row_data.get("d5_close")
        d30_close = filled.get("30d_cls") or row_data.get("30d_cls")
        
        if ipo_price and open_px and ipo_price > 0:
            premium = ((open_px - ipo_price) / ipo_price) * 100
            result["open_premium_pct"] = round(premium, 2)
            result["IPO Offer Px 1st Opn Px % Chg"] = round(premium, 2)
        
        if open_px and open_px > 0:
            if d1_close:
                result["ret_d1"] = round(((d1_close - open_px) / open_px) * 100, 2)
            if d5_close:
                result["ret_d5"] = round(((d5_close - open_px) / open_px) * 100, 2)
            if d30_close:
                result["ret_d30"] = round(((d30_close - open_px) / open_px) * 100, 2)
        
        last_px = filled.get("last_px_adj") or row_data.get("last_px_adj")
        if open_px and last_px and open_px > 0:
            result["pctChg_since_ipo_open"] = round(((last_px - open_px) / open_px) * 100, 2)
        
        shares = row_data.get(" IPO Sh Offered ")
        if ipo_price and shares and not pd.isna(shares) and not filled.get("market_cap_ipo"):
            result["market_cap_ipo"] = ipo_price * shares
        
        return result


# ============================================================================
# PARALLEL BACKFILL ENGINE
# ============================================================================

class ParallelBackfillEngine:
    """Parallel backfill engine using ThreadPoolExecutor."""
    
    ALL_COLS = [
        "d1_open", "d1_high", "d1_low", "d1_close", "d1_volume",
        "d2_open", "d2_high", "d2_low", "d2_close", "d2_volume",
        "d3_open", "d3_high", "d3_low", "d3_close", "d3_volume",
        "d4_open", "d4_high", "d4_low", "d4_close", "d4_volume",
        "d5_open", "d5_high", "d5_low", "d5_close", "d5_volume",
        "first5d_hi", "first5d_lo", "30d_hi", "30d_lo", "30d_cls",
        "d_until_30dhi", "d_until_30dlo",
        "open_px", "first_trade_time", "1st_5m_hi", "1st_5m_lo",
        "1st_5m_cls", "1st_5m_cls_time", "5mClsOrUnhaltPx", "min_to_30mHi",
        "halted_d1", "num_halts", "num_halts_d1", "num_h_1st_30m",
        "time_until_1halt", "first_halt_start", "first_halt_end",
        "1st_unhalt_px", "halt_windows",
        "GICS Sector", "GICS Ind Name", "float_shares", "market_cap_ipo", "IPO Lead",
        "open_premium_pct", "ret_d1", "ret_d5", "ret_d30",
        "IPO Offer Px 1st Opn Px % Chg", "pctChg_since_ipo_open",
        "Lifetime High", "last_px_unadj", "last_px_adj",
        "cum_split_factor_since_base", "price_change_since_ipo_pct",
        "prevTkr", "TkrChgDt", "vc_backed",
    ]
    
    def __init__(self, num_workers: int = 10, rate_limit: float = 10.0, config: dict = None):
        self.num_workers = num_workers
        self.rate_limiter = RateLimiter(rate_limit)
        self.config = config or {}
        
        # Initialize shared clients
        self.polygon = PolygonClient(rate_limiter=self.rate_limiter)
        self.bloomberg = BloombergClient()
        self.sec = SECClient(rate_limiter=self.rate_limiter)
        
        # Stats
        self.stats = ThreadSafeStats()
    
    def ensure_columns_exist(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.ALL_COLS:
            if col not in df.columns:
                df[col] = None
        return df
    
    def analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Find all missing data."""
        missing = {}
        today = datetime.now()
        
        df = self.ensure_columns_exist(df)
        
        for idx, row in df.iterrows():
            ticker = row.get("polygon_ticker", "")
            if not ticker or pd.isna(ticker):
                bbg_ticker = row.get("Ticker", "")
                if bbg_ticker and isinstance(bbg_ticker, str):
                    ticker = bbg_ticker.replace(" US Equity", "").strip()
            
            if not ticker:
                continue
            
            ipo_date = row.get("ipo_date")
            if pd.isna(ipo_date):
                ipo_date = row.get("IPO Dt")
            
            if pd.isna(ipo_date):
                continue
            
            try:
                if isinstance(ipo_date, str):
                    if "-" in ipo_date:
                        ipo_dt = datetime.strptime(ipo_date, "%Y-%m-%d")
                    else:
                        ipo_dt = datetime.strptime(ipo_date, "%m/%d/%Y")
                else:
                    ipo_dt = pd.to_datetime(ipo_date)
            except:
                continue
            
            days_since = (today - ipo_dt).days
            
            # Find missing columns
            ticker_missing = []
            for col in self.ALL_COLS:
                if pd.isna(row.get(col)):
                    # Skip 30d cols if too recent
                    if "30d" in col.lower() and days_since < 35:
                        continue
                    ticker_missing.append(col)
            
            if ticker_missing:
                missing[ticker] = {
                    "missing": ticker_missing,
                    "ipo_date": ipo_dt.strftime("%Y-%m-%d"),
                    "row_idx": idx,
                    "days_since": days_since,
                    "row_data": row.to_dict(),
                }
        
        return missing
    
    def backfill_dataframe(self, df: pd.DataFrame, tickers: List[str] = None,
                           limit: int = None) -> pd.DataFrame:
        """Backfill using parallel threads."""
        df = df.copy()
        df = self.ensure_columns_exist(df)
        
        # Analyze
        missing = self.analyze_missing_data(df)
        
        if tickers:
            missing = {k: v for k, v in missing.items() if k in tickers}
        
        if limit:
            missing = dict(list(missing.items())[:limit])
        
        if not missing:
            logger.info("No missing data to backfill")
            return df
        
        total = len(missing)
        logger.info(f"Starting parallel backfill with {self.num_workers} workers for {total} tickers")
        
        # Progress tracker
        progress = ProgressTracker(total)
        
        # Results storage
        results = {}
        
        # Create worker
        worker = BackfillWorker(self.polygon, self.bloomberg, self.sec, self.config)
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            futures = {}
            for ticker, info in missing.items():
                future = executor.submit(
                    worker.process_ticker,
                    ticker,
                    info,
                    info["row_data"]
                )
                futures[future] = ticker
            
            # Collect results as they complete
            for future in as_completed(futures):
                ticker = futures[future]
                
                try:
                    ticker, row_idx, filled = future.result()
                    results[row_idx] = filled
                    self.stats.increment("tickers_processed")
                    self.stats.increment("fields_filled", len([v for v in filled.values() if v is not None]))
                except Exception as e:
                    logger.debug(f"Error processing {ticker}: {e}")
                    self.stats.increment("errors")
                
                progress.update(ticker)
        
        progress.finish()
        
        # Apply results to DataFrame
        logger.info("Applying results to DataFrame...")
        for row_idx, filled in results.items():
            for col, value in filled.items():
                if value is not None and not pd.isna(value):
                    if col in df.columns and pd.isna(df.loc[row_idx, col]):
                        df.loc[row_idx, col] = value
        
        self.stats.increment("api_calls", self.polygon.call_count)
        
        return df
    
    def get_stats(self) -> dict:
        return self.stats.to_dict()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Parallel IPO Data Backfill v4")
    
    parser.add_argument("--input", type=str, required=True, help="Input CSV")
    parser.add_argument("--output", type=str, help="Output CSV")
    parser.add_argument("--tickers", type=str, help="Comma-separated tickers")
    parser.add_argument("--limit", type=int, help="Limit tickers")
    parser.add_argument("--dry-run", action="store_true", help="Analyze only")
    parser.add_argument("--workers", type=int, default=10, help="Number of threads (default: 10)")
    parser.add_argument("--rate-limit", type=float, default=10.0, help="API calls/sec (default: 10)")
    parser.add_argument("--skip-intraday", action="store_true")
    parser.add_argument("--skip-halts", action="store_true")
    parser.add_argument("--skip-vc", action="store_true")
    parser.add_argument("--skip-price-tracking", action="store_true")
    parser.add_argument("--skip-ticker-changes", action="store_true")
    
    args = parser.parse_args()
    
    config = {
        "backfill_intraday": not args.skip_intraday,
        "backfill_halts": not args.skip_halts,
        "backfill_vc": not args.skip_vc,
        "backfill_price_tracking": not args.skip_price_tracking,
        "backfill_ticker_changes": not args.skip_ticker_changes,
    }
    
    print("\n" + "=" * 70)
    print("PARALLEL IPO DATA BACKFILL v4")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Workers:     {args.workers}")
    print(f"  Rate limit:  {args.rate_limit} calls/sec")
    print(f"  Intraday:    {'✓' if config['backfill_intraday'] else '✗'}")
    print(f"  Halts:       {'✓' if config['backfill_halts'] else '✗'}")
    print(f"  VC:          {'✓' if config['backfill_vc'] else '✗'}")
    
    # Load
    if not os.path.exists(args.input):
        print(f"\nERROR: File not found: {args.input}")
        return
    
    print(f"\nLoading: {args.input}")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"Loaded {len(df)} rows")
    
    # Initialize engine
    engine = ParallelBackfillEngine(
        num_workers=args.workers,
        rate_limit=args.rate_limit,
        config=config
    )
    
    # Analyze
    missing = engine.analyze_missing_data(df)
    
    print(f"\n{'='*70}")
    print(f"MISSING DATA: {len(missing)} tickers")
    print("="*70)
    
    if args.dry_run:
        total = min(len(missing), args.limit) if args.limit else len(missing)
        est_time = total / (args.rate_limit / 3)  # ~3 calls per ticker
        print(f"\n[DRY RUN] Estimated time: {est_time/60:.1f} minutes for {total} tickers")
        return
    
    # Backfill
    tickers = args.tickers.split(",") if args.tickers else None
    df = engine.backfill_dataframe(df, tickers, args.limit)
    
    # Stats
    stats = engine.get_stats()
    print(f"\n{'='*70}")
    print("COMPLETE")
    print("="*70)
    print(f"  Tickers:  {stats.get('tickers_processed', 0)}")
    print(f"  Fields:   {stats.get('fields_filled', 0)}")
    print(f"  API:      {stats.get('api_calls', 0)}")
    print(f"  Errors:   {stats.get('errors', 0)}")
    
    # Save
    output_path = args.output or args.input.replace(".csv", "_complete.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()