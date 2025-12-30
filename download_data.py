import argparse
import asyncio
import json
import os
import time
import io
import zipfile
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List, Any, Tuple

import aiohttp
import pandas as pd

COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "num_trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "taker_buy_base",
    "taker_buy_quote",
]

INT_COLUMNS = ["open_time", "close_time", "num_trades"]


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


class RateLimiter:
    def __init__(self, delay: float = 0.1):
        self.delay = delay
        self.last_call = 0.0
        self.blocked_until = 0.0
        self.lock = asyncio.Lock()

    async def wait(self):
        async with self.lock:
            while True:
                now = time.monotonic()
                if now < self.blocked_until:
                    sleep_time = self.blocked_until - now
                    await asyncio.sleep(sleep_time)
                    continue
                
                elapsed = now - self.last_call
                wait_time = self.delay - elapsed
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                
                self.last_call = time.monotonic()
                break

    def block(self, duration: float) -> bool:
        """Blocks the limiter. Returns True if this call extended the block."""
        now = time.monotonic()
        target = now + duration
        if target > self.blocked_until:
            self.blocked_until = target
            return True
        return False


class ZipNotFoundError(Exception):
    pass


class ZipDataError(Exception):
    pass


def interval_to_millis(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == "m":
        return value * 60 * 1000
    if unit == "h":
        return value * 60 * 60 * 1000
    if unit == "d":
        return value * 24 * 60 * 60 * 1000
    if unit == "w":
        return value * 7 * 24 * 60 * 60 * 1000
    raise ValueError(f"Unsupported interval: {interval}")


def get_zip_gaps(
    df: pd.DataFrame,
    day_start: int,
    day_end: int,
    interval_ms: int,
) -> Optional[Tuple[List[Tuple[int, int]], int]]:
    if "open_time" not in df.columns:
        return None
    times = pd.to_numeric(df["open_time"], errors="coerce").dropna()
    if times.empty:
        return None
    times = times[(times >= day_start) & (times <= day_end)]
    if times.empty:
        return None

    times = times.astype("int64").sort_values().drop_duplicates()
    gaps: List[Tuple[int, int]] = []
    full_day = ((day_end - day_start + 1) % interval_ms == 0)

    if full_day:
        expected_last = day_start + ((day_end - day_start + 1) // interval_ms - 1) * interval_ms
        first_open = int(times.iloc[0])
        if first_open > day_start:
            gap_end = first_open - interval_ms
            if gap_end >= day_start:
                gaps.append((day_start, gap_end))

    diffs = times.values[1:] - times.values[:-1]
    gap_indices = (diffs > interval_ms).nonzero()[0]
    for i in gap_indices:
        gap_start = int(times.values[i] + interval_ms)
        gap_end = int(times.values[i + 1] - interval_ms)
        if gap_end >= gap_start:
            gaps.append((gap_start, gap_end))

    if full_day:
        last_open = int(times.iloc[-1])
        if last_open < expected_last:
            gap_start = last_open + interval_ms
            if gap_start <= expected_last:
                gaps.append((gap_start, expected_last))

    return gaps, int(times.iloc[0])


def is_month_key(value: str) -> bool:
    return (
        len(value) == 7
        and value[4] == "-"
        and value[:4].isdigit()
        and value[5:7].isdigit()
    )


def get_partition_dir(feather_dir: str, symbol: str, interval: str) -> str:
    return os.path.join(feather_dir, symbol, interval)


def list_partition_files(partition_dir: str) -> List[str]:
    if not os.path.isdir(partition_dir):
        return []
    files = []
    for name in os.listdir(partition_dir):
        if not name.endswith(".feather"):
            continue
        stem = name[:-8]
        if not is_month_key(stem):
            continue
        files.append(os.path.join(partition_dir, name))
    return sorted(files)


def normalize_open_time(df: pd.DataFrame) -> pd.DataFrame:
    if "open_time" not in df.columns:
        raise RuntimeError("Dataframe missing open_time column.")
    df = df.copy()
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df = df.dropna(subset=["open_time"])
    if not df.empty:
        df["open_time"] = df["open_time"].astype("int64")
    return df


def load_partition_open_times(partition_files: List[str]) -> Optional[List[int]]:
    if not partition_files:
        return None
    series_list = []
    for path in partition_files:
        df = pd.read_feather(path, columns=["open_time"])
        if "open_time" not in df.columns:
            raise RuntimeError(f"Missing open_time column in partition: {path}")
        series_list.append(pd.to_numeric(df["open_time"], errors="coerce"))
    times = pd.concat(series_list, ignore_index=True).dropna()
    if times.empty:
        return None
    times = times.astype("int64").drop_duplicates().sort_values()
    return times.to_list()


def write_month_partitions(df: pd.DataFrame, partition_dir: str) -> int:
    df = normalize_open_time(df)
    if df.empty:
        return 0
    os.makedirs(partition_dir, exist_ok=True)
    month_keys = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m")
    df = df.copy()
    df["_month_key"] = month_keys
    updated = 0

    for month_key, month_df in df.groupby("_month_key"):
        month_path = os.path.join(partition_dir, f"{month_key}.feather")
        month_df = month_df.drop(columns=["_month_key"])
        month_df = month_df.drop_duplicates(subset=["open_time"])
        if os.path.exists(month_path):
            existing = pd.read_feather(month_path)
            existing = normalize_open_time(existing)
            if existing.empty:
                combined = month_df.sort_values("open_time")
            else:
                combined = pd.concat([existing, month_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["open_time"], keep="last").sort_values("open_time")
        else:
            combined = month_df.sort_values("open_time")
        combined["open_time_dt"] = (
            pd.to_datetime(combined["open_time"], unit="ms", utc=True).dt.tz_convert(None)
        )
        combined.reset_index(drop=True, inplace=True)
        combined.to_feather(month_path)
        updated += 1

    return updated


def merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    ordered = [(start, end) for start, end in ranges if start <= end]
    if not ordered:
        return []
    ordered.sort()
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        last = merged[-1]
        if start <= last[1] + 1:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def ranges_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return not (end_a < start_b or start_a > end_b)


def is_in_ranges(values: pd.Series, ranges: List[Tuple[int, int]]) -> pd.Series:
    if not ranges:
        return pd.Series(False, index=values.index)
    mask = pd.Series(False, index=values.index)
    for start, end in ranges:
        mask |= (values >= start) & (values <= end)
    return mask


def get_refresh_ranges(start_ts: int, end_ts: int) -> List[Tuple[int, int]]:
    if end_ts < start_ts:
        return []

    end_dt = datetime.fromtimestamp(end_ts / 1000, timezone.utc)
    day_start = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    day_start_ms = int(day_start.timestamp() * 1000)
    prev_day_start = day_start - timedelta(days=1)
    prev_day_start_ms = int(prev_day_start.timestamp() * 1000)
    prev_day_end_ms = day_start_ms - 1

    current_month_start = day_start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    current_month_start_ms = int(current_month_start.timestamp() * 1000)
    prev_month_end = current_month_start - timedelta(milliseconds=1)
    prev_month_start = prev_month_end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    prev_month_start_ms = int(prev_month_start.timestamp() * 1000)
    prev_month_end_ms = int(prev_month_end.timestamp() * 1000)

    ranges = []
    ranges.append((max(day_start_ms, start_ts), end_ts))
    if prev_day_end_ms >= start_ts:
        ranges.append((max(prev_day_start_ms, start_ts), min(prev_day_end_ms, end_ts)))
    ranges.append((max(current_month_start_ms, start_ts), end_ts))
    if prev_month_end_ms >= start_ts:
        ranges.append((max(prev_month_start_ms, start_ts), min(prev_month_end_ms, end_ts)))

    return merge_ranges(ranges)


async def fetch_zip_data(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    date_str: str,
    cache_dir: str,
    semaphore: asyncio.Semaphore,
    force_download: bool,
) -> Optional[pd.DataFrame]:
    """
    Downloads and extracts kline data from Binance Vision daily archives, 
    caching the zip files locally to avoid redundant downloads.
    """
    zip_filename = f"{symbol}-{interval}-{date_str}.zip"
    cache_path = os.path.join(cache_dir, zip_filename)
    
    if not force_download and os.path.exists(cache_path):
        try:
            with zipfile.ZipFile(cache_path) as z:
                csv_filename = f"{symbol}-{interval}-{date_str}.csv"
                with z.open(csv_filename) as f:
                    return pd.read_csv(f, names=COLUMNS, header=None)
        except Exception:
            os.remove(cache_path)

    async with semaphore:
        url = f"https://data.binance.vision/data/futures/um/daily/klines/{symbol}/{interval}/{zip_filename}"
        try:
            async with session.get(url) as response:
                if response.status == 404:
                    raise ZipNotFoundError(f"Zip not found: {zip_filename}")
                if response.status in (429, 418):
                    return None
                if response.status != 200:
                    return None
                
                content = await response.read()
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_path, "wb") as f:
                    f.write(content)
                    
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    csv_filename = f"{symbol}-{interval}-{date_str}.csv"
                    with z.open(csv_filename) as f:
                        return pd.read_csv(f, names=COLUMNS, header=None)
        except ZipNotFoundError:
            raise
        except Exception:
            return None


async def fetch_klines(
    session: aiohttp.ClientSession,
    base_url: str,
    symbol: str,
    interval: str,
    start_time: Optional[int],
    end_time: Optional[int],
    limit: int,
    limiter: RateLimiter,
    semaphore: asyncio.Semaphore,
) -> List[Any]:
    params = {"symbol": symbol, "interval": interval, "limit": str(limit)}
    if start_time is not None:
        params["startTime"] = str(int(start_time))
    if end_time is not None:
        params["endTime"] = str(int(end_time))

    retry_delay = 1.0
    max_retries = 10

    for attempt in range(max_retries):
        async with semaphore:
            await limiter.wait()
            try:
                async with session.get(f"{base_url}/fapi/v1/klines", params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    
                    if response.status in (429, 418):
                        data = {}
                        try:
                            data = await response.json()
                        except Exception:
                            pass
                        
                        body_retry_after = data.get("retryAfter") or \
                                         data.get("data", {}).get("retryAfter") or \
                                         data.get("error", {}).get("data", {}).get("retryAfter")
                        
                        if body_retry_after:
                            sleep_time = (body_retry_after - int(time.time() * 1000)) / 1000
                        else:
                            retry_after_hdr = response.headers.get("Retry-After")
                            sleep_time = float(retry_after_hdr) if retry_after_hdr else retry_delay
                        
                        sleep_time = max(sleep_time, 1.0)
                        msg_type = "IP BAN" if response.status == 418 else "RATE LIMIT"
                        if limiter.block(sleep_time):
                            print(f"{msg_type} detected. Cooling down for {sleep_time:.1f}s...")
                        
                        retry_delay *= 2
                        continue
                    
                    print(
                        f"API error fetching {symbol} {interval} "
                        f"[{start_time}-{end_time}]: {response.status}"
                    )
                    response.raise_for_status()

            except aiohttp.ClientError:
                if attempt == max_retries - 1: raise
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            except Exception:
                if attempt == max_retries - 1: raise
                await asyncio.sleep(retry_delay)
                retry_delay *= 2

    return []


def klines_to_frame(klines: Any) -> pd.DataFrame:
    if isinstance(klines, pd.DataFrame):
        df = klines
    else:
        df = pd.DataFrame(klines, columns=COLUMNS)
        
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in INT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce", downcast="integer")

    df["open_time_dt"] = (
        pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
    )
    if "ignore" in df.columns:
        df.drop(columns=["ignore"], inplace=True)
    return df


async def update_symbol_data(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    feather_dir: str,
    cache_dir: str,
    base_url: str,
    limit: int,
    limiter: RateLimiter,
    semaphore: asyncio.Semaphore,
    start_ts: int,
    batch_size: int,
) -> Optional[pd.DataFrame]:
    interval_ms = interval_to_millis(interval)
    batch_size = max(1, batch_size)
    
    now_utc = datetime.now(timezone.utc)
    today_start_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_yesterday_ms = int(today_start_utc.timestamp() * 1000) - 1

    partition_dir = get_partition_dir(feather_dir, symbol, interval)
    legacy_path = os.path.join(feather_dir, f"{symbol}_{interval}.feather")
    existing_times: Optional[List[int]] = None

    partition_files = list_partition_files(partition_dir)
    if partition_files:
        existing_times = load_partition_open_times(partition_files)
    elif os.path.exists(legacy_path):
        try:
            legacy_df = pd.read_feather(legacy_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to read existing data: {legacy_path}") from exc
        legacy_df = normalize_open_time(legacy_df)
        if not legacy_df.empty:
            legacy_df = legacy_df.drop_duplicates(subset=["open_time"]).sort_values("open_time")
            migrated_months = write_month_partitions(legacy_df, partition_dir)
            print(f"{symbol} {interval}: migrated legacy data to {migrated_months} monthly partitions.")
            existing_times = legacy_df["open_time"].to_list()

    missing_ranges = []

    existing_min_open: Optional[int] = None
    if existing_times:
        existing_times = pd.Series(existing_times, dtype="int64").to_numpy()
        existing_times.sort()
        min_open = int(existing_times[0])
        max_open = int(existing_times[-1])
        existing_min_open = min_open

        if start_ts < min_open:
            missing_ranges.append((start_ts, min_open - 1))

        diffs = existing_times[1:] - existing_times[:-1]
        gap_indices = (diffs > interval_ms).nonzero()[0]
        for i in gap_indices:
            gap_start = int(existing_times[i] + interval_ms)
            gap_end = int(existing_times[i + 1] - interval_ms)
            if gap_end >= gap_start:
                missing_ranges.append((gap_start, gap_end))

        next_needed = max_open + interval_ms
        if next_needed < end_of_yesterday_ms:
            missing_ranges.append((next_needed, end_of_yesterday_ms))
    else:
        if start_ts < end_of_yesterday_ms:
            missing_ranges.append((start_ts, end_of_yesterday_ms))

    refresh_ranges = get_refresh_ranges(start_ts, end_of_yesterday_ms)
    if refresh_ranges:
        missing_ranges.extend(refresh_ranges)
        missing_ranges = merge_ranges(missing_ranges)

    if not missing_ranges:
        print(f"{symbol} {interval}: all data present up to yesterday.")
        return None

    zip_requests = []
    api_ranges = []
    
    for start, end in missing_ranges:
        curr_start = start
        while curr_start <= end:
            dt_start = datetime.fromtimestamp(curr_start / 1000, timezone.utc)
            dt_next_midnight = (dt_start + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            next_midnight_ms = int(dt_next_midnight.timestamp() * 1000)
            day_end_ms = next_midnight_ms - 1
            
            if day_end_ms <= end and dt_start.hour == 0 and dt_start.minute == 0:
                date_str = dt_start.strftime("%Y-%m-%d")
                force_download = any(
                    ranges_overlap(curr_start, day_end_ms, r_start, r_end)
                    for r_start, r_end in refresh_ranges
                )
                zip_requests.append((date_str, curr_start, day_end_ms, force_download))
                curr_start = next_midnight_ms
            else:
                api_end = min(day_end_ms, end)
                api_ranges.append((curr_start, api_end))
                curr_start = api_end + 1

    all_dfs = []
    if zip_requests:
        print(f"{symbol} {interval}: checking {len(zip_requests)} daily zip archives...")
        zip_failures = []
        zip_data_starts = []
        zip_gap_ranges = []
        zip_partials = 0
        for i in range(0, len(zip_requests), batch_size):
            batch = zip_requests[i:i + batch_size]
            zip_coros = [
                fetch_zip_data(session, symbol, interval, date_str, cache_dir, semaphore, force_download)
                for date_str, _, _, force_download in batch
            ]
            zip_results = await asyncio.gather(*zip_coros, return_exceptions=True)
            for (date_str, day_start, day_end, _), res_df in zip(batch, zip_results):
                if isinstance(res_df, Exception) or res_df is None:
                    zip_failures.append((date_str, day_start, day_end, res_df))
                    continue
                gaps_info = get_zip_gaps(res_df, day_start, day_end, interval_ms)
                if gaps_info is None:
                    zip_failures.append(
                        (date_str, day_start, day_end, ZipDataError(f"Invalid zip data: {date_str}"))
                    )
                    cache_path = os.path.join(cache_dir, f"{symbol}-{interval}-{date_str}.zip")
                    if os.path.exists(cache_path):
                        os.remove(cache_path)
                    continue

                gaps, first_open = gaps_info
                zip_data_starts.append(first_open)
                if gaps:
                    zip_partials += 1
                    zip_gap_ranges.extend(gaps)
                all_dfs.append(res_df)

        known_data_start = existing_min_open
        if zip_data_starts:
            earliest_zip_start = min(zip_data_starts)
            known_data_start = (
                earliest_zip_start
                if known_data_start is None
                else min(known_data_start, earliest_zip_start)
            )

        if zip_gap_ranges:
            filtered_gaps = []
            for gap_start, gap_end in zip_gap_ranges:
                if known_data_start is not None and gap_end < known_data_start:
                    continue
                if known_data_start is not None and gap_start < known_data_start:
                    gap_start = known_data_start
                filtered_gaps.append((gap_start, gap_end))
            api_ranges.extend(filtered_gaps)

        zip_fallbacks = 0
        zip_missing_skipped = 0
        for date_str, day_start, day_end, err in zip_failures:
            if isinstance(err, (ZipNotFoundError, ZipDataError)):
                if known_data_start is None or day_end < known_data_start:
                    zip_missing_skipped += 1
                    continue
            api_ranges.append((day_start, day_end))
            zip_fallbacks += 1

        if zip_fallbacks:
            print(f"{symbol} {interval}: {zip_fallbacks} daily zip archives failed; falling back to API.")
        if zip_partials:
            print(f"{symbol} {interval}: {zip_partials} zip day(s) had gaps; fetching missing ranges via API.")
        if zip_missing_skipped:
            print(f"{symbol} {interval}: skipping {zip_missing_skipped} missing/invalid zip day(s) at start of range.")

    if api_ranges:
        chunks = []
        step_ms = limit * interval_ms
        for start, end in api_ranges:
            curr = start
            while curr <= end:
                chunk_e = min(curr + step_ms - 1, end)
                chunks.append((curr, chunk_e))
                curr = chunk_e + 1
        
        if chunks:
            print(f"{symbol} {interval}: downloading {len(chunks)} fragments via API...")
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                tasks = [
                    fetch_klines(session, base_url, symbol, interval, s, e, limit, limiter, semaphore)
                    for s, e in batch
                ]
                api_results = await asyncio.gather(*tasks, return_exceptions=True)
                for (s, e), res in zip(batch, api_results):
                    if isinstance(res, Exception):
                        print(f"{symbol} {interval}: error fetching {s}-{e}: {res}")
                        continue
                    if res:
                        all_dfs.append(pd.DataFrame(res, columns=COLUMNS))

    if not all_dfs:
        return None

    new_df = pd.concat([klines_to_frame(df) for df in all_dfs], ignore_index=True)
    new_df = normalize_open_time(new_df)
    new_df = new_df.drop_duplicates(subset=["open_time"])
    if new_df.empty:
        return None

    if existing_times is not None and len(existing_times) > 0:
        existing_index = pd.Index(existing_times)
        if refresh_ranges:
            refresh_mask = is_in_ranges(new_df["open_time"], refresh_ranges)
            if refresh_mask.any():
                keep_mask = refresh_mask | ~new_df["open_time"].isin(existing_index)
                new_df = new_df[keep_mask]
            else:
                new_df = new_df[~new_df["open_time"].isin(existing_index)]
        else:
            new_df = new_df[~new_df["open_time"].isin(existing_index)]
        if new_df.empty:
            print(f"{symbol} {interval}: no new rows after deduplication.")
            return None

    updated_months = write_month_partitions(new_df, partition_dir)
    print(
        f"{symbol} {interval}: saved {len(new_df)} new rows across {updated_months} month(s) "
        f"to {partition_dir}."
    )
    return new_df


async def main_async(config_path: str, symbols_override: Optional[List[str]]) -> None:
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return

    interval = config.get("candle_interval", "1m")
    base_url = config.get("binance_base_url", "https://fapi.binance.com")
    limit = int(config.get("max_klines_per_request", 1500))
    request_delay = float(config.get("request_delay", 0.1))
    max_concurrency = max(1, int(config.get("max_concurrent_requests", 10)))
    batch_size = max(1, int(config.get("batch_size", max_concurrency * 5)))
    
    start_date_str = config.get("start_date", "2018-01-01")
    try:
        dt = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_ts = int(dt.timestamp() * 1000)
    except ValueError:
        start_ts = 1514764800000 

    data_dir = config.get("data_dir", "data")
    feather_dir = config.get("feather_dir") or os.path.join(data_dir, "feather")
    cache_dir = os.path.join(data_dir, "cache", "zips")
    
    os.makedirs(feather_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    if symbols_override:
        symbols = set(symbols_override)
    else:
        config_symbols = config.get("train_symbols") or config.get("symbols") or ["BTCUSDT"]
        symbols = set(config_symbols)

    print(
        f"Starting download: {len(symbols)} symbol(s), interval {interval}, "
        f"start {start_date_str}."
    )

    limiter = RateLimiter(delay=request_delay)
    semaphore = asyncio.Semaphore(max_concurrency)

    async with aiohttp.ClientSession() as session:
        symbols_sorted = sorted(symbols)
        tasks = [
            update_symbol_data(
                session, symbol, interval,
                feather_dir,
                cache_dir, base_url, limit, limiter, semaphore, start_ts, batch_size
            )
            for symbol in symbols_sorted
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        failures = [
            (symbol, result)
            for symbol, result in zip(symbols_sorted, results)
            if isinstance(result, Exception)
        ]
        if failures:
            print("Some symbols failed during download:")
            for symbol, err in failures:
                print(f"- {symbol}: {err}")

    print(f"Download complete for {len(symbols_sorted)} symbol(s).")


def main(config_path: str = "config.json", symbols_override: Optional[list] = None) -> None:
    asyncio.run(main_async(config_path, symbols_override))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Binance futures klines.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--symbols", nargs="*", help="Optional list of symbols.")
    args = parser.parse_args()
    main(args.config, args.symbols)
