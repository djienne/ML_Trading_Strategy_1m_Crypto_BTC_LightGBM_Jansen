import pandas as pd


def _build_volume_bars_single(df, volume_size):
    if df is None or df.empty:
        return df

    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataframe missing columns: {sorted(missing)}")

    df = df.sort_index()
    bars = []
    cum_volume = 0.0
    bar_open = None
    bar_high = None
    bar_low = None
    bar_close = None
    bar_start_time = None
    bar_count = 0

    for ts, row in df.iterrows():
        volume = row["volume"]
        if pd.isna(volume):
            continue

        if bar_open is None:
            bar_open = row["open"]
            bar_high = row["high"]
            bar_low = row["low"]
            bar_start_time = ts
            cum_volume = 0.0
            bar_count = 0

        bar_high = max(bar_high, row["high"])
        bar_low = min(bar_low, row["low"])
        bar_close = row["close"]
        cum_volume += float(volume)
        bar_count += 1

        if cum_volume >= volume_size:
            bars.append(
                {
                    "bar_start_time": bar_start_time,
                    "bar_end_time": ts,
                    "open": bar_open,
                    "high": bar_high,
                    "low": bar_low,
                    "close": bar_close,
                    "volume": cum_volume,
                    "bar_count": bar_count,
                }
            )
            bar_open = None
            bar_high = None
            bar_low = None
            bar_close = None
            bar_start_time = None
            cum_volume = 0.0
            bar_count = 0

    if not bars:
        return pd.DataFrame()

    bars_df = pd.DataFrame(bars)
    bars_df = bars_df.set_index("bar_end_time")
    bars_df.index.name = "timestamp"
    return bars_df


def build_volume_bars(df, volume_size):
    if df is None or df.empty:
        return df

    try:
        volume_size = float(volume_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid volume_bar_size: {volume_size}") from exc
    if volume_size <= 0:
        raise ValueError("volume_bar_size must be positive.")

    if isinstance(df.index, pd.MultiIndex):
        frames = []
        for symbol, symbol_df in df.groupby(level=0):
            symbol_df = symbol_df.droplevel(0)
            bars = _build_volume_bars_single(symbol_df, volume_size)
            if bars is None or bars.empty:
                continue
            bars["symbol"] = symbol
            bars = bars.set_index("symbol", append=True).swaplevel(0, 1).sort_index()
            frames.append(bars)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames).sort_index()

    return _build_volume_bars_single(df, volume_size)
