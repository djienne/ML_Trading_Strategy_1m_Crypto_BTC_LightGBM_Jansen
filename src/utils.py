import numpy as np
import pandas as pd


def get_time_index(index):
    if isinstance(index, pd.MultiIndex):
        return index.get_level_values(-1)
    return index


def get_date_key(index):
    time_index = pd.to_datetime(get_time_index(index))
    return time_index.date


def get_symbol_key(index):
    if isinstance(index, pd.MultiIndex):
        return index.get_level_values(0)
    return pd.Index(["SINGLE"] * len(index), name="symbol")


def assign_decile(x, bins=10):
    ranks = x.rank(method="first")
    n = len(ranks)
    if n == 0:
        return ranks
    if n == 1:
        return pd.Series([bins], index=x.index, dtype=int)
    if n < bins:
        scaled = (ranks - 1).div(n - 1).mul(bins - 1).add(1)
        return pd.Series(np.floor(scaled).astype(int), index=x.index)
    try:
        return pd.qcut(ranks, bins, labels=False) + 1
    except ValueError:
        scaled = (ranks - 1).div(n - 1).mul(bins - 1).add(1)
        return pd.Series(np.floor(scaled).astype(int), index=x.index)


def resolve_quantile_scope(scope, symbol_count):
    if scope == "auto":
        return "timestamp" if symbol_count > 1 else "date"
    return scope


def resolve_feature_flags(config):
    defaults = {
        "returns": True,
        "bop": True,
        "cci": True,
        "mfi": True,
        "rsi": True,
        "stochrsi": True,
        "stoch": True,
        "natr": True,
    }
    flags = dict(defaults)
    flags.update(config.get("feature_flags", {}))
    return flags


def get_train_symbols(config):
    symbols = config.get("train_symbols") or config.get("symbols") or ["BTCUSDT"]
    return [s for s in symbols if s]


def get_inference_symbol(config, train_symbols=None):
    configured = config.get("inference_symbol")
    if configured:
        return configured
    train_symbols = train_symbols or get_train_symbols(config)
    return train_symbols[0] if train_symbols else "BTCUSDT"


def get_config_symbols(config):
    return get_train_symbols(config)
