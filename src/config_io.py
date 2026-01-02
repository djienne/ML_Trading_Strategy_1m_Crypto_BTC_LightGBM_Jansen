import json
import os

from src.utils import format_volume_size, get_inference_symbol, get_train_symbols, resolve_bar_type


def load_config(path="config.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def resolve_symbol_interval(config, symbol=None, interval=None):
    symbols = get_train_symbols(config)
    default_symbol = get_inference_symbol(config, symbols)
    symbol = symbol or default_symbol
    interval = interval or config.get("candle_interval", "1m")
    return symbol, interval


def resolve_bar_id(config, interval):
    bar_type = resolve_bar_type(config)
    if bar_type == "volume":
        size = config.get("volume_bar_size")
        size_label = format_volume_size(size)
        if not size_label:
            raise ValueError("volume_bar_size is required when bar_type is 'volume'.")
        return f"vol{size_label}_{interval}"
    return interval


def resolve_paths(config, symbol, interval):
    base_data_dir = config.get("data_dir", "data")
    feather_dir = config.get("feather_dir", os.path.join(base_data_dir, "feather"))
    processed_dir = config.get("processed_dir", os.path.join(base_data_dir, "processed"))
    predictions_dir = config.get("predictions_dir", os.path.join(base_data_dir, "predictions"))
    eval_dir = config.get("eval_dir", os.path.join(base_data_dir, "eval"))
    models_dir = config.get("models_dir", "models")
    bar_id = resolve_bar_id(config, interval)
    return {
        "feather_dir": feather_dir,
        "processed_dir": processed_dir,
        "predictions_dir": predictions_dir,
        "eval_dir": eval_dir,
        "bar_id": bar_id,
        "model_dir": os.path.join(models_dir, f"{symbol}_{bar_id}"),
        "features_path": os.path.join(processed_dir, f"{symbol}_{bar_id}_model_data.feather"),
        "predictions_path": os.path.join(predictions_dir, f"{symbol}_{bar_id}_predictions.feather"),
    }


def resolve_prediction_paths(config, interval, target_symbol):
    all_paths = resolve_paths(config, "ALL", interval)
    if os.path.exists(all_paths["predictions_path"]):
        return all_paths, True
    target_paths = resolve_paths(config, target_symbol, interval)
    return target_paths, False
