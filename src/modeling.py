import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.utils import get_symbol_key


def ic_lgbm(preds, train_data):
    ic = spearmanr(preds, train_data.get_label())[0]
    if np.isnan(ic):
        ic = 0.0
    return "ic", ic, True


def train_and_predict(data, model_dir=None, resume=False, continue_rounds=50):
    print("Starting model training (rolling window)...")
    data = data.sort_index()

    target_col = "fwd1min"
    if target_col not in data.columns:
        raise ValueError(f"Missing target column: {target_col}")

    feature_cols = [c for c in data.columns if c != target_col]
    symbol_count = pd.Index(get_symbol_key(data.index)).nunique()
    print(
        f"Training data: {len(data)} rows, {symbol_count} symbol(s), "
        f"{len(feature_cols)} features."
    )
    if resume:
        print(f"Training mode: resume (+{continue_rounds} rounds per fold when available).")
    else:
        print("Training mode: fresh models per fold.")

    # Parameters from the notebook (adjusted for CPU)
    params = dict(
        objective="regression",
        metric=["rmse"],
        device="cpu",
        num_leaves=16,
        min_data_in_leaf=500,
        feature_fraction=0.8,
        verbose=-1,
        seed=42,
    )
    num_boost_round = 250

    # Rolling window setup: Train 12 months, Test 1 month (24/7 crypto)
    day = 24 * 60
    month = 30
    train_len = 12 * month * day
    test_len = month * day
    lookahead = 1

    total_rows = len(data)
    min_required = train_len + test_len + lookahead
    if total_rows < min_required:
        print("Warning: Insufficient data for 12m train + 1m test.")
        print(f"Data length: {total_rows}, Required: {min_required}")
        print("Adjusting split to 80% train / 20% test for demonstration.")

        split_idx = int(total_rows * 0.8)
        train_idx = np.arange(0, split_idx)
        test_start = split_idx + lookahead
        if test_start >= total_rows:
            test_start = split_idx
        test_idx = np.arange(test_start, total_rows)

        splits = [(train_idx, test_idx)] if len(test_idx) else []
    else:
        splits = []
        step = test_len
        max_start = total_rows - (train_len + lookahead + test_len)
        for start in range(0, max_start + 1, step):
            train_start = start
            train_end = train_start + train_len
            test_start = train_end + lookahead
            test_end = test_start + test_len
            splits.append((np.arange(train_start, train_end), np.arange(test_start, test_end)))

    print(
        f"Rolling window: train={train_len} rows, test={test_len} rows, "
        f"lookahead={lookahead}."
    )
    print(f"Generated {len(splits)} splits.")

    all_predictions = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"Processing Fold {fold + 1}/{len(splits)}...")

        X_train = data.iloc[train_idx][feature_cols]
        y_train = data.iloc[train_idx][target_col]
        X_test = data.iloc[test_idx][feature_cols]
        y_test = data.iloc[test_idx][target_col]

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        init_model = None
        if resume and model_dir:
            init_path = os.path.join(model_dir, f"fold_{fold + 1:02d}.txt")
            if os.path.exists(init_path):
                init_model = init_path
                print(f"  Fold {fold + 1}: resuming from {init_path} (+{continue_rounds} rounds)")

        boost_rounds = continue_rounds if init_model else num_boost_round
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=boost_rounds,
            valid_sets=[lgb_train, lgb_eval],
            feval=ic_lgbm,
            callbacks=callbacks,
            init_model=init_model,
        )

        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"fold_{fold + 1:02d}.txt")
            model.save_model(model_path)

        best_iter = model.best_iteration or model.current_iteration()
        preds = model.predict(X_test, num_iteration=best_iter)

        fold_preds = pd.DataFrame({"target": y_test, "prediction": preds}, index=y_test.index)
        all_predictions.append(fold_preds)

        ic, _ = spearmanr(y_test, preds)
        if np.isnan(ic):
            ic = 0.0
        print(f"  Fold {fold + 1} IC: {ic:.4f}")

    if not all_predictions:
        return pd.DataFrame()

    return pd.concat(all_predictions).sort_index()
