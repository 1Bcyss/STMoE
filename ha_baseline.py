# ha_baseline_v2.py
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd


def masked_mae(preds, labels, null_val=0.0):
    mask = labels > null_val
    mask = mask.astype(np.float32)
    if mask.mean() == 0:
        return np.nan
    mask = mask / mask.mean()
    loss = np.abs(preds - labels)
    loss = loss * mask
    loss = np.where(np.isnan(loss), 0, loss)
    return np.mean(loss)


def masked_rmse(preds, labels, null_val=0.0):
    mask = labels > null_val
    mask = mask.astype(np.float32)
    if mask.mean() == 0:
        return np.nan
    mask = mask / mask.mean()
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = np.where(np.isnan(loss), 0, loss)
    return np.sqrt(np.mean(loss))


def masked_mape(preds, labels, null_val=0.0):
    mask = labels > null_val
    mask = mask.astype(np.float32)
    if mask.mean() == 0:
        return np.nan
    mask = mask / mask.mean()

    safe_labels = np.where(labels == 0, 1, labels)
    loss = np.abs((preds - labels) / safe_labels)
    loss = loss * mask
    loss = np.where(np.isnan(loss), 0, loss)
    return np.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0)
    mape = masked_mape(pred, real, 0.0)
    rmse = masked_rmse(pred, real, 0.0)
    return mae, mape, rmse


def load_traffic_df(path):
    if path.endswith(".h5"):
        df = pd.read_hdf(path)
    else:
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index)
    return df


def get_time_slot(index, num_slots=288):
    """
    5分钟粒度时，num_slots=288。
    返回每个时间戳属于一天中的第几个时间片。
    """
    minutes = index.hour * 60 + index.minute
    slot_size = int(24 * 60 / num_slots)
    return (minutes // slot_size).astype(np.int64)


def generate_target_indices(num_timestamps, seq_length_x=12, seq_length_y=12, y_start=1):
    """
    与 generate_training_data.py 保持一致：
    x_offsets = [-11, ..., 0]
    y_offsets = [1, ..., 12]
    t 表示输入窗口最后一个时间点。
    每个样本的预测目标是 t+1 到 t+12。
    """
    x_offsets = np.sort(np.arange(-(seq_length_x - 1), 1, 1))
    y_offsets = np.sort(np.arange(y_start, seq_length_y + 1, 1))

    min_t = abs(min(x_offsets))
    max_t = abs(num_timestamps - abs(max(y_offsets)))

    target_indices = []
    for t in range(min_t, max_t):
        target_indices.append(t + y_offsets)

    return np.asarray(target_indices)  # (num_samples, H)


def split_indices(num_samples):
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    train_range = np.arange(0, num_train)
    val_range = np.arange(num_train, num_train + num_val)
    test_range = np.arange(num_samples - num_test, num_samples)

    return train_range, val_range, test_range


def build_ha_tod(df, train_target_indices, num_slots=288):
    """
    按 node + time-of-day 统计历史平均。
    table: (num_slots, N)
    """
    values = df.values.astype(np.float64)
    index = df.index
    N = values.shape[1]

    slots = get_time_slot(index, num_slots=num_slots)

    sums = np.zeros((num_slots, N), dtype=np.float64)
    counts = np.zeros((num_slots, N), dtype=np.float64)

    flat_indices = train_target_indices.reshape(-1)

    for idx in flat_indices:
        slot = slots[idx]
        val = values[idx]
        valid = val > 0
        sums[slot, valid] += val[valid]
        counts[slot, valid] += 1

    node_mean = np.zeros(N, dtype=np.float64)
    train_values = values[flat_indices]
    for n in range(N):
        valid_values = train_values[:, n]
        valid_values = valid_values[valid_values > 0]
        node_mean[n] = valid_values.mean() if len(valid_values) > 0 else 0.0

    table = np.zeros((num_slots, N), dtype=np.float64)
    for s in range(num_slots):
        for n in range(N):
            if counts[s, n] > 0:
                table[s, n] = sums[s, n] / counts[s, n]
            else:
                table[s, n] = node_mean[n]

    return table, node_mean


def build_ha_dow_tod(df, train_target_indices, num_slots=288):
    """
    按 node + weekday + time-of-day 统计历史平均。
    table: (7, num_slots, N)
    """
    values = df.values.astype(np.float64)
    index = df.index
    N = values.shape[1]

    slots = get_time_slot(index, num_slots=num_slots)
    dows = index.dayofweek.values.astype(np.int64)

    sums = np.zeros((7, num_slots, N), dtype=np.float64)
    counts = np.zeros((7, num_slots, N), dtype=np.float64)

    flat_indices = train_target_indices.reshape(-1)

    for idx in flat_indices:
        dow = dows[idx]
        slot = slots[idx]
        val = values[idx]
        valid = val > 0
        sums[dow, slot, valid] += val[valid]
        counts[dow, slot, valid] += 1

    # fallback 1: time-of-day table
    tod_table, node_mean = build_ha_tod(df, train_target_indices, num_slots=num_slots)

    table = np.zeros((7, num_slots, N), dtype=np.float64)
    for d in range(7):
        for s in range(num_slots):
            for n in range(N):
                if counts[d, s, n] > 0:
                    table[d, s, n] = sums[d, s, n] / counts[d, s, n]
                else:
                    table[d, s, n] = tod_table[s, n]

    return table, tod_table, node_mean


def predict_ha(df, test_target_indices, mode="dow_tod", num_slots=288):
    values = df.values.astype(np.float64)
    index = df.index
    N = values.shape[1]
    B, H = test_target_indices.shape

    slots = get_time_slot(index, num_slots=num_slots)
    dows = index.dayofweek.values.astype(np.int64)

    # 注意：这里会在 main 里传入已经构建好的 table，所以本函数不单独用
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traffic_df_filename", type=str, required=True)
    parser.add_argument("--seq_length_x", type=int, default=12)
    parser.add_argument("--seq_length_y", type=int, default=12)
    parser.add_argument("--y_start", type=int, default=1)
    parser.add_argument("--num_slots", type=int, default=288)
    parser.add_argument("--mode", type=str, default="dow_tod", choices=["tod", "dow_tod"])
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    df = load_traffic_df(args.traffic_df_filename)

    print("Loaded traffic data:", df.shape)
    print("Time range:", df.index[0], "->", df.index[-1])
    print("HA mode:", args.mode)

    target_indices = generate_target_indices(
        num_timestamps=len(df),
        seq_length_x=args.seq_length_x,
        seq_length_y=args.seq_length_y,
        y_start=args.y_start,
    )

    num_samples = target_indices.shape[0]
    train_range, val_range, test_range = split_indices(num_samples)

    train_target_indices = target_indices[train_range]
    test_target_indices = target_indices[test_range]

    print("num_samples:", num_samples)
    print("train samples:", len(train_range))
    print("test samples:", len(test_range))
    print("target_indices shape:", target_indices.shape)

    values = df.values.astype(np.float64)
    index = df.index
    slots = get_time_slot(index, num_slots=args.num_slots)
    dows = index.dayofweek.values.astype(np.int64)

    B, H = test_target_indices.shape
    N = values.shape[1]

    real = np.zeros((B, H, N), dtype=np.float64)
    pred = np.zeros((B, H, N), dtype=np.float64)

    if args.mode == "tod":
        tod_table, node_mean = build_ha_tod(
            df,
            train_target_indices,
            num_slots=args.num_slots
        )

        for b in range(B):
            for h in range(H):
                idx = test_target_indices[b, h]
                slot = slots[idx]
                real[b, h, :] = values[idx]
                pred[b, h, :] = tod_table[slot]

    elif args.mode == "dow_tod":
        dow_tod_table, tod_table, node_mean = build_ha_dow_tod(
            df,
            train_target_indices,
            num_slots=args.num_slots
        )

        for b in range(B):
            for h in range(H):
                idx = test_target_indices[b, h]
                dow = dows[idx]
                slot = slots[idx]
                real[b, h, :] = values[idx]
                pred[b, h, :] = dow_tod_table[dow, slot]

    amae, amape, armse = [], [], []

    for h in range(H):
        metrics = metric(pred[:, h, :], real[:, h, :])
        print(
            "Evaluate HA-{} on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}".format(
                args.mode, h + 1, metrics[0], metrics[1], metrics[2]
            )
        )
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    print(
        "On average over {} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}".format(
            H, np.mean(amae), np.mean(amape), np.mean(armse)
        )
    )

    if args.save is not None:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        np.savez_compressed(
            args.save + "_prediction.npz",
            prediction=pred,
            ground_truth=real,
        )
        print("[Saved]", args.save + "_prediction.npz")


if __name__ == "__main__":
    main()