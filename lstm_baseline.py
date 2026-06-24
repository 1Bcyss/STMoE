# lstm_baseline.py
# -*- coding: utf-8 -*-

import os
import time
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def set_seed(seed):
    if seed == -1:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_mae_torch(preds, labels, null_val=0.0):
    mask = labels > null_val
    mask = mask.float()
    mask = mask / torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse_torch(preds, labels, null_val=0.0):
    mask = labels > null_val
    mask = mask.float()
    mask = mask / torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sqrt(torch.mean(loss))


def masked_mape_torch(preds, labels, null_val=0.0):
    mask = labels > null_val
    mask = mask.float()
    mask = mask / torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds - labels) / labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric_torch(pred, real):
    mae = masked_mae_torch(pred, real, 0.0).item()
    mape = masked_mape_torch(pred, real, 0.0).item()
    rmse = masked_rmse_torch(pred, real, 0.0).item()
    return mae, mape, rmse


class NPZDataLoader:
    def __init__(self, x, y, batch_size, shuffle=False, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        if pad_with_last_sample:
            num_padding = (batch_size - (len(x) % batch_size)) % batch_size
            if num_padding > 0:
                x_padding = np.repeat(x[-1:], num_padding, axis=0)
                y_padding = np.repeat(y[-1:], num_padding, axis=0)
                x = np.concatenate([x, x_padding], axis=0)
                y = np.concatenate([y, y_padding], axis=0)

        self.x = x
        self.y = y
        self.size = len(x)
        self.num_batch = self.size // self.batch_size

    def get_iterator(self):
        indices = np.arange(self.size)
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(self.num_batch):
            batch_idx = indices[i * self.batch_size: (i + 1) * self.batch_size]
            yield self.x[batch_idx], self.y[batch_idx]


class LSTMBaseline(nn.Module):
    """
    纯 LSTM baseline。

    输入 x: (B, T_in, N, C)
    输出 y: (B, T_out, N)

    做法:
    1. 把每个节点当成一条时间序列
    2. reshape 成 (B*N, T_in, C)
    3. LSTM 编码
    4. Linear 输出未来 T_out 步
    5. reshape 回 (B, T_out, N)
    """

    def __init__(self, in_dim=2, hidden_dim=64, num_layers=2, out_steps=12, dropout=0.1):
        super().__init__()
        self.out_steps = out_steps
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.proj = nn.Linear(hidden_dim, out_steps)

    def forward(self, x):
        # x: (B, T, N, C)
        B, T, N, C = x.shape

        # (B, T, N, C) -> (B, N, T, C) -> (B*N, T, C)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B * N, T, C)

        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]  # (B*N, hidden_dim)
        pred = self.proj(last_hidden)  # (B*N, out_steps)

        # (B*N, H) -> (B, N, H) -> (B, H, N)
        pred = pred.view(B, N, self.out_steps)
        pred = pred.permute(0, 2, 1).contiguous()

        return pred


def load_npz_dataset(data_dir, batch_size):
    train = np.load(os.path.join(data_dir, "train.npz"))
    val = np.load(os.path.join(data_dir, "val.npz"))
    test = np.load(os.path.join(data_dir, "test.npz"))

    x_train, y_train = train["x"], train["y"]
    x_val, y_val = val["x"], val["y"]
    x_test, y_test = test["x"], test["y"]

    # 只用训练集速度特征计算 mean/std，和你的主模型逻辑保持一致
    mean = x_train[..., 0].mean()
    std = x_train[..., 0].std()

    def transform_x(x):
        x = x.copy()
        x[..., 0] = (x[..., 0] - mean) / std
        return x

    def transform_y_speed(y):
        # y[..., 0] 是真实速度
        return (y[..., 0] - mean) / std

    x_train_norm = transform_x(x_train)
    x_val_norm = transform_x(x_val)
    x_test_norm = transform_x(x_test)

    y_train_norm = transform_y_speed(y_train)
    y_val_norm = transform_y_speed(y_val)
    y_test_norm = transform_y_speed(y_test)

    loaders = {
        "train": NPZDataLoader(x_train_norm, y_train_norm, batch_size, shuffle=True),
        "val": NPZDataLoader(x_val_norm, y_val_norm, batch_size, shuffle=False),
        "test": NPZDataLoader(x_test_norm, y_test_norm, batch_size, shuffle=False),
    }

    raw = {
        "y_test_raw": y_test[..., 0],
        "mean": mean,
        "std": std,
        "x_train_shape": x_train.shape,
        "y_train_shape": y_train.shape,
        "x_val_shape": x_val.shape,
        "y_val_shape": y_val.shape,
        "x_test_shape": x_test.shape,
        "y_test_shape": y_test.shape,
    }

    return loaders, raw


def evaluate(model, loader, device, mean, std):
    model.eval()

    all_pred_raw = []
    all_real_raw = []

    with torch.no_grad():
        for x, y_norm in loader.get_iterator():
            x = torch.tensor(x, dtype=torch.float32, device=device)
            y_norm = torch.tensor(y_norm, dtype=torch.float32, device=device)

            pred_norm = model(x)

            pred_raw = pred_norm * std + mean
            real_raw = y_norm * std + mean

            all_pred_raw.append(pred_raw)
            all_real_raw.append(real_raw)

    pred = torch.cat(all_pred_raw, dim=0)
    real = torch.cat(all_real_raw, dim=0)

    # 去掉 padding 出来的多余样本时，在 test 阶段外面会再次对齐；
    # 这里先直接返回。
    return pred, real


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data", type=str, default="./data/METR-LA")
    parser.add_argument("--in_dim", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--save", type=str, default="./experiment/baselines/lstm/LSTM")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")

    loaders, raw = load_npz_dataset(args.data, args.batch_size)

    print("x_train shape:", raw["x_train_shape"])
    print("y_train shape:", raw["y_train_shape"])
    print("x_val shape:", raw["x_val_shape"])
    print("y_val shape:", raw["y_val_shape"])
    print("x_test shape:", raw["x_test_shape"])
    print("y_test shape:", raw["y_test_shape"])
    print("mean:", raw["mean"], "std:", raw["std"])

    out_steps = raw["y_train_shape"][1]

    model = LSTMBaseline(
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        out_steps=out_steps,
        dropout=args.dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    best_val = float("inf")
    wait = 0
    best_path = args.save + "_best.pth"

    print("Start training LSTM baseline...")
    train_times = []
    val_times = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        t1 = time.time()

        for x, y_norm in loaders["train"].get_iterator():
            x = torch.tensor(x, dtype=torch.float32, device=device)
            y_norm = torch.tensor(y_norm, dtype=torch.float32, device=device)

            optimizer.zero_grad()
            pred_norm = model(x)
            loss = criterion(pred_norm, y_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_losses.append(loss.item())

        t2 = time.time()
        train_times.append(t2 - t1)

        s1 = time.time()
        val_pred_raw, val_real_raw = evaluate(model, loaders["val"], device, raw["mean"], raw["std"])
        s2 = time.time()
        val_times.append(s2 - s1)

        # 验证集去掉 padding 后再计算
        val_size = raw["y_val_shape"][0]
        val_pred_raw = val_pred_raw[:val_size]
        val_real_raw = val_real_raw[:val_size]

        val_mae = masked_mae_torch(val_pred_raw, val_real_raw, 0.0).item()
        val_mape = masked_mape_torch(val_pred_raw, val_real_raw, 0.0).item()
        val_rmse = masked_rmse_torch(val_pred_raw, val_real_raw, 0.0).item()

        print(
            "Epoch: {:03d}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch".format(
                epoch, np.mean(train_losses), val_mae, val_mape, val_rmse, t2 - t1
            )
        )

        if val_mae < best_val:
            best_val = val_mae
            wait = 0
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1

        if wait > args.patience:
            print("Early Termination!")
            break

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_times)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_times)))

    print("Load best model:", best_path)
    model.load_state_dict(torch.load(best_path, map_location=device))

    test_pred_raw, test_real_raw = evaluate(model, loaders["test"], device, raw["mean"], raw["std"])

    test_size = raw["y_test_shape"][0]
    test_pred_raw = test_pred_raw[:test_size]
    test_real_raw = test_real_raw[:test_size]

    print("Training finished")
    print("The valid loss on best model is", round(best_val, 4))

    amae, amape, armse = [], [], []

    for h in range(test_real_raw.shape[1]):
        pred_h = test_pred_raw[:, h, :]
        real_h = test_real_raw[:, h, :]

        metrics = metric_torch(pred_h, real_h)
        print(
            "Evaluate LSTM on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}".format(
                h + 1, metrics[0], metrics[1], metrics[2]
            )
        )
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    print(
        "On average over {} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}".format(
            test_real_raw.shape[1], np.mean(amae), np.mean(amape), np.mean(armse)
        )
    )

    # 保存预测结果，方便后面画图
    np.savez_compressed(
        args.save + "_prediction.npz",
        prediction=test_pred_raw.detach().cpu().numpy(),
        ground_truth=test_real_raw.detach().cpu().numpy(),
    )
    print("[Saved]", args.save + "_prediction.npz")


if __name__ == "__main__":
    main()