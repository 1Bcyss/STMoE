# GMAN_baseline.py
# -*- coding: utf-8 -*-

import os
import time
import argparse
import random

import numpy as np
import torch
import torch.nn as nn

import util


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
    safe_labels = torch.where(labels == 0, torch.ones_like(labels), labels)
    loss = torch.abs((preds - labels) / safe_labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


class SpatialTemporalBlock(nn.Module):
    """
    稳定版 GMAN-style Block:
    1. Spatial Attention: 同一时间步下，不同节点之间做注意力
    2. Temporal Attention: 同一节点下，不同历史时间步之间做注意力
    3. FFN
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()

        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.spatial_norm = nn.LayerNorm(hidden_dim)
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )

        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (B, T, N, H)
        B, T, N, H = x.shape

        # Spatial attention: (B*T, N, H)
        xs = x.reshape(B * T, N, H)
        s_out, _ = self.spatial_attn(xs, xs, xs)
        xs = self.spatial_norm(xs + s_out)
        x = xs.reshape(B, T, N, H)

        # Temporal attention: (B*N, T, H)
        xt = x.permute(0, 2, 1, 3).contiguous().reshape(B * N, T, H)
        t_out, _ = self.temporal_attn(xt, xt, xt)
        xt = self.temporal_norm(xt + t_out)
        x = xt.reshape(B, N, T, H).permute(0, 2, 1, 3).contiguous()

        # FFN
        f = self.ffn(x)
        x = self.ffn_norm(x + f)

        return x


class GMANStableBaseline(nn.Module):
    """
    GMAN-style stable baseline.

    输入:
        x: (B, T_in, N, C)

    输出:
        pred_norm: (B, T_out, N)

    说明:
        该模型不是官方 GMAN 完整复现，而是适配当前项目的稳定注意力 baseline。
    """

    def __init__(
        self,
        num_nodes,
        in_dim=2,
        hidden_dim=64,
        out_steps=12,
        num_layers=1,
        num_heads=4,
        dropout=0.1,
        max_time_index=288,
        use_residual=True
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.out_steps = out_steps
        self.max_time_index = max_time_index
        self.use_residual = use_residual

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.time_emb = nn.Embedding(max_time_index, hidden_dim)
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)

        self.encoder_blocks = nn.ModuleList([
            SpatialTemporalBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.future_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def get_time_index(self, tod):
        return ((tod * self.max_time_index) % self.max_time_index).long()

    def forward(self, x):
        # x: (B, T, N, C)
        B, T_in, N, C = x.shape
        device = x.device

        # 历史 time-of-day
        hist_tod = x[:, :, 0, -1]  # (B, T)
        hist_time_index = self.get_time_index(hist_tod)

        # 历史时间嵌入
        hist_time_feat = self.time_emb(hist_time_index)  # (B, T, H)
        hist_time_feat = hist_time_feat.unsqueeze(2).expand(B, T_in, N, self.hidden_dim)

        # 节点嵌入
        node_ids = torch.arange(N, device=device)
        node_feat = self.node_emb(node_ids)  # (N, H)
        node_feat_hist = node_feat.view(1, 1, N, self.hidden_dim).expand(B, T_in, N, self.hidden_dim)

        # 输入映射 + 时间嵌入 + 节点嵌入
        h = self.input_proj(x)
        h = h + hist_time_feat + node_feat_hist

        # 编码历史序列
        for block in self.encoder_blocks:
            h = block(h)

        # 使用最后一个历史时刻的隐藏状态作为历史总结
        last_hidden = h[:, -1, :, :]  # (B, N, H)

        # 未来时间索引
        last_time_index = hist_time_index[:, -1]  # (B,)
        steps = torch.arange(1, self.out_steps + 1, device=device).view(1, self.out_steps)
        future_time_index = (last_time_index.view(B, 1) + steps) % self.max_time_index

        future_time_feat = self.time_emb(future_time_index.long())  # (B, T_out, H)
        future_time_feat = future_time_feat.unsqueeze(2).expand(
            B, self.out_steps, N, self.hidden_dim
        )

        node_feat_future = node_feat.view(1, 1, N, self.hidden_dim).expand(
            B, self.out_steps, N, self.hidden_dim
        )

        last_hidden_expand = last_hidden.unsqueeze(1).expand(
            B, self.out_steps, N, self.hidden_dim
        )

        future_hidden = self.future_fusion(
            torch.cat([last_hidden_expand, future_time_feat, node_feat_future], dim=-1)
        )

        delta = self.out_proj(future_hidden).squeeze(-1)  # (B, T_out, N)

        if self.use_residual:
            last_speed = x[:, -1, :, 0]  # normalized speed, (B, N)
            base = last_speed.unsqueeze(1).expand(B, self.out_steps, N)
            pred = base + delta
        else:
            pred = delta

        return pred


def to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.float().to(device)
    return torch.from_numpy(x).float().to(device)


def evaluate(model, loader, scaler, device):
    model.eval()
    preds, reals = [], []

    with torch.no_grad():
        for x, y in loader.get_iterator():
            x = to_tensor(x, device)
            y = to_tensor(y, device)

            y_speed = y[..., 0]

            pred_norm = model(x)
            pred = scaler.inverse_transform(pred_norm)

            preds.append(pred)
            reals.append(y_speed)

    preds = torch.cat(preds, dim=0)
    reals = torch.cat(reals, dim=0)

    return preds, reals


def print_horizon_metrics(pred, real):
    horizon = real.size(1)

    amae = []
    amape = []
    armse = []

    for h in range(horizon):
        p = pred[:, h, :]
        r = real[:, h, :]

        mae = masked_mae_torch(p, r, 0.0).item()
        mape = masked_mape_torch(p, r, 0.0).item()
        rmse = masked_rmse_torch(p, r, 0.0).item()

        amae.append(mae)
        amape.append(mape)
        armse.append(rmse)

        print(
            "Evaluate GMAN on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}".format(
                h + 1, mae, mape, rmse
            )
        )

    print(
        "On average over {} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}".format(
            horizon,
            np.mean(amae),
            np.mean(amape),
            np.mean(armse)
        )
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data", type=str, default="./data/METR-LA")
    parser.add_argument("--in_dim", type=int, default=2)
    parser.add_argument("--num_nodes", type=int, default=207)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=99)

    parser.add_argument("--max_time_index", type=int, default=288)
    parser.add_argument("--no_residual", action="store_true")

    parser.add_argument("--save", type=str, default="./experiment/metrla/baselines/gman/GMAN")

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")

    dataloader = util.load_dataset(
        args.data,
        args.batch_size,
        args.batch_size,
        args.batch_size
    )

    scaler = dataloader["scaler"]
    out_steps = dataloader["y_train"].shape[1]

    model = GMANStableBaseline(
        num_nodes=args.num_nodes,
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        out_steps=out_steps,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_time_index=args.max_time_index,
        use_residual=not args.no_residual
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    best_path = args.save + "_best.pth"

    best_val = float("inf")
    wait = 0

    train_times = []
    val_times = []

    print(args)
    print("Train GMAN-stable baseline with {} parameters".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    ))
    print("Start training GMAN-stable baseline...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        dataloader["train_loader"].shuffle()

        train_losses = []
        t1 = time.time()

        for x, y in dataloader["train_loader"].get_iterator():
            x = to_tensor(x, device)
            y = to_tensor(y, device)

            y_speed = y[..., 0]

            optimizer.zero_grad()

            pred_norm = model(x)
            pred = scaler.inverse_transform(pred_norm)

            loss = masked_mae_torch(pred, y_speed, 0.0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_losses.append(loss.item())

        t2 = time.time()
        train_times.append(t2 - t1)

        s1 = time.time()
        val_pred, val_real = evaluate(model, dataloader["val_loader"], scaler, device)
        s2 = time.time()
        val_times.append(s2 - s1)

        val_size = dataloader["y_val"].shape[0]
        val_pred = val_pred[:val_size]
        val_real = val_real[:val_size]

        val_mae = masked_mae_torch(val_pred, val_real, 0.0).item()
        val_mape = masked_mape_torch(val_pred, val_real, 0.0).item()
        val_rmse = masked_rmse_torch(val_pred, val_real, 0.0).item()

        print(
            "Epoch: {:03d}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch".format(
                epoch,
                np.mean(train_losses),
                val_mae,
                val_mape,
                val_rmse,
                t2 - t1
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

    test_pred, test_real = evaluate(model, dataloader["test_loader"], scaler, device)

    test_size = dataloader["y_test"].shape[0]
    test_pred = test_pred[:test_size]
    test_real = test_real[:test_size]

    print("Training finished")
    print("The valid loss on best model is {:.4f}".format(best_val))

    print_horizon_metrics(test_pred, test_real)

    save_npz = args.save + "_prediction.npz"

    np.savez_compressed(
        save_npz,
        prediction=test_pred.detach().cpu().numpy().transpose(1, 0, 2)[:, :, None, :],
        ground_truth=test_real.detach().cpu().numpy().transpose(1, 0, 2)[:, :, None, :]
    )

    print("[Saved]", save_npz)


if __name__ == "__main__":
    main()