# PDFormer_baseline.py
# -*- coding: utf-8 -*-

"""
PDFormer-style baseline for NYCBike inflow prediction.

适配当前项目数据格式：
    data/NYCBike/train.npz
    data/NYCBike/val.npz
    data/NYCBike/test.npz
    data/NYCBike/adj_mx.npy

实验设置：
    - 只使用 inflow
    - 输入 12 步，预测 1 步
    - 输入维度 in_dim=1
    - 节点数 num_nodes=295

推荐运行命令：
    python PDFormer_baseline.py --data ./data/NYCBike --adjdata ./data/NYCBike/adj_mx.npy --in_dim 1 --num_nodes 295 --batch_size 16 --epochs 100 --save ./experiment/nycbike/pdformer/PDFormer

先测试能否跑通：
    python PDFormer_baseline.py --data ./data/NYCBike --adjdata ./data/NYCBike/adj_mx.npy --in_dim 1 --num_nodes 295 --batch_size 16 --epochs 2 --save ./experiment/nycbike/pdformer/PDFormer_test
"""

import os
import time
import argparse
import random

import numpy as np
import torch
import torch.nn as nn

import util


# ============================================================
# 1. Reproducibility
# ============================================================

def set_seed(seed):
    if seed == -1:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. Metrics
#    注意：NYCBike inflow 中 0 是正常值，不应按缺失值屏蔽。
# ============================================================

def mae_torch(preds, labels):
    return torch.mean(torch.abs(preds - labels))


def rmse_torch(preds, labels):
    return torch.sqrt(torch.mean((preds - labels) ** 2))


def mape_torch(preds, labels, eps=1e-5):
    mask = labels.abs() > eps
    if mask.sum().item() == 0:
        return torch.tensor(0.0, device=labels.device)
    return torch.mean(torch.abs((preds[mask] - labels[mask]) / labels[mask]))


# ============================================================
# 3. Graph construction
# ============================================================

def build_graph_tensors(adjdata, adjtype, num_nodes, short_top_k):
    """
    构建 PDFormer-style baseline 使用的图结构信息。

    支持三类邻接矩阵：
        1. .npy，例如 ./data/NYCBike/adj_mx.npy
        2. .csv，例如 ./data/Bike/adj.csv
        3. 原项目 util.load_adj 支持的邻接文件，例如 .pkl

    返回：
        adj_norm:   (N, N)，行归一化邻接矩阵，用于传播延迟上下文
        short_mask: (N, N)，短距离空间注意力可见掩码
        num_nodes:  根据邻接矩阵自动修正后的节点数
    """

    if adjdata is not None and os.path.exists(adjdata):
        if adjdata.endswith(".npy"):
            A = np.load(adjdata).astype(np.float32)

        elif adjdata.endswith(".csv"):
            A = np.genfromtxt(adjdata, delimiter=",").astype(np.float32)

        else:
            # 兼容 METR-LA / PEMS-BAY 等常见 pkl 邻接矩阵
            sensor_ids, sensor_id_to_ind, supports = util.load_adj(adjdata, adjtype)
            num_nodes = len(sensor_ids)

            A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
            for support in supports:
                A += np.asarray(support, dtype=np.float32)

        A = np.squeeze(A)
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        A = np.maximum(A, 0.0)

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"邻接矩阵必须是方阵，但当前 shape={A.shape}")

        num_nodes = A.shape[0]

    else:
        print("Invalid or missing adjdata. Use identity adjacency.")
        A = np.eye(num_nodes, dtype=np.float32)

    N = A.shape[0]

    # 短距离空间注意力 mask
    # short_mask[i, j] = True 表示节点 i 可以关注节点 j
    short_mask = np.zeros((N, N), dtype=bool)

    if short_top_k is not None and short_top_k > 0 and short_top_k < N:
        for i in range(N):
            row = A[i].copy()
            row[i] = max(row[i], 1.0)
            idx = np.argsort(-row)[:short_top_k]
            short_mask[i, idx] = True
    else:
        short_mask = A > 0

    np.fill_diagonal(short_mask, True)

    # 传播延迟上下文用的归一化邻接矩阵
    A_prop = A.copy()
    A_prop = A_prop * short_mask.astype(np.float32)
    np.fill_diagonal(A_prop, np.maximum(np.diag(A_prop), 1.0))

    row_sum = A_prop.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    adj_norm = A_prop / row_sum

    adj_norm = torch.tensor(adj_norm, dtype=torch.float32)
    short_mask = torch.tensor(short_mask, dtype=torch.bool)

    print("Loaded adjacency shape:", A.shape)
    print("short_mask true ratio:", short_mask.float().mean().item())

    return adj_norm, short_mask, num_nodes


# ============================================================
# 4. Model modules
# ============================================================

class MaskedSpatialAttention(nn.Module):
    """
    PDFormer-style 空间注意力。

    输入：
        x: (B, T, N, H)

    核心思想：
        1. short_mask 表示基于邻接矩阵得到的短距离可见范围；
        2. node_embeddings 相似度构造长距离语义相关节点；
        3. 短距离 mask 与长距离 semantic mask 取并集。
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, long_top_k=20):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.long_top_k = long_top_k

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def build_semantic_mask(self, node_embeddings, N, device):
        if self.long_top_k is None or self.long_top_k <= 0:
            return torch.zeros(N, N, dtype=torch.bool, device=device)

        if self.long_top_k >= N:
            return torch.ones(N, N, dtype=torch.bool, device=device)

        sim = torch.matmul(node_embeddings, node_embeddings.transpose(0, 1))
        _, idx = torch.topk(sim, k=self.long_top_k, dim=-1)

        semantic_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
        semantic_mask.scatter_(1, idx, True)
        semantic_mask.fill_diagonal_(True)

        return semantic_mask

    def forward(self, x, node_embeddings, short_mask):
        B, T, N, H = x.shape
        device = x.device

        z = x.reshape(B * T, N, H)

        q = self.q_proj(z)
        k = self.k_proj(z)
        v = self.v_proj(z)

        q = q.view(B * T, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B * T, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B * T, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)

        semantic_mask = self.build_semantic_mask(node_embeddings, N, device)
        allowed_mask = short_mask.to(device) | semantic_mask
        allowed_mask.fill_diagonal_(True)

        disallowed = ~allowed_mask
        scores = scores.masked_fill(disallowed.view(1, 1, N, N), -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B * T, N, H)
        out = self.out_proj(out)

        out = out.reshape(B, T, N, H)

        return out


class TemporalAttention(nn.Module):
    """
    时间注意力模块。

    输入：
        x: (B, T, N, H)
    输出：
        out: (B, T, N, H)
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x):
        B, T, N, H = x.shape

        xt = x.permute(0, 2, 1, 3).contiguous().reshape(B * N, T, H)
        out, _ = self.attn(xt, xt, xt)
        out = out.reshape(B, N, T, H).permute(0, 2, 1, 3).contiguous()

        return out


class PDFormerBlock(nn.Module):
    """
    PDFormer-style Block。

    包含：
        1. Masked Spatial Attention
        2. Temporal Attention
        3. Feed Forward Network
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, long_top_k=20):
        super().__init__()

        self.spatial_attn = MaskedSpatialAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            long_top_k=long_top_k
        )

        self.temporal_attn = TemporalAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
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

    def forward(self, x, node_embeddings, short_mask):
        s_out = self.spatial_attn(x, node_embeddings, short_mask)
        x = self.spatial_norm(x + s_out)

        t_out = self.temporal_attn(x)
        x = self.temporal_norm(x + t_out)

        f = self.ffn(x)
        x = self.ffn_norm(x + f)

        return x


class PDFormerStableBaseline(nn.Module):
    """
    PDFormer-style stable baseline.

    输入：
        x: (B, T_in, N, C)

    输出：
        pred_norm: (B, T_out, N)

    针对 NYCBike inflow 的关键改动：
        1. 不强制使用 time-of-day 特征；
        2. 当输入只有 inflow 一个通道时，时间嵌入置零；
        3. 邻接矩阵通过 adj_mx.npy 读取；
        4. 输出步长由 y_train.shape[1] 自动决定。
    """

    def __init__(
        self,
        num_nodes,
        adj_norm,
        short_mask,
        in_dim=1,
        hidden_dim=64,
        out_steps=1,
        num_layers=1,
        num_heads=4,
        dropout=0.2,
        max_time_index=288,
        max_delay=12,
        long_top_k=20,
        use_residual=True,
        use_time_emb=False
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.out_steps = out_steps
        self.max_time_index = max_time_index
        self.max_delay = max_delay
        self.use_residual = use_residual
        self.use_time_emb = use_time_emb

        self.register_buffer("adj_norm", adj_norm.float())
        self.register_buffer("short_mask", short_mask.bool())

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.time_emb = nn.Embedding(max_time_index, hidden_dim)
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.delay_emb = nn.Embedding(max_delay + 1, hidden_dim)

        self.blocks = nn.ModuleList([
            PDFormerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                long_top_k=long_top_k
            )
            for _ in range(num_layers)
        ])

        # last_hidden, future_time_feat, node_feat_future, delay_context, delay_feat
        self.future_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.future_norm = nn.LayerNorm(hidden_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def get_time_index(self, tod):
        """
        tod: time-of-day, 范围通常为 [0, 1)。
        当前 NYCBike 只用 inflow 时不会启用该函数。
        """
        idx = ((tod * self.max_time_index) % self.max_time_index).long()
        idx = torch.clamp(idx, min=0, max=self.max_time_index - 1)
        return idx

    def build_delay_context(self, last_hidden):
        """
        使用邻接矩阵幂近似不同预测步的传播延迟影响。

        输入：
            last_hidden: (B, N, H)

        输出：
            delay_context: (B, T_out, N, H)
        """

        B, N, H = last_hidden.shape
        device = last_hidden.device
        dtype = last_hidden.dtype

        A = self.adj_norm.to(device=device, dtype=dtype)
        A_power = torch.eye(N, device=device, dtype=dtype)

        contexts = []

        for _ in range(self.out_steps):
            A_power = torch.matmul(A_power, A)
            prop = torch.einsum("nm,bmh->bnh", A_power, last_hidden)
            contexts.append(prop)

        delay_context = torch.stack(contexts, dim=1)

        return delay_context

    def forward(self, x):
        # x: (B, T, N, C)
        B, T_in, N, C = x.shape
        device = x.device

        if N != self.num_nodes:
            raise ValueError(f"输入节点数 N={N} 与模型 num_nodes={self.num_nodes} 不一致")

        # --------------------------------------------------------
        # 历史时间嵌入
        # 当前 NYCBike 输入只有 inflow 一个通道，因此不使用 time-of-day。
        # 如果以后你把 time-of-day 拼到最后一个通道，并设置 --use_time_emb，才会启用。
        # --------------------------------------------------------
        if self.use_time_emb and C >= 2:
            hist_tod = x[:, :, 0, -1]
            hist_time_index = self.get_time_index(hist_tod)

            hist_time_feat = self.time_emb(hist_time_index)
            hist_time_feat = hist_time_feat.unsqueeze(2).expand(
                B, T_in, N, self.hidden_dim
            )
        else:
            hist_time_index = None
            hist_time_feat = torch.zeros(
                B, T_in, N, self.hidden_dim,
                device=device,
                dtype=x.dtype
            )

        # 节点嵌入
        node_ids = torch.arange(N, device=device)
        node_feat = self.node_emb(node_ids)
        node_feat_hist = node_feat.view(1, 1, N, self.hidden_dim).expand(
            B, T_in, N, self.hidden_dim
        )

        # 输入投影 + 时间嵌入 + 节点嵌入
        h = self.input_proj(x)
        h = h + hist_time_feat + node_feat_hist

        for block in self.blocks:
            h = block(h, self.node_emb.weight, self.short_mask)

        # 历史编码结果
        last_hidden = h[:, -1, :, :]  # (B, N, H)

        # --------------------------------------------------------
        # 未来时间嵌入
        # 当前 NYCBike 没有未来 time-of-day，因此置零。
        # --------------------------------------------------------
        if self.use_time_emb and hist_time_index is not None:
            last_time_index = hist_time_index[:, -1]
            steps = torch.arange(1, self.out_steps + 1, device=device).view(1, self.out_steps)
            future_time_index = (last_time_index.view(B, 1) + steps) % self.max_time_index

            future_time_feat = self.time_emb(future_time_index.long())
            future_time_feat = future_time_feat.unsqueeze(2).expand(
                B, self.out_steps, N, self.hidden_dim
            )
        else:
            future_time_feat = torch.zeros(
                B, self.out_steps, N, self.hidden_dim,
                device=device,
                dtype=x.dtype
            )

        # 节点嵌入复制到未来步
        node_feat_future = node_feat.view(1, 1, N, self.hidden_dim).expand(
            B, self.out_steps, N, self.hidden_dim
        )

        # 历史最后隐藏状态复制到未来步
        last_hidden_expand = last_hidden.unsqueeze(1).expand(
            B, self.out_steps, N, self.hidden_dim
        )

        # 传播延迟上下文
        delay_context = self.build_delay_context(last_hidden)

        # delay embedding
        delay_ids = torch.arange(1, self.out_steps + 1, device=device)
        delay_ids = torch.clamp(delay_ids, max=self.max_delay)
        delay_feat = self.delay_emb(delay_ids)
        delay_feat = delay_feat.view(1, self.out_steps, 1, self.hidden_dim).expand(
            B, self.out_steps, N, self.hidden_dim
        )

        future_hidden = self.future_fusion(
            torch.cat(
                [
                    last_hidden_expand,
                    future_time_feat,
                    node_feat_future,
                    delay_context,
                    delay_feat
                ],
                dim=-1
            )
        )

        future_hidden = self.future_norm(future_hidden + last_hidden_expand)

        delta = self.out_proj(future_hidden).squeeze(-1)  # (B, T_out, N)

        if self.use_residual:
            last_value = x[:, -1, :, 0]
            base = last_value.unsqueeze(1).expand(B, self.out_steps, N)
            pred = base + delta
        else:
            pred = delta

        return pred


# ============================================================
# 5. Train / Evaluate utilities
# ============================================================

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

            y_true = y[..., 0]

            pred_norm = model(x)
            pred = scaler.inverse_transform(pred_norm)

            preds.append(pred)
            reals.append(y_true)

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

        mae = mae_torch(p, r).item()
        mape = mape_torch(p, r).item()
        rmse = rmse_torch(p, r).item()

        amae.append(mae)
        amape.append(mape)
        armse.append(rmse)

        print(
            "Evaluate PDFormer on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}".format(
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


# ============================================================
# 6. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data", type=str, default="./data/NYCBike")
    parser.add_argument("--adjdata", type=str, default="./data/NYCBike/adj_mx.npy")
    parser.add_argument("--adjtype", type=str, default="doubletransition")

    parser.add_argument("--in_dim", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=295)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=99)

    parser.add_argument("--max_time_index", type=int, default=288)
    parser.add_argument("--max_delay", type=int, default=12)

    parser.add_argument("--short_top_k", type=int, default=20)
    parser.add_argument("--long_top_k", type=int, default=20)

    parser.add_argument("--no_residual", action="store_true")
    parser.add_argument("--use_time_emb", action="store_true")

    parser.add_argument("--save", type=str, default="./experiment/nycbike/pdformer/PDFormer")

    args = parser.parse_args()

    set_seed(args.seed)

    if torch.cuda.is_available() and "cpu" not in args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    adj_norm, short_mask, inferred_num_nodes = build_graph_tensors(
        adjdata=args.adjdata,
        adjtype=args.adjtype,
        num_nodes=args.num_nodes,
        short_top_k=args.short_top_k
    )
    args.num_nodes = inferred_num_nodes

    dataloader = util.load_dataset(
        args.data,
        args.batch_size,
        args.batch_size,
        args.batch_size
    )

    scaler = dataloader["scaler"]
    out_steps = dataloader["y_train"].shape[1]

    print("Detected output steps:", out_steps)
    print("Detected train x shape:", dataloader["x_train"].shape)
    print("Detected train y shape:", dataloader["y_train"].shape)

    model = PDFormerStableBaseline(
        num_nodes=args.num_nodes,
        adj_norm=adj_norm,
        short_mask=short_mask,
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        out_steps=out_steps,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_time_index=args.max_time_index,
        max_delay=args.max_delay,
        long_top_k=args.long_top_k,
        use_residual=not args.no_residual,
        use_time_emb=args.use_time_emb
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
    print("Train PDFormer-style baseline with {} parameters".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    ))
    print("Start training PDFormer-style baseline...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        dataloader["train_loader"].shuffle()

        train_losses = []
        t1 = time.time()

        for x, y in dataloader["train_loader"].get_iterator():
            x = to_tensor(x, device)
            y = to_tensor(y, device)

            y_true = y[..., 0]

            optimizer.zero_grad()

            pred_norm = model(x)
            pred = scaler.inverse_transform(pred_norm)

            loss = mae_torch(pred, y_true)

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

        val_mae = mae_torch(val_pred, val_real).item()
        val_mape = mape_torch(val_pred, val_real).item()
        val_rmse = rmse_torch(val_pred, val_real).item()

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
            print("[Saved] Best model -> {}".format(best_path))
        else:
            wait += 1
            print("[EarlyStopping] No improvement. wait={}/{}".format(wait, args.patience))

        if wait >= args.patience:
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
