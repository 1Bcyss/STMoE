# rgdan_baseline.py
# -*- coding: utf-8 -*-
"""
Self-contained RGDAN baseline for TESTAM/STGateMoE npz data format.

特点：
1. 不需要 RGDAN 官方仓库的 model.py；
2. 不需要 RGDAN 官方仓库的 utils.py；
3. 可以直接放在 TESTAM-main 根目录运行；
4. 读取现有的 train.npz / val.npz / test.npz；
5. 输出每个 horizon 的 MAE / MAPE / RMSE；
6. 保存 prediction.npz，格式与 stgcn_baseline.py 类似。
"""

import os
import math
import time
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import util


class GCN(nn.Module):
    def __init__(self, k, d):
        super().__init__()
        D = k * d
        self.fc = nn.Linear(2 * D, D)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, X, STE, A):
        X = torch.cat((X, STE), dim=-1)
        H = F.gelu(self.fc(X))
        H = torch.einsum("btnd,nm->btmd", H, A)
        return self.dropout(H.contiguous())


class RandomGAT(nn.Module):
    def __init__(self, k, d, adj, device):
        super().__init__()
        D = k * d
        self.d = d
        self.K = k
        self.device = device

        num_nodes = adj.shape[0]
        self.fc = nn.Linear(2 * D, D)
        self.register_buffer("adj", adj.float())

        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10, device=device))
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes, device=device))

    def forward(self, X, STE):
        X = torch.cat((X, STE), dim=-1)
        H = F.gelu(self.fc(X))
        H = torch.cat(torch.split(H, self.d, dim=-1), dim=0)

        adp = torch.mm(self.nodevec1, self.nodevec2)
        zero_vec = torch.tensor(-9e15, device=self.device)
        adp = torch.where(self.adj > 0, adp, zero_vec)
        adj = F.softmax(adp, dim=-1)

        H = torch.einsum("nm,btmd->btnd", adj, H)
        H = torch.cat(torch.split(H, H.shape[0] // self.K, dim=0), dim=-1)
        return F.gelu(H.contiguous())


class STEmbModel(nn.Module):
    def __init__(self, se_dim, te_dim, out_dim, device):
        super().__init__()
        self.te_dim = te_dim
        self.device = device

        self.se_fc1 = nn.Linear(se_dim, out_dim)
        self.se_fc2 = nn.Linear(out_dim, out_dim)

        self.te_fc1 = nn.Linear(te_dim, out_dim)
        self.te_fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, SE, TE):
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.se_fc2(F.gelu(self.se_fc1(SE)))

        dayofweek = F.one_hot(TE[..., 0], num_classes=7)
        timeofday = F.one_hot(TE[..., 1], num_classes=self.te_dim - 7)

        TE = torch.cat((dayofweek, timeofday), dim=-1)
        TE = TE.unsqueeze(2).float().to(self.device)
        TE = self.te_fc2(F.gelu(self.te_fc1(TE)))

        return SE + TE


class TemporalAttentionModel(nn.Module):
    def __init__(self, k, d, device):
        super().__init__()
        D = k * d
        self.K = k
        self.d = d
        self.device = device

        self.fc_q = nn.Linear(2 * D, D)
        self.fc_k = nn.Linear(2 * D, D)
        self.fc_v = nn.Linear(2 * D, D)
        self.fc_o1 = nn.Linear(D, D)
        self.fc_o2 = nn.Linear(D, D)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, X, STE, mask=True):
        X = torch.cat((X, STE), dim=-1)

        query = F.gelu(self.fc_q(X))
        key = F.gelu(self.fc_k(X))
        value = F.gelu(self.fc_v(X))

        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)

        query = query.transpose(2, 1)
        key = key.transpose(1, 2).transpose(2, 3)
        value = value.transpose(2, 1)

        attention = torch.matmul(query, key) / (self.d ** 0.5)

        if mask:
            num_steps = X.shape[1]
            m = torch.ones(num_steps, num_steps, device=self.device)
            m = torch.tril(m).bool()
            zero_vec = torch.tensor(-9e15, device=self.device)
            attention = torch.where(m, attention, zero_vec)

        attention = F.softmax(attention, dim=-1)
        H = torch.matmul(attention, value)

        H = H.transpose(2, 1)
        H = torch.cat(torch.split(H, H.shape[0] // self.K, dim=0), dim=-1)
        H = self.fc_o2(F.gelu(self.fc_o1(H)))
        return self.dropout(H)


class GatedFusionModel(nn.Module):
    def __init__(self, k, d):
        super().__init__()
        D = k * d

        self.fc_s = nn.Linear(D, D)
        self.fc_t = nn.Linear(D, D)
        self.fc_o1 = nn.Linear(D, D)
        self.fc_o2 = nn.Linear(D, D)

    def forward(self, HS, HT):
        XS = self.fc_s(HS)
        XT = self.fc_t(HT)

        z = torch.sigmoid(XS + XT)
        H = z * HS + (1.0 - z) * HT
        H = self.fc_o2(F.gelu(self.fc_o1(H)))
        return H


class STAttModel(nn.Module):
    def __init__(self, k, d, supports, device):
        super().__init__()
        D = k * d

        self.fc = nn.Linear(7 * D, D)

        self.gcn = GCN(k, d)
        self.gat1 = RandomGAT(k, d, supports[0], device)
        self.gat2 = RandomGAT(k, d, supports[0], device)
        self.gat3 = RandomGAT(k, d, supports[1], device)
        self.gat4 = RandomGAT(k, d, supports[1], device)

        self.temporal_attn = TemporalAttentionModel(k, d, device)
        self.fusion = GatedFusionModel(k, d)

    def forward(self, X, STE, adaptive_adj, mask=True):
        HS1 = self.gat1(X, STE)
        HS2 = self.gat2(HS1, STE)

        HS3 = self.gat3(X, STE)
        HS4 = self.gat4(HS3, STE)

        HS5 = self.gcn(X, STE, adaptive_adj)
        HS6 = self.gcn(HS5, STE, adaptive_adj)

        HS = torch.cat((X, HS1, HS2, HS3, HS4, HS5, HS6), dim=-1)
        HS = F.gelu(self.fc(HS))

        HT = self.temporal_attn(X, STE, mask=mask)
        H = self.fusion(HS, HT)

        return X + H


class TransformAttentionModel(nn.Module):
    def __init__(self, k, d):
        super().__init__()
        D = k * d
        self.K = k
        self.d = d

        self.fc_q = nn.Linear(D, D)
        self.fc_k = nn.Linear(D, D)
        self.fc_v = nn.Linear(D, D)
        self.fc_o1 = nn.Linear(D, D)
        self.fc_o2 = nn.Linear(D, D)

    def forward(self, X, STE_P, STE_Q):
        query = F.gelu(self.fc_q(STE_Q))
        key = F.gelu(self.fc_k(STE_P))
        value = F.gelu(self.fc_v(X))

        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)

        query = query.transpose(2, 1)
        key = key.transpose(1, 2).transpose(2, 3)
        value = value.transpose(2, 1)

        attention = torch.matmul(query, key) / (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)

        H = torch.matmul(attention, value)
        H = H.transpose(2, 1)

        H = torch.cat(torch.split(H, H.shape[0] // self.K, dim=0), dim=-1)
        H = self.fc_o2(F.gelu(self.fc_o1(H)))
        return H


class RGDAN(nn.Module):
    def __init__(self, k, d, se_dim, te_dim, p, layers, device, supports, num_nodes):
        super().__init__()

        D = k * d
        self.p = p
        self.layers = layers
        self.device = device

        self.input_fc1 = nn.Linear(1, D)
        self.input_fc2 = nn.Linear(D, D)

        self.st_embedding = STEmbModel(se_dim, te_dim, D, device)

        self.encoder_blocks = nn.ModuleList([
            STAttModel(k, d, supports, device)
            for _ in range(layers)
        ])

        self.transform_attn = TransformAttentionModel(k, d)

        self.decoder_blocks = nn.ModuleList([
            STAttModel(k, d, supports, device)
            for _ in range(layers)
        ])

        self.output_fc1 = nn.Linear(D, D)
        self.output_fc2 = nn.Linear(D, 1)

        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10, device=device))
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes, device=device))

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, X, SE, TE):
        adaptive_adj = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

        X = X.unsqueeze(-1)
        X = self.input_fc2(F.gelu(self.input_fc1(X)))

        STE = self.st_embedding(SE, TE)
        STE_P = STE[:, :self.p]
        STE_Q = STE[:, self.p:]

        for block in self.encoder_blocks:
            X = block(X, STE_P, adaptive_adj, mask=True)

        X = self.transform_attn(X, STE_P, STE_Q)

        for block in self.decoder_blocks:
            X = block(X, STE_Q, adaptive_adj, mask=True)

        X = self.output_fc2(self.dropout(F.gelu(self.output_fc1(X))))
        return X.squeeze(-1)


def set_seed(seed):
    if seed == -1:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_tensor(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, dtype=dtype, device=device)


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


def _tod_to_slot(tod_array, num_slots):
    slots = np.floor(tod_array * num_slots + 1e-6).astype(np.int64)
    return np.mod(slots, num_slots)


def build_te_from_npz(x, y, global_start_idx, init_slot, start_dow, num_slots):
    B, P, N, C = x.shape
    Q = y.shape[1]

    if C < 2:
        raise ValueError("x 的最后一维少于 2，缺少 time-of-day 特征。")

    hist_tod = x[:, :, 0, 1]

    if y.shape[-1] >= 2:
        fut_tod = y[:, :, 0, 1]
    else:
        last_slot = _tod_to_slot(hist_tod[:, -1], num_slots)
        fut_slots = (last_slot[:, None] + np.arange(1, Q + 1)[None, :]) % num_slots
        fut_tod = fut_slots / float(num_slots)

    tod = np.concatenate([hist_tod, fut_tod], axis=1)
    timeofday = _tod_to_slot(tod, num_slots)

    offsets = np.arange(P + Q, dtype=np.int64)[None, :]
    sample_starts = global_start_idx + np.arange(B, dtype=np.int64)[:, None]
    absolute_slots = init_slot + sample_starts + offsets
    dayofweek = (start_dow + absolute_slots // num_slots) % 7

    te = np.stack([dayofweek, timeofday], axis=-1).astype(np.int64)
    return te


def load_npz_rgdan_dataset(data_dir, P, Q, start_dow, time_slot):
    num_slots = 24 * 60 // time_slot

    train = np.load(os.path.join(data_dir, "train.npz"))
    val = np.load(os.path.join(data_dir, "val.npz"))
    test = np.load(os.path.join(data_dir, "test.npz"))

    x_train, y_train = train["x"], train["y"]
    x_val, y_val = val["x"], val["y"]
    x_test, y_test = test["x"], test["y"]

    if x_train.shape[1] != P:
        raise ValueError(f"P={P} 与数据输入长度不一致，x_train.shape[1]={x_train.shape[1]}")
    if y_train.shape[1] != Q:
        raise ValueError(f"Q={Q} 与数据预测长度不一致，y_train.shape[1]={y_train.shape[1]}")

    mean = x_train[..., 0].mean()
    std = x_train[..., 0].std()

    def norm_x(x):
        return ((x[..., 0] - mean) / std).astype(np.float32)

    trainX = norm_x(x_train)
    valX = norm_x(x_val)
    testX = norm_x(x_test)

    trainY = y_train[..., 0].astype(np.float32)
    valY = y_val[..., 0].astype(np.float32)
    testY = y_test[..., 0].astype(np.float32)

    init_slot = int(_tod_to_slot(x_train[0:1, 0, 0, 1], num_slots).reshape(-1)[0])

    num_train = x_train.shape[0]
    num_val = x_val.shape[0]

    trainTE = build_te_from_npz(x_train, y_train, 0, init_slot, start_dow, num_slots)
    valTE = build_te_from_npz(x_val, y_val, num_train, init_slot, start_dow, num_slots)
    testTE = build_te_from_npz(x_test, y_test, num_train + num_val, init_slot, start_dow, num_slots)

    return trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, float(mean), float(std)


def load_se_file(path, num_nodes):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    start = 0
    first = lines[0].split()

    if len(first) == 2:
        try:
            int(first[0])
            int(first[1])
            start = 1
        except Exception:
            start = 0

    for line in lines[start:]:
        parts = [float(v) for v in line.split()]
        if len(parts) > 2:
            parts = parts[1:]
        data.append(parts)

    se = np.asarray(data, dtype=np.float32)

    if se.shape[0] != num_nodes:
        raise ValueError(f"SE 文件节点数不匹配：读取到 {se.shape[0]}，期望 {num_nodes}")

    return se


def build_spectral_se_from_adj(adj_matrix, se_dim):
    """
    根据已经归一化后的邻接矩阵构造谱嵌入。
    adj_matrix 应该是二维矩阵，不再直接读取 pkl。
    """
    A = np.asarray(adj_matrix, dtype=np.float64)
    A = np.nan_to_num(A)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 1.0)

    N = A.shape[0]
    deg = A.sum(axis=1)
    deg[deg <= 1e-12] = 1e-12

    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    L = np.eye(N) - D_inv_sqrt @ A @ D_inv_sqrt

    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        idx = np.argsort(eigvals)
        eigvecs = eigvecs[:, idx]
        se = eigvecs[:, 1:se_dim + 1]
    except Exception:
        rng = np.random.RandomState(99)
        se = rng.normal(0, 0.1, size=(N, se_dim))

    if se.shape[1] < se_dim:
        pad = np.zeros((N, se_dim - se.shape[1]), dtype=np.float64)
        se = np.concatenate([se, pad], axis=1)

    se = se.astype(np.float32)
    se = (se - se.mean(axis=0, keepdims=True)) / (se.std(axis=0, keepdims=True) + 1e-6)
    return se


def batch_predict(model, X, TE, SE, mean, std, device, batch_size):
    model.eval()
    preds = []

    num_samples = X.shape[0]
    num_batch = math.ceil(num_samples / batch_size)

    with torch.no_grad():
        for batch_idx in range(num_batch):
            start = batch_idx * batch_size
            end = min(num_samples, (batch_idx + 1) * batch_size)

            batchX = to_tensor(X[start:end], device, dtype=torch.float32)
            batchTE = to_tensor(TE[start:end], device, dtype=torch.long)

            pred_norm = model(batchX, SE, batchTE)
            pred = pred_norm * std + mean
            preds.append(pred.detach().cpu())

    return torch.cat(preds, dim=0)


def print_horizon_metrics(pred, real):
    horizon = real.size(1)
    amae, amape, armse = [], [], []

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
            "Evaluate RGDAN on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}".format(
                h + 1, mae, mape, rmse
            )
        )

    print(
        "On average over {} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}".format(
            horizon, np.mean(amae), np.mean(amape), np.mean(armse)
        )
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data", type=str, default="./data/METR-LA")
    parser.add_argument("--adjdata", type=str, default="./data/METR-LA/adj_mx.pkl")
    parser.add_argument("--adjtype", type=str, default="doubletransition")
    parser.add_argument("--dataset", type=str, default="METR-LA")

    parser.add_argument("--P", type=int, default=12)
    parser.add_argument("--Q", type=int, default=12)
    parser.add_argument("--time_slot", type=int, default=5)
    parser.add_argument("--start_dow", type=int, default=0)

    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--se_file", type=str, default=None)
    parser.add_argument("--se_dim", type=int, default=64)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--decay_epoch", type=int, default=20)
    parser.add_argument("--seed", type=int, default=99)

    parser.add_argument("--save", type=str, default="./experiment/rgdan/metrla/RGDAN_METRLA")

    args = parser.parse_args()
    print(args)

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")

    print("Loading npz dataset...")

    trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, mean, std = load_npz_rgdan_dataset(
        args.data,
        args.P,
        args.Q,
        args.start_dow,
        args.time_slot
    )

    print("mean={:.6f}, std={:.6f}".format(mean, std))
    print("trainX:", trainX.shape, "trainTE:", trainTE.shape, "trainY:", trainY.shape)
    print("valX:  ", valX.shape, "valTE:  ", valTE.shape, "valY:  ", valY.shape)
    print("testX: ", testX.shape, "testTE:", testTE.shape, "testY:", testY.shape)

    if args.adjdata and os.path.exists(args.adjdata):
        sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
        supports = [torch.tensor(np.asarray(a), dtype=torch.float32, device=device) for a in adj_mx]
        num_nodes = len(sensor_ids)
    else:
        num_nodes = trainX.shape[-1]
        supports = [
            torch.eye(num_nodes, dtype=torch.float32, device=device),
            torch.eye(num_nodes, dtype=torch.float32, device=device)
        ]

    if len(supports) == 1:
        supports = [supports[0], supports[0]]

    print("num_nodes:", num_nodes)

    if args.se_file is not None and os.path.exists(args.se_file):
        SE_np = load_se_file(args.se_file, num_nodes)
        print("Loaded SE from file:", args.se_file, SE_np.shape)
    else:
        SE_np = build_spectral_se_from_adj(supports[0].detach().cpu().numpy(), args.se_dim)
        print("Built spectral SE from adjacency:", SE_np.shape)

    SE = torch.tensor(SE_np, dtype=torch.float32, device=device)

    TEmbsize = (24 * 60 // args.time_slot) + 7

    model = RGDAN(
        args.K,
        args.d,
        SE.shape[1],
        TEmbsize,
        args.P,
        args.L,
        device,
        supports,
        num_nodes
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epoch, gamma=0.3)

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    best_path = args.save + "_best.pth"

    print("Train RGDAN baseline with {} parameters".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    ))
    print("Start training RGDAN baseline...")

    best_val = float("inf")
    wait = 0
    train_times = []
    val_times = []

    num_train = trainX.shape[0]

    for epoch in range(1, args.epochs + 1):
        model.train()

        permutation = np.random.permutation(num_train)
        trainX = trainX[permutation]
        trainTE = trainTE[permutation]
        trainY = trainY[permutation]

        train_losses = []
        t1 = time.time()

        num_batch = math.ceil(num_train / args.batch_size)

        for batch_idx in range(num_batch):
            start = batch_idx * args.batch_size
            end = min(num_train, (batch_idx + 1) * args.batch_size)

            batchX = to_tensor(trainX[start:end], device, dtype=torch.float32)
            batchTE = to_tensor(trainTE[start:end], device, dtype=torch.long)
            batchY = to_tensor(trainY[start:end], device, dtype=torch.float32)

            optimizer.zero_grad()

            pred_norm = model(batchX, SE, batchTE)
            pred = pred_norm * std + mean

            loss = masked_mae_torch(pred, batchY, 0.0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_losses.append(loss.item())

        t2 = time.time()
        train_times.append(t2 - t1)

        s1 = time.time()
        val_pred = batch_predict(model, valX, valTE, SE, mean, std, device, args.batch_size)
        s2 = time.time()
        val_times.append(s2 - s1)

        val_real = torch.tensor(valY, dtype=torch.float32)

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

        scheduler.step()

        if wait >= args.patience:
            print("Early Termination!")
            break

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_times)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_times)))
    print("Load best model:", best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))

    test_pred = batch_predict(model, testX, testTE, SE, mean, std, device, args.batch_size)
    test_real = torch.tensor(testY, dtype=torch.float32)

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