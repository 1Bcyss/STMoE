# dcrnn_baseline.py
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


class DiffusionConv1D(nn.Module):
    def __init__(self, input_dim, output_dim, supports_len=2, max_diffusion_step=2):
        super().__init__()
        self.max_diffusion_step = max_diffusion_step
        self.supports_len = supports_len
        total_dim = input_dim * (1 + supports_len * max_diffusion_step)
        self.linear = nn.Linear(total_dim, output_dim)

    def nconv(self, x, A):
        # x: (B, N, C), A: (N, N)
        return torch.einsum("bnc,nm->bmc", x, A).contiguous()

    def forward(self, x, supports):
        # x: (B, N, C)
        out = [x]
        for A in supports:
            x1 = self.nconv(x, A)
            out.append(x1)
            for _ in range(2, self.max_diffusion_step + 1):
                x1 = self.nconv(x1, A)
                out.append(x1)

        h = torch.cat(out, dim=-1)
        h = self.linear(h)
        return h


class DCGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, supports_len, max_diffusion_step=2):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.gate_conv = DiffusionConv1D(
            input_dim=input_dim + hidden_dim,
            output_dim=2 * hidden_dim,
            supports_len=supports_len,
            max_diffusion_step=max_diffusion_step
        )

        self.update_conv = DiffusionConv1D(
            input_dim=input_dim + hidden_dim,
            output_dim=hidden_dim,
            supports_len=supports_len,
            max_diffusion_step=max_diffusion_step
        )

    def forward(self, x, h, supports):
        # x: (B, N, input_dim)
        # h: (B, N, hidden_dim)
        combined = torch.cat([x, h], dim=-1)
        gates = torch.sigmoid(self.gate_conv(combined, supports))
        r, u = torch.chunk(gates, 2, dim=-1)

        candidate_input = torch.cat([x, r * h], dim=-1)
        c = torch.tanh(self.update_conv(candidate_input, supports))

        h_new = u * h + (1.0 - u) * c
        return h_new


class DCRNNBaseline(nn.Module):
    """
    DCRNN-style baseline:
    - encoder: DCGRU over historical sequence
    - output: use final hidden state to directly predict 12 horizons
    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
        out_steps,
        supports_len,
        num_layers=2,
        max_diffusion_step=2,
        dropout=0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_steps = out_steps
        self.dropout = nn.Dropout(dropout)

        cells = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            cells.append(
                DCGRUCell(
                    input_dim=cur_input_dim,
                    hidden_dim=hidden_dim,
                    supports_len=supports_len,
                    max_diffusion_step=max_diffusion_step
                )
            )
        self.cells = nn.ModuleList(cells)
        self.proj = nn.Linear(hidden_dim, out_steps)

    def forward(self, x, supports):
        # x: (B, T, N, C)
        B, T, N, C = x.shape
        device = x.device

        hidden_states = [
            torch.zeros(B, N, self.hidden_dim, device=device)
            for _ in range(self.num_layers)
        ]

        for t in range(T):
            input_t = x[:, t, :, :]  # (B, N, C)

            for layer_idx, cell in enumerate(self.cells):
                h = hidden_states[layer_idx]
                h_new = cell(input_t, h, supports)
                h_new = self.dropout(h_new)
                hidden_states[layer_idx] = h_new
                input_t = h_new

        final_h = hidden_states[-1]      # (B, N, hidden)
        out = self.proj(final_h)         # (B, N, H)
        out = out.permute(0, 2, 1)       # (B, H, N)
        return out


def to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.float().to(device)
    return torch.from_numpy(x).float().to(device)


def evaluate(model, loader, scaler, supports, device):
    model.eval()
    preds, reals = [], []

    with torch.no_grad():
        for x, y in loader.get_iterator():
            x = to_tensor(x, device)
            y = to_tensor(y, device)

            y_speed = y[..., 0]  # (B, H, N)
            pred_norm = model(x, supports)
            pred = scaler.inverse_transform(pred_norm)

            preds.append(pred)
            reals.append(y_speed)

    preds = torch.cat(preds, dim=0)
    reals = torch.cat(reals, dim=0)
    return preds, reals


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
            "Evaluate DCRNN on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}".format(
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
    parser.add_argument("--in_dim", type=int, default=2)
    parser.add_argument("--num_nodes", type=int, default=207)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--max_diffusion_step", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--save", type=str, default="./experiment/metrla/baselines/dcrnn/METR-LA_DCRNN")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")

    if args.adjdata and os.path.exists(args.adjdata):
        sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
        supports = [torch.tensor(np.array(a), dtype=torch.float32, device=device) for a in adj_mx]
        args.num_nodes = len(sensor_ids)
    else:
        supports = [torch.eye(args.num_nodes, dtype=torch.float32, device=device)]

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader["scaler"]
    out_steps = dataloader["y_train"].shape[1]

    model = DCRNNBaseline(
        input_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        out_steps=out_steps,
        supports_len=len(supports),
        num_layers=args.num_layers,
        max_diffusion_step=args.max_diffusion_step,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    best_path = args.save + "_best.pth"

    best_val = float("inf")
    wait = 0
    train_times = []
    val_times = []

    print(args)
    print("Train DCRNN baseline with {} parameters".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    ))
    print("Start training DCRNN baseline...")

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
            pred_norm = model(x, supports)
            pred = scaler.inverse_transform(pred_norm)
            loss = masked_mae_torch(pred, y_speed, 0.0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_losses.append(loss.item())

        t2 = time.time()
        train_times.append(t2 - t1)

        s1 = time.time()
        val_pred, val_real = evaluate(model, dataloader["val_loader"], scaler, supports, device)
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

    test_pred, test_real = evaluate(model, dataloader["test_loader"], scaler, supports, device)

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