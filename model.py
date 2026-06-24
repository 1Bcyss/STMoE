import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy as cp


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('nvlc,vw->nwlc', (x, A))
        return x.contiguous()

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, supports_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * supports_len + 1) * c_in
        self.mlp = nn.Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=-1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class QKVAttention(nn.Module):
    def __init__(self, in_dim, hidden_size, dropout, num_heads=4):
        super(QKVAttention, self).__init__()
        self.query = nn.Linear(in_dim, hidden_size, bias=False)
        self.key = nn.Linear(in_dim, hidden_size, bias=False)
        self.value = nn.Linear(in_dim, hidden_size, bias=False)
        self.num_heads = num_heads
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        assert hidden_size % num_heads == 0

    def forward(self, x, kv=None):
        if kv is None:
            kv = x

        query = self.query(x)
        key = self.key(kv)
        value = self.value(kv)
        num_heads = self.num_heads

        if num_heads > 1:
            query = torch.cat(torch.chunk(query, num_heads, dim=-1), dim=0)
            key = torch.cat(torch.chunk(key, num_heads, dim=-1), dim=0)
            value = torch.cat(torch.chunk(value, num_heads, dim=-1), dim=0)

        d = value.size(-1)
        energy = torch.matmul(query, key.transpose(-1, -2))
        energy = energy / (d ** 0.5)
        score = torch.softmax(energy, dim=-1)
        head_out = torch.matmul(score, value)
        out = torch.cat(torch.chunk(head_out, num_heads, dim=0), dim=-1)
        return self.dropout(self.proj(out))

#层归一化
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(*normalized_shape))
        self.beta = nn.Parameter(torch.zeros(*normalized_shape))

    def forward(self, x):
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]
        mean = x.mean(dim=dims, keepdims=True)
        std = x.std(dim=dims, keepdims=True, unbiased=False)
        x_norm = (x - mean) / (std + self.eps)
        out = x_norm * self.gamma + self.beta
        return out

#残差连接
class SkipConnection(nn.Module):
    def __init__(self, module, norm):
        super(SkipConnection, self).__init__()
        self.module = module
        self.norm = norm

    def forward(self, x, aux=None):
        return self.norm(x + self.module(x, aux))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, in_dim, hidden_size, dropout, activation=nn.GELU()):
        super(PositionwiseFeedForward, self).__init__()
        self.act = activation
        self.l1 = nn.Linear(in_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, in_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, kv=None):
        return self.dropout(self.l2(self.act(self.l1(x))))

#把时间编号从 (B, T) 变成 (B, T, 32)
class TemporalInformationEmbedding(nn.Module):
    def __init__(self, hidden_size, vocab_size, freq_act=torch.sin, n_freq=1):
        super(TemporalInformationEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.freq_act = freq_act
        self.n_freq = n_freq

    def forward(self, x):
        x_emb = self.embedding(x)
        x_weight = self.linear(x_emb)
        if self.n_freq == 0:
            return x_weight
        if self.n_freq == x_emb.size(-1):
            return self.freq_act(x_weight)
        x_linear = x_weight[..., self.n_freq:]
        x_act = self.freq_act(x_weight[..., :self.n_freq])
        return torch.cat([x_linear, x_act], dim=-1)


class TemporalLSTMExpert(nn.Module):
    """
    改进版 LSTM 时间专家：Encoder-Decoder 结构。
    输入：
        history_time_index: (B, T_in)
        speed:              (B, N, T_in, C)
        future_time_index:  (B, T_out)
    输出：
        out:          (B, N, T_out, out_dim)
        hidden_list:  [future_hidden]，用于门控网络
    """
    def __init__(
        self,
        hidden_size,
        num_nodes,
        layers,
        dropout,
        in_dim=1,
        out_dim=1,
        vocab_size=288,
        activation=nn.ReLU()
    ):
        super(TemporalLSTMExpert, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_nodes = num_nodes
        self.layers = layers
        self.act = activation
        self.embedding = TemporalInformationEmbedding(hidden_size, vocab_size=vocab_size)
        #负责把原始速度/流量特征映射到隐藏维度
        self.spd_proj = nn.Linear(in_dim, hidden_size)
        self.encoder_fuse = nn.Linear(hidden_size * 2, hidden_size)
        self.encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )

        self.future_fuse = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )

        self.norm = nn.LayerNorm(hidden_size)

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_dim)
        )

    def forward(self, history_time_index, speed=None, future_time_index=None):
        B, T_in = history_time_index.shape

        if future_time_index is None:
            future_time_index = history_time_index

        T_out = future_time_index.shape[1]

        hist_time_feat = self.embedding(history_time_index)
        hist_time_feat = hist_time_feat.unsqueeze(1).expand(
            B, self.num_nodes, T_in, self.hidden_size
        )

        if speed is None:
            speed = torch.zeros(
                B,
                self.num_nodes,
                T_in,
                self.in_dim,
                device=hist_time_feat.device,
                dtype=hist_time_feat.dtype
            )

        spd_feat = self.spd_proj(speed)
        enc_feat = self.encoder_fuse(torch.cat([hist_time_feat, spd_feat], dim=-1))
        enc_feat = enc_feat.reshape(B * self.num_nodes, T_in, self.hidden_size)
        _, (h_n, _) = self.encoder(enc_feat)
        last_hidden = h_n[-1]
        last_hidden = last_hidden.reshape(B, self.num_nodes, self.hidden_size)
        future_time_feat = self.embedding(future_time_index)
        future_time_feat = future_time_feat.unsqueeze(1).expand(
            B, self.num_nodes, T_out, self.hidden_size
        )
        hidden_expand = last_hidden.unsqueeze(2).expand(
            B, self.num_nodes, T_out, self.hidden_size
        )
        future_hidden = self.future_fuse(
            torch.cat([hidden_expand, future_time_feat], dim=-1)
        )
        future_hidden = self.norm(future_hidden + hidden_expand)
        out = self.out_proj(self.act(future_hidden))
        return out, [future_hidden]

#加工时间向量，并扩展到所有节点(B, T, 32) → (B, N, T, 32)
class FutureTimeEmbedding(nn.Module):
    """
    轻量未来时间隐藏表示生成器。
    用未来时间索引 next_time_index 生成 h_future，
    给图专家和注意力专家提供未来时间隐藏表示。
    输入:
        time_index: (B, T)
    输出:
        time_feat: (B, N, T, hidden_size)
    """
    def __init__(self, hidden_size, num_nodes, vocab_size=288, dropout=0.1):
        super(FutureTimeEmbedding, self).__init__()
        self.hidden_size = hidden_size
        self.num_nodes = num_nodes
        self.time_embedding = TemporalInformationEmbedding(
            hidden_size=hidden_size,
            vocab_size=vocab_size
        )

        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, time_index):
        B, T = time_index.shape

        time_feat = self.time_embedding(time_index)
        time_feat = self.norm(self.proj(time_feat))

        time_feat = time_feat.unsqueeze(1).expand(
            B, self.num_nodes, T, self.hidden_size
        )

        return time_feat
"""
输入历史交通数据 x: (B, C, N, T)
        ↓
调整维度 + 输入投影
(B, C, N, T) → (B, N, T, hidden_size)
        ↓
多层时空编码
每层包括：
    时间注意力 QKVAttention
    空间图卷积 GCN
    未来时间交叉注意力 ed_layer
    前馈网络 FFN
        ↓
得到最终隐藏表示 x
        ↓
投影成 hidden residual + prediction
        ↓
输出：
    更新隐藏表示
    预测结果
    每层隐藏状态
"""
class STModel(nn.Module):
    def __init__(
        self,
        hidden_size,
        supports_len,
        num_nodes,
        dropout,
        layers,
        out_dim=1,
        in_dim=2,
        spatial=False,
        activation=nn.ReLU()
    ):
        super(STModel, self).__init__()
        self.spatial = spatial
        self.act = activation
        self.out_dim = out_dim

        s_gcn = gcn(
            c_in=hidden_size,
            c_out=hidden_size,
            dropout=dropout,
            supports_len=supports_len,
            order=2
        )
        t_attn = QKVAttention(
            in_dim=hidden_size,
            hidden_size=hidden_size,
            dropout=dropout
        )
        ff = PositionwiseFeedForward(
            in_dim=hidden_size,
            hidden_size=4 * hidden_size,
            dropout=dropout
        )
        norm = LayerNorm(normalized_shape=(hidden_size,))

        self.start_linear = nn.Linear(in_dim, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size + out_dim)

        self.temporal_layers = nn.ModuleList()
        self.spatial_layers = nn.ModuleList()
        self.ed_layers = nn.ModuleList()
        self.ff = nn.ModuleList()

        for _ in range(layers):
            self.temporal_layers.append(SkipConnection(cp(t_attn), cp(norm)))
            self.spatial_layers.append(SkipConnection(cp(s_gcn), cp(norm)))
            self.ed_layers.append(SkipConnection(cp(t_attn), cp(norm)))
            self.ff.append(SkipConnection(cp(ff), cp(norm)))

    def forward(self, x, prev_hidden, supports):
        x = self.start_linear(x.permute(0, 2, 3, 1))
        x_start = x
        hiddens = []

        for temporal_layer, spatial_layer, ed_layer, ff in zip(
            self.temporal_layers, self.spatial_layers, self.ed_layers, self.ff
        ):
            if not self.spatial:
                x1 = temporal_layer(x)
                x_attn = spatial_layer(x1, supports)
            else:
                x1 = spatial_layer(x, supports)
                x_attn = temporal_layer(x1)

            if prev_hidden is not None:
                x_attn = ed_layer(x_attn, prev_hidden[-1])

            x = ff(x_attn)
            hiddens.append(x)

        out = self.proj(self.act(x))
        res, out = torch.split(out, [out.size(-1) - self.out_dim, self.out_dim], dim=-1)
        return x_start - res, out.contiguous(), hiddens


class GMANStyleBlock(nn.Module):
    """
    用于 AttentionModel 内部的 GMAN-style 时空注意力块。
    1. Spatial Attention：同一时间步内，不同节点之间做注意力；
    2. Temporal Attention：同一节点内，不同历史时间步之间做注意力；
    3. FFN + Residual + LayerNorm。
    """
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super(GMANStyleBlock, self).__init__()

        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.spatial_norm = nn.LayerNorm(hidden_size)
        self.temporal_norm = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout)
        )

        self.ffn_norm = nn.LayerNorm(hidden_size)

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
        # Feed Forward
        f = self.ffn(x)
        x = self.ffn_norm(x + f)
        return x


class AttentionModel(nn.Module):
    """
    注意力专家。
    1. 加入节点嵌入 node_emb；
    2. 加入时段嵌入 time_emb；
    3. 采用 GMAN-style 空间注意力 + 时间注意力；
    4. 利用未来时间隐藏表示进行未来步解码；
    5. 使用残差预测，增强训练稳定性。
    输入:
        x: (B, C, N, T)
    输出:
        out:    (B, N, T, out_dim)
        hidden: (B, N, T, hidden_size)
    """
    def __init__(
        self,
        hidden_size,
        layers,
        dropout,
        in_dim=2,
        out_dim=1,
        spatial=False,
        activation=nn.ReLU(),
        num_nodes=207,
        max_time_index=288,
        num_heads=4
    ):
        super(AttentionModel, self).__init__()

        self.spatial = spatial
        self.act = activation
        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_nodes = num_nodes
        self.max_time_index = max_time_index
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        self.time_emb = nn.Embedding(max_time_index, hidden_size)
        self.node_emb = nn.Embedding(num_nodes, hidden_size)

        self.blocks = nn.ModuleList([
            GMANStyleBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(layers)
        ])

        self.future_fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )

        self.future_norm = nn.LayerNorm(hidden_size)

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_dim)
        )

    def get_time_index(self, tod):
        return ((tod * self.max_time_index) % self.max_time_index).long()

    def forward(self, x, prev_hidden=None):
        """
        x: (B, C, N, T)
        prev_hidden:
            来自 FutureTimeEmbedding 的未来时间隐藏表示，
            形状通常为 [ (B, N, T, H) ]
        """
        B, C, N, T = x.shape
        device = x.device

        # (B, C, N, T) -> (B, T, N, C)
        x_seq = x.permute(0, 3, 2, 1).contiguous()

        # 历史 time-of-day，默认最后一个输入通道是 TOD
        hist_tod = x_seq[:, :, 0, -1]  # (B, T)
        hist_time_index = self.get_time_index(hist_tod)
        # 历史时间嵌入: (B, T, H) -> (B, T, N, H)
        hist_time_feat = self.time_emb(hist_time_index)
        hist_time_feat = hist_time_feat.unsqueeze(2).expand(
            B, T, N, self.hidden_size
        )
        # 节点嵌入: (N, H) -> (B, T, N, H)
        node_ids = torch.arange(N, device=device)
        node_feat = self.node_emb(node_ids)
        node_feat_hist = node_feat.view(1, 1, N, self.hidden_size).expand(
            B, T, N, self.hidden_size
        )

        # 输入映射 + 时间嵌入 + 节点嵌入
        h = self.input_proj(x_seq)
        h = h + hist_time_feat + node_feat_hist

        # GMAN-style 编码
        for block in self.blocks:
            h = block(h)

        # 使用最后一个历史时间步作为历史总结
        last_hidden = h[:, -1, :, :]  # (B, N, H)

        # 未来时间隐藏表示
        if prev_hidden is not None and len(prev_hidden) > 0:
            # prev_hidden[-1]: (B, N, T, H) -> (B, T, N, H)
            future_time_feat = prev_hidden[-1].permute(0, 2, 1, 3).contiguous()
        else:
            last_time_index = hist_time_index[:, -1]
            steps = torch.arange(1, T + 1, device=device).view(1, T)
            future_time_index = (last_time_index.view(B, 1) + steps) % self.max_time_index
            future_time_feat = self.time_emb(future_time_index.long())
            future_time_feat = future_time_feat.unsqueeze(2).expand(
                B, T, N, self.hidden_size
            )

        node_feat_future = node_feat.view(1, 1, N, self.hidden_size).expand(
            B, T, N, self.hidden_size
        )

        last_hidden_expand = last_hidden.unsqueeze(1).expand(
            B, T, N, self.hidden_size
        )

        future_hidden = self.future_fusion(
            torch.cat(
                [last_hidden_expand, future_time_feat, node_feat_future],
                dim=-1
            )
        )

        future_hidden = self.future_norm(future_hidden + last_hidden_expand)
        # 预测归一化空间中的速度变化量
        delta = self.out_proj(self.act(future_hidden))  # (B, T, N, out_dim)
        # 残差预测：未来速度 = 当前速度 + 变化量
        last_speed = x_seq[:, -1, :, 0]  # (B, N)
        base = last_speed.unsqueeze(1).unsqueeze(-1).expand(
            B, T, N, self.out_dim
        )
        out = base + delta  # (B, T, N, out_dim)

        # 转为 MoE 统一格式: (B, N, T, out_dim)
        out = out.permute(0, 2, 1, 3).contiguous()
        # hidden 用于门控网络: (B, N, T, H)
        hidden = future_hidden.permute(0, 2, 1, 3).contiguous()
        return out, hidden


class ExpertAdapter(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim)
        )
    def forward(self, x):
        return x + self.net(x)


class STContextGate(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_nodes,
        mem_hid=32,
        in_dim=2,
        out_dim=1,
        memory_size=20,
        use_time_gate=True,
        top_k=1,
        sparse_dispatch=False,
        max_tod=288,
        num_experts=3,
        time_gate_scale_init=0.01,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.use_time_gate = use_time_gate
        self.top_k = top_k
        self.sparse_dispatch = sparse_dispatch
        self.num_experts = num_experts
        self.time_gate_scale_init=time_gate_scale_init

        self.memory = nn.Parameter(torch.empty(memory_size, mem_hid))
        self.input_query = nn.Linear(in_dim, mem_hid, bias=False)

        self.time_emb = nn.Embedding(max_tod, mem_hid)
        self.time_norm = nn.LayerNorm(mem_hid)
        self.time_gate_scale = nn.Parameter(torch.tensor(time_gate_scale_init))
        branch_count = 1 + int(use_time_gate)
        self.context_fusion = nn.Sequential(
            nn.Linear(mem_hid * branch_count, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.expert_proj = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_experts)]
        )

        self.score_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1)
            )
            for _ in range(num_experts)
        ])

        self.conf_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1)
            )
            for _ in range(num_experts)
        ])

        self.time_gate_scale = nn.Parameter(torch.tensor(time_gate_scale_init))
        self.reset_parameters()

    def reset_parameters(self):
        for _, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if hasattr(self.time_norm, "weight") and self.time_norm.weight is not None:
            nn.init.ones_(self.time_norm.weight)
        if hasattr(self.time_norm, "bias") and self.time_norm.bias is not None:
            nn.init.zeros_(self.time_norm.bias)

        with torch.no_grad():
            self.time_gate_scale.fill_(self.time_gate_scale_init)

    def query_mem(self, x):
        q = self.input_query(x)
        energy = torch.matmul(q, self.memory.T)
        score = torch.softmax(energy, dim=-1)
        return torch.matmul(score, self.memory)

    def forward(self, input_x, hidden_list, time_index=None):
        """
        input_x:     (B, N, T, C)
        hidden_list: list of expert hidden states, each is (B, N, T, H)
        time_index:  (B, T), optional
        """
        B, N, T, _ = input_x.size()

        if len(hidden_list) != self.num_experts:
            raise ValueError(
                f"Gate expected {self.num_experts} experts, but got {len(hidden_list)} hidden states."
            )

        mem_feat = self.query_mem(input_x)
        contexts = [mem_feat]

        if self.use_time_gate and time_index is not None:
            time_feat = self.time_emb(time_index.long())
            time_feat = self.time_norm(time_feat)
            time_feat = time_feat.unsqueeze(1).expand(B, N, T, -1)
            contexts.append(self.time_gate_scale * time_feat)

        gate_context = self.context_fusion(torch.cat(contexts, dim=-1))

        scores = []
        for i, h in enumerate(hidden_list):
            h_proj = self.expert_proj[i](h)
            score_input = torch.cat([gate_context, h_proj], dim=-1)
            logits = self.score_heads[i](score_input).squeeze(-1)
            conf = self.conf_heads[i](h_proj).squeeze(-1)
            scores.append(logits + conf)

        scores = torch.stack(scores, dim=-1)

        if self.sparse_dispatch and self.top_k < scores.size(-1):
            _, topk_idx = torch.topk(scores, k=self.top_k, dim=-1)
            sparse_mask = torch.zeros_like(scores).scatter_(-1, topk_idx, 1.0)
            scores = scores.masked_fill(sparse_mask == 0, float('-inf'))

        return scores.unsqueeze(-2).expand(B, N, T, self.out_dim, scores.size(-1))


class STGateMoE(nn.Module):
    def __init__(
        self,
        num_nodes,
        dropout=0.3,
        in_dim=2,
        out_dim=1,
        hidden_size=32,
        layers=3,
        prob_mul=False,
        max_time_index=288,
        use_time_gate=True,
        top_k=1,
        sparse_dispatch=False,
        use_time_expert=True,
        use_graph_expert=True,
        use_attention_expert=True,
        time_gate_scale_init=0.01,
        **args
    ):
        super().__init__()
        self.dropout = dropout
        self.prob_mul = prob_mul
        self.supports_len = 2
        self.max_time_index = max_time_index
        self.sparse_dispatch = sparse_dispatch
        self.top_k = top_k
        self.use_time_gate = use_time_gate
        self.use_time_expert = use_time_expert
        self.use_graph_expert = use_graph_expert
        self.use_attention_expert = use_attention_expert
        self.time_expert_bias = -0.1
        self.num_experts = (
            int(use_time_expert)
            + int(use_graph_expert)
            + int(use_attention_expert)
        )
        if self.num_experts == 0:
            raise ValueError("At least one expert must be enabled.")

        if self.use_time_expert:
            self.time_expert = TemporalLSTMExpert(
                hidden_size=hidden_size,
                num_nodes=num_nodes,
                in_dim=in_dim - 1,
                out_dim=out_dim,
                layers=layers,
                dropout=dropout,
                vocab_size=max_time_index
            )
            self.id_adapter = ExpertAdapter(max(1, in_dim - 1), hidden_size, dropout)
        else:
            self.time_expert = None
            self.id_adapter = None

        if self.use_graph_expert:
            self.adaptive_expert = STModel(
                hidden_size=hidden_size,
                supports_len=self.supports_len,
                num_nodes=num_nodes,
                in_dim=in_dim,
                out_dim=out_dim,
                layers=layers,
                dropout=dropout
            )
            self.st_adapter = ExpertAdapter(in_dim, hidden_size, dropout)
        else:
            self.adaptive_expert = None
            self.st_adapter = None

        if self.use_attention_expert:
            self.attention_expert = AttentionModel(
                hidden_size=hidden_size,
                in_dim=in_dim,
                out_dim=out_dim,
                layers=layers,
                dropout=dropout,
                num_nodes=num_nodes,
                max_time_index=max_time_index,
                num_heads=4
            )
            self.attn_adapter = None
        else:
            self.attention_expert = None
            self.attn_adapter = None

        if self.use_graph_expert or self.use_attention_expert:
            self.future_time_encoder = FutureTimeEmbedding(
                hidden_size=hidden_size,
                num_nodes=num_nodes,
                vocab_size=max_time_index,
                dropout=dropout
            )
        else:
            self.future_time_encoder = None

        self.graph_token_1 = nn.Parameter(torch.empty(num_nodes, hidden_size))
        self.graph_token_2 = nn.Parameter(torch.empty(num_nodes, hidden_size))

        self.gate_network = STContextGate(
            hidden_size=hidden_size,
            num_nodes=num_nodes,
            mem_hid=hidden_size,
            in_dim=in_dim,
            out_dim=out_dim,
            memory_size=20,
            use_time_gate=use_time_gate,
            top_k=top_k,
            sparse_dispatch=sparse_dispatch,
            max_tod=max_time_index,
            num_experts=self.num_experts,
            time_gate_scale_init = time_gate_scale_init
        )

        for model in [
            self.time_expert,
            self.adaptive_expert,
            self.attention_expert,
            self.future_time_encoder
        ]:
            if model is not None:
                for _, p in model.named_parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)

        nn.init.xavier_uniform_(self.graph_token_1)
        nn.init.xavier_uniform_(self.graph_token_2)

    def build_supports(self):
        g1 = torch.softmax(torch.relu(torch.mm(self.graph_token_1, self.graph_token_2.T)), dim=-1)
        g2 = torch.softmax(torch.relu(torch.mm(self.graph_token_2, self.graph_token_1.T)), dim=-1)
        return [g1, g2]

    def forward(self, input_x, gate_out=False):
        # 只有 Graph Expert 需要自适应图结构。
        # 如果图专家关闭，则不再构建 supports。
        new_supports = self.build_supports() if self.use_graph_expert else None

        time_index = input_x[:, -1, 0]
        max_t = self.max_time_index

        cur_time_index = ((time_index * max_t) % max_t).long()
        next_time_index = ((time_index * max_t + time_index.size(-1)) % max_t).long()
        expert_outputs = []
        expert_hiddens = []

        h_future = None
        if self.future_time_encoder is not None:
            h_future = [self.future_time_encoder(next_time_index)]

        if self.use_time_expert:
            id_speed = input_x[:, :-1].permute(0, 2, 3, 1)
            id_speed = self.id_adapter(id_speed)

            o_time, h_time = self.time_expert(
                cur_time_index,
                id_speed,
                future_time_index=next_time_index
            )

            expert_outputs.append(o_time)
            expert_hiddens.append(h_time[-1])

        if self.use_graph_expert:
            st_input = self.st_adapter(input_x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            _, o_adaptive, h_adaptive = self.adaptive_expert(st_input, h_future, new_supports)

            expert_outputs.append(o_adaptive)
            expert_hiddens.append(h_adaptive[-1])

        if self.use_attention_expert:
            # 新版 AttentionModel 直接使用原始 input_x，
            # 因为它内部需要 time-of-day 通道和速度通道。
            o_attention, h_attention = self.attention_expert(input_x, h_future)
            expert_outputs.append(o_attention)
            expert_hiddens.append(h_attention)

        if len(expert_outputs) == 0:
            raise ValueError("At least one expert must be enabled.")

        ind_out = torch.stack(expert_outputs, dim=-1)

        gate_input = input_x.permute(0, 2, 3, 1)
        time_index_input = cur_time_index if self.use_time_gate else None

        gate_logits = self.gate_network(
            gate_input,
            expert_hiddens,
            time_index=time_index_input,
        )
        if self.use_time_expert and self.time_expert_bias != 0:
            expert_bias = torch.zeros_like(gate_logits)
            expert_bias[..., 0] = self.time_expert_bias
            gate_logits = gate_logits + expert_bias

        gate_prob = torch.softmax(gate_logits, dim=-1)

        out = (ind_out * gate_prob).sum(dim=-1)
        out = out.permute(0, 3, 1, 2).contiguous()

        if self.training or gate_out:
            return out, gate_prob, ind_out
        return out
