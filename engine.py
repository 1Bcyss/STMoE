import torch
import torch.optim as optim
from model import STGateMoE
import util
class trainer():
    def __init__(
        self,
        scaler,
        in_dim,
        out_dim,
        num_nodes,
        nhid,
        dropout,
        device,
        lr_mul=1.,
        n_warmup_steps=2000,
        quantile=0.7,
        is_quantile=False,
        warmup_epoch=0,
        use_time_gate=True,
        top_k=1,
        sparse_dispatch=False,
        lb_weight=0.0,
        use_time_expert=True,
        use_graph_expert=True,
        use_attention_expert=True,
        time_gate_scale_init=0.01,
    ):
        self.model = STGateMoE(
            num_nodes=num_nodes,
            dropout=dropout,
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_size=nhid,
            use_time_gate=use_time_gate,
            top_k=top_k,
            sparse_dispatch=sparse_dispatch,
            use_time_expert=use_time_expert,
            use_graph_expert=use_graph_expert,
            use_attention_expert=use_attention_expert,
            time_gate_scale_init=time_gate_scale_init,
        )

        self.model.to(device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.98),
            eps=1e-9
        )

        self.schedule = util.CosineWarmupScheduler(
            self.optimizer,
            d_model=nhid,
            n_warmup_steps=n_warmup_steps,
            lr_mul=lr_mul
        )

        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5
        self.flag = is_quantile
        self.quantile = quantile
        self.warmup_epoch = warmup_epoch
        self.threshold = 0.
        self.lb_weight = lb_weight

    def load_balance_loss(self, gate_prob):
        expert_usage = gate_prob.mean(dim=(0, 1, 2, 3))
        uniform = torch.full_like(expert_usage, 1.0 / expert_usage.size(0))
        return torch.sum((expert_usage - uniform) ** 2)

    def train(self, input_x, real, cur_epoch):
        self.model.train()
        self.schedule.zero_grad()
#前向传播  output最终融合后的预测结果  gate_prob	门控网络给每个专家分配的权重      res每个专家各自的预测结果
        output, gate_prob, res = self.model(input_x)
        predict = self.scaler.inverse_transform(output)

        ind_loss = self.loss(
            self.scaler.inverse_transform(res),
            real.permute(0, 2, 3, 1).unsqueeze(-1),
            self.threshold,
            reduce=None
        )

        if self.flag:
            gated_loss = self.loss(predict, real, reduce=None).permute(0, 2, 3, 1)
            l_worst_avoidance, l_best_choice = self.get_quantile_label(
                gated_loss,
                gate_prob,
                real
            )
        else:
            l_worst_avoidance, l_best_choice = self.get_label(
                ind_loss,
                gate_prob,
                real
            )

        worst_avoidance = (
                -0.5
                * l_worst_avoidance
                * torch.log(gate_prob + 1e-8)
        )

        best_choice = (
                -0.5
                * l_best_choice
                * torch.log(gate_prob + 1e-8)
        )

        if cur_epoch <= self.warmup_epoch:
            loss = ind_loss.mean()
        else:
            loss = ind_loss.mean() + worst_avoidance.mean() + best_choice.mean()

        if self.lb_weight > 0:
            loss = loss + self.lb_weight * self.load_balance_loss(gate_prob)

        loss.backward()
        #梯度裁剪
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.schedule.step_and_update_lr()

        mape = util.masked_mape(predict, real, self.threshold).item()
        rmse = util.masked_rmse(predict, real, self.threshold).item()
        return loss.item(), mape, rmse
#模型切换到验证模式
    def eval(self, input_x, real):
        self.model.eval()
        output = self.model(input_x)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, self.threshold)
        mape = util.masked_mape(predict, real, self.threshold).item()
        rmse = util.masked_rmse(predict, real, self.threshold).item()
        return loss.item(), mape, rmse

    def get_quantile_label(self, gated_loss, gate, real):
        gated_loss = gated_loss.unsqueeze(dim=-1)
        real = real.unsqueeze(dim=-1)

        max_quantile = gated_loss.quantile(self.quantile)
        min_quantile = gated_loss.quantile(1 - self.quantile)

        incorrect = (gated_loss > max_quantile).expand_as(gate)

        correct = (
            (gated_loss < min_quantile)
            & (real.permute(0, 2, 3, 1, 4) > self.threshold)
        ).expand_as(gate)

        cur_expert = gate.argmax(dim=-1, keepdim=True)

        not_chosen = gate.topk(
            dim=-1,
            k=min(2, gate.size(-1)),
            largest=False
        ).indices

        selected = torch.zeros_like(gate).scatter_(
            -1,
            cur_expert,
            1.0
        )

        scaling = torch.zeros_like(gate).scatter_(
            -1,
            not_chosen,
            0.5 if gate.size(-1) > 1 else 1.0
        )

        selected[incorrect] = scaling[incorrect]
        l_worst_avoidance = selected.detach()

        selected = torch.zeros_like(gate).scatter(
            -1,
            cur_expert,
            1.0
        ) * correct

        l_best_choice = selected.detach()

        return l_worst_avoidance, l_best_choice

    def get_label(self, ind_loss, gate, real):
        empty_val = (
            real.permute(0, 2, 3, 1)
            .unsqueeze(-1)
            .expand_as(gate)
        ) <= self.threshold

        max_error = ind_loss.argmax(dim=-1, keepdim=True)
        cur_expert = gate.argmax(dim=-1, keepdim=True)

        incorrect = max_error == cur_expert

        selected = torch.zeros_like(gate).scatter(
            -1,
            cur_expert,
            1.0
        )

        scaling = torch.ones_like(gate) * ind_loss
        scaling = scaling.scatter(-1, max_error, 0.)
        scaling = scaling / (scaling.sum(dim=-1, keepdim=True) + 1e-8) * (1 - selected)

        l_worst_avoidance = torch.where(
            incorrect,
            scaling,
            selected
        )

        l_worst_avoidance = torch.where(
            empty_val,
            torch.zeros_like(gate),
            l_worst_avoidance
        ).detach()

        min_error = ind_loss.argmin(dim=-1, keepdim=True)
        correct = min_error == cur_expert

        scaling = torch.zeros_like(gate).scatter(
            -1,
            min_error,
            1.
        )

        l_best_choice = torch.where(
            correct,
            selected,
            scaling
        )

        l_best_choice = torch.where(
            empty_val,
            torch.zeros_like(gate),
            l_best_choice
        ).detach()

        return l_worst_avoidance, l_best_choice