# train.py
import os
import time
import json
import argparse
import numpy as np
import torch
import util
from engine import trainer
from copy import deepcopy as cp
# 命令行参数解析器
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data', type=str, default='data/METR-LA', help='data path')
parser.add_argument('--adjdata', type=str, default=None, help='邻接矩阵路径')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='邻接矩阵的处理方式')
parser.add_argument('--out_dim', type=int, default=1)
parser.add_argument('--nhid', type=int, default=32)
parser.add_argument('--in_dim', type=int, default=2)
parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--epochs', type=int, default=100)
#每训练 50 个 iteration 输出一次 loss、MAPE、RMSE 等信息
parser.add_argument('--print_every', type=int, default=50)
parser.add_argument('--seed', type=int, default=99, help='随机种子')
parser.add_argument('--save', type=str, default='./experiment/METR-LA_STMoE', help='保存路径')
parser.add_argument('--load_path', type=str, default=None)
# ===== 改进早停策略 =====
parser.add_argument('--patience', type=int, default=8, help='早停')
parser.add_argument('--min_delta', type=float, default=0.001, help='最小提升')
parser.add_argument('--val_every', type=int, default=1, help='validate every N epochs')
parser.add_argument('--lr_mul', type=float, default=1)
parser.add_argument('--n_warmup_steps', type=int, default=4000)
parser.add_argument('--quantile', type=float, default=0.7)
parser.add_argument('--is_quantile', action='store_true')
parser.add_argument('--warmup_epoch', type=int, default=0)
# ===== 保存 history =====
parser.add_argument('--history_save_every', type=int, default=5, help='save history every N epochs')
# 门控开关
parser.add_argument('--use_time_gate', action='store_true',default=True)
parser.add_argument('--top_k', type=int, default=1)
parser.add_argument('--sparse_dispatch', action='store_true')
parser.add_argument('--lb_weight', type=float, default=0.0)
# 专家开关
parser.add_argument('--disable_time_expert', action='store_true')
parser.add_argument('--disable_graph_expert', action='store_true')
parser.add_argument('--disable_attention_expert', action='store_true')
# gate 权重初始化：
parser.add_argument(  '--time_gate_scale_init',  dest='time_gate_scale_init',type=float,default=0.01)
args = parser.parse_args()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_history_json(history, save_prefix):
    history_path = save_prefix + '_history.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f'[Saved] Training history -> {history_path}')
#把数据转成模型能用的Tensor，并放到正确的训练设备上
def to_device_tensor(array, device):
    if isinstance(array, torch.Tensor):
        return array.float().to(device)
    return torch.from_numpy(array).float().to(device)

def main():
#固定训练参数
    if args.seed != -1:
        print('Start Deterministic Training with seed {}'.format(args.seed))
        torch.manual_seed(args.seed)
        #训练时数据打乱顺序保持一致
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    device = torch.device(args.device)

    if args.adjdata:
        if os.path.exists(args.adjdata):
            sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
            args.num_nodes = len(sensor_ids)
        else:
            print('无效的文件路径；改用用户提供的 args.num_nodes')
#调用 util.py 里的 load_dataset() 函数，加载训练集、验证集和测试集。
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    use_time_expert = not args.disable_time_expert
    use_graph_expert = not args.disable_graph_expert
    use_attention_expert = not args.disable_attention_expert

    engine = trainer(
        scaler,
        args.in_dim,
        args.out_dim,
        args.num_nodes,
        args.nhid,
        args.dropout,
        device,
        lr_mul=args.lr_mul,
        n_warmup_steps=args.n_warmup_steps,
        quantile=args.quantile,
        is_quantile=args.is_quantile,
        warmup_epoch=args.warmup_epoch,
        use_time_gate=args.use_time_gate,
        top_k=args.top_k,
        sparse_dispatch=args.sparse_dispatch,
        lb_weight=args.lb_weight,
        use_time_expert=use_time_expert,
        use_graph_expert=use_graph_expert,
        use_attention_expert=use_attention_expert,
        time_gate_scale_init=args.time_gate_scale_init,
    )
#模型创建完成后、正式训练开始前 的初始化部分
    print('Train the model with {} parameters'.format(count_parameters(engine.model)))
    if args.load_path is not None:
        engine.model.load_state_dict(torch.load(args.load_path, map_location=device))
        engine.model.to(device)
    print('start training...', flush=True)

    his_loss, val_time, train_time = [], [], []
    wait, best = 0, 1e9
    best_path = None
    last_valid_loss = None
    last_valid_mape = None
    last_valid_rmse = None
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mape': [],
        'val_mape': [],
        'train_rmse': [],
        'val_rmse': []
    }

    for i in range(1, args.epochs + 1):
        #保存当前 epoch 中每个 batch 的训练指标
        train_loss, train_mape, train_rmse = [], [], []
        t1 = time.time()
        dataloader['train_loader'].shuffle()

        for iteration, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            #交换第 1 维和第 3 维  (B, T, N, C)变成(B, C, N, T)
            trainx = to_device_tensor(x, device).transpose(1, 3)
            trainy = to_device_tensor(y, device).transpose(1, 3)
            #核心训练步骤。调用trainer.train(...)
            metrics = engine.train(trainx, trainy[:, :args.out_dim, :, :], i)
            #把当前 batch 的指标保存到列表中
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])

            if iteration % args.print_every == 0:
                print(
                    'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'.format(
                        iteration, train_loss[-1], train_mape[-1], train_rmse[-1]
                    ),
                    #立刻打印
                    flush=True
                )
        #在一个 epoch 训练结束后，计算本轮训练集的平均指标
        t2 = time.time()
        train_time.append(t2 - t1)
        mtrain_loss = float(np.mean(train_loss))
        mtrain_mape = float(np.mean(train_mape))
        mtrain_rmse = float(np.mean(train_rmse))
        #判断是否进行验证
        do_validate = (i == 1) or (i % args.val_every == 0) or (i == args.epochs)
        if do_validate:
            valid_loss, valid_mape, valid_rmse = [], [], []
            s1 = time.time()

            for iteration, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
                valx = to_device_tensor(x, device).transpose(1, 3)
                valy = to_device_tensor(y, device).transpose(1, 3)
                metrics = engine.eval(valx, valy[:, :args.out_dim, :, :])

                valid_loss.append(metrics[0])
                valid_mape.append(metrics[1])
                valid_rmse.append(metrics[2])

            s2 = time.time()
            val_time.append(s2 - s1)
            mvalid_loss = float(np.mean(valid_loss))
            mvalid_mape = float(np.mean(valid_mape))
            mvalid_rmse = float(np.mean(valid_rmse))
            last_valid_loss = mvalid_loss
            last_valid_mape = mvalid_mape
            last_valid_rmse = mvalid_rmse
            #记录历史验证集损失
            his_loss.append(mvalid_loss)
            print('第 {:03d} 轮验证完成，用时 {:.4f} 秒'.format(i, (s2 - s1)))
            improvement = best - mvalid_loss

            if improvement > args.min_delta:
                best = mvalid_loss
                wait = 0
                best_path = args.save + '_epoch_' + str(i) + '_' + str(round(mvalid_loss, 2)) + '.pth'
                torch.save(engine.model.state_dict(), best_path)
                print('[Saved] 当前最佳模型 -> {}'.format(best_path))
                print('[EarlyStopping] Valid Loss improved by {:.6f}. Best={:.6f}, wait={}/{}'.format(
                    improvement, best, wait, args.patience
                ))
            else:
                wait += 1
                print('[早停] 无有效提升：当前={:.6f}，最优={:.6f}，提升={:.6f}，阈值={}，等待={}/{}'.format(
                    mvalid_loss, best, improvement, args.min_delta, wait, args.patience
                ))

        else:
            mvalid_loss = last_valid_loss if last_valid_loss is not None else 0.0
            mvalid_mape = last_valid_mape if last_valid_mape is not None else 0.0
            mvalid_rmse = last_valid_rmse if last_valid_rmse is not None else 0.0
            print('Epoch: {:03d}, Validation skipped. Next validation every {} epochs.'.format(i, args.val_every))

        history['train_loss'].append(mtrain_loss)
        history['val_loss'].append(mvalid_loss)
        history['train_mape'].append(mtrain_mape)
        history['val_mape'].append(mvalid_mape)
        history['train_rmse'].append(mtrain_rmse)
        history['val_rmse'].append(mvalid_rmse)

        print(
            'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, '
            'Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'.format(
                i, mtrain_loss, mtrain_mape, mtrain_rmse,
                mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)
            ),
            flush=True
        )

        if args.history_save_every > 0 and i % args.history_save_every == 0:
            save_history_json(history, args.save)

        if do_validate and wait >= args.patience:
            print('触发早停')
            print('[EarlyStopping] Stop at epoch {}. Best Valid Loss={:.6f}'.format(i, best))
            break

    print('Average Training Time: {:.4f} secs/epoch'.format(np.mean(train_time)))
    if len(val_time) > 0:
        print('Average Inference Time: {:.4f} secs'.format(np.mean(val_time)))
    else:
        print('Average Inference Time: 0.0000 secs')
    if best_path is None:
        raise RuntimeError('No best model was saved. Please check validation settings.')

    print('Load best model:', best_path)
    #训练结束后，准备进入测试集评估阶段 准备测试集真实标签 realy 和预测结果列表 outputs
    engine.model.load_state_dict(torch.load(best_path, map_location=device))
    engine.model.to(device)
    outputs = []
    realy = to_device_tensor(dataloader['y_test'], device).transpose(1, 3)[:, :args.out_dim, :, :]

    for iteration, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = to_device_tensor(x, device).transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx)
        outputs.append(preds)

    yhat = torch.cat(outputs, dim=0)[:realy.size(0), ...]

    print('训练完成')
    print('最佳模型的验证集损失为：', str(round(best, 4)))

    amae, amape, armse = [], [], []
    results = {
        'prediction': [],
        'ground_truth': []
    }

    for horizon_i in range(realy.size(-1)):
        pred = scaler.inverse_transform(yhat[..., horizon_i])
        real = realy[..., horizon_i]
        results['prediction'].append(cp(pred).cpu().numpy())
        results['ground_truth'].append(cp(real).cpu().numpy())
        metrics = util.metric(pred, real)
        print(
            'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'.format(
                horizon_i + 1, metrics[0], metrics[1], metrics[2]
            )
        )
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    print(
        'On average over {} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'.format(
            realy.size(-1), np.mean(amae), np.mean(amape), np.mean(armse)
        )
    )

    results['prediction'] = np.asarray(results['prediction'])
    results['ground_truth'] = np.asarray(results['ground_truth'])

    np.savez_compressed(args.save + '_prediction.npz', **results)
    print('[Saved] Prediction results -> {}'.format(args.save + '_prediction.npz'))
    save_history_json(history, args.save)


if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    print('总时间: {:.4f}'.format(t2 - t1))