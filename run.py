import os
import argparse

# 创建命令行参数解析器
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None, type=str, required=True)
parser.add_argument('--device', default=-1, type=int)
parser.add_argument('--exp_id', default=0, type=int)
args = parser.parse_args()
device = 'cuda:' + str(args.device)

BATCH_DICT = {
    'METR-LA': 32,
    'PEMS-BAY': 32,
    'PEMS08': 32
}
INIT_DICT = {
    'METR-LA': 5,
    'PEMS-BAY': 5,
    'PEMS08': 5
}
NUM_NODES_DICT = {
    'METR-LA': 207,
    'PEMS-BAY': 325,
    'PEMS08': 170
}

if args.device < 0:
    print("采用 CPU 训练...")
    device = 'cpu'

if args.dataset not in BATCH_DICT.keys():
    raise ValueError(
        "输入的数据集名称不合法，请选择 METR-LA、PEMS-BAY 或 PEMS08"
    )

save_dir = 'experiment/{}_{}'.format(args.dataset, args.exp_id)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

batch_size = BATCH_DICT[args.dataset]
warmup_epoch = INIT_DICT[args.dataset]
num_nodes = NUM_NODES_DICT[args.dataset]

log = (
    "python -u train.py "
    "--device {} "
    "--data ./data/{} "
    "--adjdata ./data/{}/adj_mx.pkl "
    "--adjtype doubletransition "
    "--nhid 32 "
    "--in_dim 2 "
    "--out_dim 1 "
    "--num_nodes {} "
    "--batch_size {} "
    "--dropout 0.1 "
    "--epochs 100 "
    "--seed 99 "
    "--save ./experiment/{}_{}/STMoE "
    "--quantile 0.7 "
    "--is_quantile "
    "--warmup_epoch {} "
    "--lb_weight 0.001 "
    "--val_every 1 "
    "--patience 8 "
    "--min_delta 0.001"
)
cmd = log.format(
    device,
    args.dataset,
    args.dataset,
    num_nodes,
    batch_size,
    args.dataset,
    args.exp_id,
    warmup_epoch
)

print(cmd)
os.system(cmd)