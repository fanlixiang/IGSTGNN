import os
import argparse
import numpy as np
from datetime import datetime

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

from src.models.igstgnn import IGSTGNN
from src.engines.igstgnn_engine import IGSTGNN_Engine
from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from src.utils.graph_algo import normalize_adj_mx
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

def get_config():
    parser = get_public_config()

    parser.add_argument('--num_feat', type=int, default=1)
    parser.add_argument('--num_hidden', type=int, default=32)
    parser.add_argument('--node_hidden', type=int, default=12)
    parser.add_argument('--time_emb_dim', type=int, default=12)
    parser.add_argument('--layer', type=int, default=5)
    parser.add_argument('--k_t', type=int, default=3)
    parser.add_argument('--k_s', type=int, default=2)
    parser.add_argument('--gap', type=int, default=3)
    parser.add_argument('--cl_epoch', type=int, default=3)
    parser.add_argument('--warm_epoch', type=int, default=30)
    parser.add_argument('--tpd', type=int, default=288) # 288 = 12 * 24

    parser.add_argument('--lrate', type=float, default=2e-3)
    parser.add_argument('--wdecay', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--clip_grad_value', type=float, default=5)
    
    # Incident related parameters
    parser.add_argument('--icsf_dim', type=int, default=64)
    parser.add_argument('--module_name', type=str, default='igstgnn')

    # TIID parameters. lambda_incident is kept as an ablation scale and defaults
    # to the paper-equivalent value of 1.0.
    parser.add_argument('--lambda_incident', type=float, default=1.0, help='Ablation scale for TIID incident context')
    parser.add_argument('--sigma_t', type=float, default=1.0, help='Temporal decay parameter')
    args = parser.parse_args()

    # Log directory configuration
    args.module_name = IGSTGNN.__module__.split('.')[-1]
    args.run_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = '{}_{}_s{}_{}'.format(args.dataset, args.module_name, args.seed, args.run_time)
    if args.use_sensor_info:
        log_dir = './experiments/{}/{}/'.format(args.model_name, run_name)
    else:
        log_dir = './experiments/{}/{}_nosensor/'.format(args.model_name, run_name)
    logger = get_logger(log_dir, __name__, 'record_{}_s{}_{}.log'.format(args.module_name, args.seed, args.run_time))
    print("model_name", args.module_name)
    logger.info(args)
    
    return args, log_dir, logger

def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)
    
    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    args.data_path = data_path
    args.node_num = node_num
    logger.info('Adj path: ' + adj_path)

    # Load adjacency matrix
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = normalize_adj_mx(adj_mx, 'doubletransition')
    args.adjs = [torch.tensor(i).to(device) for i in adj_mx]
    
    # Load dataset
    dataloader, scaler = load_dataset(data_path, args, logger)
    
    # Calculate curriculum learning related parameters
    cl_step = args.cl_epoch * dataloader['train_loader'].num_batch
    warm_step = args.warm_epoch * dataloader['train_loader'].num_batch

    # Create model
    model = IGSTGNN(node_num=node_num,
                         input_dim=args.input_dim,
                         output_dim=args.output_dim,
                         seq_len=args.seq_len,
                         horizon=args.horizon,
                         model_args=vars(args),
                         dataset=args.dataset,
                         data_path=data_path,
                         use_sensor_info=args.use_sensor_info)



    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 38, 46, 54, 62, 70, 80], gamma=0.5)

    # Create engine
    engine = IGSTGNN_Engine(device=device,
                                model=model,
                                dataloader=dataloader,
                                scaler=scaler,
                                sampler=None,
                                loss_fn=loss_fn,
                                lrate=args.lrate,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                clip_grad_value=args.clip_grad_value,
                                max_epochs=args.max_epochs,
                                patience=args.patience,
                                log_dir=log_dir,
                                logger=logger,
                                seed=args.seed,
                                cl_step=cl_step,
                                warm_step=warm_step,
                                horizon=args.horizon,
                                incident=args.incident)

    if args.mode == 'train':
        engine.train()
    else:
        engine.evaluate(args.mode)

if __name__ == "__main__":
    main()
