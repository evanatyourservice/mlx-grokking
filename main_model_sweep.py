import warnings
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Optional
import os

from models import Transformer
from data import grokking_data
from quad import QUAD

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('highest')


parser = argparse.ArgumentParser(add_help=True)
# data args
parser.add_argument('--p', type=int, default=97, help='prime number')
parser.add_argument('--op', type=str, default='/', choices=['*', '/', '+', '-'], help='operation')
parser.add_argument('--train-fraction', type=float, default=0.4, help='train fraction')
# model args
parser.add_argument('--depth', type=int, default=2, help='depth')
parser.add_argument('--dim', type=int, default=128, help='dimension')
parser.add_argument('--heads', type=int, default=1, help='heads')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
# optimizer args
parser.add_argument('--optimizer', type=str, default='quad', choices=['adamw', 'quad'], help='optimizer')
parser.add_argument('--lr', type=float, default=3e-3, help='learning rate')
parser.add_argument('--weight-decay', type=float, default=0.3, help='weight decay')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1')
parser.add_argument('--beta2', type=float, default=0.99, help='beta2')
# training args
parser.add_argument('-b', '--batch_size', type=int, default=512, help='batch size')
parser.add_argument('-e', '--epochs', type=int, default=400, help='number of epochs')
# misc args
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--early-stop', action='store_true', help='stop training early when goal validation accuracy is reached')
# model sweep args
parser.add_argument('--dim-sweep', type=str, default=None, help='comma separated list of dimensions to sweep')
parser.add_argument('--depth-sweep', type=str, default=None, help='comma separated list of depths to sweep')
parser.add_argument('--heads-sweep', type=str, default=None, help='comma separated list of heads to sweep')
parser.add_argument('--dropout-sweep', type=str, default=None, help='comma separated list of dropout values to sweep')
parser.add_argument('--batch-size-sweep', type=str, default=None, help='comma separated list of batch sizes to sweep')


class NeuralNetwork:
    def __init__(self, model: nn.Module, optimizer, device: torch.device, batch_size: int = 64):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.device = device
        
        self.model = torch.compile(self.model)

        # traces
        self.train_error_trace = []
        self.train_acc_trace = []
        self.val_error_trace = []
        self.val_acc_trace = []

    # batching ---------------------------------------------------------------
    def _make_batches(self, X: torch.Tensor, T: torch.Tensor):
        bs = self.batch_size if self.batch_size != -1 else X.size(0)
        for i in range(0, X.size(0), bs):
            yield X[i:i + bs], T[i:i + bs]

    # evaluation -------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, X: torch.Tensor, T: torch.Tensor):
        self.model.eval()
        total_loss, total_correct = 0.0, 0
        
        X_device, T_device = X.to(self.device), T.to(self.device)
        
        for i in range(0, X.size(0), self.batch_size):
            x_b = X_device[i:i + self.batch_size]
            t_b = T_device[i:i + self.batch_size]
            
            logits = self.model(x_b)
            loss = self.loss_fn(logits, t_b)
            
            total_loss += loss.item() * x_b.size(0)
            total_correct += (logits.argmax(dim=1) == t_b).sum().item()
            
        n = X.size(0)
        return total_loss / n, total_correct / n

    # training --------------------------------------------------------------
    GOAL_VAL_ACC = 0.90  # 90% accuracy threshold to consider problem solved

    def train(
        self,
        Xtrain: torch.Tensor,
        Ttrain: torch.Tensor,
        Xval: torch.Tensor,
        Tval: torch.Tensor,
        epochs: int,
        enable_early_stop: bool = False,
    ):
        Xval_device, Tval_device = Xval.to(self.device), Tval.to(self.device)
        
        # pre-create permutation tensor
        n_train = Xtrain.size(0)
        
        solved_epoch: Optional[int] = None
        pbar = tqdm(range(1, epochs + 1), desc='Training', unit='epoch')
        for epoch in pbar:
            self.model.train()
            # shuffle indices
            perm = torch.randperm(n_train)
            
            total_loss, total_correct = 0.0, 0
            
            # move shuffled data to device once per epoch
            Xtrain_shuffled = Xtrain[perm].to(self.device)
            Ttrain_shuffled = Ttrain[perm].to(self.device)
            
            for i in range(0, n_train, self.batch_size):
                x_b = Xtrain_shuffled[i:i + self.batch_size]
                t_b = Ttrain_shuffled[i:i + self.batch_size]
                
                self.optimizer.zero_grad(set_to_none=True)
                
                logits = self.model(x_b)
                loss = self.loss_fn(logits, t_b)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * x_b.size(0)
                total_correct += (logits.argmax(dim=1) == t_b).sum().item()

            avg_train_loss = total_loss / n_train
            avg_train_acc = total_correct / n_train
            self.train_error_trace.append(avg_train_loss)
            self.train_acc_trace.append(avg_train_acc)

            # validation
            val_loss, val_acc = self.evaluate(Xval_device, Tval_device)
            self.val_error_trace.append(val_loss)
            self.val_acc_trace.append(val_acc)

            pbar.set_postfix(val_acc=f"{val_acc * 100:.2f}%")

            # check solved ----------------------------------------------------
            if solved_epoch is None and val_acc >= self.GOAL_VAL_ACC:
                solved_epoch = epoch
                pbar.write(
                    f"solved at epoch {epoch}: val_acc {val_acc * 100:.2f}% ≥ {self.GOAL_VAL_ACC * 100:.0f}%"
                )
                if enable_early_stop:
                    break

        if solved_epoch is None:
            solved_epoch = epoch  # unsolved, returns final epoch (likely epochs)

        return solved_epoch


def build_optimizer(args, model_params):
    if args.optimizer == 'adamw':
        return optim.AdamW(model_params, lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    if args.optimizer == 'quad':
        return QUAD(
            list(model_params),
            lr=args.lr,
            momentum=args.beta1,
            weight_decay=args.weight_decay,
            preconditioner_lr=1.0,
            max_size_dense=float('inf'),
            max_skew_dense=1.0,
            store_triu_vector=False,
            precondition_largest_two_dims=True,
            normalize_grads=True,
            dtype=torch.float32,
        )
    raise ValueError(f"unknown optimizer {args.optimizer}")


def main(args):
    # set all seeds for determinism
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

    Xtrain, Ttrain, Xtest, Ttest = grokking_data(args.p, op=args.op, train_fraction=args.train_fraction, seed=args.seed)

    # model ------------------------------------------------------------------
    model = Transformer(depth=args.depth, dim=args.dim, heads=args.heads, n_tokens=args.p + 2, seq_len=Xtrain.size(1), dropout=args.dropout)

    # optimizer --------------------------------------------------------------
    optimizer = build_optimizer(args, model.parameters())

    # network/trainer --------------------------------------------------------
    net = NeuralNetwork(model, optimizer, device=device, batch_size=args.batch_size)
    solved_epoch = net.train(
        Xtrain, Ttrain, Xtest, Ttest, epochs=args.epochs, enable_early_stop=args.early_stop
    )

    print(
        f"summary | opt {args.optimizer} depth {args.depth} dim {args.dim} heads {args.heads} dropout {args.dropout} | solved_epoch {solved_epoch}"
    )

    return solved_epoch


def _parse_sweep_list(value: str | None):
    if value is None:
        return None
    return [float(v) for v in value.split(',') if v]


def _parse_int_list(value: str | None):
    if value is None:
        return None
    return [int(v) for v in value.split(',') if v]


def _parse_string_list(value: str | None):
    if value is None:
        return None
    return [v.strip() for v in value.split(',') if v.strip()]


def plot_comparison(sweep_param_name: str, sweep_values: list, adamw_solved_epochs: list, quad_solved_epochs: list, args):
    """Plot comparison of AdamW vs QUAD solved epochs across swept parameter values."""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # line plot with markers
    ax.plot(sweep_values, adamw_solved_epochs, 'o-', label='AdamW', color='#1b9e77', linewidth=2, markersize=8)
    ax.plot(sweep_values, quad_solved_epochs, 's-', label='QUAD', color='#d95f02', linewidth=2, markersize=8)
    
    # clean parameter name for display
    param_display_names = {
        'dim': 'Model Dimension',
        'depth': 'Model Depth',
        'heads': 'Attention Heads',
        'dropout': 'Dropout Rate',
        'batch_size': 'Batch Size'
    }
    
    ax.set_xlabel(param_display_names.get(sweep_param_name, sweep_param_name.replace('-', ' ').title()))
    ax.set_ylabel('Epochs to Solve (90% Val Acc)')
    ax.set_title(param_display_names.get(sweep_param_name, sweep_param_name.replace('-', ' ').title()) + ' (400 max iters)')
    
    # set log scale for batch size
    if sweep_param_name == 'batch_size':
        ax.set_xscale('log')
    
    # set x-axis ticks to only show swept values
    ax.set_xticks(sweep_values)
    ax.set_xticklabels(sweep_values)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    # add value labels on points
    for x, y in zip(sweep_values, adamw_solved_epochs):
        ax.annotate(f'{y}', xy=(x, y), xytext=(0, 5), textcoords='offset points', 
                   ha='center', fontsize=8, color='#1b9e77')
    
    for x, y in zip(sweep_values, quad_solved_epochs):
        ax.annotate(f'{y}', xy=(x, y), xytext=(0, -15), textcoords='offset points', 
                   ha='center', fontsize=8, color='#d95f02')
    
    fig.tight_layout()
    
    # save plot
    os.makedirs('media', exist_ok=True)
    op_safe = args.op.replace('/', 'div').replace('*', 'mul').replace('+', 'add').replace('-', 'sub')
    filename = f"media/grokking_p{args.p}_op{op_safe}_{sweep_param_name}_sweep_comparison.png"
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"saved comparison plot to {filename}")


if __name__ == '__main__':
    args = parser.parse_args()

    # override with best hypers found in opt sweep
    if args.optimizer == 'quad':
        args.lr = 3e-3
        args.weight_decay = 0.3
        args.beta1 = 0.9
    elif args.optimizer == 'adamw':
        args.lr = 3e-3
        args.weight_decay = 0.3
        args.beta1 = 0.9
        args.beta2 = 0.99
    
    # determine which parameter to sweep
    sweep_configs = []
    
    if args.dim_sweep:
        sweep_param = 'dim'
        sweep_values = _parse_int_list(args.dim_sweep)
        for value in sweep_values:
            config = vars(args).copy()
            config['dim'] = value
            sweep_configs.append((sweep_param, value, config))
    elif args.depth_sweep:
        sweep_param = 'depth'
        sweep_values = _parse_int_list(args.depth_sweep)
        for value in sweep_values:
            config = vars(args).copy()
            config['depth'] = value
            sweep_configs.append((sweep_param, value, config))
    elif args.heads_sweep:
        sweep_param = 'heads'
        sweep_values = _parse_int_list(args.heads_sweep)
        for value in sweep_values:
            config = vars(args).copy()
            config['heads'] = value
            sweep_configs.append((sweep_param, value, config))
    elif args.dropout_sweep:
        sweep_param = 'dropout'
        sweep_values = _parse_sweep_list(args.dropout_sweep)
        for value in sweep_values:
            config = vars(args).copy()
            config['dropout'] = value
            sweep_configs.append((sweep_param, value, config))
    elif args.batch_size_sweep:
        sweep_param = 'batch_size'
        sweep_values = _parse_int_list(args.batch_size_sweep)
        for value in sweep_values:
            config = vars(args).copy()
            config['batch_size'] = value
            sweep_configs.append((sweep_param, value, config))
    else:
        # no sweep, just run once with default parameters
        print("no sweep parameters specified, running single configuration")
        for opt in ['adamw', 'quad']:
            run_args = argparse.Namespace(**vars(args))
            run_args.optimizer = opt
            solved_epoch = main(run_args)
        exit(0)
    
    # run sweep with both optimizers
    adamw_solved_epochs = []
    quad_solved_epochs = []
    
    for param_name, param_value, config in sweep_configs:
        print(f"\n{'='*60}")
        print(f"running {param_name}={param_value}")
        print(f"{'='*60}")
        
        # run adamw
        adamw_args = argparse.Namespace(**config)
        adamw_args.optimizer = 'adamw'
        adamw_solved_epoch = main(adamw_args)
        adamw_solved_epochs.append(adamw_solved_epoch)
        
        # run quad
        quad_args = argparse.Namespace(**config)
        quad_args.optimizer = 'quad'
        quad_solved_epoch = main(quad_args)
        quad_solved_epochs.append(quad_solved_epoch)
    
    # plot comparison
    plot_comparison(sweep_param, sweep_values, adamw_solved_epochs, quad_solved_epochs, args)
    
    # print summary
    print(f"\n{'='*60}")
    print(f"sweep summary for {sweep_param}")
    print(f"{'='*60}")
    for i, value in enumerate(sweep_values):
        print(f"{sweep_param}={value}: adamw={adamw_solved_epochs[i]}, quad={quad_solved_epochs[i]}")
    
    # find best configurations
    best_adamw_idx = np.argmin(adamw_solved_epochs)
    best_quad_idx = np.argmin(quad_solved_epochs)
    print(f"\nbest adamw: {sweep_param}={sweep_values[best_adamw_idx]} with solved_epoch={adamw_solved_epochs[best_adamw_idx]}")
    print(f"best quad: {sweep_param}={sweep_values[best_quad_idx]} with solved_epoch={quad_solved_epochs[best_quad_idx]}")
