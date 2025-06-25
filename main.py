import warnings
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Optional

from models import Transformer
from data import grokking_data
from quad import QUAD

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('highest')


parser = argparse.ArgumentParser(add_help=True)
# data args
parser.add_argument('--p', type=int, default=97, help='prime number')
parser.add_argument('--op', type=str, default='/', choices=['*', '/', '+', '-'], help='operation')
parser.add_argument('--train-fraction', type=float, default=0.33, help='train fraction')
# model args
parser.add_argument('--depth', type=int, default=3, help='depth')
parser.add_argument('--dim', type=int, default=128, help='dimension')
parser.add_argument('--heads', type=int, default=1, help='heads')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
# optimizer args
parser.add_argument('--optimizer', type=str, default='quad', choices=['adamw', 'quad'], help='optimizer')
parser.add_argument('--lr', type=float, default=3e-3, help='learning rate')
parser.add_argument('--weight-decay', type=float, default=0.1, help='weight decay')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1')
parser.add_argument('--beta2', type=float, default=0.99, help='beta2')
# training args
parser.add_argument('-b', '--batch_size', type=int, default=512, help='batch size')
parser.add_argument('-e', '--epochs', type=int, default=500, help='number of epochs')
# misc args
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--disable-early-stop', action='store_false', help='continue training even after goal validation accuracy is reached')
parser.add_argument('--opt-sweep', type=str, default='adamw,quad', help='comma separated list of optimizers to sweep, overrides --optimizer')
parser.add_argument('--lr-sweep', type=str, default=None, help='comma separated list of learning rates to sweep, overrides --lr')
parser.add_argument('--weight-decay-sweep', type=str, default=None, help='comma separated list of weight decay values to sweep, overrides --weight-decay')
parser.add_argument('--beta1-sweep', type=str, default=None, help='comma separated list of beta1 values to sweep, overrides --beta1')
parser.add_argument('--beta2-sweep', type=str, default=None, help='comma separated list of beta2 values to sweep, overrides --beta2')


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
            preconditioner_lr=0.9,
            max_size_dense=float('inf'),
            max_skew_dense=1.0,
            store_triu_vector=False,
            precondition_largest_two_dims=True,
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
        Xtrain, Ttrain, Xtest, Ttest, epochs=args.epochs, enable_early_stop=not args.disable_early_stop
    )

    print(
        f"summary | lr {args.lr:.2e} wd {args.weight_decay} b1 {args.beta1} b2 {args.beta2} | solved_epoch {solved_epoch}"
    )

    # plotting ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 3.5))
    lw = 2
    ax.plot(np.array(net.train_acc_trace) * 100, label='train', color='#1b9e77', lw=lw)
    ax.plot(np.array(net.val_acc_trace) * 100, label='val', color='#d95f02', lw=lw)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    fig.tight_layout()
    op_safe = args.op.replace('/', 'div').replace('*', 'mul').replace('+', 'add').replace('-', 'sub')
    filename = f"media/grokking_p{args.p}_op{op_safe}_opt{args.optimizer}_lr{args.lr}_d{args.depth}_dim{args.dim}_h{args.heads}_wd{args.weight_decay}.png"
    fig.savefig(filename, dpi=300)
    plt.close(fig)

    return solved_epoch


def _parse_sweep_list(value: str | None):
    if value is None:
        return None
    return [float(v) for v in value.split(',') if v]


def _parse_string_list(value: str | None):
    if value is None:
        return None
    return [v.strip() for v in value.split(',') if v.strip()]


if __name__ == '__main__':
    args = parser.parse_args()

    opt_list = _parse_string_list(args.opt_sweep) or [args.optimizer]
    lr_list = _parse_sweep_list(args.lr_sweep) or [args.lr]
    wd_list = _parse_sweep_list(args.weight_decay_sweep) or [args.weight_decay]
    beta1_list = _parse_sweep_list(args.beta1_sweep) or [args.beta1]
    beta2_list = _parse_sweep_list(args.beta2_sweep) or [args.beta2]

    best_solved_epoch = args.epochs  # initialize with max epochs
    best_cfg = None

    for opt in opt_list:
        for lr in lr_list:
            for wd in wd_list:
                for beta1 in beta1_list:
                    for beta2 in beta2_list:
                        run_args = argparse.Namespace(**vars(args))
                        run_args.optimizer = opt
                        run_args.lr = lr
                        run_args.weight_decay = wd
                        run_args.beta1 = beta1
                        run_args.beta2 = beta2
                        solved_epoch = main(run_args)
                        if solved_epoch < best_solved_epoch:
                            best_solved_epoch = solved_epoch
                            best_cfg = (opt, lr, wd, beta1, beta2)

    if best_cfg is not None:
        opt, lr, wd, beta1, beta2 = best_cfg
        print(
            f"best config | opt {opt} lr {lr:.2e} wd {wd} b1 {beta1} b2 {beta2} | solved_epoch {best_solved_epoch}"
        )
