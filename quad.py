"""
simple torchrun for 8xGPUs 1 node:
torchrun --standalone --nproc_per_node=8 optimizers/quad_batched_optimized.py
"""

import math
import torch

def matmul_transpose(x, transpose=False):
    """Matrix multiplication with optional transpose of x."""
    if transpose:
        return x.mT @ x
    return x @ x.mT


PACK_THRESHOLD = 1024


class QUAD(torch.optim.Optimizer):
    """PSGD-QUAD optimizer.

    Args:
        params: list of parameters to optimize
        lr: learning rate
        momentum: momentum beta
        weight_decay: weight decay
        preconditioner_lr: preconditioner learning rate
        max_size_dense: dimensions larger than this will have diagonal preconditioners, otherwise
            dense.
        max_skew_dense: dimensions with skew larger than this compared to the other dimension will
            have diagonal preconditioners, otherwise dense.
        store_triu_vector: if True, store the triu of dense preconditioners as vector between
            optimizer steps saving 50% memory between steps.
        precondition_largest_two_dims: Precondition largest two dimensions instead of last two
            dimensions after reshaping layer to 3 dimensions.
        dtype: dtype for all computations and states in QUAD.
    """
    def __init__(
        self,
        params: list[torch.nn.Parameter],
        lr: float = 0.001,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        preconditioner_lr: float = 0.8,
        max_size_dense: int = 8192,
        max_skew_dense: float = 1.0,
        store_triu_vector: bool = False,
        precondition_largest_two_dims: bool = False,
        dtype: torch.dtype | None = None,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            preconditioner_lr=preconditioner_lr,
            max_size_dense=max_size_dense,
            max_skew_dense=max_skew_dense,
            store_triu_vector=store_triu_vector,
            precondition_largest_two_dims=precondition_largest_two_dims,
            dtype=dtype,
        )
        super().__init__(params, defaults)

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        momentum_buffers,
        merged_shapes,
        permute_orders,
        permute_backs,
        Qs,
        Ls,
        diags,
        packeds,
        state_steps,
    ):
        group_dtype = group['dtype']
        for p in group["params"]:
            if p.grad is None:
                continue
                
            # use grad dtype if group dtype is None
            dtype = p.grad.dtype if group_dtype is None else group_dtype
            
            params_with_grad.append(p)
            grads.append(p.grad if group_dtype is None else p.grad.to(dtype=dtype))
            state = self.state[p]
            
            if "momentum_buffer" not in state:
                state["step"] = 0
                state["momentum_buffer"] = p.grad.clone() if group_dtype is None else p.grad.clone().to(dtype=dtype)
                state["merged_shape"] = merge_adjacent_dims_to_three(state["momentum_buffer"])
                
                g_reshaped = state["momentum_buffer"].view(state["merged_shape"])
                
                if group["precondition_largest_two_dims"] and g_reshaped.ndim == 3:
                    sizes = g_reshaped.shape
                    permute_order = sorted(range(3), key=lambda i: sizes[i])
                    # if dim 0 stays first, largest two dims are already last
                    if permute_order[0] == 0:
                        state["permute_order"] = None
                        state["permute_back"] = None
                    else:
                        permute_back = [None] * 3
                        for new_pos, old_axis in enumerate(permute_order):
                            permute_back[old_axis] = new_pos
                        state["permute_order"] = permute_order
                        state["permute_back"] = permute_back
                else:
                    state["permute_order"] = None
                    state["permute_back"] = None
                
                if state.get("permute_order") is not None:
                    g_reshaped = g_reshaped.permute(*state["permute_order"])
                
                scale = ((torch.mean((torch.abs(g_reshaped))**4))**(-1/8))**(1/2)  # 2 preconds
                if g_reshaped.ndim <= 1:
                    state["Q"] = [scale * torch.ones_like(g_reshaped, dtype=dtype)]
                    state["L"] = [torch.zeros_like(g_reshaped, dtype=dtype)]
                    state["diag"] = [True]
                else:
                    Qs_new = []
                    Ls_new = []
                    diag_new = []
                    batch_shape = g_reshaped.shape[:-2]
                    for size in g_reshaped.shape[-2:]:
                        if size > group["max_size_dense"] or size**2 > group["max_skew_dense"] * g_reshaped.numel():
                            Qs_new.append(scale * torch.ones(*batch_shape, size, dtype=dtype, device=g_reshaped.device))
                            Ls_new.append(torch.zeros(*batch_shape, 1, dtype=dtype, device=g_reshaped.device))
                            diag_new.append(True)
                        else:
                            Qs_new.append(scale * torch.eye(size, dtype=dtype, device=g_reshaped.device).repeat(*batch_shape, 1, 1))
                            Ls_new.append(torch.zeros(*batch_shape, 1, 1, dtype=dtype, device=g_reshaped.device))
                            diag_new.append(False)
                    state["Q"] = Qs_new
                    state["L"] = Ls_new
                    state["diag"] = diag_new
                state["packed"] = [False] * len(state["diag"])
                
            momentum_buffers.append(state['momentum_buffer'])
            merged_shapes.append(state["merged_shape"])
            permute_orders.append(state.get("permute_order"))
            permute_backs.append(state.get("permute_back"))
            Qs.append(state["Q"])
            Ls.append(state["L"])
            diags.append(state["diag"])
            packeds.append(state["packed"])
            state_steps.append(state["step"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            momentum_buffers: list[torch.Tensor] = []
            merged_shapes: list[tuple] = []
            permute_orders: list[list | None] = []
            permute_backs: list[list | None] = []
            Qs: list[list | None] = []
            Ls: list[list | None] = []
            diags: list[list | None] = []
            packeds: list[list | None] = []
            state_steps: list[int] = []

            self._init_group(
                group,
                params_with_grad,
                grads,
                momentum_buffers,
                merged_shapes,
                permute_orders,
                permute_backs,
                Qs,
                Ls,
                diags,
                packeds,
                state_steps,
            )

            if len(params_with_grad) == 0:
                continue
            
            torch._foreach_lerp_(momentum_buffers, grads, 1 - group['momentum'])
            
            dtype = group['dtype']

            preconditioned_grads = []
            for p, g, merged_shape, perm_order, perm_back, Q, L, diag, packed in zip(
                params_with_grad, momentum_buffers, merged_shapes, permute_orders, permute_backs,
                Qs, Ls, diags, packeds
            ):
                dtype = g.dtype
                state = self.state[p]
                
                state["step"] += 1
                
                original_shape = g.shape
                g_reshaped = g.view(merged_shape)
                
                if perm_order is not None:
                    g_reshaped = g_reshaped.permute(*perm_order)

                if group["store_triu_vector"]:
                    dims = g_reshaped.shape[-2:]
                    for idx, (q, is_diag, is_packed) in enumerate(zip(Q, diag, packed)):
                        if not is_diag and is_packed:
                            N = dims[idx]
                            Q[idx] = unpack_triu(q, N)
                    state["Q"] = Q
                
                startup_steps = 10
                if g_reshaped.ndim <= 1:
                    for _ in range(startup_steps if state["step"] == 1 else 1):
                        g_preconditioned = update_diag_solo(
                            Q[0], L[0], g_reshaped, group["preconditioner_lr"]
                        )
                else:
                    if state["step"] % 250 == 0:
                        ql, qr = Q[0], Q[1]
                        max_l = ql.abs().flatten(1).amax(dim=1)
                        max_r = qr.abs().flatten(1).amax(dim=1)
                        rho = (max_l / max_r).sqrt()
                        rho_l = rho.view(-1, *[1] * (ql.ndim - 1))
                        rho_r = rho.view(-1, *[1] * (qr.ndim - 1))
                        Q[0] /= rho_l
                        Q[1] *= rho_r
                    
                    for _ in range(startup_steps if state["step"] == 1 else 1):
                        if not diag[0] and not diag[1]:
                            g_preconditioned = precondition_DD(
                                Ql=Q[0],
                                Qr=Q[1],
                                Ll=L[0],
                                Lr=L[1],
                                G=g_reshaped,
                                precond_lr=group["preconditioner_lr"]
                            )
                        elif diag[0] and not diag[1]:
                            g_preconditioned = precondition_dD(
                                Ql=Q[0],
                                Qr=Q[1],
                                Ll=L[0],
                                Lr=L[1],
                                G=g_reshaped,
                                precond_lr=group["preconditioner_lr"]
                            )
                        elif not diag[0] and diag[1]:
                            g_preconditioned = precondition_Dd(
                                Ql=Q[0],
                                Qr=Q[1],
                                Ll=L[0],
                                Lr=L[1],
                                G=g_reshaped,
                                precond_lr=group["preconditioner_lr"]
                            )
                        else:
                            g_preconditioned = precondition_dd(
                                Ql=Q[0],
                                Qr=Q[1],
                                Ll=L[0],
                                Lr=L[1],
                                G=g_reshaped,
                                precond_lr=group["preconditioner_lr"]
                            )

                if group["store_triu_vector"]:
                    for idx, (q_full, is_diag) in enumerate(zip(Q, diag)):
                        if not is_diag:
                            N = q_full.shape[-1]
                            if N >= PACK_THRESHOLD:
                                state["Q"][idx] = pack_triu(q_full)
                                state["packed"][idx] = True
                            else:
                                state["packed"][idx] = False

                if perm_back is not None:
                    g_preconditioned = g_preconditioned.permute(*perm_back)
                
                original_shape = p.grad.shape
                if original_shape != g_preconditioned.shape:
                    g_preconditioned = g_preconditioned.view(original_shape)
                
                assert g_preconditioned.dtype == dtype
                preconditioned_grads.append(g_preconditioned.to(dtype=p.dtype))
            
            if group["weight_decay"] > 0:
                torch._foreach_mul_(params_with_grad, 1 - group["lr"] * group["weight_decay"])
            
            # divide by 10 to match adam
            torch._foreach_add_(params_with_grad, preconditioned_grads, alpha=-group["lr"] / 3.0)

        return loss


precond_beta = 0.95

@torch.compile
def update_diag_solo(Q, L, G, precond_lr):
    Pg = Q * Q * G
    term1 = Pg * Pg
    term2 = 1.0
    ell = torch.amax(term1, keepdim=True) + term2
    L.copy_(torch.max(precond_beta * L + (1 - precond_beta) * ell, ell))
    gain = 1 - precond_lr/2/L * (term1 - term2)
    Q.mul_(gain * gain)
    return Pg


@torch.compile
def precondition_dd(Ql, Qr, Ll, Lr, G, precond_lr):
    """Diagonal-diagonal preconditioning."""
    Pg = (Ql * Ql).unsqueeze(-1) * G * (Qr * Qr).unsqueeze(-2)
    
    # left diagonal update
    term1_l = (Pg * Pg).sum(-1)
    term2_l = G.numel() / Ql.shape[-1]
    ell_l = torch.amax(term1_l, dim=-1, keepdim=True) + term2_l
    Ll.copy_(torch.maximum(precond_beta * Ll + (1 - precond_beta) * ell_l, ell_l))
    gain_l = 1 - precond_lr/2/Ll * (term1_l - term2_l)
    Ql.mul_(gain_l * gain_l)
    
    # right diagonal update
    term1_r = (Pg * Pg).sum(-2)
    term2_r = G.numel() / Qr.shape[-1]
    ell_r = torch.amax(term1_r, dim=-1, keepdim=True) + term2_r
    Lr.copy_(torch.maximum(precond_beta * Lr + (1 - precond_beta) * ell_r, ell_r))
    gain_r = 1 - precond_lr/2/Lr * (term1_r - term2_r)
    Qr.mul_(gain_r * gain_r)
    
    return Pg


@torch.compile
def precondition_dD(Ql, Qr, Ll, Lr, G, precond_lr):
    """Diagonal-dense preconditioning."""
    Pg = (Ql * Ql).unsqueeze(-1) * G @ matmul_transpose(Qr)
    
    # left diagonal update
    term1_l = (Pg * Pg).sum(-1)
    term2_l = G.numel() / Ql.shape[-1]
    ell_l = torch.amax(term1_l, dim=-1, keepdim=True) + term2_l
    Ll.copy_(torch.maximum(precond_beta * Ll + (1 - precond_beta) * ell_l, ell_l))
    gain_l = 1 - precond_lr/2/Ll * (term1_l - term2_l)
    Ql.mul_(gain_l * gain_l)
    
    # right dense update
    term1_r = matmul_transpose(Pg, transpose=True)
    term2_r = G.numel() / Qr.shape[-1]
    row_norm_sq = torch.sum(term1_r * term1_r, dim=-1, keepdim=True)
    max_row_norm_sq, i_max = torch.max(row_norm_sq, dim=-2, keepdim=True)
    v_r = torch.gather(term1_r, dim=-2, index=i_max.expand(-1, -1, term1_r.shape[-1]))
    v_r = v_r.transpose(-2, -1) / torch.sqrt(max_row_norm_sq)
    ell_r = torch.norm(term1_r @ v_r, dim=(-2, -1), keepdim=True) + term2_r
    Lr.copy_(torch.maximum(precond_beta * Lr + (1 - precond_beta) * ell_r, ell_r))
    p = Qr - precond_lr/2/Lr * (term1_r @ Qr - term2_r * Qr)
    p = p - precond_lr/2/Lr * (p @ term1_r - p * term2_r)
    Qr.copy_((p + p.mT) / 2)
    
    return Pg


@torch.compile
def precondition_Dd(Ql, Qr, Ll, Lr, G, precond_lr):
    """Dense-diagonal preconditioning."""
    Pg = matmul_transpose(Ql) @ G * (Qr * Qr).unsqueeze(-2)
    
    # left dense update
    term1_l = matmul_transpose(Pg)
    term2_l = G.numel() / Ql.shape[-1]
    row_norm_sq = torch.sum(term1_l * term1_l, dim=-1, keepdim=True)
    max_row_norm_sq, i_max = torch.max(row_norm_sq, dim=-2, keepdim=True)
    v_r = torch.gather(term1_l, dim=-2, index=i_max.expand(-1, -1, term1_l.shape[-1]))
    v_r = v_r.transpose(-2, -1) / torch.sqrt(max_row_norm_sq)
    ell_l = torch.norm(term1_l @ v_r, dim=(-2, -1), keepdim=True) + term2_l
    Ll.copy_(torch.maximum(precond_beta * Ll + (1 - precond_beta) * ell_l, ell_l))
    p = Ql - precond_lr/2/Ll * (term1_l @ Ql - term2_l * Ql)
    p = p - precond_lr/2/Ll * (p @ term1_l - p * term2_l)
    Ql.copy_((p + p.mT) / 2)
    
    # right diagonal update
    term1_r = (Pg * Pg).sum(-2)
    term2_r = G.numel() / Qr.shape[-1]
    ell_r = torch.amax(term1_r, dim=-1, keepdim=True) + term2_r
    Lr.copy_(torch.maximum(precond_beta * Lr + (1 - precond_beta) * ell_r, ell_r))
    gain_r = 1 - precond_lr/2/Lr * (term1_r - term2_r)
    Qr.mul_(gain_r * gain_r)
    
    return Pg


@torch.compile
def precondition_DD(Ql, Qr, Ll, Lr, G, precond_lr):
    """Dense-dense preconditioning."""
    Pg = matmul_transpose(Ql) @ G @ matmul_transpose(Qr)
    
    # left dense update
    term1_l = matmul_transpose(Pg)
    term2_l = G.numel() / Ql.shape[-1]
    row_norm_sq = torch.sum(term1_l * term1_l, dim=-1, keepdim=True)
    max_row_norm_sq, i_max = torch.max(row_norm_sq, dim=-2, keepdim=True)
    v_r = torch.gather(term1_l, dim=-2, index=i_max.expand(-1, -1, term1_l.shape[-1]))
    v_r = v_r.transpose(-2, -1) / torch.sqrt(max_row_norm_sq)
    ell_l = torch.norm(term1_l @ v_r, dim=(-2, -1), keepdim=True) + term2_l
    Ll.copy_(torch.maximum(precond_beta * Ll + (1 - precond_beta) * ell_l, ell_l))
    p = Ql - precond_lr/2/Ll * (term1_l @ Ql - term2_l * Ql)
    p = p - precond_lr/2/Ll * (p @ term1_l - p * term2_l)
    Ql.copy_((p + p.mT) / 2)
    
    # right dense update
    term1_r = matmul_transpose(Pg, transpose=True)
    term2_r = G.numel() / Qr.shape[-1]
    row_norm_sq = torch.sum(term1_r * term1_r, dim=-1, keepdim=True)
    max_row_norm_sq, i_max = torch.max(row_norm_sq, dim=-2, keepdim=True)
    v_r = torch.gather(term1_r, dim=-2, index=i_max.expand(-1, -1, term1_r.shape[-1]))
    v_r = v_r.transpose(-2, -1) / torch.sqrt(max_row_norm_sq)
    ell_r = torch.norm(term1_r @ v_r, dim=(-2, -1), keepdim=True) + term2_r
    Lr.copy_(torch.maximum(precond_beta * Lr + (1 - precond_beta) * ell_r, ell_r))
    p = Qr - precond_lr/2/Lr * (term1_r @ Qr - term2_r * Qr)
    p = p - precond_lr/2/Lr * (p @ term1_r - p * term2_r)
    Qr.copy_((p + p.mT) / 2)
    
    return Pg


def merge_adjacent_dims_to_three(tensor):
    """We return either a rank-1 or rank-3 tensor."""
    if tensor.ndim == 1:
        return tensor.shape
    if tensor.ndim == 2:
        return (1,) + tensor.shape
    if tensor.ndim == 3:
        return tensor.shape
    
    dims = list(tensor.shape)
    best_ratio = float('inf')
    best = None
    for s1 in range(1, len(dims) - 1):
        p1 = math.prod(dims[:s1])
        for s2 in range(s1 + 1, len(dims)):
            p2 = math.prod(dims[s1:s2])
            p3 = math.prod(dims[s2:])
            ratio = max(p1, p2, p3) / min(p1, p2, p3)
            if ratio < best_ratio:
                best_ratio = ratio
                best = (p1, p2, p3)
    if best is None:
        return tensor.shape
    return best


@torch.compile
def pack_triu(x):
    n = x.size(-1)
    row, col = torch.triu_indices(n, n, device=x.device, dtype=torch.int64)
    return x[..., row, col]   


@torch.compile
def unpack_triu(v, n):
    *batch, n_triu = v.shape
    row, col = torch.triu_indices(n, n, device=v.device, dtype=torch.int64)
    full = v.new_zeros(*batch, n, n)
    full[..., row, col] = v
    full[..., col, row] = v
    return full
