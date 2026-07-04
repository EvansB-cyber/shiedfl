"""
Advanced federated learning algorithms for Non-IID data.
- FedProx: proximal term stabilizes local training against global model drift
- FedOpt: server-side adaptive optimizer (Adam) on aggregated updates
"""
import copy
import torch
import torch.optim as optim


def apply_fedprox_loss(model, global_state: dict, mu: float) -> torch.Tensor:
    """Proximal regularization: (mu/2) * ||w - w_global||^2."""
    if mu <= 0 or not global_state:
        return torch.tensor(0.0)
    prox = torch.tensor(0.0)
    for name, param in model.named_parameters():
        if name in global_state and global_state[name].dtype.is_floating_point:
            prox = prox + torch.sum((param - global_state[name]) ** 2)
    return (mu / 2.0) * prox


def fedopt_step(server_state: dict, aggregated_weights: dict, round_id: int,
                lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.99, eps: float = 1e-8) -> dict:
    """
    FedAdam-style server update: treat (aggregated - current) as pseudo-gradient.
    """
    if not server_state:
        return aggregated_weights

    if not hasattr(fedopt_step, "_m"):
        fedopt_step._m = {}
        fedopt_step._v = {}

    new_state = {}
    for key in aggregated_weights:
        if not aggregated_weights[key].dtype.is_floating_point:
            new_state[key] = aggregated_weights[key].clone()
            continue

        grad = aggregated_weights[key] - server_state[key]
        if key not in fedopt_step._m:
            fedopt_step._m[key] = torch.zeros_like(grad)
            fedopt_step._v[key] = torch.zeros_like(grad)

        fedopt_step._m[key] = beta1 * fedopt_step._m[key] + (1 - beta1) * grad
        fedopt_step._v[key] = beta2 * fedopt_step._v[key] + (1 - beta2) * (grad ** 2)

        m_hat = fedopt_step._m[key] / (1 - beta1 ** (round_id + 1))
        v_hat = fedopt_step._v[key] / (1 - beta2 ** (round_id + 1))
        new_state[key] = server_state[key] + lr * m_hat / (v_hat.sqrt() + eps)

    return new_state


def reset_fedopt_state():
    """Clear FedOpt momentum buffers (call when switching algorithms)."""
    if hasattr(fedopt_step, "_m"):
        del fedopt_step._m
    if hasattr(fedopt_step, "_v"):
        del fedopt_step._v
