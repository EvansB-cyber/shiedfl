import torch


class ServerOptimizer:
    """
    Base class for FedOpt-style server-side adaptive aggregation strategies.
    (Reddi et al., "Adaptive Federated Optimization", ICLR 2021)

    Treats the difference between the client-averaged model and the current
    global model as a pseudo-gradient, then applies an adaptive update rule
    on top of standard FedAvg aggregation. This generally improves convergence
    speed and stability under non-IID client data, compared to plain FedAvg.
    """

    def __init__(self, server_lr=0.1, beta1=0.9, beta2=0.99, tau=1e-3):
        self.server_lr = server_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.m = {}
        self.v = {}

    def _init_state(self, key, shape, device):
        if key not in self.m:
            self.m[key] = torch.zeros(shape, device=device)
            self.v[key] = torch.zeros(shape, device=device)

    def reset(self):
        """Clears momentum/variance buffers (called when switching strategies)."""
        self.m = {}
        self.v = {}

    def step(self, current_weights, averaged_client_weights):
        raise NotImplementedError


class FedAvgOptimizer(ServerOptimizer):
    """Plain FedAvg: the new global model is simply the client average. No state kept."""

    def step(self, current_weights, averaged_client_weights):
        return {k: v.clone() for k, v in averaged_client_weights.items()}


class FedAdamOptimizer(ServerOptimizer):
    """Server-side Adam: m and v track first/second moments of the pseudo-gradient."""

    def step(self, current_weights, averaged_client_weights):
        new_weights = {}
        for key, old_val in current_weights.items():
            if not old_val.dtype.is_floating_point:
                new_weights[key] = averaged_client_weights[key].clone()
                continue
            delta = averaged_client_weights[key] - old_val
            self._init_state(key, delta.shape, delta.device)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * delta
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (delta ** 2)
            new_weights[key] = old_val + self.server_lr * self.m[key] / (torch.sqrt(self.v[key]) + self.tau)
        return new_weights


class FedYogiOptimizer(ServerOptimizer):
    """
    Server-side Yogi: like Adam, but the second-moment update uses a
    multiplicative sign correction, which controls variance growth better
    on highly non-IID / sparse client updates.
    """

    def step(self, current_weights, averaged_client_weights):
        new_weights = {}
        for key, old_val in current_weights.items():
            if not old_val.dtype.is_floating_point:
                new_weights[key] = averaged_client_weights[key].clone()
                continue
            delta = averaged_client_weights[key] - old_val
            self._init_state(key, delta.shape, delta.device)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * delta
            delta_sq = delta ** 2
            sign_term = torch.sign(self.v[key] - delta_sq)
            self.v[key] = self.v[key] - (1 - self.beta2) * sign_term * delta_sq
            new_weights[key] = old_val + self.server_lr * self.m[key] / (torch.sqrt(self.v[key]) + self.tau)
        return new_weights


class FedAdagradOptimizer(ServerOptimizer):
    """Server-side Adagrad: second moment only accumulates, never decays."""

    def step(self, current_weights, averaged_client_weights):
        new_weights = {}
        for key, old_val in current_weights.items():
            if not old_val.dtype.is_floating_point:
                new_weights[key] = averaged_client_weights[key].clone()
                continue
            delta = averaged_client_weights[key] - old_val
            self._init_state(key, delta.shape, delta.device)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * delta
            self.v[key] = self.v[key] + (delta ** 2)
            new_weights[key] = old_val + self.server_lr * self.m[key] / (torch.sqrt(self.v[key]) + self.tau)
        return new_weights


STRATEGY_REGISTRY = {
    "fedavg": FedAvgOptimizer,
    "fedadam": FedAdamOptimizer,
    "fedyogi": FedYogiOptimizer,
    "fedadagrad": FedAdagradOptimizer,
}


def build_optimizer(name, server_lr=0.1, beta1=0.9, beta2=0.99, tau=1e-3):
    name = name.lower()
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown server optimizer strategy: '{name}'. Choose from {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name](server_lr=server_lr, beta1=beta1, beta2=beta2, tau=tau)