"""
Secure aggregation via pairwise masking (Bonawitz et al. simplified simulation).

Each client masks its weight update so the aggregator only sees masked values.
Pairwise masks cancel when summed, revealing the true aggregate without exposing
individual client weights.
"""
import hashlib
import torch


def _seed_for(client_id: str, round_id: int, key: str) -> int:
    raw = f"{client_id}:{round_id}:{key}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def generate_pairwise_mask(client_id: str, peer_id: str, round_id: int, template: dict) -> dict:
    """Deterministic pseudo-random mask shared between a client pair."""
    mask = {}
    for key, tensor in template.items():
        if not tensor.dtype.is_floating_point:
            continue
        seed = _seed_for(client_id, round_id, key) ^ _seed_for(peer_id, round_id, key)
        gen = torch.Generator()
        gen.manual_seed(seed)
        mask[key] = torch.randn(tensor.shape, generator=gen, dtype=tensor.dtype)
    return mask


def mask_client_weights(client_id: str, weights: dict, client_ids: list, round_id: int) -> dict:
    """
    Apply pairwise masking to client weights before transmission.
    Client i adds +M(i,j) for j > i and subtracts -M(j,i) for j < i.
    """
    if len(client_ids) <= 1:
        return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in weights.items()}

    sorted_ids = sorted(client_ids)
    idx = sorted_ids.index(client_id)
    masked = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in weights.items()}

    for j, peer_id in enumerate(sorted_ids):
        if peer_id == client_id:
            continue
        pair_mask = generate_pairwise_mask(client_id, peer_id, round_id, weights)
        sign = 1.0 if idx < j else -1.0
        for key in pair_mask:
            masked[key] = masked[key] + sign * pair_mask[key]
    return masked


def secure_aggregate(masked_weights_list: list) -> dict:
    """Sum masked client weights; pairwise masks cancel to yield FedAvg result."""
    if not masked_weights_list:
        return {}
    aggregated = {}
    for key in masked_weights_list[0].keys():
        tensors = [w[key] for w in masked_weights_list]
        if tensors[0].dtype.is_floating_point:
            aggregated[key] = torch.stack(tensors, dim=0).sum(dim=0) / len(tensors)
        else:
            aggregated[key] = tensors[0].clone()
    return aggregated
