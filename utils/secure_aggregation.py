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


def provider_to_global_mask(
    provider_id: str,
    peer_provider_ids: list,
    round_id: int,
    weights: dict,
) -> dict:
    """
    Apply pairwise SecAgg masking at the provider->global transit layer.

    This is the Step 2 'missing piece': the same pairwise-mask protocol used
    at the client->provider layer is now also applied at the provider->global
    layer so that even the GlobalServer cannot inspect individual provider
    weight contributions — it only sees the sum.

    Usage (in FlowerProviderBridge.fit or ProviderServer.aggregate_local_updates
    before sending weights to the global tier):

        all_provider_ids = ["P1", "P2", "P3"]
        masked = provider_to_global_mask(
            provider_id      = "P1",
            peer_provider_ids = all_provider_ids,
            round_id         = round_idx,
            weights          = agg_sms_sd,
        )

    The GlobalServer calls secure_aggregate(all_masked_providers) to recover
    the true aggregate — pairwise masks cancel in the sum.

    Parameters
    ----------
    provider_id      : ID of the provider applying the mask.
    peer_provider_ids: Full list of provider IDs participating this round
                       (including provider_id itself).
    round_id         : Current FL round number (makes masks round-unique).
    weights          : The provider's aggregated weight dict to mask.

    Returns
    -------
    dict : Masked weight dict — safe to send to the GlobalServer.
    """
    return mask_client_weights(
        client_id  = provider_id,
        weights    = weights,
        client_ids = peer_provider_ids,
        round_id   = round_id,
    )
