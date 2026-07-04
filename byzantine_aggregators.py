import torch
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

def _flatten(state_dict):
    return torch.cat([v.float().flatten() for v in state_dict.values()])

def _unflatten(flat, reference):
    result = {}
    offset = 0
    for k, v in reference.items():
        numel = v.numel()
        result[k] = flat[offset: offset + numel].reshape(v.shape).to(v.dtype)
        offset += numel
    return result

class TrimmedMeanAggregator:
    def __init__(self, trim_fraction=0.1):
        if not (0.0 < trim_fraction < 0.5):
            raise ValueError("trim_fraction must be in (0, 0.5)")
        self.trim_fraction = trim_fraction

    def aggregate(self, updates, provider_id=None):
        n = len(updates)
        k = int(self.trim_fraction * n)
        remaining = n - 2 * k
        if remaining <= 0:
            raise ValueError("trim_fraction removes all updates.")
        stacked = torch.stack([_flatten(u) for u in updates], dim=0)
        sorted_indices = torch.argsort(stacked, dim=0)
        tail_low = sorted_indices[:k, :]
        tail_high = sorted_indices[n-k:, :]
        mask = torch.ones_like(stacked)
        for i in range(k):
            mask.scatter_(0, tail_low[i:i+1, :], 0.0)
            mask.scatter_(0, tail_high[i:i+1, :], 0.0)
        trimmed_mean = (stacked * mask).sum(dim=0) / remaining
        trimmed_set = set(tail_low.unique().tolist() + tail_high.unique().tolist())
        trimmed_indices = sorted(int(x) for x in trimmed_set)
        if trimmed_indices:
            logger.warning("[TrimmedMean][provider=%s] Trimmed %d client(s): %s",
                           provider_id or "?", len(trimmed_indices), trimmed_indices)
        return _unflatten(trimmed_mean, updates[0]), trimmed_indices


class KrumAggregator:
    def __init__(self, num_byzantine=1, multi_k=2):
        self.f = num_byzantine
        self.m = multi_k

    def aggregate(self, updates, provider_ids=None):
        n = len(updates)
        if n < 2 * self.f + 3:
            logger.warning("[Krum] n=%d does not satisfy n >= 2f+3=%d. Proceeding with Multi-Krum (m=%d).",
                           n, 2 * self.f + 3, self.m)
        flat = torch.stack([_flatten(u) for u in updates], dim=0)
        diff = flat.unsqueeze(0) - flat.unsqueeze(1)
        dist_sq = (diff ** 2).sum(dim=-1)
        neighbours = max(1, n - self.f - 2)
        scores = torch.zeros(n)
        for i in range(n):
            row = dist_sq[i].clone()
            row[i] = float("inf")
            sorted_dists, _ = torch.sort(row)
            scores[i] = sorted_dists[:neighbours].sum()
        _, ranked = torch.sort(scores)
        selected_indices = ranked[:self.m].tolist()
        rejected_indices = ranked[self.m:].tolist()
        if rejected_indices:
            rej_names = ([provider_ids[i] for i in rejected_indices]
                         if provider_ids else rejected_indices)
            logger.warning("[Krum] Rejected as potentially Byzantine: %s", rej_names)
        selected_flat = flat[selected_indices].mean(dim=0)
        return _unflatten(selected_flat, updates[0]), selected_indices, rejected_indices


class ByzantineEscrowMonitor:
    def __init__(self, threshold=3, db_conn=None, escrow_callback=None):
        self.threshold = threshold
        self.db = db_conn
        self.escrow_callback = escrow_callback
        self._flag_counts = {}
        self._audit_log = {}

    def record_flags(self, flagged_ids, round_num, reason="byzantine"):
        for pid in flagged_ids:
            self._flag_counts[pid] = self._flag_counts.get(pid, 0) + 1
            entry = {
                "round": round_num,
                "timestamp": datetime.utcnow().isoformat(),
                "reason": reason,
                "consecutive_flags": self._flag_counts[pid],
            }
            self._audit_log.setdefault(pid, []).append(entry)
            logger.warning("[ByzantineMonitor] %s flagged (%d/%d) - round %d",
                           pid, self._flag_counts[pid], self.threshold, round_num)
            if self._flag_counts[pid] >= self.threshold:
                self._trigger_escrow_hold(pid, round_num)

    def clear_flags(self, provider_id):
        self._flag_counts.pop(provider_id, None)
        logger.info("[ByzantineMonitor] Cleared flags for %s", provider_id)

    def get_audit_log(self, provider_id=None):
        if provider_id:
            return self._audit_log.get(provider_id, [])
        return self._audit_log

    def _trigger_escrow_hold(self, provider_id, round_num):
        reason = (
            f"Provider {provider_id} flagged as Byzantine for "
            f"{self._flag_counts[provider_id]} consecutive rounds "
            f"(threshold={self.threshold}). Escrow settlement withheld."
        )
        logger.error("[EscrowHold] %s", reason)
        if self.escrow_callback:
            self.escrow_callback(provider_id, reason)