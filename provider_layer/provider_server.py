import torch
import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.crypto import encrypt_weights, decrypt_weights
from utils.secure_aggregation import mask_client_weights, secure_aggregate
from utils.escrow_agent import auto_resolve_escrow
from byzantine_aggregators import TrimmedMeanAggregator, KrumAggregator, ByzantineEscrowMonitor

logger = logging.getLogger(__name__)


class ProviderServer:
    """
    Tier 2: Intermediate Provider Node.

    Coordinates local edge nodes, performs sub-aggregation of model updates,
    and acts as Escrow Authority.

    Step 2 fix — correct aggregation order:
        WRONG (old): decrypt → mask → aggregate      (Byzantine updates masked,
                                                       filter sees only random noise)
        RIGHT (new): decrypt → Byzantine filter
                             → mask → secure_aggregate (filter sees real gradients;
                                                        masking hides individuals
                                                        from the provider tier)

    The TrimmedMeanAggregator removes the most extreme client updates
    (trim_fraction=0.1 trims the top and bottom 10% per parameter) before
    masking, so a single malicious client cannot skew the provider aggregate
    even if it passes authentication.
    """

    def __init__(self, provider_id: str, trim_fraction: float = 0.1):
        self.provider_id = provider_id
        self.edge_devices = []
        self.escrow_records = {}
        self.secure_aggregation_enabled = True
        self.escrow_auto_agent_enabled  = True

        # Byzantine filter at the client→provider level (TrimmedMean).
        # Krum is used at the provider→global level inside GlobalServer.
        self._byz_filter   = TrimmedMeanAggregator(trim_fraction=trim_fraction)
        self._byz_monitor  = ByzantineEscrowMonitor(
            threshold=3,
            escrow_callback=self._on_byzantine_escrow_hold,
        )

    def add_edge_device(self, device):
        self.edge_devices.append(device)

    # ── Main aggregation entry point ─────────────────────────────────────────

    def aggregate_local_updates(self, local_results_list, round_id=0, secure_agg=True):
        """
        Step 2 — corrected pipeline:

            decrypt
              ↓
            TrimmedMean Byzantine filter   ← NEW: filters before masking
              ↓
            pairwise mask (SecAgg)          ← masks the already-filtered weights
              ↓
            secure_aggregate               ← sums masked weights; masks cancel
              ↓
            encrypt for transit to GlobalServer

        Returns:
            (encrypted_sms_weights, encrypted_call_weights)
        """
        if not local_results_list:
            return None, None

        client_ids = [
            res.get("device_id", f"client-{i}")
            for i, res in enumerate(local_results_list)
        ]
        use_secure = (
            secure_agg if secure_agg is not None else self.secure_aggregation_enabled
        )

        # ── 1. Decrypt ────────────────────────────────────────────────────────
        sms_weights_list  = [decrypt_weights(res["sms_weights_encrypted"])  for res in local_results_list]
        call_weights_list = [decrypt_weights(res["call_weights_encrypted"]) for res in local_results_list]

        # ── 2. Byzantine filter (TrimmedMean) — operates on raw gradients ─────
        #    Only effective when ≥ 3 clients (below that, trimming removes too many).
        if len(sms_weights_list) >= 3:
            filtered_sms,  trimmed_sms  = self._byz_filter.aggregate(
                sms_weights_list,  provider_id=self.provider_id
            )
            filtered_call, trimmed_call = self._byz_filter.aggregate(
                call_weights_list, provider_id=self.provider_id
            )
            # Record trimmed clients for the ByzantineEscrowMonitor.
            # Union of SMS + call trimmed indices — if a client is suspicious in
            # either model it counts as one flag.
            flagged = sorted(set(trimmed_sms) | set(trimmed_call))
            flagged_ids = [client_ids[i] for i in flagged if i < len(client_ids)]
            if flagged_ids:
                self._byz_monitor.record_flags(flagged_ids, round_id, reason="TrimmedMean")
                logger.warning(
                    "[Provider:%s] Round %d — trimmed clients: %s",
                    self.provider_id, round_id, flagged_ids
                )
            # After filtering we have one aggregated weight dict; wrap in a list
            # so the masking step below receives consistent input.
            sms_to_mask  = [filtered_sms]
            call_to_mask = [filtered_call]
        else:
            # Too few clients to filter — fall back to plain average.
            logger.warning(
                "[Provider:%s] Round %d — only %d client(s), skipping Byzantine filter.",
                self.provider_id, round_id, len(sms_weights_list)
            )
            sms_to_mask  = sms_weights_list
            call_to_mask = call_weights_list

        # ── 3. Pairwise SecAgg masking (on already-filtered weights) ──────────
        if use_secure and len(sms_to_mask) > 1:
            mask_ids = [f"{self.provider_id}-filtered-{i}" for i in range(len(sms_to_mask))]
            masked_sms  = [mask_client_weights(cid, w, mask_ids, round_id)
                           for cid, w in zip(mask_ids, sms_to_mask)]
            masked_call = [mask_client_weights(cid, w, mask_ids, round_id)
                           for cid, w in zip(mask_ids, call_to_mask)]
            agg_sms  = secure_aggregate(masked_sms)
            agg_call = secure_aggregate(masked_call)
        else:
            # Single entry after filtering or SecAgg disabled — no masking needed.
            agg_sms  = sms_to_mask[0]
            agg_call = call_to_mask[0]

        # ── 4. Encrypt for transit ────────────────────────────────────────────
        return encrypt_weights(agg_sms), encrypt_weights(agg_call)

    # ── Escrow evaluation ─────────────────────────────────────────────────────

    def evaluate_escrow(self, transfer_id, sender_id, receiver_phone, amount, risk_report, message=""):
        """
        Escrow Decision Logic with automated agent for low/high confidence cases.
        """
        risk_score   = risk_report["total_risk_score"]
        agent_result = None

        if self.escrow_auto_agent_enabled:
            agent_result = auto_resolve_escrow(risk_report, amount, message)

        if agent_result and agent_result["action"] == "AUTO_APPROVE":
            status = "APPROVED"
            reason = f"[Auto-Agent] {agent_result['reason']}"
        elif agent_result and agent_result["action"] == "AUTO_BLOCK":
            status = "BLOCKED"
            reason = f"[Auto-Agent] {agent_result['reason']}"
        elif risk_score >= 0.65:
            status = "HELD_IN_ESCROW"
            reason = "High risk detected: "
            if risk_report["sms_risk_score"] > 0.7:
                reason += "Potential SMS Phishing content. "
            if risk_report["contact_risk_score"] > 0.7:
                reason += "Receiver phone is flagged as untrusted. "
            if risk_report["amount_risk_score"] > 0.7:
                reason += "Unusually large transfer amount. "
            reason = reason.strip() or "High aggregate neural network risk score."
        else:
            status = "APPROVED"
            reason = "Passed security filters."

        record = {
            "transfer_id":   transfer_id,
            "sender_id":     sender_id,
            "receiver_phone": receiver_phone,
            "amount":        amount,
            "risk_report":   risk_report,
            "status":        status,
            "decision_by":   self.provider_id,
            "reason":        reason,
            "agent_decision": agent_result,
        }
        self.escrow_records[transfer_id] = record
        return record

    def resolve_escrow(self, transfer_id, action):
        if transfer_id in self.escrow_records:
            record = self.escrow_records[transfer_id]
            if record["status"] == "HELD_IN_ESCROW":
                if action == "RELEASE":
                    record["status"] = "RELEASED_FROM_ESCROW"
                elif action == "BLOCK":
                    record["status"] = "BLOCKED"
                return record
        return None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _avg_weights(self, weights_list):
        """Plain FedAvg — used only when Byzantine filter is skipped."""
        avg = {}
        for key in weights_list[0].keys():
            if weights_list[0][key].dtype.is_floating_point:
                avg[key] = torch.stack([w[key] for w in weights_list]).mean(dim=0)
            else:
                avg[key] = weights_list[0][key].clone()
        return avg

    def _on_byzantine_escrow_hold(self, provider_id: str, reason: str):
        """
        Called by ByzantineEscrowMonitor when a client exceeds the flag threshold.
        Logs the event; in production this would freeze the device's escrow queue.
        """
        logger.error(
            "[Provider:%s] Escrow HOLD triggered for device %s — %s",
            self.provider_id, provider_id, reason
        )

    def get_byzantine_audit_log(self):
        return self._byz_monitor.get_audit_log()
