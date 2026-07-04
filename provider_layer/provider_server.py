import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.crypto import encrypt_weights, decrypt_weights
from utils.secure_aggregation import mask_client_weights, secure_aggregate
from utils.escrow_agent import auto_resolve_escrow

class ProviderServer:
    """
    Tier 2: Intermediate Provider Node.
    Coordinates local edge nodes, performs sub-aggregation of model updates, and acts as Escrow Authority.
    """
    def __init__(self, provider_id):
        self.provider_id = provider_id
        self.edge_devices = []
        self.escrow_records = {}
        self.secure_aggregation_enabled = True
        self.escrow_auto_agent_enabled = True

    def add_edge_device(self, device):
        self.edge_devices.append(device)

    def aggregate_local_updates(self, local_results_list, round_id=0, secure_agg=True):
        """
        Federated Averaging with optional secure aggregation (pairwise masking).
        """
        if not local_results_list:
            return None, None

        client_ids = [res.get("device_id", f"client-{i}") for i, res in enumerate(local_results_list)]
        use_secure = secure_agg if secure_agg is not None else self.secure_aggregation_enabled

        sms_weights_list = [decrypt_weights(res["sms_weights_encrypted"]) for res in local_results_list]
        call_weights_list = [decrypt_weights(res["call_weights_encrypted"]) for res in local_results_list]

        if use_secure and len(sms_weights_list) > 1:
            masked_sms = [
                mask_client_weights(cid, w, client_ids, round_id)
                for cid, w in zip(client_ids, sms_weights_list)
            ]
            masked_call = [
                mask_client_weights(cid, w, client_ids, round_id)
                for cid, w in zip(client_ids, call_weights_list)
            ]
            aggregated_sms_weights = secure_aggregate(masked_sms)
            aggregated_call_weights = secure_aggregate(masked_call)
        else:
            aggregated_sms_weights = self._avg_weights(sms_weights_list)
            aggregated_call_weights = self._avg_weights(call_weights_list)

        return encrypt_weights(aggregated_sms_weights), encrypt_weights(aggregated_call_weights)

    def _avg_weights(self, weights_list):
        avg_weights = {}
        for key in weights_list[0].keys():
            if weights_list[0][key].dtype.is_floating_point:
                tensors = [w[key] for w in weights_list]
                avg_weights[key] = torch.stack(tensors, dim=0).mean(dim=0)
            else:
                avg_weights[key] = weights_list[0][key].clone()
        return avg_weights

    def evaluate_escrow(self, transfer_id, sender_id, receiver_phone, amount, risk_report, message=""):
        """
        Escrow Decision Logic with automated agent for low/high confidence cases.
        """
        risk_score = risk_report["total_risk_score"]
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
                reason += "Receiver phone is flagged as untrusted in contacts database. "
            if risk_report["amount_risk_score"] > 0.7:
                reason += "Unusually large transfer amount. "
            if reason.endswith(" "):
                reason += "High aggregate neural network risk score."
            else:
                reason = reason.strip()
        else:
            status = "APPROVED"
            reason = "Passed security filters."

        record = {
            "transfer_id": transfer_id,
            "sender_id": sender_id,
            "receiver_phone": receiver_phone,
            "amount": amount,
            "risk_report": risk_report,
            "status": status,
            "decision_by": self.provider_id,
            "reason": reason,
            "agent_decision": agent_result
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
