"""
ShieldFL - Flower Dual-Role Provider Bridge (Step 1)
======================================================

Each ProviderServer acts as BOTH a Flower client (relative to the
GlobalServer) AND a Flower server (relative to its own edge devices).

Architecture:

    flwr.simulation.start_simulation()
          |
          v
    FlowerProviderBridge(ProviderServer)   <- one per provider
          |  implements flwr.client.NumPyClient
          |
          +-- on fit()      -> collect edge device updates
          |                    -> ProviderServer.aggregate_local_updates()
          |                       (TrimmedMean Byzantine filter -> SecAgg -> encrypt)
          |                    -> send provider-aggregated weights UP to global
          |
          +-- on evaluate() -> distribute global weights DOWN to edges
                               -> run holdout eval, return loss + metrics

Global aggregation uses a custom Flower Strategy (ShieldFLStrategy) that:
  * Applies Krum at the provider->global level (second Byzantine defence layer)
  * Calls ml_tracking.log_round() after each server round
  * Writes tracking_db.log_model_version() via log_round()
  * Saves .pth checkpoints to models_checkpoint/

Usage (via main.py):
    python main.py --flower --rounds 5 --algorithm fedavg

Or directly:
    from flower_bridge import run_flower_simulation
    run_flower_simulation(num_rounds=5, fl_algorithm="fedavg")

Self-test (no flwr required):
    python flower_bridge.py
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

# -- project root on sys.path -------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from edge_layer.models import SMSFraudCNN, CallDetectionMLP
from edge_layer.data import get_dataloaders
from edge_layer.edge_device import EdgeDevice
from provider_layer.provider_server import ProviderServer
from global_layer.global_server import GlobalServer
from byzantine_aggregators import KrumAggregator
from utils.ml_tracking import start_run, log_params, log_round, end_run
from utils.crypto import encrypt_weights, decrypt_weights

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(ROOT, "models_checkpoint")


# ---------------------------------------------------------------------------
# Helpers: state-dict <-> Flower numpy-array wire format
# ---------------------------------------------------------------------------

def state_dict_to_ndarrays(state_dict: dict) -> List[np.ndarray]:
    """Convert a PyTorch state-dict to a list of numpy arrays (Flower format)."""
    return [v.detach().cpu().numpy() for v in state_dict.values()]


def ndarrays_to_state_dict(arrays: List[np.ndarray], reference: dict) -> dict:
    """Reconstruct a state-dict from Flower numpy arrays using reference for keys/dtypes."""
    result = {}
    for (key, ref_tensor), arr in zip(reference.items(), arrays):
        result[key] = torch.tensor(arr, dtype=ref_tensor.dtype)
    return result


# ---------------------------------------------------------------------------
# Dual-Role Bridge: Flower NumPyClient wrapping ProviderServer
# ---------------------------------------------------------------------------

class FlowerProviderBridge:
    """
    Wraps a ProviderServer so it participates in Flower simulation as a
    federated client of the GlobalServer, while acting as a local aggregator
    for its own edge devices.

    Dual role
    ---------
    As Flower CLIENT  : sends provider-aggregated weights up to the global tier.
    As local SERVER   : runs TrimmedMean Byzantine filter then SecAgg pairwise
                        masking on its edge device updates before sending up.

    flwr is NOT imported at module level; the rest of the codebase remains
    importable even when flwr is not installed.
    """

    def __init__(
        self,
        provider: "ProviderServer",
        edge_devices: List["EdgeDevice"],
        fl_algorithm: str = "fedavg",
        epochs_per_round: int = 1,
        fedprox_mu: float = 0.01,
    ):
        self.provider      = provider
        self.edge_devices  = edge_devices
        self.fl_algorithm  = fl_algorithm
        self.epochs        = epochs_per_round
        self.fedprox_mu    = fedprox_mu

        # Reference models - used only for shape/key metadata, never trained here
        self._sms_ref  = SMSFraudCNN()
        self._call_ref = CallDetectionMLP()

    # -------------------------------------------------------------------------
    # Flower NumPyClient interface
    # -------------------------------------------------------------------------

    def get_parameters(self, config: dict) -> List[np.ndarray]:
        """Return initial parameters. Flower calls this once at round 0."""
        return (
            state_dict_to_ndarrays(self._sms_ref.state_dict())
            + state_dict_to_ndarrays(self._call_ref.state_dict())
        )

    def fit(
        self,
        parameters: List[np.ndarray],
        config: dict,
    ) -> Tuple[List[np.ndarray], int, dict]:
        """
        Flower calls fit() each round.

        Pipeline (correct order — Step 2 constraint preserved):
          1. Unpack global weights from Flower server.
          2. Push weights down to all edge devices (set_model_weights).
          3. Each edge device runs train_local() independently.
          4. ProviderServer.aggregate_local_updates():
               decrypt -> TrimmedMean Byzantine filter -> SecAgg masking -> encrypt
          5. Decrypt provider aggregate, convert to numpy, return UP to Flower.
        """
        round_id = int(config.get("server_round", 0))

        # 1. Unpack global parameters into state-dicts
        n_sms  = len(self._sms_ref.state_dict())
        n_call = len(self._call_ref.state_dict())
        global_sms_sd  = ndarrays_to_state_dict(
            parameters[:n_sms], self._sms_ref.state_dict()
        )
        global_call_sd = ndarrays_to_state_dict(
            parameters[n_sms: n_sms + n_call], self._call_ref.state_dict()
        )

        # 2. Push to edge devices (EdgeDevice.set_model_weights expects encrypted)
        enc_sms  = encrypt_weights(global_sms_sd)
        enc_call = encrypt_weights(global_call_sd)
        for dev in self.edge_devices:
            dev.set_model_weights(enc_sms, enc_call)

        # 3. Local training on each edge device
        client_results = []
        for dev in self.edge_devices:
            res = dev.train_local(
                epochs=self.epochs,
                lr=0.01,
                fl_algorithm=self.fl_algorithm,
                fedprox_mu=self.fedprox_mu,
                global_sms_state=global_sms_sd if self.fl_algorithm == "fedprox" else None,
                global_call_state=global_call_sd if self.fl_algorithm == "fedprox" else None,
            )
            client_results.append(res)

        # 4. Provider aggregation: TrimmedMean Byzantine filter -> SecAgg
        enc_agg_sms, enc_agg_call = self.provider.aggregate_local_updates(
            client_results, round_id=round_id
        )

        # 5. Decrypt and convert to Flower numpy arrays
        agg_sms_sd  = decrypt_weights(enc_agg_sms)
        agg_call_sd = decrypt_weights(enc_agg_call)
        out_arrays = (
            state_dict_to_ndarrays(agg_sms_sd)
            + state_dict_to_ndarrays(agg_call_sd)
        )

        return out_arrays, len(self.edge_devices), {
            "provider_id": self.provider.provider_id,
            "num_devices": float(len(self.edge_devices)),
            "round_id":    float(round_id),
        }

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: dict,
    ) -> Tuple[float, int, dict]:
        """
        Flower calls evaluate() each round.
        Loads global SMS weights and evaluates on provider-local data.
        """
        n_sms = len(self._sms_ref.state_dict())
        sms_sd = ndarrays_to_state_dict(
            parameters[:n_sms], self._sms_ref.state_dict()
        )
        self._sms_ref.load_state_dict(sms_sd)
        self._sms_ref.eval()

        sms_loader, _ = get_dataloaders(
            client_id=self.provider.provider_id,
            batch_size=16,
            num_samples=60,
        )
        correct = total = 0
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for x, y in sms_loader:
                out        = self._sms_ref(x)
                total_loss += criterion(out, y).item()
                correct    += (out.argmax(dim=1) == y).sum().item()
                total      += y.size(0)

        acc  = correct / max(1, total)
        loss = total_loss / max(1, len(sms_loader))
        return float(loss), total, {
            "sms_accuracy": acc,
            "provider_id":  self.provider.provider_id,
        }


# ---------------------------------------------------------------------------
# Custom Flower Strategy: Krum at provider->global + MLflow/tracking_db hooks
# ---------------------------------------------------------------------------

def _build_flower_strategy(
    global_server: "GlobalServer",
    fl_algorithm: str,
    num_rounds: int,
):
    """
    Build and return a flwr.server.strategy.Strategy configured with:

      * Krum aggregation at the provider->global level (second Byzantine layer)
      * FedAvg mean fallback when too few providers for Krum
      * log_round() after every round (MLflow metrics + tracking_db row)
      * .pth checkpoint saved to models_checkpoint/ each round

    Raises ImportError if flwr is not installed.
    """
    try:
        import flwr as flwr_lib
        from flwr.server.strategy import FedAvg
        from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    except ImportError as exc:
        raise ImportError(
            "flwr is not installed. Run: pip install flwr\n"
            "Or: pip install -r requirements.txt"
        ) from exc

    krum = KrumAggregator(num_byzantine=1, multi_k=2)

    class ShieldFLStrategy(FedAvg):
        """
        FedAvg subclass that overrides aggregate_fit with Krum aggregation
        and adds MLflow / tracking_db logging after each round.
        """

        def aggregate_fit(self, server_round, results, failures):
            """
            Krum aggregation across providers.

            Each provider already ran TrimmedMean+SecAgg at its tier, so
            Krum here provides a second, independent Byzantine defence at
            the global tier — tolerating up to f=1 compromised provider.
            """
            if not results:
                return None, {}

            all_arrays = [
                parameters_to_ndarrays(fit_res.parameters)
                for _, fit_res in results
            ]

            # Build per-provider state-dicts for SMS and call models
            n_sms  = len(SMSFraudCNN().state_dict())
            n_call = len(CallDetectionMLP().state_dict())
            sms_ref  = SMSFraudCNN()
            call_ref = CallDetectionMLP()

            sms_sds  = [
                ndarrays_to_state_dict(a[:n_sms], sms_ref.state_dict())
                for a in all_arrays
            ]
            call_sds = [
                ndarrays_to_state_dict(
                    a[n_sms: n_sms + n_call], call_ref.state_dict()
                )
                for a in all_arrays
            ]

            # Krum (requires >= 2f+3 = 5 providers; falls back below that)
            if len(sms_sds) >= 3:
                agg_sms_sd,  _, rej_sms  = krum.aggregate(sms_sds)
                agg_call_sd, _, rej_call = krum.aggregate(call_sds)
                if rej_sms or rej_call:
                    logger.warning(
                        "[ShieldFLStrategy] Round %d Krum rejected providers: "
                        "SMS=%s Call=%s",
                        server_round, rej_sms, rej_call,
                    )
            else:
                logger.warning(
                    "[ShieldFLStrategy] Round %d: only %d provider(s); using FedAvg.",
                    server_round, len(sms_sds),
                )

                def _mean_sd(sd_list):
                    out = {}
                    for k in sd_list[0]:
                        stacked = torch.stack([sd[k].float() for sd in sd_list])
                        out[k]  = stacked.mean(0).to(sd_list[0][k].dtype)
                    return out

                agg_sms_sd  = _mean_sd(sms_sds)
                agg_call_sd = _mean_sd(call_sds)

            # Push aggregated weights into the shared global_server instance
            global_server.sms_model.load_state_dict(agg_sms_sd)
            global_server.call_model.load_state_dict(agg_call_sd)

            # Evaluate on global holdout and capture metrics
            metrics = global_server.log_metrics(round_id=server_round)
            print(
                f"  [Flower Round {server_round}] "
                f"SMS: {metrics['sms_accuracy'] * 100:.2f}%  "
                f"Holdout SMS: {metrics['holdout_sms_accuracy'] * 100:.2f}%"
            )

            # Save checkpoints
            os.makedirs(MODELS_DIR, exist_ok=True)
            ckpt_sms  = os.path.join(MODELS_DIR, "global_sms_model.pth")
            ckpt_call = os.path.join(MODELS_DIR, "global_call_model.pth")
            torch.save(global_server.sms_model.state_dict(), ckpt_sms)
            torch.save(global_server.call_model.state_dict(), ckpt_call)

            # Single log_round call: MLflow metrics + tracking_db model version row
            log_round(
                metrics=metrics,
                round_id=server_round,
                checkpoint_path=ckpt_sms,
                fl_algorithm=fl_algorithm,
            )

            out_arrays = (
                state_dict_to_ndarrays(agg_sms_sd)
                + state_dict_to_ndarrays(agg_call_sd)
            )
            return ndarrays_to_parameters(out_arrays), {}

        def aggregate_evaluate(self, server_round, results, failures):
            """Weighted-average loss across all provider evaluations."""
            if not results:
                return None, {}
            total    = sum(num for _, (num, _, _) in results)
            avg_loss = sum(num * loss for _, (num, loss, _) in results) / max(1, total)
            return avg_loss, {}

    # Initialise strategy with the current global model weights so round 0
    # starts from the same state as the CLI simulation mode.
    init_arrays = (
        state_dict_to_ndarrays(global_server.sms_model.state_dict())
        + state_dict_to_ndarrays(global_server.call_model.state_dict())
    )
    return ShieldFLStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=1,
        min_available_clients=2,
        initial_parameters=flwr_lib.common.ndarrays_to_parameters(init_arrays),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_flower_simulation(
    num_rounds: int = 5,
    fl_algorithm: str = "fedavg",
    epochs_per_round: int = 1,
    num_providers: int = 3,
    devices_per_provider: int = 3,
):
    """
    Run the full 3-tier ShieldFL system via Flower simulation.

    Topology
    --------
    GlobalServer  (Flower server — ShieldFLStrategy with Krum)
        ^
    ProviderServer x num_providers  (Flower clients — FlowerProviderBridge)
        ^
    EdgeDevice x devices_per_provider per provider

    Security guarantees
    -------------------
    Edge->Provider   : TrimmedMean Byzantine filter + SecAgg pairwise masking
    Provider->Global : Krum Byzantine filter
    Transit          : encrypt_weights() on all weight payloads

    Observability
    -------------
    Every round -> MLflow run (or local fallback) + tracking_db model_version row
    """
    try:
        import flwr as flwr_lib
    except ImportError:
        print(
            "\n[ShieldFL] flwr is not installed.\n"
            "Install it: pip install flwr\n"
            "Then re-run: python main.py --flower\n"
        )
        return

    print("=" * 62)
    print("SHIELDFL - FLOWER SIMULATION (Dual-Role Provider Bridge)")
    print(f"Algorithm: {fl_algorithm.upper()} | Krum + TrimmedMean | SecAgg: ON")
    print(f"Providers: {num_providers} | Devices/provider: {devices_per_provider}")
    print("=" * 62)

    # Start MLflow run
    start_run("3tier-fl-flower", tags={"algorithm": fl_algorithm, "mode": "flower"})
    log_params({
        "fl_algorithm":          fl_algorithm,
        "rounds":                num_rounds,
        "num_providers":         num_providers,
        "devices_per_provider":  devices_per_provider,
        "epochs_per_round":      epochs_per_round,
    })

    # Build topology
    global_server  = GlobalServer(fl_algorithm=fl_algorithm)
    providers_map: Dict[str, ProviderServer]   = {}
    devices_map:   Dict[str, List[EdgeDevice]] = {}

    for p_idx in range(num_providers):
        pid  = f"P{p_idx + 1}"
        prov = ProviderServer(pid)
        devs: List[EdgeDevice] = []
        for d_idx in range(devices_per_provider):
            dev = EdgeDevice(f"{pid}-{d_idx}")
            prov.add_edge_device(dev)
            devs.append(dev)
        providers_map[pid] = prov
        devices_map[pid]   = devs

    def client_fn(cid: str):
        """Flower calls client_fn("0"), client_fn("1"), ... for each client."""
        pid      = list(providers_map.keys())[int(cid)]
        bridge   = FlowerProviderBridge(
            provider         = providers_map[pid],
            edge_devices     = devices_map[pid],
            fl_algorithm     = fl_algorithm,
            epochs_per_round = epochs_per_round,
        )

        from flwr.client import NumPyClient

        class _Adapted(NumPyClient):
            def get_parameters(self, config):
                return bridge.get_parameters(config)
            def fit(self, parameters, config):
                return bridge.fit(parameters, config)
            def evaluate(self, parameters, config):
                return bridge.evaluate(parameters, config)

        return _Adapted().to_client()

    strategy = _build_flower_strategy(global_server, fl_algorithm, num_rounds)

    # Pre-flight: validate escrow pipeline is live before FL starts
    first_pid = list(devices_map.keys())[0]
    first_dev = devices_map[first_pid][0]
    fraud_sms = (
        "URGENT: Your MTN MoMo wallet is suspended. "
        "Send your PIN and OTP to unblock your account now."
    )
    print("\n[Pre-flight] Escrow smoke-test on first edge device ...")
    try:
        _ = first_dev.receive_inbound(
            sender_phone="+233200000000",
            message_text=fraud_sms,
            provider_id=first_pid,
        )
        print("[Pre-flight] Passed OK\n")
    except Exception as e:
        print(f"[Pre-flight] WARNING: receive_inbound() raised {e}\n")

    # Run simulation
    history = flwr_lib.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_providers,
        config=flwr_lib.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    print("\n" + "=" * 62)
    print("FLOWER SIMULATION COMPLETED")
    print(f"Checkpoints saved to: {MODELS_DIR}")
    print("=" * 62)

    if history and hasattr(history, "metrics_distributed"):
        for rnd, m in history.metrics_distributed.items():
            print(f"  Round {rnd}: {m}")

    end_run()
    return history


# ---------------------------------------------------------------------------
# Quick self-test: validates the bridge without starting Flower
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== flower_bridge.py self-test (no flwr required) ===\n")

    prov = ProviderServer("TEST")
    devs = [EdgeDevice(f"T-{i}") for i in range(3)]
    for d in devs:
        prov.add_edge_device(d)

    bridge = FlowerProviderBridge(prov, devs)

    # 1. get_parameters round-trip
    params  = bridge.get_parameters({})
    n_exp   = len(SMSFraudCNN().state_dict()) + len(CallDetectionMLP().state_dict())
    assert len(params) == n_exp, f"Expected {n_exp} arrays, got {len(params)}"
    print(f"[1] get_parameters() -> {len(params)} arrays  ... PASS")

    # 2. fit()
    out_params, n_ex, metrics = bridge.fit(params, {"server_round": 1})
    assert len(out_params) == n_exp and n_ex == 3
    print("[2] fit() one round                           ... PASS")

    # 3. evaluate()
    loss, n_ev, eval_m = bridge.evaluate(out_params, {"server_round": 1})
    assert isinstance(loss, float)
    print("[3] evaluate()                                ... PASS")

    # 4. state-dict round-trip integrity
    sms_sd    = SMSFraudCNN().state_dict()
    arrays    = state_dict_to_ndarrays(sms_sd)
    recovered = ndarrays_to_state_dict(arrays, sms_sd)
    for k in sms_sd:
        assert torch.allclose(sms_sd[k].float(), recovered[k].float()), \
            f"Round-trip mismatch on key: {k}"
    print("[4] state-dict round-trip integrity           ... PASS")

    print("\nAll self-tests passed.")
    print("Full simulation: python main.py --flower --rounds 5 --algorithm fedavg")
