import sys
import os
import argparse
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from global_layer.global_server import GlobalServer
from provider_layer.provider_server import ProviderServer
from edge_layer.edge_device import EdgeDevice
from utils.ml_tracking import start_run, log_params, log_round, end_run
from flower_bridge import run_flower_simulation
import utils.tracking_db as tracking_db

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_checkpoint")


def run_simulation(num_rounds=5, epochs_per_round=1, fl_algorithm="fedprox"):
    print("=" * 60)
    print("STARTING 3-TIER FEDERATED LEARNING SIMULATION")
    print(f"Algorithm: {fl_algorithm.upper()} | Secure Agg: ON | Holdout Eval: ON")
    print("=" * 60)

    start_run("3tier-fl-cli", tags={"algorithm": fl_algorithm})
    log_params({"fl_algorithm": fl_algorithm, "rounds": num_rounds})

    global_server = GlobalServer(fl_algorithm=fl_algorithm)

    providers = {
        "S1": ProviderServer("S1"),
        "S2": ProviderServer("S2"),
        "S3": ProviderServer("S3")
    }

    devices = {}
    for prov_id, provider in providers.items():
        for client_idx in range(3):
            dev_id = f"{prov_id}-{client_idx}"
            device = EdgeDevice(dev_id)
            provider.add_edge_device(device)
            devices[dev_id] = device

    g_sms, g_call = global_server.get_global_weights()
    for dev in devices.values():
        dev.set_model_weights(g_sms, g_call)

    holdout = global_server.evaluate_holdout()
    print(f"Global holdout: {holdout['holdout_size']} samples (never used in client training)")

    initial_metrics = global_server.metrics_history[-1]
    print(f"Initial Metrics (Round 0) - SMS: {initial_metrics['sms_accuracy']*100:.2f}%, "
          f"Holdout SMS: {initial_metrics['holdout_sms_accuracy']*100:.2f}%")

    for round_idx in range(1, num_rounds + 1):
        print(f"\n--- FL Round {round_idx} / {num_rounds} ---")
        provider_updates = []
        global_sms, global_call = global_server.get_global_state_dicts()

        for prov_id, provider in providers.items():
            client_results = []
            for dev in provider.edge_devices:
                res = dev.train_local(
                    epochs=epochs_per_round,
                    lr=0.01,
                    fl_algorithm=fl_algorithm,
                    fedprox_mu=0.01,
                    global_sms_state=global_sms if fl_algorithm == "fedprox" else None,
                    global_call_state=global_call if fl_algorithm == "fedprox" else None,
                )
                client_results.append(res)

            p_sms, p_call = provider.aggregate_local_updates(client_results, round_id=round_idx)
            provider_updates.append({"sms_weights": p_sms, "call_weights": p_call})

        global_server.aggregate_provider_updates(provider_updates, round_id=round_idx)
        metrics = global_server.log_metrics(round_id=round_idx)
        print(f"  >> Round {round_idx} - SMS: {metrics['sms_accuracy']*100:.2f}%, "
              f"Holdout SMS: {metrics['holdout_sms_accuracy']*100:.2f}%")

        # ── Step 5: log round to MLflow + tracking_db ─────────────────────────────
        os.makedirs(MODELS_DIR, exist_ok=True)
        ckpt_sms  = os.path.join(MODELS_DIR, "global_sms_model.pth")
        ckpt_call = os.path.join(MODELS_DIR, "global_call_model.pth")
        torch.save(global_server.sms_model.state_dict(),  ckpt_sms)
        torch.save(global_server.call_model.state_dict(), ckpt_call)
        log_round(
            metrics=metrics,
            round_id=round_idx,
            checkpoint_path=ckpt_sms,    # SMS model is the primary fraud detector
            fl_algorithm=fl_algorithm,
        )
        # ───────────────────────────────────────────────────────────────

        # ── Step 4: escrow smoke-test ─────────────────────────────────────
        # Fire receive_message() on the highest-risk device (S1-0) once per
        # round so suspicious_messages and tier_escalations are populated,
        # the tracking_db → MLflow path can be verified (Step 5), and the
        # FL loop is confirmed to be intercepting real fraud signals.
        fraud_sms = (
            "URGENT: Your MTN MoMo wallet is suspended. "
            "Send your PIN and OTP to unblock your account now."
        )
        intercept = devices["S1-0"].receive_message(
            sender_phone="+233200000000",
            message_text=fraud_sms,
            amount=500.0,
            provider_id="S1",
            model_round_id=round_idx,
            is_ground_truth_spam=True,
        )
        verdict  = intercept["escrow"]["action"]
        risk_str = f"{intercept['total_risk_score']:.3f}"
        db_ok    = "[OK] persisted" if intercept["persisted"] else "[FAIL] db-write failed"
        esc_str  = "escalated" if intercept["escalated"] else "not escalated"
        print(f"  [Escrow] verdict={verdict} risk={risk_str} {db_ok} {esc_str}")
        # ───────────────────────────────────────────────────────────────

        g_sms, g_call = global_server.get_global_weights()
        for dev in devices.values():
            dev.set_model_weights(g_sms, g_call)

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETED SUCCESSFULLY")
    print(f"Checkpoints in: {MODELS_DIR}")
    print("=" * 60)
    end_run()


def run_server(host="127.0.0.1", port=8000):
    import uvicorn
    print(f"Starting ShieldFL API server at http://{host}:{port}")
    print("Login: admin / password  |  Device token: edge-node / edge-secret-2026")
    uvicorn.run("api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ShieldFL 3-Tier Federated Learning System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py                               # CLI simulation (plain)\n"
            "  python main.py --flower                      # Flower simulation (Krum+SecAgg)\n"
            "  python main.py --flower --rounds 10          # 10-round Flower simulation\n"
            "  python main.py --serve                       # Start FastAPI server\n"
        ),
    )
    parser.add_argument("--serve",   action="store_true", help="Start the FastAPI web server")
    parser.add_argument("--flower",  action="store_true", help="Run Flower simulation (requires: pip install flwr)")
    parser.add_argument("--host",    default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port",    type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--rounds",  type=int, default=5)
    parser.add_argument("--algorithm", choices=["fedavg", "fedprox", "fedopt"], default="fedprox")
    parser.add_argument("--providers",    type=int, default=3, help="Number of provider nodes (Flower mode)")
    parser.add_argument("--devices",      type=int, default=3, help="Edge devices per provider (Flower mode)")
    parser.add_argument("--epochs",       type=int, default=1, help="Local epochs per round")
    args = parser.parse_args()

    if args.serve:
        run_server(args.host, args.port)
    elif args.flower:
        run_flower_simulation(
            num_rounds=args.rounds,
            fl_algorithm=args.algorithm,
            epochs_per_round=args.epochs,
            num_providers=args.providers,
            devices_per_provider=args.devices,
        )
    else:
        run_simulation(num_rounds=args.rounds, fl_algorithm=args.algorithm)
