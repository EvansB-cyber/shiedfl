import sys
import os
import argparse
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from global_layer.global_server import GlobalServer
from provider_layer.provider_server import ProviderServer
from edge_layer.edge_device import EdgeDevice
from utils.ml_tracking import start_run, log_params, end_run


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

        g_sms, g_call = global_server.get_global_weights()
        for dev in devices.values():
            dev.set_model_weights(g_sms, g_call)

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETED SUCCESSFULLY")
    print("=" * 60)

    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_checkpoint")
    os.makedirs(models_dir, exist_ok=True)
    torch.save(global_server.sms_model.state_dict(), os.path.join(models_dir, "global_sms_model.pth"))
    torch.save(global_server.call_model.state_dict(), os.path.join(models_dir, "global_call_model.pth"))
    print(f"Saved global model checkpoints to: {models_dir}")
    end_run()


def run_server(host="127.0.0.1", port=8000):
    import uvicorn
    print(f"Starting ShieldFL API server at http://{host}:{port}")
    print("Login: admin / password  |  Device token: edge-node / edge-secret-2026")
    uvicorn.run("api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3-Tier Federated Learning System")
    parser.add_argument("--serve", action="store_true", help="Start the FastAPI web server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--algorithm", choices=["fedavg", "fedprox", "fedopt"], default="fedprox")
    args = parser.parse_args()

    if args.serve:
        run_server(args.host, args.port)
    else:
        run_simulation(num_rounds=args.rounds, fl_algorithm=args.algorithm)
