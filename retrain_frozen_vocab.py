"""
retrain_frozen_vocab.py — Checkbox 5: one-time cost of freezing VOCAB_SIZE.

Run this ONCE after the vocab freeze to replace the old checkpoints (which
used embedding size 1000) with new ones (embedding size = len(VOCAB) = 110).

Usage:
    python retrain_frozen_vocab.py [--rounds N]

The script:
  1. Prints a vocab audit so you can confirm the freeze is correct.
  2. Runs a compact FL simulation (3 providers × 3 clients, default 5 rounds).
  3. Saves new checkpoints to models_checkpoint/.
  4. Writes a models_checkpoint/vocab_lock.json recording the vocab version
     and checksum that the new weights were trained against — any future
     vocab change that doesn't bump VOCAB_VERSION will be caught by CI.

After this script succeeds you can delete any *.pth files that pre-date it.
"""

import sys
import os
import json
import argparse
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from edge_layer.data import (
    VOCAB, VOCAB_MAP, VOCAB_SIZE, VOCAB_VERSION,
    get_vocab_info, export_vocab_json,
    get_dataloaders, generate_global_test_data,
)
from edge_layer.models import SMSFraudCNN, CallDetectionMLP
from edge_layer.edge_device import EdgeDevice
from provider_layer.provider_server import ProviderServer
from global_layer.global_server import GlobalServer
from utils.crypto import encrypt_weights


def vocab_audit():
    print("=" * 60)
    print("VOCAB AUDIT — confirming freeze is correct")
    print("=" * 60)
    info = get_vocab_info()
    print(f"  vocab_version : {info['vocab_version']}")
    print(f"  vocab_size    : {info['vocab_size']}  (embedding rows after freeze)")
    print(f"  pad_id        : {info['pad_id']}  (<PAD>)")
    print(f"  unk_id        : {info['unk_id']}  (<UNK>)")
    print(f"  export_path   : {info['export_path']}")
    print(f"  checksum_path : {info['checksum_path']}")

    # Spot-check MoMo tokens
    required = ["mtn", "telecel", "airteltigo", "momo", "pin", "otp",
                "verify", "urgent", "suspend", "blocked"]
    missing = [t for t in required if t not in VOCAB_MAP]
    if missing:
        print(f"\n  [WARN] Missing expected tokens: {missing}")
    else:
        print(f"\n  [OK]   All {len(required)} spot-check tokens present.")

    print()
    return info


def run_retrain(num_rounds: int = 5):
    info = vocab_audit()

    print("=" * 60)
    print(f"RETRAINING against frozen vocab v{VOCAB_VERSION} ({VOCAB_SIZE} tokens)")
    print("=" * 60)

    global_server = GlobalServer(fl_algorithm="fedprox")

    providers = {pid: ProviderServer(pid) for pid in ("S1", "S2", "S3")}
    devices = {}
    for prov_id, provider in providers.items():
        for idx in range(3):
            dev_id = f"{prov_id}-{idx}"
            dev = EdgeDevice(dev_id)
            provider.add_edge_device(dev)
            devices[dev_id] = dev

    g_sms, g_call = global_server.get_global_weights()
    for dev in devices.values():
        dev.set_model_weights(g_sms, g_call)

    for round_idx in range(1, num_rounds + 1):
        print(f"  Round {round_idx}/{num_rounds} ...", end=" ", flush=True)
        provider_updates = []
        global_sms, global_call = global_server.get_global_state_dicts()

        for prov_id, provider in providers.items():
            client_results = []
            for dev in provider.edge_devices:
                res = dev.train_local(
                    epochs=1, lr=0.01,
                    fl_algorithm="fedprox", fedprox_mu=0.01,
                    global_sms_state=global_sms,
                    global_call_state=global_call,
                )
                client_results.append(res)
            p_sms, p_call = provider.aggregate_local_updates(client_results, round_id=round_idx)
            provider_updates.append({"sms_weights": p_sms, "call_weights": p_call})

        global_server.aggregate_provider_updates(provider_updates, round_id=round_idx)
        metrics = global_server.log_metrics(round_id=round_idx)
        print(f"SMS={metrics['sms_accuracy']*100:.1f}%  Call={metrics['call_accuracy']*100:.1f}%")

        g_sms, g_call = global_server.get_global_weights()
        for dev in devices.values():
            dev.set_model_weights(g_sms, g_call)

    # ── Save new checkpoints ─────────────────────────────────────────────────
    models_dir = os.path.join(ROOT, "models_checkpoint")
    os.makedirs(models_dir, exist_ok=True)

    sms_path  = os.path.join(models_dir, "global_sms_model.pth")
    call_path = os.path.join(models_dir, "global_call_model.pth")
    torch.save(global_server.sms_model.state_dict(), sms_path)
    torch.save(global_server.call_model.state_dict(), call_path)
    print(f"\n  Saved: {sms_path}")
    print(f"  Saved: {call_path}")

    # ── Write vocab lock so future CI can detect drift ───────────────────────
    sha_path = info["checksum_path"]
    checksum = open(sha_path).read().strip() if os.path.exists(sha_path) else "unknown"

    lock = {
        "vocab_version"  : VOCAB_VERSION,
        "vocab_size"     : VOCAB_SIZE,
        "vocab_checksum" : checksum,
        "trained_rounds" : num_rounds,
        "sms_checkpoint" : sms_path,
        "call_checkpoint": call_path,
        "final_metrics"  : global_server.metrics_history[-1],
    }
    lock_path = os.path.join(models_dir, "vocab_lock.json")
    with open(lock_path, "w") as f:
        json.dump(lock, f, indent=2)
    print(f"  Lock : {lock_path}")

    print("\n" + "=" * 60)
    print("RETRAIN COMPLETE — old checkpoints can now be deleted.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain after vocab freeze")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of FL rounds (default: 5)")
    args = parser.parse_args()
    run_retrain(num_rounds=args.rounds)
