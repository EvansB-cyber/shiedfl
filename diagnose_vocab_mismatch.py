import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("VOCAB SIZE DIAGNOSTIC")
print("=" * 60)

# 1. What does edge_layer/data.py actually define?
try:
    from edge_layer.data import VOCAB, VOCAB_SIZE
    print(f"\n[data.py] len(VOCAB)      = {len(VOCAB)}")
    print(f"[data.py] VOCAB_SIZE      = {VOCAB_SIZE}")
    if len(VOCAB) != VOCAB_SIZE:
        print("  !! MISMATCH: VOCAB_SIZE constant does not match actual len(VOCAB)")
except Exception as e:
    print(f"\n[data.py] FAILED TO IMPORT: {e}")

# 2. What does the exported vocab.json say?
vocab_json_path = os.path.join("edge_layer", "vocab.json")
if os.path.exists(vocab_json_path):
    with open(vocab_json_path, "r") as f:
        vj = json.load(f)
    print(f"\n[vocab.json] vocab_size field = {vj.get('vocab_size')}")
    print(f"[vocab.json] actual token count = {len(vj.get('tokens', {}))}")
    print(f"[vocab.json] vocab_version = {vj.get('vocab_version')}")
else:
    print(f"\n[vocab.json] NOT FOUND at {vocab_json_path}")

# 3. What does the actual checkpoint's embedding layer say?
try:
    import torch
    ckpt = torch.load("models_checkpoint/global_sms_model.pth", map_location="cpu")
    key = "embedding.weight"
    if key not in ckpt:
        candidates = [k for k in ckpt.keys() if "embed" in k.lower() and "weight" in k.lower()]
        key = candidates[0] if candidates else None
    if key:
        print(f"\n[checkpoint] {key} shape = {ckpt[key].shape}")
    else:
        print(f"\n[checkpoint] No embedding weight key found. Keys: {list(ckpt.keys())}")

    # Also show total file size and number of tensors, to sanity check the checkpoint isn't a stub
    total_params = sum(v.numel() for v in ckpt.values() if hasattr(v, "numel"))
    print(f"[checkpoint] total tensor entries = {len(ckpt)}")
    print(f"[checkpoint] total parameter count = {total_params}")
except Exception as e:
    print(f"\n[checkpoint] FAILED TO LOAD: {e}")

print("\n" + "=" * 60)
print("If these three numbers don't all agree, that's the bug.")
print("=" * 60)