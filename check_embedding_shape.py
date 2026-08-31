import torch

checkpoint = torch.load("models_checkpoint/global_sms_model.pth", map_location="cpu")

# Try the common key name first; fall back to scanning all keys for an embedding weight
key = "embedding.weight"
if key not in checkpoint:
    candidates = [k for k in checkpoint.keys() if "embedding" in k.lower() and "weight" in k.lower()]
    if candidates:
        key = candidates[0]
        print(f"Note: 'embedding.weight' not found directly, using detected key: '{key}'")
    else:
        print("Could not find an embedding weight key. Available keys:")
        for k in checkpoint.keys():
            print(" -", k)
        raise SystemExit(1)

print("SMS embedding shape:", checkpoint[key].shape)