# edge/sync.py
def compute_delta(new_state: dict, old_state: dict) -> dict:
    return {k: new_state[k] - old_state[k] for k in new_state}

def prepare_for_sync(model, prev_state_dict, dp_noise_fn=None):
    delta = compute_delta(model.state_dict(), prev_state_dict)
    if dp_noise_fn:
        delta = dp_noise_fn(delta)
    return delta