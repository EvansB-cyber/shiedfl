"""
ML experiment tracking via MLflow (with optional Weights & Biases backend).
Falls back to structured console logging when neither is available.
"""
import os
import json
from datetime import datetime

_mlflow = None
_wandb = None
_active_run = None
_fallback_log = []


def _init_backends():
    global _mlflow, _wandb
    if _mlflow is None:
        try:
            import mlflow
            _mlflow = mlflow
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
            _mlflow.set_tracking_uri(tracking_uri)
        except ImportError:
            _mlflow = False
    if _wandb is None:
        try:
            import wandb
            _wandb = wandb
        except ImportError:
            _wandb = False


def start_run(experiment_name: str = "3tier-fl", tags: dict = None):
    global _active_run
    _init_backends()
    tags = tags or {}

    if _mlflow and _mlflow is not False:
        _mlflow.set_experiment(experiment_name)
        _active_run = _mlflow.start_run(run_name=f"round-{datetime.now():%Y%m%d-%H%M%S}")
        _mlflow.set_tags(tags)
        return "mlflow"

    if _wandb and _wandb is not False and os.environ.get("WANDB_API_KEY"):
        _wandb.init(project=experiment_name, tags=tags, reinit=True)
        _active_run = True
        return "wandb"

    _active_run = {"experiment": experiment_name, "tags": tags}
    return "local"


def log_metrics(metrics: dict, step: int = None):
    _init_backends()
    if _mlflow and _mlflow is not False and _active_run and hasattr(_active_run, "info"):
        _mlflow.log_metrics(metrics, step=step)
    elif _wandb and _wandb is not False and _active_run is True:
        _wandb.log(metrics, step=step)
    else:
        entry = {"step": step, **metrics, "ts": datetime.now().isoformat()}
        _fallback_log.append(entry)
        print(f"[ML Tracking] {json.dumps(entry)}")


def log_params(params: dict):
    _init_backends()
    if _mlflow and _mlflow is not False and _active_run and hasattr(_active_run, "info"):
        _mlflow.log_params(params)
    elif _wandb and _wandb is not False and _active_run is True:
        _wandb.config.update(params)
    else:
        print(f"[ML Tracking] params: {json.dumps(params)}")


def end_run():
    global _active_run
    _init_backends()
    if _mlflow and _mlflow is not False and _active_run and hasattr(_active_run, "info"):
        _mlflow.end_run()
    elif _wandb and _wandb is not False and _active_run is True:
        _wandb.finish()
    _active_run = None


def get_fallback_log():
    return list(_fallback_log)
