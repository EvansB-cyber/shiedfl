"""
ML experiment tracking via MLflow (with optional Weights & Biases backend).
Falls back to structured console logging when neither is available.

v2 additions (Step 5):
  log_artifact(path)   — logs a file/directory to the active run
  log_round(metrics, round_id, checkpoint_path) — single call per FL round
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


def log_artifact(local_path: str):
    """
    Log a local file or directory as a run artifact.
    On MLflow: uploads to the artifact store.
    On W&B:    saves as a file artifact.
    On local:  prints the path (no upload possible without a backend).
    """
    _init_backends()
    if not os.path.exists(local_path):
        print(f"[ML Tracking] log_artifact: path not found: {local_path}")
        return
    if _mlflow and _mlflow is not False and _active_run and hasattr(_active_run, "info"):
        if os.path.isdir(local_path):
            _mlflow.log_artifacts(local_path)
        else:
            _mlflow.log_artifact(local_path)
    elif _wandb and _wandb is not False and _active_run is True:
        artifact = _wandb.Artifact(name=os.path.basename(local_path), type="model")
        if os.path.isdir(local_path):
            artifact.add_dir(local_path)
        else:
            artifact.add_file(local_path)
        _wandb.log_artifact(artifact)
    else:
        print(f"[ML Tracking] artifact: {local_path}")


def log_round(
    metrics: dict,
    round_id: int,
    checkpoint_path: str = None,
    fl_algorithm: str = None,
    dp_enabled: bool = False,
    dp_noise: float = None,
):
    """
    Convenience wrapper called once per FL round. Does three things:

      1. log_metrics(metrics, step=round_id)  — round-stamped metrics to
         MLflow / W&B / local log so charts plot against round number.

      2. tracking_db.log_model_version(...)   — writes one row to
         global_model_versions so every suspicious_messages detection
         can be traced back to the exact round that trained it.

      3. log_artifact(checkpoint_path)        — uploads the .pth file to
         the MLflow artifact store (skipped when checkpoint_path is None).

    This is the single entry point main.py calls after aggregate_provider_updates().
    """
    # ── 1. Metrics to MLflow / W&B / local ──────────────────────────────────
    log_metrics(metrics, step=round_id)

    # ── 2. tracking_db row (imports lazily to avoid circular imports) ────────
    try:
        import sys
        import os as _os
        sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        import utils.tracking_db as _tracking_db
        _tracking_db.log_model_version(
            round_id=round_id,
            fl_algorithm=fl_algorithm or "unknown",
            metrics=metrics,
            checkpoint_path=checkpoint_path,
            dp_enabled=dp_enabled,
            dp_noise=dp_noise,
        )
    except Exception as db_err:
        print(f"[ML Tracking] tracking_db.log_model_version failed: {db_err}")

    # ── 3. Checkpoint artifact ───────────────────────────────────────────────
    if checkpoint_path:
        log_artifact(checkpoint_path)
