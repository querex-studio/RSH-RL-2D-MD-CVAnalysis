import json
import os
import time

try:
    import config
except Exception:
    config = None


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))


def default_time_tag():
    return time.strftime("%Y%m%d-%H%M%S")


def _resolve_runs_root(root=None):
    if root:
        base = root
    else:
        base = getattr(config, "RUNS_DIR", "analysis_runs") if config else "analysis_runs"
    if not os.path.isabs(base):
        base = os.path.join(ROOT_DIR, base)
    return base


def prepare_run_dir(time_tag, root=None):
    runs_root = _resolve_runs_root(root)
    run_dir = os.path.join(runs_root, time_tag)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _snapshot_config():
    if config is None:
        return {}
    fields = [
        "ATOM1_INDEX", "ATOM2_INDEX", "ATOM3_INDEX", "ATOM4_INDEX",
        "CURRENT_DISTANCE", "FINAL_TARGET", "CURRENT_DISTANCE_2", "FINAL_TARGET_2",
        "TARGET_MIN", "TARGET_MAX", "TARGET2_MIN", "TARGET2_MAX",
        "stepsize", "dcdfreq_mfpt", "sim_steps_mfpt",
        "RESULTS_DIR", "RESULTS_TRAJ_DIR",
    ]
    snap = {}
    unit_mod = None
    if hasattr(config, "unit"):
        unit_mod = config.unit
    elif hasattr(config, "u"):
        unit_mod = config.u
    for f in fields:
        if hasattr(config, f):
            val = getattr(config, f)
            try:
                if hasattr(val, "value_in_unit") and unit_mod is not None:
                    val = float(val.value_in_unit(unit_mod.picoseconds))
            except Exception:
                pass
            snap[f] = val
    return snap


def write_run_metadata(run_dir, meta):
    payload = dict(meta or {})
    payload.setdefault("time_tag", os.path.basename(run_dir))
    payload.setdefault("config", _snapshot_config())
    out_path = os.path.join(run_dir, "run.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def cleanup_empty_dirs(root_dir):
    removed = 0
    for root, dirs, files in os.walk(root_dir, topdown=False):
        if files:
            continue
        if any(os.path.isdir(os.path.join(root, d)) for d in dirs):
            continue
        if root == root_dir:
            continue
        try:
            os.rmdir(root)
            removed += 1
        except OSError:
            pass
    return removed
