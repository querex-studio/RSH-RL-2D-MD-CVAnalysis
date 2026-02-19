import os


def remove_empty_dirs(root_dir):
    removed = 0
    for root, dirs, files in os.walk(root_dir, topdown=False):
        # Skip if any files exist
        if files:
            continue
        # Skip if any non-empty subdirs remain
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


if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
    runs_root = os.path.join(ROOT_DIR, "results_PPO", "analysis_runs")
    if not os.path.isdir(runs_root):
        print(f"No analysis_runs directory at: {runs_root}")
    else:
        count = remove_empty_dirs(runs_root)
        print(f"Removed {count} empty directories under {runs_root}")
