import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config

def dist(a, b):
    return np.linalg.norm(a - b)

def plot_2d_path():
    # 1. Setup paths
    top = config.psf_file
    # Assuming standard trajectory location
    traj_pattern = os.path.join(config.RESULTS_TRAJ_DIR, "*.dcd")
    traj_list = sorted(glob.glob(traj_pattern))
    
    if not traj_list:
        print(f"No trajectories found in {traj_pattern}")
        return

    print(f"Found {len(traj_list)} DCD files.")
    
    # 2. Load Universe
    try:
        u = mda.Universe(top, traj_list)
    except Exception as e:
        print(f"Error loading universe: {e}")
        # Fallback for testing/dev if paths are different
        print("Ensure 'step3_input.psf' and 'dcd_trajs' are accessible.")
        return

    # 3. Select Atoms using config indices
    # config indices are 0-based OpenMM indices, which match MDAnalysis 0-based indices
    try:
        atom1 = u.atoms[config.ATOM1_INDEX] # P
        atom2 = u.atoms[config.ATOM2_INDEX] # Mg
        atom3 = u.atoms[config.ATOM3_INDEX] # Gln
        atom4 = u.atoms[config.ATOM4_INDEX] # Asn
    except IndexError:
        print("Error: Atom indices in config.py out of range for this topology.")
        return

    print(f"CV1: {atom1.name}-{atom1.resname} vs {atom2.name}-{atom2.resname}")
    print(f"CV2: {atom3.name}-{atom3.resname} vs {atom4.name}-{atom4.resname}")

    # 4. Calculate CVs
    cv1_vals = []
    cv2_vals = []
    
    print("Computing trajectory path...")
    for ts in u.trajectory:
        d1 = dist(atom1.position, atom2.position)
        d2 = dist(atom3.position, atom4.position)
        cv1_vals.append(d1)
        cv2_vals.append(d2)
        
    cv1_vals = np.array(cv1_vals)
    cv2_vals = np.array(cv2_vals)

    # 5. Plot
    plot_dir = getattr(config, "PLOTS_DIR", "./plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)
    # time in ps
    unit_mod = None
    if hasattr(config, "unit"):
        unit_mod = config.unit
    elif hasattr(config, "u"):
        unit_mod = config.u
    if unit_mod is not None and hasattr(config, "stepsize"):
        report_interval = int(getattr(config, "DCD_REPORT_INTERVAL", getattr(config, "dcdfreq_mfpt", 1)))
        dt_ps = float(config.stepsize.value_in_unit(unit_mod.picoseconds)) * report_interval
        time_ps = np.arange(len(cv1_vals), dtype=np.float32) * dt_ps
    else:
        time_ps = np.arange(len(cv1_vals), dtype=np.float32)
        
    plt.figure(figsize=(8, 8))
    
    # Plot path with color mapping to time
    sc = plt.scatter(cv1_vals, cv2_vals, c=time_ps, cmap='viridis', s=10, alpha=0.6)
    cbar = plt.colorbar(sc)
    cbar.set_label('Time (ps)')
    
    # Add start/end markers
    plt.plot(cv1_vals[0], cv2_vals[0], 'r*', markersize=15, label='Start')
    plt.plot(cv1_vals[-1], cv2_vals[-1], 'kx', markersize=15, label='End')
    
    # Annotate Goals
    plt.axvline(x=config.FINAL_TARGET, color='r', linestyle='--', alpha=0.3, label='CV1 Target')
    plt.axhline(y=config.FINAL_TARGET_2, color='g', linestyle='--', alpha=0.3, label='CV2 Target')

    label_cv1 = getattr(config, "CV1_LABEL", "CV1 distance")
    label_cv2 = getattr(config, "CV2_LABEL", "CV2 distance")
    plt.xlabel(f"{label_cv1} (Å)")
    plt.ylabel(f"{label_cv2} (Å)")
    plt.title('2D Trajectory Path')
    plt.legend()
    plt.grid(True)
    
    out_path = os.path.join(plot_dir, "2d_trajectory_path.png")
    plt.savefig(out_path)
    print(f"Saved 2D path plot to {out_path}")
    plt.close()

if __name__ == "__main__":
    plot_2d_path()
