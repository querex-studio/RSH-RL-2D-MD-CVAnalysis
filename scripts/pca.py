"""
Wrapper entrypoint for PCA on PPO trajectories.

Usage:
  python scripts/pca.py --max-traj 50 --write-structures
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from analysis.pca_md import main


if __name__ == "__main__":
    main()
