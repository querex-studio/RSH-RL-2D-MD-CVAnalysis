"""
Wrapper entrypoint for post-processing PPO trajectories.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from analysis.post_process import main


if __name__ == "__main__":
    main()
