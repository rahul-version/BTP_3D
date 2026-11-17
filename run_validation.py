"""
One-click shortcut to run dedicated validation on Dataset/Validation/
and generate 'validation_checkpoints_plot.png' with gradient-descent style loss curve.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(PROJECT_ROOT, "Evaluation")
if EVAL_DIR not in sys.path:
    sys.path.append(EVAL_DIR)

from validate import run_validation, replot_validation_from_csv

if __name__ == "__main__":
    if "--replot" in sys.argv:
        replot_validation_from_csv()
    else:
        run_validation()
