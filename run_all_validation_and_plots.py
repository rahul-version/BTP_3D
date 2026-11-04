"""
Master Connected Pipeline Script:
- Connects Training, Testing, Validation datasets
- Evaluates all models/checkpoints in model/
- Determines whether each checkpoint improved or degraded
- Connects LR, SR, HR, and error visualization
- Outputs all comparison graphs and summary tables directly into this parent folder
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(PROJECT_ROOT, "Evaluation")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "training")

if EVAL_DIR not in sys.path:
    sys.path.append(EVAL_DIR)
if TRAIN_DIR not in sys.path:
    sys.path.append(TRAIN_DIR)

from checkpoint_evaluator import run_checkpoint_evaluations, replot_from_csv
from evaluate import main as run_evaluate


def main():
    if "--replot" in sys.argv:
        replot_from_csv()
        return

    print("=" * 80)
    print("3D CRYO-EM SUPER-RESOLUTION: MASTER VALIDATION & GRAPH SUITE")
    print("All outputs, graphs, and reports will be saved in:")
    print(f"  {PROJECT_ROOT}")
    print("=" * 80)
    print()

    # Step 1: Run Checkpoint Validation & Progress Evaluation across all models
    print("[STEP 1/2] Evaluating all model checkpoints on Validation split...")
    run_checkpoint_evaluations(split="val")

    # Step 2: Check if test results exist in result/ folder and evaluate them
    result_dir = os.path.join(PROJECT_ROOT, "result")
    mrc_results = [f for f in os.listdir(result_dir) if f.lower().endswith(".mrc")] if os.path.exists(result_dir) else []

    if len(mrc_results) > 0:
        print("\n" + "=" * 80)
        print(f"[STEP 2/2] Evaluating {len(mrc_results)} generated SR files in result/ against Dataset/HR...")
        print("=" * 80)
        sys.argv = ["evaluate.py"]
        run_evaluate()
    else:
        print("\n[STEP 2/2] Note: No .mrc files in result/ folder yet.")
        print("To generate super-resolved files, run: python testing/test.py")

    print("\n" + "=" * 80)
    print("SUCCESS: Master Validation & Graph Suite Complete.")
    print("Generated graphs in this folder:")
    print("  - checkpoint_metric_comparison.png       (PSNR, SSIM, FSC across all checkpoints)")
    print("  - fsc_comparison_all_models.png          (Overlay of FSC curves for all checkpoints)")
    print("  - connected_pipeline_visualization.png   (LR -> SR -> HR -> Error comparison)")
    print("  - evaluation_metrics_summary.png         (Summary of test volume results if present)")
    print("  - training_validation_curves.png         (Real-time curves during training)")
    print("=" * 80)


if __name__ == "__main__":
    main()
