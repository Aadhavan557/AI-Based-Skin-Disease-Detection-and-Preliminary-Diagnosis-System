"""
run_all.py
-----------
Runs the entire preprocessing pipeline sequentially.
Each step is logged with timing information.

Usage:
    python preprocessing/run_all.py
    python preprocessing/run_all.py --steps 1 2 3   # run specific steps
    python preprocessing/run_all.py --skip 3         # skip step 3

Steps:
    1  → Dataset exploration
    2  → Remove corrupted images
    3  → Remove duplicates
    4  → Resize images
    5  → Compute normalization stats
    6  → Data augmentation
    7  → Split dataset
    8  → Visualize dataset
    9  → Create TF dataset
    10 → Generate HTML report
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

PREPROCESSING_DIR = Path(__file__).resolve().parent

STEPS = {
    1:  ("01_dataset_exploration.py",      "Dataset Exploration"),
    2:  ("02_remove_corrupted_images.py",  "Remove Corrupted Images"),
    3:  ("03_remove_duplicates.py",        "Remove Duplicates"),
    4:  ("04_resize_images.py",            "Resize Images"),
    5:  ("05_normalize_images.py",         "Compute Normalization Stats"),
    6:  ("06_data_augmentation.py",        "Data Augmentation"),
    7:  ("07_split_dataset.py",            "Split Dataset"),
    8:  ("08_visualize_dataset.py",        "Visualize Dataset"),
    9:  ("09_create_tf_dataset.py",        "Create TF Dataset"),
    10: ("10_generate_report.py",          "Generate HTML Report"),
}


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def run_step(step_num: int, script: str, name: str) -> bool:
    script_path = PREPROCESSING_DIR / script
    print(f"\n{'─'*62}")
    print(f"  STEP {step_num:02d} / 10 — {name}")
    print(f"  Script: {script}")
    print(f"{'─'*62}")

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PREPROCESSING_DIR),
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  ✓ Step {step_num} completed in {fmt_time(elapsed)}")
        return True
    else:
        print(f"\n  ✗ Step {step_num} FAILED (exit code {result.returncode}) after {fmt_time(elapsed)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run the skin disease preprocessing pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--steps", nargs="+", type=int, metavar="N",
        help="Only run these step numbers (e.g., --steps 1 2 4)"
    )
    parser.add_argument(
        "--skip", nargs="+", type=int, metavar="N",
        help="Skip these step numbers (e.g., --skip 3 6)"
    )
    parser.add_argument(
        "--stop-on-error", action="store_true",
        help="Stop pipeline on first step failure (default: continue)"
    )
    args = parser.parse_args()

    # Determine which steps to run
    selected = set(args.steps) if args.steps else set(STEPS.keys())
    skipped  = set(args.skip) if args.skip else set()
    to_run   = sorted(selected - skipped)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   AI-Based Skin Disease Detection                        ║")
    print("║   Preprocessing Pipeline                                 ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                      ║")
    print(f"║  Steps     : {', '.join(map(str, to_run)):<48}║")
    print("╚══════════════════════════════════════════════════════════╝")

    pipeline_start = time.time()
    results = {}

    for step_num in to_run:
        if step_num not in STEPS:
            print(f"  [WARN] Step {step_num} not defined — skipping.")
            continue
        script, name = STEPS[step_num]
        ok = run_step(step_num, script, name)
        results[step_num] = ok
        if not ok and args.stop_on_error:
            print("\n  Pipeline stopped due to error (--stop-on-error set).")
            break

    # ── Final summary ──────────────────────────────────────
    total_time = time.time() - pipeline_start
    print(f"\n{'═'*62}")
    print("  PIPELINE SUMMARY")
    print(f"{'═'*62}")
    for step_num, ok in results.items():
        _, name = STEPS[step_num]
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  Step {step_num:02d} — {name:<38} {status}")

    total_pass = sum(results.values())
    total_fail = len(results) - total_pass
    print(f"{'─'*62}")
    print(f"  Passed: {total_pass}  Failed: {total_fail}  Total time: {fmt_time(total_time)}")
    print(f"{'═'*62}\n")

    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
