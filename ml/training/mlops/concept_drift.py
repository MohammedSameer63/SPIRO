#!/usr/bin/env python3
"""SPIRO ML — training/mlops/concept_drift.py
Run concept drift detection with Page-Hinkley + CUSUM.

Usage
-----
    python training/mlops/concept_drift.py \
        --model-id yolov11s \
        --baseline-npy baseline_confs.npy \
        --current-npy current_confs.npy
"""
import argparse, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

def main():
    from lib.ml.mlops.drift.concept_drift import ConceptDriftDetector
    from lib.ml.core.logger import get_logger
    log = get_logger("concept_drift")

    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--baseline-npy", required=True)
    p.add_argument("--current-npy", required=True)
    args = p.parse_args()

    baseline = np.load(args.baseline_npy)
    current  = np.load(args.current_npy)

    detector = ConceptDriftDetector()
    detector.set_baseline(baseline)
    events = detector.update_batch(current, model_id=args.model_id)
    if events:
        print(f"\n⚠ {len(events)} concept drift event(s) detected:")
        for e in events:
            print(f"  [{e.detector}] {e.details}")
    else:
        print("\nNo concept drift detected.")
    out = detector.export_report(args.model_id)
    print(f"Report: {out}")

if __name__ == "__main__":
    main()
