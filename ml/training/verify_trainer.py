#!/usr/bin/env python3
"""
SPIRO ML — training/verify_trainer.py
Convenience wrapper for VerifyTrainer.

Usage
-----
    python training/verify_trainer.py --variant s
    python training/verify_trainer.py --variant m --device 0 --batch 16
    python training/verify_trainer.py --config configs/verification/effnetv2_s.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Delegate to train_effnet.py main()
from training.train_effnet import main  # type: ignore

if __name__ == "__main__":
    main()
