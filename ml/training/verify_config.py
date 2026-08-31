#!/usr/bin/env python3
"""
SPIRO ML — training/verify_config.py
Inspect and validate verification configs.

Usage
-----
    python training/verify_config.py show --config configs/verification/effnetv2_s.yaml
    python training/verify_config.py validate --config configs/verification/effnetv2_m.yaml
    python training/verify_config.py list
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args():
    p = argparse.ArgumentParser(description="Verification config tools")
    sub = p.add_subparsers(dest="command", required=True)
    show = sub.add_parser("show")
    show.add_argument("--config", required=True)
    val = sub.add_parser("validate")
    val.add_argument("--config", required=True)
    sub.add_parser("list")
    return p.parse_args()


def main():
    args = parse_args()
    from lib.ml.core.logger import get_logger
    from lib.ml.verification.verify_config import VerifyConfig
    log = get_logger("verify_config")

    if args.command == "show":
        cfg = VerifyConfig.load(args.config)
        print(cfg.to_yaml())

    elif args.command == "validate":
        try:
            cfg = VerifyConfig.load(args.config)
            log.info(f"Config is VALID ✓")
            log.info(f"  Variant:    {cfg.model.variant}")
            log.info(f"  timm_name:  {cfg.timm_name}")
            log.info(f"  input_size: {cfg.input_size}")
            log.info(f"  num_classes:{cfg.model.num_classes}")
            log.info(f"  loss:       {cfg.loss.name}")
            log.info(f"  fusion:     {cfg.fusion.method}")
        except Exception as e:
            log.error(f"Config INVALID: {e}")
            sys.exit(1)

    elif args.command == "list":
        cfg_dir = Path("configs/verification")
        if not cfg_dir.exists():
            log.error(f"configs/verification/ not found")
            return
        configs = sorted(cfg_dir.glob("*.yaml"))
        print(f"\nAvailable verification configs ({len(configs)}):")
        for c in configs:
            try:
                cfg = VerifyConfig.load(c)
                print(f"  {c.name:<35} → {cfg.model.variant} | {cfg.input_size}px")
            except Exception as e:
                print(f"  {c.name:<35} → ERROR: {e}")


if __name__ == "__main__":
    main()
