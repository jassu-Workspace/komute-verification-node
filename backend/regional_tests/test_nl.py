#!/usr/bin/env python
"""
Standalone Brutal OCR Verification Test Runner for Newfoundland and Labrador (NL).
Executes the 102 brutal test cases against ground truth with 92% pass threshold.
Displays real-time comparative proof: Expected Ground Truth vs. Extracted OCR.
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from core.regional_test_runner import run_single_region_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Newfoundland and Labrador (NL) Brutal DL OCR Verification Test")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Run in compact mode (single-line status per image without 3-line comparative proof tree)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.92,
        help="Regional passing threshold (default: 0.92 for 92.0%%)"
    )
    args = parser.parse_args()

    result = run_single_region_test(
        "NL",
        threshold=args.threshold,
        verbose=True,
        show_proof=not args.compact
    )
    sys.exit(0 if result["verdict"] == "PASSED" else 1)
