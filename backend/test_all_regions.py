"""
United Multi-Province Brutal OCR Verification Test Runner.
Executes all 13 Canadian issuing jurisdictions (1,326 test cases) with hardware acceleration.
Enforces strict 92.0% regional and 80.0% overall country pass thresholds.
Updates real-time image-by-image telemetry with comparative proof (Expected vs Extracted)
and produces consolidated markdown audit scorecard.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = SCRIPT_DIR
ROOT_DIR = os.path.dirname(BACKEND_DIR)
PROGRESS_FILE = os.path.join(BACKEND_DIR, "test_progress.json")
PROGRESS_TMP = os.path.join(BACKEND_DIR, "test_progress.tmp")

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from core.hardware_profile import apply_hardware_acceleration, print_hardware_banner, get_hardware_profile
from core.provincial_templates import PROVINCIAL_CONFIGS
from core.regional_test_runner import run_single_region_test, save_test_progress


def parse_args():
    parser = argparse.ArgumentParser(description="United Canadian Driving License Brutal OCR Test Runner")
    parser.add_argument(
        "--regions",
        type=str,
        default=None,
        help="Comma-separated list of province codes to test (e.g. 'ON,QC,BC,AB'). Defaults to all 13."
    )
    parser.add_argument(
        "--threshold-region",
        type=float,
        default=0.92,
        help="Regional passing threshold (default: 0.92 for 92.0%%)"
    )
    parser.add_argument(
        "--threshold-overall",
        type=float,
        default=0.80,
        help="Overall national passing threshold (default: 0.80 for 80.0%%)"
    )
    parser.add_argument(
        "--report",
        type=str,
        default=os.path.join(ROOT_DIR, "provincial_brutal_test_report.md"),
        help="Path to output markdown report file"
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Run in compact mode (single-line status per image without 3-line comparative proof tree)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress real-time per-image console logs (only print regional summaries)"
    )
    return parser.parse_args()


def run_united_suite(
    regions: Optional[List[str]] = None,
    threshold_region: float = 0.92,
    threshold_overall: float = 0.80,
    report_path: Optional[str] = None,
    verbose: bool = True,
    show_proof: bool = True
):
    # Apply hardware auto-tuning before any computation
    profile = apply_hardware_acceleration()
    print_hardware_banner(profile)

    target_provs = [p.strip().upper() for p in regions] if regions else sorted(PROVINCIAL_CONFIGS.keys())
    total_expected_cases = len(target_provs) * 102

    print("=" * 105, flush=True)
    print("      UNITED CANADIAN MULTI-PROVINCE BRUTAL OCR VERIFICATION TEST SUITE (13 REGIONS)      ", flush=True)
    print("=" * 105, flush=True)
    print(f"  * Target Jurisdictions     : {len(target_provs)} ({', '.join(target_provs)})", flush=True)
    print(f"  * Cases per Jurisdiction   : 102 brutal distortion test images", flush=True)
    print(f"  * Grand Total Test Cases   : {total_expected_cases} images", flush=True)
    print(f"  * Regional Pass Benchmark  : >= {threshold_region * 100:.1f}%", flush=True)
    print(f"  * National Pass Benchmark  : >= {threshold_overall * 100:.1f}%", flush=True)
    print(f"  * Comparative Proof Mode   : {'ENABLED (Expected vs Extracted)' if show_proof else 'COMPACT'}", flush=True)
    print(f"  * Real-Time Telemetry File : {PROGRESS_FILE}", flush=True)
    print("=" * 105 + "\n", flush=True)

    t_suite_start = time.time()
    provincial_summaries = []
    total_all_cases = 0
    total_all_passed = 0
    total_all_failed = 0

    passed_regions = []
    failed_regions = []

    for idx, prov in enumerate(target_provs):
        prov_name = PROVINCIAL_CONFIGS.get(prov, {}).get("name", prov)
        print(f"\n[{idx+1:2d}/{len(target_provs):2d}] Starting Jurisdiction: {prov} ({prov_name}) ...", flush=True)

        res = run_single_region_test(
            prov,
            threshold=threshold_region,
            verbose=verbose,
            show_proof=show_proof,
            global_offset=total_all_cases,
            global_total=total_expected_cases,
            print_banner=False
        )
        provincial_summaries.append(res)

        total_all_cases += res["total_cases"]
        total_all_passed += res["passed_count"]
        total_all_failed += res["failed_count"]

        verdict = res["verdict"]
        rate = res["pass_rate_pct"]
        if verdict == "PASSED":
            passed_regions.append((prov, prov_name, rate))
            print(f"  >>> [PASSED] {prov}: {res['passed_count']}/{res['total_cases']} passed ({rate:.1f}% >= {threshold_region*100:.1f}%)\n", flush=True)
        else:
            failed_regions.append((prov, prov_name, rate, res["failed_count"]))
            print(f"  >>> [FAILED] {prov}: {res['passed_count']}/{res['total_cases']} passed ({rate:.1f}% < {threshold_region*100:.1f}%)\n", flush=True)

    t_suite_total = time.time() - t_suite_start
    overall_pass_rate = (total_all_passed / total_all_cases) * 100.0 if total_all_cases > 0 else 0.0
    is_overall_passed = (overall_pass_rate / 100.0) >= threshold_overall

    # Print Master Summary Scorecard
    print("\n" + "=" * 105, flush=True)
    print("                           MASTER NATIONAL VERIFICATION SCORECARD                           ", flush=True)
    print("=" * 105, flush=True)
    print(f"{'Region':<6} | {'Jurisdiction Name':<26} | {'Total':<6} | {'Passed':<6} | {'Failed':<6} | {'Pass Rate':<10} | {'Field Acc':<10} | {'Status':<8}", flush=True)
    print("-" * 105, flush=True)

    for r in provincial_summaries:
        status_str = "PASS" if r["verdict"] == "PASSED" else "FAIL"
        f_acc = r.get("field_accuracy_pct", r["pass_rate_pct"])
        print(f"{r['province']:<6} | {r['province_name']:<26} | {r['total_cases']:<6} | {r['passed_count']:<6} | {r['failed_count']:<6} | {r['pass_rate_pct']:>6.2f}%    | {f_acc:>6.2f}%    | {status_str:<8}", flush=True)

    print("-" * 105, flush=True)
    print(f"{'TOTAL':<6} | {'ALL 13 CANADIAN JURISDICTIONS':<26} | {total_all_cases:<6} | {total_all_passed:<6} | {total_all_failed:<6} | {overall_pass_rate:>6.2f}%    | {'-':>6}       | {'PASS' if is_overall_passed else 'FAIL':<8}", flush=True)
    print("=" * 105, flush=True)

    # Regional Benchmark Breakdown
    print(f"\n[REGIONAL AUDIT - Benchmark: >= {threshold_region*100:.1f}%]:")
    print(f"  * Passed Jurisdictions ({len(passed_regions)}/{len(target_provs)}):")
    for p_code, p_name, p_rate in passed_regions:
        print(f"      [+] {p_code:2s} ({p_name}): {p_rate:.2f}%")

    if failed_regions:
        print(f"\n  * FAILED Jurisdictions ({len(failed_regions)}/{len(target_provs)} - Did not meet {threshold_region*100:.1f}%):")
        for p_code, p_name, p_rate, p_fails in failed_regions:
            print(f"      [!] {p_code:2s} ({p_name}): {p_rate:.2f}% ({p_fails} failures)")
    else:
        print(f"\n  * FAILED Jurisdictions: NONE! (All {len(target_provs)} jurisdictions met the {threshold_region*100:.1f}% threshold!)")

    # National Benchmark Breakdown
    print(f"\n[NATIONAL AUDIT - Benchmark: >= {threshold_overall*100:.1f}%]:")
    print(f"  * Grand Total Images Tested : {total_all_cases}")
    print(f"  * Total Successful Passes   : {total_all_passed}")
    print(f"  * Overall National Pass Rate: {overall_pass_rate:.2f}% (Benchmark: {threshold_overall*100:.1f}%)")
    print(f"  * National Suite Verdict    : {'>>> OVERALL PASSED <<<' if is_overall_passed else '>>> OVERALL FAILED <<<'}")
    print(f"  * Average Test Speed        : {total_all_cases / max(0.001, t_suite_total):.1f} img/s")
    print(f"  * Total Test Duration       : {t_suite_total:.2f}s (~{t_suite_total/60:.1f} mins)")
    print("=" * 105 + "\n", flush=True)

    # Save Consolidated JSON with Proof Highlights
    master_results = {
        "timestamp": datetime.now().isoformat(),
        "total_cases": total_all_cases,
        "total_passed": total_all_passed,
        "total_failed": total_all_failed,
        "overall_pass_rate_pct": round(overall_pass_rate, 2),
        "threshold_overall_pct": round(threshold_overall * 100.0, 1),
        "threshold_region_pct": round(threshold_region * 100.0, 1),
        "national_verdict": "PASSED" if is_overall_passed else "FAILED",
        "passed_regions": [p[0] for p in passed_regions],
        "failed_regions": [p[0] for p in failed_regions],
        "regional_summaries": provincial_summaries,
        "hardware_profile": profile
    }

    results_json_path = os.path.join(BACKEND_DIR, "test_results_all.json")
    try:
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(master_results, f, indent=2)
        print(f"Consolidated results saved to: {results_json_path}")
    except Exception:
        pass

    # Save final progress state
    save_test_progress({
        "status": "ALL_REGIONS_COMPLETED",
        "total_cases": total_all_cases,
        "total_passed": total_all_passed,
        "overall_pass_rate_pct": round(overall_pass_rate, 2),
        "national_verdict": master_results["national_verdict"],
        "passed_regions_count": len(passed_regions),
        "failed_regions_count": len(failed_regions),
        "total_duration_seconds": round(t_suite_total, 2),
        "timestamp": datetime.now().isoformat()
    })

    # Generate Markdown Report if requested
    if report_path:
        generate_markdown_report(report_path, master_results, passed_regions, failed_regions)
        print(f"Audit report generated at: {report_path}\n")

    return is_overall_passed and (len(failed_regions) == 0)


def generate_markdown_report(report_path: str, data: Dict[str, Any], passed_regs: list, failed_regs: list):
    hw = data.get("hardware_profile", {})
    cpu_str = hw.get("cpu_name", "Intel Core i7-1255U")
    gpu_str = hw.get("gpu_name", "Intel Iris Xe")
    workers_str = hw.get("active_worker_threads", 8)

    lines = [
        "# Canadian Multi-Province Brutal Driving Licence OCR Audit Report",
        "",
        f"**Date:** {data['timestamp']}  ",
        f"**Hardware Platform:** {cpu_str} ({workers_str} Worker Threads) + {gpu_str}  ",
        f"**Total Test Cases:** {data['total_cases']} across {len(data['regional_summaries'])} issuing jurisdictions  ",
        f"**Overall Pass Rate:** {data['overall_pass_rate_pct']}% (Benchmark: {data['threshold_overall_pct']}%)  ",
        f"**National Verdict:** **{data['national_verdict']}**  ",
        "",
        "---",
        "",
        "## Master Scorecard by Jurisdiction",
        "",
        "| Region | Jurisdiction | Cases | Passed | Failed | Pass Rate | Field Accuracy | Regional Threshold | Status |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in data["regional_summaries"]:
        status = "PASSED" if r["verdict"] == "PASSED" else "**FAILED**"
        f_acc = r.get("field_accuracy_pct", r["pass_rate_pct"])
        lines.append(
            f"| `{r['province']}` | {r['province_name']} | {r['total_cases']} | {r['passed_count']} | {r['failed_count']} | {r['pass_rate_pct']:.2f}% | {f_acc:.2f}% | {data['threshold_region_pct']}% | {status} |"
        )

    lines.extend([
        f"| **TOTAL** | **All Jurisdictions** | **{data['total_cases']}** | **{data['total_passed']}** | **{data['total_failed']}** | **{data['overall_pass_rate_pct']:.2f}%** | - | **{data['threshold_overall_pct']}%** | **{data['national_verdict']}** |",
        "",
        "---",
        "",
        "## Regional Breakdown",
        "",
        f"### Passed Jurisdictions ({len(passed_regs)})",
    ])
    for p_code, p_name, p_rate in passed_regs:
        lines.append(f"- **{p_code}** ({p_name}): **{p_rate:.2f}%**")

    lines.append("")
    lines.append(f"### Failed Jurisdictions ({len(failed_regs)})")
    if failed_regs:
        for p_code, p_name, p_rate, p_fails in failed_regs:
            lines.append(f"- **{p_code}** ({p_name}): **{p_rate:.2f}%** ({p_fails} failures)")
    else:
        lines.append("- *None! All regions met or exceeded the 92.0% threshold.*")

    lines.extend([
        "",
        "---",
        "",
        "## Sample Comparative Verification Proofs (Expected vs. Extracted)",
        "",
        "| Region | Case | Category | Expected DL | Extracted DL | Expected Name | Extracted Name | Verdict |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |",
    ])

    for r in data.get("regional_summaries", []):
        prov = r.get("province", "")
        proofs = r.get("case_proofs", [])
        for cp in proofs[:3]:  # Top 3 sample proofs per region
            p_data = cp.get("proof", {})
            exp = p_data.get("expected", {})
            ext = p_data.get("extracted", {})
            v = "PASS" if cp.get("passed") else "FAIL"
            lines.append(
                f"| `{prov}` | `{cp.get('filename')}` | {cp.get('category')} | `{exp.get('dl_number')}` | `{ext.get('dl_number')}` | `{exp.get('name')}` | `{ext.get('name')}` | {v} |"
            )

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception:
        pass


if __name__ == "__main__":
    args = parse_args()
    selected_regions = [r.strip() for r in args.regions.split(",")] if args.regions else None
    success = run_united_suite(
        regions=selected_regions,
        threshold_region=args.threshold_region,
        threshold_overall=args.threshold_overall,
        report_path=args.report,
        verbose=not args.quiet,
        show_proof=not args.compact
    )
    sys.exit(0 if success else 1)
