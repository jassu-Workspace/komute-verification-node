"""
Canadian Multi-Province 102 Brutal Driving License Test Runner.
Executes 1,326 test cases across all 13 Canadian provinces and territories.
Features:
  - In-memory on-the-fly image synthesis (zero disk bottleneck).
  - Real-time live status reporting on EVERY test case (Count, Remaining, Elapsed, Speed, ETA).
  - Live progress JSON file persistence (backend/test_progress.json) for instant external polling.
  - Per-region (92%) and overall (80%) threshold evaluations.
  - Markdown audit scorecard generation.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

# Clamped threading for predictable performance & stability
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["ORT_INTRA_OP_NUM_THREADS"] = "2"
os.environ["ORT_INTER_OP_NUM_THREADS"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import cv2
import psutil

cv2.setNumThreads(2)

try:
    from termcolor import colored
except ImportError:
    def colored(text, color=None, on_color=None, attrs=None):
        return text

from app.schemas import LicenseDetails, PersonalInfo
from core.dl_ocr import dl_ocr_engine
from core.provincial_templates import PROVINCIAL_CONFIGS
from generate_brutal_provincial_tests import (
    apply_blur,
    apply_glare,
    apply_low_contrast,
    apply_noise,
    apply_perspective,
    apply_rotation,
    build_102_distortion_specs,
    generate_base_card,
)

process = psutil.Process(os.getpid())
PROGRESS_JSON_PATH = os.path.join(SCRIPT_DIR, "test_progress.json")
REPORT_PATH = os.path.join(ROOT_DIR, "provincial_brutal_test_report.md")
BRAIN_REPORT_PATH = os.path.join(r"C:\Users\jaswa\.gemini\antigravity-cli\brain\80eae86a-5acf-4ebc-9622-918b4d9327e8", "provincial_brutal_test_report.md")


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def save_progress_file(data: Dict[str, Any]):
    try:
        with open(PROGRESS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def run_all_brutal_tests(provinces_to_run: List[str] = None):
    specs = build_102_distortion_specs()
    target_provs = provinces_to_run or sorted(PROVINCIAL_CONFIGS.keys())
    total_cases = len(target_provs) * len(specs)

    print("\n" + "=" * 105, flush=True)
    print("      CANADIAN MULTI-PROVINCE 102 BRUTAL TEST SUITE (13 JURISDICTIONS - 1,326 CASES)      ", flush=True)
    print("=" * 105, flush=True)
    print(f"  • Total Jurisdictions to Test : {len(target_provs)} ({', '.join(target_provs)})", flush=True)
    print(f"  • Cases per Jurisdiction      : {len(specs)} (10 Clean, 20 Blur, 20 Noise, 20 Skew, 10 Glare, 10 Warp, 12 Brutal)", flush=True)
    print(f"  • Grand Total Test Cases      : {total_cases}", flush=True)
    print(f"  • Regional Pass Benchmark     : >= 92.0%", flush=True)
    print(f"  • Overall Country Benchmark   : >= 80.0%", flush=True)
    print("=" * 105 + "\n", flush=True)

    completed_count = 0
    total_passed = 0
    t_start = time.time()
    provincial_results = []
    latencies = []

    progress_state = {
        "status": "RUNNING",
        "started_at": datetime.now().isoformat(),
        "total_cases": total_cases,
        "completed": 0,
        "remaining": total_cases,
        "passed": 0,
        "failed": 0,
        "current_pass_rate": 0.0,
        "elapsed_seconds": 0.0,
        "elapsed_human": "0s",
        "eta_seconds": 0.0,
        "eta_human": "calculating...",
        "speed_sec_per_test": 0.0,
        "current_province": target_provs[0],
        "current_case_idx": 0,
        "last_case_summary": "Starting suite...",
        "provincial_summary": {}
    }
    save_progress_file(progress_state)

    for prov_idx, prov in enumerate(target_provs):
        cfg = PROVINCIAL_CONFIGS[prov]
        prov_name = cfg["name"]
        profiles = cfg["profiles"]
        num_profiles = len(profiles)

        print(f"\n[INIT] Pre-rendering base card textures for {prov} ({prov_name})...", flush=True)
        base_cards = [generate_base_card(prov, profile_idx=p_idx)[0] for p_idx in range(num_profiles)]
        print(f"[READY] Base cards initialized. Launching 102 test cases for {prov}...\n", flush=True)

        prov_passed = 0
        prov_failures = []
        prov_cat_stats = {}
        prov_latencies = []

        for c_idx, spec in enumerate(specs):
            cat = spec["category"]
            if cat not in prov_cat_stats:
                prov_cat_stats[cat] = {"total": 0, "passed": 0}
            prov_cat_stats[cat]["total"] += 1

            # 1. Synthesize distorted image directly in memory
            base_img = base_cards[spec["profile_idx"] % num_profiles]
            img = base_img.copy()

            for op, val in spec["ops"]:
                if op == "blur": img = apply_blur(img, val)
                elif op == "noise": img = apply_noise(img, val)
                elif op == "rotate": img = apply_rotation(img, val)
                elif op == "glare": img = apply_glare(img)
                elif op == "contrast": img = apply_low_contrast(img)
                elif op == "perspective": img = apply_perspective(img)

            # 2. Build driver credentials
            profile = profiles[spec["profile_idx"] % num_profiles]
            name, dl, dob, exp = profile

            personal = PersonalInfo(
                full_name=name,
                date_of_birth=dob,
                email_address="driver@komute.com",
                mobile_number="+10000000000",
                role="Driver"
            )
            license_d = LicenseDetails(
                license_number=dl,
                issuing_province=prov,
                license_expiry_date=exp
            )

            # 3. Execute OCR verification
            t_case_start = time.time()
            res = dl_ocr_engine.verify_license(img, personal, license_d)
            dt = (time.time() - t_case_start) * 1000.0

            completed_count += 1
            latencies.append(dt)
            prov_latencies.append(dt)

            is_pass = res.passed
            if is_pass:
                prov_passed += 1
                total_passed += 1
                prov_cat_stats[cat]["passed"] += 1
                verdict_str = colored("PASS", "green")
            else:
                verdict_str = colored("FAIL", "red")
                prov_failures.append({
                    "case": c_idx + 1,
                    "desc": spec["desc"],
                    "category": cat,
                    "details": res.details
                })

            # Calculate real-time ETA & velocity
            elapsed = time.time() - t_start
            remaining = total_cases - completed_count
            avg_sec_per_test = elapsed / completed_count if completed_count > 0 else 0
            eta_sec = remaining * avg_sec_per_test
            pct_done = (completed_count / total_cases) * 100.0
            overall_rate = (total_passed / completed_count) * 100.0

            # Progress bar visualization (20 chars)
            bar_fill = int(pct_done / 5)
            bar_str = "█" * bar_fill + "░" * (20 - bar_fill)

            case_summary = f"[{bar_str}] {completed_count:4d}/{total_cases} ({pct_done:5.1f}%) | Left: {remaining:4d} | Elapsed: {format_seconds(elapsed):>7} | ETA: {format_seconds(eta_sec):>7} | {prov} #{c_idx+1:03d} ({cat:<15}) -> {verdict_str} ({dt:5.0f}ms)"
            print(case_summary, flush=True)

            # Update live JSON progress file
            progress_state.update({
                "completed": completed_count,
                "remaining": remaining,
                "percentage": round(pct_done, 1),
                "passed": total_passed,
                "failed": completed_count - total_passed,
                "current_pass_rate": round(overall_rate, 1),
                "elapsed_seconds": round(elapsed, 1),
                "elapsed_human": format_seconds(elapsed),
                "eta_seconds": round(eta_sec, 1),
                "eta_human": format_seconds(eta_sec),
                "speed_sec_per_test": round(avg_sec_per_test, 2),
                "current_province": prov,
                "current_case_idx": c_idx + 1,
                "last_case_summary": f"{prov} #{c_idx+1} ({cat}) -> {'PASS' if is_pass else 'FAIL'}",
            })
            save_progress_file(progress_state)

        prov_rate = (prov_passed / len(specs)) * 100.0
        prov_status = colored("PASSED (>=92%)", "green", attrs=["bold"]) if prov_rate >= 92.0 else colored("FAILED (<92%)", "red", attrs=["bold"])
        avg_prov_lat = sum(prov_latencies) / len(prov_latencies) if prov_latencies else 0.0

        print(f"\n--> [{prov} COMPLETE] {prov_name}: {prov_passed}/{len(specs)} Passed ({prov_rate:5.1f}%) | Status: {prov_status} | Avg Latency: {avg_prov_lat:.1f}ms\n" + "-" * 105, flush=True)

        prov_result_dict = {
            "province": prov,
            "name": prov_name,
            "total": len(specs),
            "passed": prov_passed,
            "pass_rate": prov_rate,
            "avg_latency_ms": avg_prov_lat,
            "category_stats": prov_cat_stats,
            "failures": prov_failures,
        }
        provincial_results.append(prov_result_dict)
        progress_state["provincial_summary"][prov] = {
            "name": prov_name,
            "passed": prov_passed,
            "total": len(specs),
            "pass_rate": round(prov_rate, 1),
            "status": "PASSED" if prov_rate >= 92.0 else "FAILED",
        }
        save_progress_file(progress_state)

    total_time = time.time() - t_start
    final_overall_rate = (total_passed / total_cases) * 100.0 if total_cases > 0 else 0.0
    overall_verdict = colored("PASSED (>=80%)", "green", attrs=["bold"]) if final_overall_rate >= 80.0 else colored("FAILED (<80%)", "red", attrs=["bold"])

    # Final Scorecard
    print("\n" + "=" * 105, flush=True)
    print("                             13-JURISDICTION FINAL AUDIT SCORECARD                                ", flush=True)
    print("=" * 105, flush=True)
    print(f"{'Code':<6} {'Jurisdiction':<28} {'Passed':<12} {'Pass Rate':<14} {'Threshold (92%)':<22} {'Avg Latency'}", flush=True)
    print("-" * 105, flush=True)

    failed_regions = []
    passed_regions = []

    for r in provincial_results:
        code = r["province"]
        name = r["name"]
        score_str = f"{r['passed']}/{r['total']}"
        rate_str = f"{r['pass_rate']:.1f}%"
        if r["pass_rate"] >= 92.0:
            colored_status = colored("PASSED", "green")
            passed_regions.append((code, name, r["pass_rate"]))
        else:
            colored_status = colored("FAILED", "red")
            failed_regions.append((code, name, r["pass_rate"]))

        lat_str = f"{r['avg_latency_ms']:.1f}ms"
        print(f"{code:<6} {name:<28} {score_str:<12} {rate_str:<14} {colored_status:<30} {lat_str}", flush=True)

    print("-" * 105, flush=True)
    print(f"Total Test Cases Executed  : {total_cases}", flush=True)
    print(f"Total Passed               : {total_passed} ({final_overall_rate:.1f}%)", flush=True)
    print(f"Overall Country Status     : {overall_verdict} (Threshold: 80.0%)", flush=True)
    print(f"Total Execution Time       : {format_seconds(total_time)} (Average: {total_time/total_cases*1000:.1f}ms per test)", flush=True)
    print("=" * 105 + "\n", flush=True)

    # Print Failed vs Passed Regions Summary
    print(colored("REGIONAL THRESHOLD ANALYSIS SUMMARY:", attrs=["bold"]), flush=True)
    if failed_regions:
        print(colored(f"  [FAIL] FAILED REGIONS (<92.0% Threshold): {len(failed_regions)}", "red", attrs=["bold"]), flush=True)
        for c, n, rt in failed_regions:
            print(f"     - [{c}] {n}: {rt:.1f}% (Needs {92.0 - rt:.1f}% improvement)", flush=True)
    else:
        print(colored("  [PASS] ALL 13 REGIONS EXCEEDED THE 92.0% THRESHOLD!", "green", attrs=["bold"]), flush=True)

    print(colored(f"\n  [PASS] PASSED REGIONS (>=92.0% Threshold): {len(passed_regions)}", "green", attrs=["bold"]), flush=True)
    for c, n, rt in passed_regions:
        print(f"     - [{c}] {n}: {rt:.1f}%", flush=True)

    print("\n" + "=" * 105 + "\n", flush=True)

    # Mark progress file as DONE
    progress_state["status"] = "COMPLETED"
    progress_state["finished_at"] = datetime.now().isoformat()
    save_progress_file(progress_state)

    # Generate Markdown Report
    generate_markdown_report(provincial_results, total_cases, total_passed, final_overall_rate, total_time, failed_regions, passed_regions)


def generate_markdown_report(results, total_tests, total_passed, overall_rate, elapsed, failed_regions, passed_regions):
    md = f"""# Canadian Multi-Province 102 Brutal Driving License Test Report

**Execution Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Total Test Cases Executed**: {total_tests} ({len(results)} jurisdictions × 102 test cases)  
**Total Passed**: {total_passed} ({overall_rate:.1f}%)  
**Overall Country Benchmark Status**: **{'PASSED (>= 80.0%)' if overall_rate >= 80.0 else 'FAILED (< 80.0%)'}**  
**Total Execution Time**: {format_seconds(elapsed)} (Average {elapsed/total_tests*1000:.1f}ms per test case)  

---

## 1. Executive Summary & Regional Threshold Evaluation

- **Per-Region Verification Threshold**: $\ge 92.0\%$ pass rate
- **Overall All-Regions Combined Threshold**: $\ge 80.0\%$ pass rate

### Performance Breakdown by Jurisdiction:

| Jurisdiction Code | Province / Territory Name | Passed / Total | Pass Rate | Regional Status ($\ge 92\%$) | Avg Latency |
|:---:|:---|:---:|:---:|:---:|:---:|
"""
    for r in results:
        status_md = "**PASS**" if r["pass_rate"] >= 92.0 else "<span style='color:red;'>**FAIL**</span>"
        md += f"| **{r['province']}** | {r['name']} | {r['passed']} / {r['total']} | **{r['pass_rate']:.1f}%** | {status_md} | {r['avg_latency_ms']:.1f}ms |\n"

    md += f"""
---

## 2. Regional Pass/Fail Scorecard

### Failed Regions (< 92.0% Threshold):
"""
    if failed_regions:
        for c, n, rt in failed_regions:
            md += f"- ❌ **[{c}] {n}**: **{rt:.1f}%** (Deficit: -{92.0 - rt:.1f}%)\n"
    else:
        md += "- **None! All 13 jurisdictions achieved $\ge 92.0\%$ pass rate.**\n"

    md += "\n### Passed Regions ($\ge 92.0\%$ Threshold):\n"
    for c, n, rt in passed_regions:
        md += f"- ✓ **[{c}] {n}**: **{rt:.1f}%**\n"

    md += """
---

## 3. Category Breakdown Across All Jurisdictions

| Category | Clean Base | Gaussian Blur | Sensor Noise | Skew Rotation | Glare & Contrast | Perspective Warp | Brutal Combo |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for r in results:
        stats = r["category_stats"]
        def get_stat(cat_name):
            if cat_name in stats:
                st = stats[cat_name]
                return f"{st['passed']}/{st['total']}"
            elif cat_name == "Glare & Contrast":
                p = stats.get("Optical Glare", {}).get("passed", 0) + stats.get("Low Contrast", {}).get("passed", 0)
                t = stats.get("Optical Glare", {}).get("total", 0) + stats.get("Low Contrast", {}).get("total", 0)
                return f"{p}/{t}"
            return "N/A"

        md += f"| **{r['province']}** | {get_stat('Clean Base')} | {get_stat('Gaussian Blur')} | {get_stat('Sensor Noise')} | {get_stat('Skew Rotation')} | {get_stat('Glare & Contrast')} | {get_stat('Perspective Warp')} | {get_stat('Brutal Combo')} |\n"

    md += """
---

## 4. Failure Analysis & Diagnostics
The main dropoffs under extreme conditions:
1. **Severe Sensor Noise ($\sigma \ge 85$)**: High frequency noise scrambles stroke connectivity on fine fonts.
2. **Heavy Gaussian Blur ($k \ge 16$)**: Optical character segmentation merges adjacent digits.
3. **Brutal Combinations**: Simultaneous blur + noise + rotation + homography warp pushes edge-case degradation.
"""
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    try:
        with open(BRAIN_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(md)
    except Exception:
        pass
    print(f"\nAudit Report written to: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 102 brutal DL tests across Canadian provinces")
    parser.add_argument("--provinces", nargs="+", help="Specific province codes to run, e.g. QC BC ON")
    args = parser.parse_args()

    run_all_brutal_tests(provinces_to_run=args.provinces)
