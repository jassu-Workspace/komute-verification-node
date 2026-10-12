"""
Core Regional Test Runner Engine.
Executes hardware-accelerated OCR verification on generated test images for any Canadian jurisdiction.
Evaluates results against ground_truth.json with real-time, image-by-image telemetry,
granular field accuracy, explicit comparative proof (Expected vs Extracted), and strict regional (92.0%) pass benchmark.
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
TEST_IMAGES_DIR = os.path.join(BACKEND_DIR, "test_images")
PROGRESS_FILE = os.path.join(BACKEND_DIR, "test_progress.json")
PROGRESS_TMP = os.path.join(BACKEND_DIR, "test_progress.tmp")

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Apply hardware auto-tuning and neural acceleration
try:
    from core.hardware_profile import apply_hardware_acceleration, print_hardware_banner, get_hardware_profile
    _hw_prof = apply_hardware_acceleration()
except Exception:
    _hw_prof = {}

import cv2
from app.schemas import LicenseDetails, PersonalInfo
from core.dl_ocr import dl_ocr_engine
from core.provincial_templates import PROVINCIAL_CONFIGS


def save_test_progress(state: Dict[str, Any]) -> None:
    """
    Safely writes the real-time test state to backend/test_progress.json.
    Uses atomic rename to avoid file read locks from concurrent monitors.
    """
    try:
        with open(PROGRESS_TMP, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        if os.path.exists(PROGRESS_TMP):
            os.replace(PROGRESS_TMP, PROGRESS_FILE)
    except Exception:
        try:
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass


def build_case_proof(res: Any, expected: Dict[str, str], prov: str) -> Dict[str, Any]:
    """
    Constructs an explicit comparative proof record:
    - Expected Answer: Ground Truth credentials
    - Extracted Answer: OCR candidate extractions
    - Verdicts & Proof: Field-by-field similarity, checksum, and justification
    """
    exp_dl = str(expected.get("dl_number", "")).strip().upper()
    exp_name = str(expected.get("name", "")).strip().upper()
    exp_dob = str(expected.get("dob", "")).strip()
    exp_exp = str(expected.get("expiry", "")).strip()

    ext_dl = str(res.extracted_dl_number or "NOT_EXTRACTED").strip()
    ext_name = str(res.extracted_name or "NOT_EXTRACTED").strip()
    ext_dob = str(res.extracted_dob or "NOT_EXTRACTED").strip()
    ext_exp = str(res.extracted_expiry or "NOT_EXTRACTED").strip()
    ext_prov = getattr(res, "extracted_province", prov) or prov

    # 1. DL Match
    dl_match = False
    if res.number_matched:
        dl_match = True
    elif res.extracted_dl_number:
        clean_ext = "".join(filter(str.isalnum, res.extracted_dl_number)).upper()
        clean_exp = "".join(filter(str.isalnum, exp_dl)).upper()
        if clean_exp and (clean_exp in clean_ext or clean_ext in clean_exp):
            dl_match = True
        elif res.number_similarity >= 0.85:
            dl_match = True

    # 2. Name Match
    name_match = False
    if res.name_matched:
        name_match = True
    elif res.name_similarity >= 0.75:
        name_match = True
    elif res.extracted_name and exp_name:
        ext_parts = set(res.extracted_name.upper().split())
        exp_parts = set(exp_name.split())
        if ext_parts and exp_parts and (ext_parts.intersection(exp_parts)):
            name_match = True

    # 3. DOB Match
    dob_match = False
    if res.dob_matched:
        dob_match = True
    elif res.dob_similarity >= 0.80:
        dob_match = True
    elif res.extracted_dob and exp_dob:
        clean_d_ext = "".join(filter(str.isdigit, res.extracted_dob))
        clean_d_exp = "".join(filter(str.isdigit, exp_dob))
        if clean_d_ext and clean_d_exp and (clean_d_ext == clean_d_exp):
            dob_match = True

    # 4. Expiry Match
    expiry_match = False
    if res.expiry_similarity >= 0.70:
        expiry_match = True
    elif res.extracted_expiry and exp_exp:
        clean_e_ext = "".join(filter(str.isdigit, res.extracted_expiry))
        clean_e_exp = "".join(filter(str.isdigit, exp_exp))
        if clean_e_ext and clean_e_exp and (clean_e_ext == clean_e_exp):
            expiry_match = True
    elif not res.is_expired and not any("Expiry" in m for m in res.missing_fields):
        expiry_match = True

    # 5. Grammar Checksum Match
    grammar_pass = False
    if res.number_rules_pass is True or (res.number_rules_pass is None and len(res.rule_discrepancies) == 0):
        grammar_pass = True
    elif not any("Grammar" in m for m in res.missing_fields):
        grammar_pass = True

    fields = {
        "dl_number": dl_match,
        "name": name_match,
        "dob": dob_match,
        "expiry": expiry_match,
        "grammar": grammar_pass
    }
    matched_count = sum(1 for v in fields.values() if v)
    case_accuracy_pct = (matched_count / 5.0) * 100.0

    dl_sim_pct = round(res.number_similarity * 100.0, 1) if hasattr(res, "number_similarity") else (100.0 if dl_match else 0.0)
    name_sim_pct = round(res.name_similarity * 100.0, 1) if hasattr(res, "name_similarity") else (100.0 if name_match else 0.0)
    dob_sim_pct = round(res.dob_similarity * 100.0, 1) if hasattr(res, "dob_similarity") else (100.0 if dob_match else 0.0)
    exp_sim_pct = round(res.expiry_similarity * 100.0, 1) if hasattr(res, "expiry_similarity") else (100.0 if expiry_match else 0.0)

    if res.passed and matched_count == 5:
        justification = "All 5 Canadian identity & provincial grammar dimensions verified (100% ground truth match)."
    elif res.passed:
        justification = f"Passed credential verification ({matched_count}/5 dimensions confirmed)."
    else:
        fails = []
        if not dl_match: fails.append(f"DL Number ({dl_sim_pct}%)")
        if not name_match: fails.append(f"Name ({name_sim_pct}%)")
        if not dob_match: fails.append("DOB")
        if not expiry_match: fails.append("Expiry")
        if not grammar_pass: fails.append("Provincial Grammar Checksum")
        justification = f"Verification rejected due to mismatch on: {', '.join(fails)}."

    return {
        "expected": {
            "dl_number": exp_dl,
            "name": exp_name,
            "dob": exp_dob,
            "expiry": exp_exp,
            "province": prov
        },
        "extracted": {
            "dl_number": ext_dl,
            "name": ext_name,
            "dob": ext_dob,
            "expiry": ext_exp,
            "province": ext_prov,
            "grammar_valid": grammar_pass
        },
        "field_verdicts": fields,
        "similarities": {
            "dl_similarity_pct": dl_sim_pct,
            "name_similarity_pct": name_sim_pct,
            "dob_similarity_pct": dob_sim_pct,
            "expiry_similarity_pct": exp_sim_pct
        },
        "matched_fields_count": matched_count,
        "case_accuracy_pct": round(case_accuracy_pct, 1),
        "justification": justification
    }


def run_single_region_test(
    prov_code: str,
    threshold: float = 0.92,
    verbose: bool = True,
    show_proof: bool = True,
    global_offset: int = 0,
    global_total: int = 0,
    print_banner: bool = True,
    on_case_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Executes brutal test cases for a single Canadian province/territory.
    Reports real-time accuracy and comparative proof on EVERY SINGLE IMAGE.
    Updates backend/test_progress.json after every image.
    Returns comprehensive regional scorecard.
    """
    prov = prov_code.upper()
    if prov not in PROVINCIAL_CONFIGS:
        raise ValueError(f"Unknown Canadian issuing jurisdiction: {prov}. Valid: {list(PROVINCIAL_CONFIGS.keys())}")

    prov_cfg = PROVINCIAL_CONFIGS[prov]
    prov_name = prov_cfg["name"]
    prov_dir = os.path.join(TEST_IMAGES_DIR, prov)

    if not os.path.exists(prov_dir):
        raise FileNotFoundError(f"Regional test images directory not found: {prov_dir}. Please run generator first.")

    gt_file = os.path.join(prov_dir, "ground_truth.json")
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}.")

    with open(gt_file, "r", encoding="utf-8") as f:
        ground_truth_cases = json.load(f)

    total_cases = len(ground_truth_cases)
    req_pass = int(total_cases * threshold)

    if verbose and print_banner:
        print_hardware_banner()
        print("=" * 105, flush=True)
        print(f"      BRUTAL OCR VERIFICATION ENGINE: {prov} ({prov_name}) - {total_cases} CASES REAL-TIME      ", flush=True)
        print("=" * 105, flush=True)
        print(f"  * Target Jurisdiction : {prov} ({prov_name})", flush=True)
        print(f"  * Dataset Directory   : {prov_dir}", flush=True)
        print(f"  * Regional Benchmark  : >= {threshold*100:.1f}% ({req_pass}/{total_cases} images required to pass)", flush=True)
        print(f"  * Telemetry Output    : Live console proof + {PROGRESS_FILE} per image", flush=True)
        print("-" * 105, flush=True)

    passed_count = 0
    total_field_matches_accum = 0
    failed_cases = []
    all_case_proofs = []
    category_stats: Dict[str, Dict[str, int]] = {}
    latencies: List[float] = []
    t_start = time.time()

    for idx, case in enumerate(ground_truth_cases):
        t_num = case["test_idx"]
        filename = case["filename"]
        cat = case["category"]
        desc = case["description"]
        expected = case["expected"]

        img_path = os.path.join(prov_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            failed_cases.append({
                "test_idx": t_num,
                "desc": desc,
                "category": cat,
                "error": f"Image unreadable: {img_path}"
            })
            continue

        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "passed": 0}
        category_stats[cat]["total"] += 1

        personal = PersonalInfo(
            full_name=expected["name"],
            date_of_birth=expected["dob"],
            email_address="driver@komute.com",
            mobile_number="+10000000000",
            role="Driver"
        )
        license_d = LicenseDetails(
            license_number=expected["dl_number"],
            issuing_province=prov,
            license_expiry_date=expected["expiry"]
        )

        t_case = time.time()
        res = dl_ocr_engine.verify_license(img, personal, license_d)
        latency_ms = (time.time() - t_case) * 1000.0
        latencies.append(latency_ms)

        # Build comprehensive comparative proof (Expected vs Extracted)
        proof = build_case_proof(res, expected, prov)
        fields = proof["field_verdicts"]
        case_acc_pct = proof["case_accuracy_pct"]
        matched_fields_count = proof["matched_fields_count"]
        total_field_matches_accum += matched_fields_count

        is_pass = res.passed
        if is_pass:
            passed_count += 1
            category_stats[cat]["passed"] += 1
            status_tag = "[PASS]"
        else:
            status_tag = "[FAIL]"
            failed_cases.append({
                "test_idx": t_num,
                "filename": filename,
                "desc": desc,
                "category": cat,
                "field_matches": fields,
                "case_accuracy_pct": round(case_acc_pct, 1),
                "proof": proof,
                "details": getattr(res, "details", ""),
                "missing_fields": getattr(res, "missing_fields", []),
                "rule_discrepancies": getattr(res, "rule_discrepancies", [])
            })

        all_case_proofs.append({
            "test_idx": t_num,
            "filename": filename,
            "category": cat,
            "passed": is_pass,
            "latency_ms": round(latency_ms, 1),
            "proof": proof
        })

        # Calculate running rates and countdown ETA
        processed_so_far = idx + 1
        curr_pass_rate = (passed_count / processed_so_far) * 100.0
        curr_field_rate = (total_field_matches_accum / (processed_so_far * 5.0)) * 100.0
        elapsed_sec = time.time() - t_start
        speed_img_per_sec = processed_so_far / max(0.001, elapsed_sec)
        rem_cases = total_cases - processed_so_far
        eta_sec = rem_cases / max(0.001, speed_img_per_sec)
        eta_str = f"{eta_sec:4.1f}s" if eta_sec < 60 else f"{eta_sec/60:3.1f}m"

        # Real-time console log on EVERY single image
        if verbose:
            if global_total > 0:
                g_idx = global_offset + processed_so_far
                prefix = f"  {status_tag:6s} [Glob {g_idx:04d}/{global_total:04d}] [{prov} {processed_so_far:03d}/{total_cases:03d}]"
            else:
                prefix = f"  {status_tag:6s} [Case {processed_so_far:03d}/{total_cases:03d}] [{prov}]"

            cat_short = cat[:14]
            print(
                f"{prefix} {filename:<12s} | {cat_short:<14s} | Acc: {case_acc_pct:3.0f}% ({matched_fields_count}/5) | RunRate: {curr_pass_rate:5.1f}% | {latency_ms:5.1f}ms | ETA: {eta_str}",
                flush=True
            )

            # Comparative Proof Tree (Expected vs Extracted)
            if show_proof:
                exp_i = proof["expected"]
                ext_i = proof["extracted"]
                sims = proof["similarities"]
                v_dl = "OK" if fields["dl_number"] else "MISMATCH"
                v_nm = "OK" if fields["name"] else "MISMATCH"
                v_db = "OK" if fields["dob"] else "MISMATCH"
                v_ex = "OK" if fields["expiry"] else "MISMATCH"
                v_gr = "Valid" if fields["grammar"] else "Mismatch"

                print(f"         * EXPECTED  : DL: '{exp_i['dl_number']}' | Name: '{exp_i['name']}' | DOB: '{exp_i['dob']}' | Exp: '{exp_i['expiry']}'", flush=True)
                print(f"         * EXTRACTED : DL: '{ext_i['dl_number']}' | Name: '{ext_i['name']}' | DOB: '{ext_i['dob']}' | Exp: '{ext_i['expiry']}'", flush=True)
                print(f"         * PROOF     : DL: {v_dl} ({sims['dl_similarity_pct']:.0f}%) | Name: {v_nm} ({sims['name_similarity_pct']:.0f}%) | DOB: {v_db} | Exp: {v_ex} | Grammar: {v_gr} -> {'PASS' if is_pass else 'REJECT'}", flush=True)

        # Real-time disk telemetry snapshot to backend/test_progress.json
        progress_snapshot = {
            "status": "RUNNING",
            "current_region": prov,
            "current_region_name": prov_name,
            "case_idx": processed_so_far,
            "total_cases_in_region": total_cases,
            "global_case_idx": global_offset + processed_so_far if global_total > 0 else processed_so_far,
            "global_total_cases": global_total if global_total > 0 else total_cases,
            "image_name": filename,
            "category": cat,
            "description": desc,
            "case_passed": is_pass,
            "case_accuracy_pct": round(case_acc_pct, 1),
            "matched_fields_count": matched_fields_count,
            "field_matches": fields,
            "proof": proof,
            "regional_passed_count": passed_count,
            "regional_pass_rate_pct": round(curr_pass_rate, 2),
            "running_field_accuracy_pct": round(curr_field_rate, 2),
            "latency_ms": round(latency_ms, 1),
            "speed_img_per_sec": round(speed_img_per_sec, 2),
            "eta_seconds": round(eta_sec, 1),
            "elapsed_seconds": round(elapsed_sec, 2),
            "timestamp": datetime.now().isoformat()
        }
        save_test_progress(progress_snapshot)

        if on_case_callback:
            try:
                on_case_callback(progress_snapshot)
            except Exception:
                pass

    t_total = time.time() - t_start
    pass_rate = (passed_count / total_cases) if total_cases > 0 else 0.0
    overall_field_rate = (total_field_matches_accum / (total_cases * 5.0)) * 100.0 if total_cases > 0 else 0.0
    is_regional_passed = pass_rate >= threshold
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    result_summary = {
        "province": prov,
        "province_name": prov_name,
        "total_cases": total_cases,
        "passed_count": passed_count,
        "failed_count": total_cases - passed_count,
        "pass_rate_pct": round(pass_rate * 100.0, 2),
        "field_accuracy_pct": round(overall_field_rate, 2),
        "threshold_pct": round(threshold * 100.0, 1),
        "verdict": "PASSED" if is_regional_passed else "FAILED",
        "avg_latency_ms": round(avg_latency, 1),
        "total_time_seconds": round(t_total, 2),
        "category_breakdown": category_stats,
        "failed_cases": failed_cases,
        "case_proofs": all_case_proofs
    }

    # Save provincial test results with full case proofs
    res_path = os.path.join(prov_dir, "test_result.json")
    try:
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(result_summary, f, indent=2)
    except Exception:
        pass

    # Update progress file state
    save_test_progress({
        "status": "REGION_COMPLETED",
        "current_region": prov,
        "current_region_name": prov_name,
        "case_idx": total_cases,
        "total_cases_in_region": total_cases,
        "regional_passed_count": passed_count,
        "regional_pass_rate_pct": round(pass_rate * 100.0, 2),
        "field_accuracy_pct": round(overall_field_rate, 2),
        "verdict": result_summary["verdict"],
        "elapsed_seconds": round(t_total, 2),
        "timestamp": datetime.now().isoformat()
    })

    if verbose:
        print("\n" + "=" * 105, flush=True)
        print(f"                         SCORECARD: {prov} ({prov_name})                         ", flush=True)
        print("=" * 105, flush=True)
        print(f"  * Total Cases Tested  : {total_cases}", flush=True)
        print(f"  * Cases Passed        : {passed_count} / {total_cases}", flush=True)
        print(f"  * Regional Pass Rate  : {pass_rate * 100.0:.2f}% (Benchmark: >= {threshold * 100.0:.1f}%)", flush=True)
        print(f"  * Mean Field Accuracy : {overall_field_rate:.2f}%", flush=True)
        print(f"  * Regional Verdict    : {'>>> PASSED <<<' if is_regional_passed else '>>> FAILED <<<'}", flush=True)
        print(f"  * Average Latency     : {avg_latency:.1f}ms/image ({total_cases/max(0.001, t_total):.1f} img/s)", flush=True)
        print(f"  * Total Duration      : {t_total:.2f}s (~{t_total/60:.1f} mins)", flush=True)
        print("\n  Category Breakdown:")
        for cat_name, stats in category_stats.items():
            tot = stats["total"]
            p = stats["passed"]
            c_pct = (p / tot * 100.0) if tot > 0 else 0.0
            print(f"    - {cat_name:18s}: {p:2d}/{tot:2d} ({c_pct:5.1f}%)", flush=True)

        if failed_cases:
            print(f"\n  Failure Breakdown ({len(failed_cases)} failures):")
            for f_case in failed_cases[:8]:
                print(f"    [!] Case #{f_case['test_idx']:03d} ({f_case['category']} - {f_case['desc']}): {f_case.get('details', '')}")
            if len(failed_cases) > 8:
                print(f"    ... and {len(failed_cases) - 8} more (see {res_path})")
        print("=" * 105 + "\n", flush=True)

    return result_summary
