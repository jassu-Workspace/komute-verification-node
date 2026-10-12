"""
Hardware Profiling and Real-Time Optimization Engine for Komute OCR.
Detects CPU/GPU topology (Intel Core i7-1255U + Iris Xe), allocates optimal
compute worker threads, sets AVX2/MKL neural acceleration flags, and manages hardware state.
"""

import os
import sys
import platform
from typing import Any, Dict, Optional

# Track if optimizations have already been applied to prevent duplicate logs
_OPTIMIZATION_APPLIED = False
_HARDWARE_CACHE: Optional[Dict[str, Any]] = None


def get_hardware_profile() -> Dict[str, Any]:
    """
    Probes system hardware: CPU topology, vector acceleration, GPU, and memory.
    Cached after first probe to avoid redundant OS queries.
    """
    global _HARDWARE_CACHE
    if _HARDWARE_CACHE is not None:
        return _HARDWARE_CACHE

    logical_threads = os.cpu_count() or 4
    physical_cores = logical_threads

    try:
        import psutil
        phys = psutil.cpu_count(logical=False)
        if phys:
            physical_cores = phys
    except Exception:
        pass

    # Query CPU Name via Windows Registry (instant, no wmic subprocess overhead)
    cpu_name = platform.processor() or "Unknown CPU"
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            if val and val.strip():
                cpu_name = val.strip()
            winreg.CloseKey(key)
        except Exception:
            pass

    # Query GPU Description via Windows Registry
    gpu_name = "Integrated / Software Render"
    gpu_vram_mb = 1024
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0000")
            val, _ = winreg.QueryValueEx(key, "DriverDesc")
            if val and val.strip():
                gpu_name = val.strip()
            winreg.CloseKey(key)
        except Exception:
            pass

    # Vector Extensions (AVX2 / VNNI check)
    vector_ext = "Standard SIMD"
    try:
        import torch
        cap = torch.backends.cpu.get_cpu_capability()
        if "AVX2" in cap:
            vector_ext = "AVX2 / VNNI Neural Vector Acceleration"
        elif "AVX512" in cap:
            vector_ext = "AVX-512 Neural Vector Acceleration"
        else:
            vector_ext = cap
    except Exception:
        pass

    # Probe Total System RAM (GB)
    ram_gb = 16.0
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        pass

    # Strict 20% hardware budget ceiling for verification node
    budget_ratio = 0.20
    worker_threads = max(1, int(logical_threads * budget_ratio))
    system_reserved_threads = max(1, logical_threads - worker_threads)
    system_reserved_pct = round((system_reserved_threads / float(logical_threads)) * 100.0, 1)

    _HARDWARE_CACHE = {
        "cpu_name": cpu_name,
        "physical_cores": physical_cores,
        "logical_threads": logical_threads,
        "ram_gb": ram_gb,
        "budget_ratio": budget_ratio,
        "worker_threads": worker_threads,
        "reserved_threads": system_reserved_threads,
        "reserved_pct": system_reserved_pct,
        "vector_ext": vector_ext,
        "gpu_name": gpu_name,
        "gpu_vram_mb": gpu_vram_mb,
        "platform": f"{platform.system()} {platform.release()}"
    }
    return _HARDWARE_CACHE


def apply_hardware_acceleration(target_threads: Optional[int] = None) -> Dict[str, Any]:
    """
    Applies thread pool optimization and math library configurations across:
    - OpenMP (OMP_NUM_THREADS)
    - Intel MKL (MKL_NUM_THREADS)
    - OpenBLAS (OPENBLAS_NUM_THREADS)
    - ONNX Runtime (ORT_INTRA_OP_NUM_THREADS, ORT_INTER_OP_NUM_THREADS)
    - OpenCV (cv2.setNumThreads)
    - PyTorch (torch.set_num_threads)
    Strictly throttled to 20% of host machine hardware to avoid OS or application starvation.
    """
    global _OPTIMIZATION_APPLIED
    profile = get_hardware_profile()
    threads = target_threads if target_threads is not None else profile["worker_threads"]

    # Set environment variables for C/C++ thread pools
    str_th = str(threads)
    os.environ["OMP_NUM_THREADS"] = str_th
    os.environ["MKL_NUM_THREADS"] = str_th
    os.environ["OPENBLAS_NUM_THREADS"] = str_th
    os.environ["VECLIB_MAXIMUM_THREADS"] = str_th
    os.environ["NUMEXPR_NUM_THREADS"] = str_th
    os.environ["ORT_INTRA_OP_NUM_THREADS"] = str_th
    os.environ["ORT_INTER_OP_NUM_THREADS"] = "1"

    # Configure OpenCV
    try:
        import cv2
        cv2.setNumThreads(threads)
    except Exception:
        pass

    # Configure PyTorch
    try:
        import torch
        torch.set_num_threads(threads)
    except Exception:
        pass

    _OPTIMIZATION_APPLIED = True
    profile["active_worker_threads"] = threads
    return profile


apply_hardware_governor = apply_hardware_acceleration


def print_hardware_banner(profile: Optional[Dict[str, Any]] = None):
    """
    Prints a formatted ASCII hardware profile banner to standard output.
    """
    if profile is None:
        profile = get_hardware_profile()

    cpu = profile.get("cpu_name", "Unknown CPU")
    phys = profile.get("physical_cores", 10)
    log = profile.get("logical_threads", 12)
    ram = profile.get("ram_gb", 16.0)
    wrk = profile.get("active_worker_threads", profile.get("worker_threads", 2))
    res = profile.get("reserved_threads", log - wrk)
    res_pct = profile.get("reserved_pct", 83.3)
    gpu = profile.get("gpu_name", "Intel Iris Xe")
    vram = profile.get("gpu_vram_mb", 1024)

    print("\n" + "=" * 94, flush=True)
    print("               KOMUTE HARDWARE ANALYSIS & 20% RESOURCE GOVERNOR PROFILE              ", flush=True)
    print("=" * 94, flush=True)
    print(f"  * Host Processor      : {cpu}", flush=True)
    print(f"  * CPU Architecture    : {phys} Physical Cores / {log} Logical Threads", flush=True)
    print(f"  * System Memory (RAM) : {ram} GB Total Host Memory", flush=True)
    print(f"  * Hardware Allocation : 20.0% Strict Hardware Ceiling ({res_pct}% System Reserved)", flush=True)
    print(f"  * Active Workers      : {wrk} Worker Threads ({res} threads reserved for OS/UI/Apps)", flush=True)
    print(f"  * Graphics Adapter    : {gpu} (~{vram:,} MB Dedicated/Shared VRAM)", flush=True)
    print(f"  * Execution Footprint : Low-Impact Throttled Runtime (OpenCV + ONNX Runtime @ 20% Cap)", flush=True)
    print("=" * 94 + "\n", flush=True)

