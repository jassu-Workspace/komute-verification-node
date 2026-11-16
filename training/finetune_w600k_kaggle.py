# %% [markdown]
# # Fine-tune `w600k_r50` for age-gap face matching (Komüte-Local)
#
# One-click, unattended run on Kaggle. It downloads everything, prepares the data, splits it by person,
# fine-tunes on both GPUs, tests and exports one ONNX file. You only download `w600k_r50_ft.onnx` and `report.md`.
#
# **Method:** partial fine-tuning (first half of the network and all BatchNorm statistics frozen) + cosine
# distillation to the frozen original model + ArcFace head initialised from class means, trained with
# DistributedDataParallel (one process per T4) and fp16 automatic mixed precision.
#
# **Settings before you start (right-hand panel):** Accelerator = **GPU T4 x2**, Internet = **On**.
# Then click **Save Version → Save & Run All (Commit) → Save**. The run continues in the cloud after you close
# the laptop. Results: the version's **Output** tab.
#
# Data (downloaded and checked by SHA-256 automatically):
# - Training: CACD (2,000 celebrities, 163,446 photos, ages 14–62), public mirror on Hugging Face.
# - Testing: AgeDB-30, CALFW, CPLFW, LFW, CFP-FP (InsightFace `.bin` test files), public mirror on Hugging Face.
# - Start model: `w600k_r50.onnx` from InsightFace `buffalo_l` (same SHA-256 as Komüte's `models.lock`).
#
# These datasets and InsightFace models are for non-commercial research. Get legal review before production use.

# %%
import os
import subprocess
import sys
import time

T0 = time.time()
SMOKE = os.environ.get("KOMUTE_SMOKE") == "1"  # tiny CPU run for testing this notebook; never set on Kaggle


def pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])


_missing = []
for _mod, _spec in (("onnx", "onnx"), ("onnx2torch", "onnx2torch==1.5.15"), ("onnxruntime", "onnxruntime")):
    try:
        __import__(_mod)
    except ImportError:
        _missing.append(_spec)
if _missing:
    pip(*_missing)

# YuNet 2023mar needs OpenCV 4.8 or later. Check in a child process so cv2 is not imported before a fix.
_cv = subprocess.run([sys.executable, "-c", "import cv2; print(cv2.__version__)"], capture_output=True, text=True)
_ver = tuple(int(p) for p in _cv.stdout.strip().split(".")[:2]) if _cv.returncode == 0 else (0, 0)
if _ver < (4, 8):
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "-q",
                     "opencv-python", "opencv-contrib-python", "opencv-python-headless"])
    pip("opencv-python-headless==4.10.0.84")

import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import urllib.request  # noqa: E402
import zipfile  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from onnx2torch import convert  # noqa: E402

ON_KAGGLE = os.name != "nt" and Path("/kaggle/working").is_dir()
OUT = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("KOMUTE_OUT", "kaggle_out"))
DATA = Path(os.environ.get("KOMUTE_DATA", "/tmp/komute_ft"))
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
NGPU = torch.cuda.device_count()  # no CUDA context is created here; training runs in child processes
print(f"torch {torch.__version__}, cv2 {cv2.__version__}, onnx {onnx.__version__}, ort {ort.__version__}; "
      f"GPUs {NGPU}, CPUs {os.cpu_count()}, smoke={SMOKE}")
if ON_KAGGLE and NGPU == 0:
    raise SystemExit("No GPU. Set Accelerator = GPU T4 x2 in the notebook settings and run again.")

# %%
CFG = dict(
    seed=42,
    val_ids=200,            # people held out for model selection (never trained on)
    epochs=6,
    batch=256,              # total over all GPUs (128 per T4)
    lr_backbone=1e-3,
    lr_head=1e-2,
    weight_decay=5e-4,
    warmup_steps=300,
    freeze_fraction=0.5,    # first half of the weight tensors stay frozen
    margin=0.5,             # ArcFace
    scale=64.0,
    distill=1.0,            # keeps embeddings close to the original model (prevents forgetting)
    min_age_gap=7,          # validation same-person pairs: CACD spans 2004-2013, so gaps are 7-9 years at most
    prod_threshold=0.40,    # Komüte's current face threshold
    time_budget_h=10.5,     # Kaggle stops a session at 12 h
    eval_sets=["agedb_30", "calfw", "cplfw", "lfw", "cfp_fp"],
    eval_max_pairs=None,
    max_steps_per_epoch=None,
    workers=2,              # data-loader processes per GPU process (Kaggle has 4 CPUs)
)
if SMOKE:
    CFG.update(val_ids=20, epochs=1, batch=16, warmup_steps=2, eval_sets=["agedb_30"], eval_max_pairs=300,
               max_steps_per_epoch=4, workers=0)
np.random.seed(CFG["seed"])
print(json.dumps(CFG, indent=1))

# %% [markdown]
# ## 1. Download and verify

# %%
HF_CACD = ("https://huggingface.co/datasets/caojingtian1216/CACD/resolve/"
           "82ba751b2d2f1331a109be0437b9969e28616913/data/jpg.zip")
HF_VAL = "https://huggingface.co/datasets/Icar/val_sets/resolve/1f7800d9d5ba295554573a4e292e870b64a49b74/{}.bin"
BUFFALO = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
YUNET = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
         "face_detection_yunet_2023mar.onnx")
SHA = {
    "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
    "face_detection_yunet_2023mar.onnx": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    "cacd_jpg.zip": "d72e01dcdb83df407a0287767a1237ff34a3b2d61fb20cf375314252796e2ce0",
    "agedb_30.bin": "1af672c0ea829b95951b12628687956fe54263614b60b0ec5af9cdfdc4480b30",
    "calfw.bin": "f643466e875865670a91d6d836dcd0af7ac41730f279ba13de1ed5ae6546c4db",
    "cplfw.bin": "2d18b80537ef3dce1ba163d8829c5994d1408d4c2a84eee499bba5563d8da2fc",
    "lfw.bin": "9b711dca71a47fc9f295a28514abe8cc4e457632a8d634264691398c99569593",
    "cfp_fp.bin": "76306c783c2ef59c8569ebdcdd2f529f450bcc3fad57c94a5fc2b91df0f10370",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path, sha: str | None) -> Path:
    """Download with retries; keep an existing file only if its checksum matches."""
    if dest.exists() and (sha is None or sha256(dest) == sha):
        return dest
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "komute-finetune"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            break
        except Exception as e:  # network hiccup: retry, then fail loudly
            if attempt == 4:
                raise
            print(f"download retry {attempt + 1} for {dest.name}: {e}")
            time.sleep(15 * (attempt + 1))
    if sha is not None and sha256(tmp) != sha:
        tmp.unlink()
        raise RuntimeError(f"checksum mismatch for {dest.name}; refusing to use it")
    tmp.replace(dest)
    print(f"downloaded {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


W600K = DATA / "w600k_r50.onnx"
if not (W600K.exists() and sha256(W600K) == SHA["w600k_r50.onnx"]):
    bz = fetch(BUFFALO, DATA / "buffalo_l.zip", None)
    with zipfile.ZipFile(bz) as z:
        entry = next(n for n in z.namelist() if n.endswith("w600k_r50.onnx"))
        with z.open(entry) as src, open(W600K, "wb") as dst:
            shutil.copyfileobj(src, dst)
    if sha256(W600K) != SHA["w600k_r50.onnx"]:
        raise RuntimeError("w600k_r50.onnx checksum mismatch")
    bz.unlink()
YUNET_PATH = fetch(YUNET, DATA / "face_detection_yunet_2023mar.onnx", SHA["face_detection_yunet_2023mar.onnx"])
CACD_ZIP = fetch(HF_CACD, DATA / "cacd_jpg.zip", None if SMOKE else SHA["cacd_jpg.zip"])
EVAL_BINS = {s: fetch(HF_VAL.format(s), DATA / f"{s}.bin", SHA[f"{s}.bin"]) for s in CFG["eval_sets"]}
print(f"downloads done in {(time.time() - T0) / 60:.1f} min")

# %% [markdown]
# ## 2. Split by person and align every face exactly as Komüte does (YuNet + 5-point affine to 112×112)

# %%
NAME_RE = re.compile(r"^(\d+)_(.+)_(\d+)\.jpg$", re.IGNORECASE)
with zipfile.ZipFile(CACD_ZIP) as z:
    names = sorted(n for n in z.namelist() if n.lower().endswith(".jpg"))
recs = []
for n in names:
    m = NAME_RE.match(n.rsplit("/", 1)[-1])
    if not m:
        raise RuntimeError(f"unexpected file name {n}")
    recs.append((n, int(m.group(1)), m.group(2)))
if not SMOKE and len(recs) != 163446:
    raise RuntimeError(f"expected 163,446 CACD photos, found {len(recs)}")
ages = np.array([r[1] for r in recs])
people = sorted({r[2] for r in recs})
perm = np.random.default_rng(CFG["seed"]).permutation(len(people))
val_people = {people[i] for i in perm[:CFG["val_ids"]]}
train_people = sorted(set(people) - val_people)
pid = {p: i for i, p in enumerate(train_people)}
train_idx = np.array([i for i, r in enumerate(recs) if r[2] in pid])
train_lab = np.array([pid[recs[i][2]] for i in train_idx])
val_idx = np.array([i for i, r in enumerate(recs) if r[2] in val_people])
assert not ({recs[i][2] for i in train_idx} & {recs[i][2] for i in val_idx}), "person overlap"
print(f"{len(recs)} photos, {len(people)} people: train {len(train_people)} people / {len(train_idx)} photos, "
      f"validation {len(val_people)} people / {len(val_idx)} photos")

REF = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655],
                [70.7299, 92.2041]], dtype=np.float32)
_Z = _DET = None


def _align_init(zip_path: str, det_path: str) -> None:
    global _Z, _DET
    cv2.setNumThreads(1)
    _Z = zipfile.ZipFile(zip_path)
    _DET = cv2.FaceDetectorYN.create(det_path, "", (320, 320), 0.7, 0.3, 50)  # production settings


def _align_chunk(chunk: list[str]) -> tuple[np.ndarray, int]:
    out = np.empty((len(chunk), 112, 112, 3), np.uint8)
    found = 0
    for k, n in enumerate(chunk):
        im = cv2.imdecode(np.frombuffer(_Z.read(n), np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"cannot decode {n}")
        _DET.setInputSize((im.shape[1], im.shape[0]))
        _, faces = _DET.detect(im)
        if faces is not None and len(faces):
            f = max(faces, key=lambda r: r[2] * r[3])  # largest face, as pick_licence_face()
            M, _ = cv2.estimateAffinePartial2D(f[4:14].reshape(5, 2).astype(np.float32), REF, method=cv2.LMEDS)
            if M is not None:
                out[k] = cv2.warpAffine(im, M, (112, 112), borderValue=(0, 0, 0))
                found += 1
                continue
        out[k] = im if im.shape[:2] == (112, 112) else cv2.resize(im, (112, 112))  # crops are pre-aligned
    return out, found


# All photos go to one memory-mapped file (6.2 GB for CACD) that every training process reads.
_need = len(recs) * 112 * 112 * 3 + (1 << 30)
if shutil.disk_usage(DATA).free < _need:
    raise RuntimeError(f"not enough disk in {DATA}: need {_need / 1e9:.1f} GB")
X = np.lib.format.open_memmap(DATA / "X.npy", mode="w+", dtype=np.uint8, shape=(len(recs), 112, 112, 3))
chunks = [names[i:i + 512] for i in range(0, len(names), 512)]
found = pos = 0
if os.name == "nt":  # Windows cannot fork notebook-defined workers; local tests only
    _align_init(str(CACD_ZIP), str(YUNET_PATH))
    results, pool = map(_align_chunk, chunks), None
else:
    pool = Pool(os.cpu_count(), initializer=_align_init, initargs=(str(CACD_ZIP), str(YUNET_PATH)))
    results = pool.imap(_align_chunk, chunks)
for c, (arr, f) in enumerate(results):
    X[pos:pos + len(arr)] = arr
    pos += len(arr)
    found += f
    if c % 40 == 0:
        print(f"aligned {pos}/{len(X)} ({(time.time() - T0) / 60:.1f} min)", flush=True)
if pool is not None:
    pool.close()
    pool.join()
assert pos == len(X)
X.flush()
del X
X = np.load(DATA / "X.npy", mmap_mode="r")
print(f"aligned {len(X)} photos; YuNet found a face in {found} ({100 * found / len(X):.1f}%), "
      f"the rest keep the pre-aligned crop")

# %% [markdown]
# ## 3. Validation pairs, training inputs, and the training module

# %%
# Validation (model selection): held-out CACD people, same-person pairs >= min_age_gap years apart.
_rng = np.random.default_rng(CFG["seed"] + 1)
_by_person: dict[str, list[int]] = {}
for i in val_idx:
    _by_person.setdefault(recs[i][2], []).append(int(i))
POS = []
for items in _by_person.values():
    cand = [(a, b) for x, a in enumerate(items) for b in items[x + 1:] if abs(ages[a] - ages[b]) >= CFG["min_age_gap"]]
    _rng.shuffle(cand)
    cand.sort(key=lambda p: -abs(ages[p[0]] - ages[p[1]]))  # widest gaps first, random among ties
    POS += cand[:20]
NEG = []
_vp = np.array([recs[i][2] for i in val_idx])
while len(NEG) < min(20 * len(POS), 50000):
    a, b = _rng.integers(0, len(val_idx), 2)
    if _vp[a] != _vp[b]:
        NEG.append((int(val_idx[a]), int(val_idx[b])))
POS, NEG = np.array(POS, dtype=np.int64), np.array(NEG, dtype=np.int64)
print(f"validation: {len(POS)} same-person age-gap pairs, {len(NEG)} different-person pairs")
if len(POS) < 10:
    raise RuntimeError("too few validation pairs")

np.savez(DATA / "meta.npz", train_idx=train_idx, train_lab=train_lab, val_idx=val_idx, pos=POS, neg=NEG,
         n_classes=len(train_people))
(DATA / "cfg.json").write_text(json.dumps(CFG), encoding="utf-8")

# BEGIN-MODULE (the build script inlines training/komute_ft.py here)
MODULE_SOURCE = (Path(__file__).resolve().parent / "komute_ft.py").read_text(encoding="utf-8")
# END-MODULE
(DATA / "komute_ft.py").write_text(MODULE_SOURCE, encoding="utf-8")
sys.path.insert(0, str(DATA))
import komute_ft as kft  # noqa: E402

# %% [markdown]
# ## 4. Prove the ONNX → PyTorch conversion is exact (CPU, so no GPU is held during training)

# %%
_onnx = onnx.load(str(W600K))
# Komüte reads the input by session.get_inputs()[0].name and the output by position. The original output name
# ("683") is not a valid TorchScript name, so the export calls it "embedding".
IN_NAME, OUT_NAME = _onnx.graph.input[0].name, "embedding"


def ort_embed(path: Path, imgs: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rgb = imgs[..., ::-1].astype(np.float32)
    blob = ((rgb - 127.5) / 127.5).transpose(0, 3, 1, 2)
    out = np.concatenate([sess.run(None, {sess.get_inputs()[0].name: blob[i:i + 1]})[0] for i in range(len(blob))])
    return out / np.linalg.norm(out, axis=1, keepdims=True)


_probe = np.ascontiguousarray(X[val_idx[:16]])
_cos = np.sum(ort_embed(W600K, _probe) * kft.torch_embed(convert(_onnx), _probe, torch.device("cpu")), axis=1)
print(f"ONNX vs converted PyTorch: lowest cosine {_cos.min():.6f}")
if _cos.min() < 0.9999:
    raise RuntimeError("conversion is not exact; stopping")

# %% [markdown]
# ## 5. Train: DDP, one process per GPU, fp16 AMP (output streams below)

# %%
_nproc = max(1, NGPU)
_env = dict(os.environ, KOMUTE_DATA=str(DATA), KOMUTE_T0=str(T0), OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
_cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={_nproc}",
        str(DATA / "komute_ft.py")]
print(" ".join(_cmd), flush=True)
_p = subprocess.Popen(_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=_env, cwd=str(DATA))
for _line in _p.stdout:
    print(_line, end="", flush=True)
if _p.wait() != 0:
    raise RuntimeError(f"training failed with exit code {_p.returncode}; see the log above")
TRAIN = json.loads((DATA / "train_result.json").read_text(encoding="utf-8"))
history, best_epoch = TRAIN["history"], TRAIN["best_epoch"]
if best_epoch is None or not (DATA / "best.pt").exists():
    raise RuntimeError("no epoch finished; nothing to export")
print(f"best epoch {best_epoch}: {history[best_epoch]}")

# %% [markdown]
# ## 6. Test original vs fine-tuned on AgeDB-30, CALFW, CPLFW, LFW, CFP-FP

# %%
DEV = torch.device("cuda:0" if NGPU else "cpu")
teacher = convert(onnx.load(str(W600K))).to(DEV).eval()
student = convert(onnx.load(str(W600K)))
student.load_state_dict(torch.load(DATA / "best.pt", map_location="cpu"))
student = student.to(DEV).eval()

_ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
            ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"),
            ("numpy._core.multiarray", "scalar")}


class SafeUnpickler(pickle.Unpickler):  # the .bin files are third-party pickles: numpy pieces only
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def load_bin(path: Path, max_pairs: int | None) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        bins, same = SafeUnpickler(f, encoding="bytes").load()
    same = np.asarray(same, bool)
    sel = np.arange(len(same)) if not max_pairs else np.linspace(0, len(same) - 1, max_pairs).astype(int)
    imgs = np.empty((2 * len(sel), 112, 112, 3), np.uint8)
    for k, i in enumerate(sel):
        for j in (0, 1):
            im = cv2.imdecode(np.frombuffer(bytes(bins[2 * i + j]), np.uint8), cv2.IMREAD_COLOR)
            imgs[2 * k + j] = im if im.shape[:2] == (112, 112) else cv2.resize(im, (112, 112))
    return imgs, same[sel]


def ten_fold(scores: np.ndarray, same: np.ndarray) -> float:
    folds = np.array_split(np.arange(len(same)), 10)
    grid = np.arange(-1, 1, 0.0025)
    accs = []
    for k in range(10):
        tr = np.concatenate([folds[j] for j in range(10) if j != k])
        acc_tr = ((scores[tr][:, None] >= grid[None]) == same[tr][:, None]).mean(0)
        t = grid[int(np.argmax(acc_tr))]
        accs.append(np.mean((scores[folds[k]] >= t) == same[folds[k]]))
    return float(np.mean(accs))


EVAL = {s: load_bin(p, CFG["eval_max_pairs"]) for s, p in EVAL_BINS.items()}
thr = CFG["prod_threshold"]


def external_eval(model) -> dict:
    res = {}
    for s, (imgs, same) in EVAL.items():
        e = kft.torch_embed(model, imgs, DEV)
        sc = np.sum(e[0::2] * e[1::2], axis=1)
        res[s] = dict(acc10=ten_fold(sc, same), pairs=int(len(same)), same=int(same.sum()), diff=int((~same).sum()),
                      false_reject=int(np.sum(sc[same] < thr)), false_accept=int(np.sum(sc[~same] >= thr)),
                      table={f"{t:.2f}": [int(np.sum(sc[same] < t)), int(np.sum(sc[~same] >= t))]
                             for t in (0.25, 0.30, 0.35, 0.40, 0.45)},
                      rejected=np.where(same & (sc < thr))[0].tolist())
    return res


base_ext, ft_ext = external_eval(teacher), external_eval(student)
for s in EVAL:
    b, f = base_ext[s], ft_ext[s]
    print(f"{s:9s} acc10 {100 * b['acc10']:.2f} -> {100 * f['acc10']:.2f}  false rejects@{thr} "
          f"{b['false_reject']} -> {f['false_reject']}  false accepts {b['false_accept']} -> {f['false_accept']}")

# %% [markdown]
# ## 7. Export ONNX and verify it with ONNX Runtime

# %%
MODEL_OUT = OUT / "w600k_r50_ft.onnx"
_cpu = copy.deepcopy(student).cpu().float().eval()
_kw = dict(input_names=[IN_NAME], output_names=[OUT_NAME], opset_version=17, do_constant_folding=True,
           dynamic_axes={IN_NAME: {0: "batch"}, OUT_NAME: {0: "batch"}})
with torch.no_grad():
    try:
        torch.onnx.export(_cpu, torch.randn(1, 3, 112, 112), str(MODEL_OUT), dynamo=False, **_kw)
    except TypeError:  # older torch without the dynamo argument
        torch.onnx.export(_cpu, torch.randn(1, 3, 112, 112), str(MODEL_OUT), **_kw)
onnx.checker.check_model(str(MODEL_OUT))
_cos = np.sum(ort_embed(MODEL_OUT, _probe) * kft.torch_embed(_cpu, _probe, torch.device("cpu")), axis=1)
print(f"exported ONNX vs PyTorch: lowest cosine {_cos.min():.6f}")
if _cos.min() < 0.9999:
    raise RuntimeError("exported ONNX does not match the trained model")
MODEL_SHA = sha256(MODEL_OUT)

# %% [markdown]
# ## 8. Report

# %%
a_b, a_f = base_ext.get("agedb_30"), ft_ext.get("agedb_30")
fa_ok = all(ft_ext[s]["false_accept"] <= base_ext[s]["false_accept"] for s in EVAL)
lfw_ok = "lfw" not in EVAL or ft_ext["lfw"]["acc10"] >= base_ext["lfw"]["acc10"] - 0.002
agedb_ok = a_f is None or a_f["acc10"] > a_b["acc10"]
improved = fa_ok and lfw_ok and agedb_ok
lines = [
    "# Fine-tuned w600k_r50: report",
    "",
    f"- Model: `w600k_r50_ft.onnx`, SHA-256 `{MODEL_SHA}`, input `{IN_NAME}`, output `{OUT_NAME}`.",
    f"- Method: partial fine-tuning ({int(100 * CFG['freeze_fraction'])}% of weight tensors frozen, BatchNorm "
    f"frozen) + distillation to the original + ArcFace; DDP on {max(1, NGPU)} GPU(s), fp16 AMP.",
    f"- Trained on CACD: {len(train_people)} people, {len(train_idx)} photos; best epoch {best_epoch} "
    f"of {len(history) - 1}; run time {(time.time() - T0) / 3600:.1f} h.",
    f"- Validation (held-out {len(val_people)} CACD people, {len(POS)} age-gap pairs >= {CFG['min_age_gap']} years, "
    f"{len(NEG)} different-person pairs): TAR at FAR 0.1% "
    f"{100 * history[0]['tar_at_far_1e3']:.2f}% -> {100 * history[best_epoch]['tar_at_far_1e3']:.2f}%.",
    f"- **Verdict:** {'improved: test it in Komüte' if improved else 'NOT better than the original: keep w600k_r50.onnx'}.",
    "",
    f"## Test sets (production threshold {thr:.2f})",
    "",
    "| Set | Pairs | 10-fold accuracy (orig -> new) | Same person rejected | Different people accepted |",
    "|---|---|---|---|---|",
]
for s in EVAL:
    b, f = base_ext[s], ft_ext[s]
    lines.append(f"| {s} | {b['pairs']} | {100 * b['acc10']:.2f}% -> {100 * f['acc10']:.2f}% | "
                 f"{b['false_reject']} -> {f['false_reject']} of {b['same']} | "
                 f"{b['false_accept']} -> {f['false_accept']} of {b['diff']} |")
if a_b is not None:
    recovered = len(set(a_b["rejected"]) - set(a_f["rejected"]))
    lines += ["", "## AgeDB-30 by threshold (same person rejected / different people accepted)", "",
              "| Threshold | Original | Fine-tuned |", "|---|---|---|"]
    for t in a_b["table"]:
        lines.append(f"| {t} | {a_b['table'][t][0]} / {a_b['table'][t][1]} | {a_f['table'][t][0]} / {a_f['table'][t][1]} |")
    lines += ["", f"Of the {a_b['false_reject']} AgeDB-30 same-person pairs the original rejects at {thr:.2f}, "
              f"{recovered} now pass."]
lines += ["", "## Per-epoch validation", "", "| Epoch | TAR at FAR 0.1% | Same person accepted at 0.40 | "
          "Different people accepted at 0.40 |", "|---|---|---|---|"]
for h in history:
    lines.append(f"| {h['epoch']} | {100 * h['tar_at_far_1e3']:.2f}% | {100 * h['tar_at_prod']:.2f}% | "
                 f"{100 * h['far_at_prod']:.3f}% |")
lines += ["", "## Limits", "",
          "- CACD, AgeDB and LFW are all celebrity photos; some people may appear in both training and test sets.",
          "- Test pairs are pre-aligned crops, the same path as Komüte's earlier benchmark (YuNet not exercised).",
          "- Before production: re-run `scripts/bench_face_folder.py` and the full benchmark locally, recalibrate "
          "the threshold, and record the new SHA-256 in `backend/models/models.lock`.",
          "- Training data is for non-commercial research use; get legal review."]
(OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
(OUT / "report.json").write_text(json.dumps(dict(cfg=CFG, model_sha256=MODEL_SHA, best_epoch=best_epoch,
                                                 history=history, original=base_ext, finetuned=ft_ext,
                                                 improved=improved), indent=1), encoding="utf-8")
print("\n".join(lines))
print(f"\nDONE in {(time.time() - T0) / 3600:.2f} h. Files: {sorted(p.name for p in OUT.iterdir())}")
