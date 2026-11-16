"""Shared helpers and the DDP training entry point for the w600k_r50 fine-tune.

The Kaggle notebook writes this file to disk, imports it for evaluation, and starts training with
    python -m torch.distributed.run --standalone --nproc_per_node=<GPUs> komute_ft.py
Inputs come from $KOMUTE_DATA: X.npy (aligned 112x112 BGR uint8), meta.npz, cfg.json, w600k_r50.onnx.
Outputs to $KOMUTE_DATA: best.pt (student state_dict of the best epoch) and train_result.json.
Method: partial fine-tuning (first part frozen, BatchNorm statistics frozen), ArcFace head initialised from
class means, cosine distillation to the frozen original model, fp16 AMP, DistributedDataParallel.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from onnx2torch import convert


def prep(xb: torch.Tensor, dev: torch.device) -> torch.Tensor:
    """uint8 NHWC BGR -> float NCHW RGB 0..255 (the same input as Komüte's embed())."""
    return xb.to(dev, non_blocking=True).permute(0, 3, 1, 2).float()[:, [2, 1, 0]]


def norm(x: torch.Tensor) -> torch.Tensor:
    return (x - 127.5) / 127.5


@torch.no_grad()
def torch_embed(model: nn.Module, arr: np.ndarray, dev: torch.device, idx: np.ndarray | None = None,
                bs: int = 512) -> np.ndarray:
    """L2-normalised fp32 embeddings for arr (or arr[idx]), in order."""
    model.eval()
    n = len(arr) if idx is None else len(idx)
    outs = []
    for i in range(0, n, bs):
        sl = arr[i:i + bs] if idx is None else arr[np.asarray(idx[i:i + bs])]
        x = norm(prep(torch.from_numpy(np.ascontiguousarray(sl)), dev))
        outs.append(F.normalize(model(x).float(), dim=1).cpu())
    return torch.cat(outs).numpy() if outs else np.zeros((0, 512), np.float32)


class ArcHead(nn.Module):
    """ArcFace margin head (m additive angular margin, s scale), computed in fp32."""

    def __init__(self, w: torch.Tensor, m: float, s: float):
        super().__init__()
        self.W = nn.Parameter(w.detach().clone().float())
        self.s, self.cos_m, self.sin_m = s, math.cos(m), math.sin(m)
        self.th, self.mm = math.cos(math.pi - m), math.sin(math.pi - m) * m

    def forward(self, e: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cos = F.linear(F.normalize(e.float()), F.normalize(self.W)).clamp(-1 + 1e-7, 1 - 1e-7)
        sin = torch.sqrt(1 - cos * cos)
        phi = torch.where(cos > self.th, cos * self.cos_m - sin * self.sin_m, cos - self.mm)
        return self.s * torch.where(F.one_hot(y, cos.size(1)).bool(), phi, cos)


class Net(nn.Module):
    """Student plus head in one module, so DDP syncs the gradients of both."""

    def __init__(self, student: nn.Module, head: ArcHead):
        super().__init__()
        self.student, self.head = student, head

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e = self.student(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = self.head(e.float(), y)
        return e.float(), logits


def set_train(m: nn.Module) -> None:
    """Train mode, but BatchNorm keeps the statistics InsightFace trained (no drift on small batches)."""
    m.train()
    for mod in m.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.eval()


_BLUR: dict[torch.device, torch.Tensor] = {}


def augment(x: torch.Tensor) -> torch.Tensor:
    """GPU augmentation on RGB 0..255: flip, grayscale (old photos), brightness and contrast, blur."""
    n, d = x.size(0), x.device
    if d not in _BLUR:
        k = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32, device=d)
        _BLUR[d] = (torch.outer(k, k) / 256).expand(3, 1, 5, 5).contiguous()
    x = torch.where((torch.rand(n, device=d) < 0.5)[:, None, None, None], x.flip(3), x)
    g = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1).expand_as(x)
    x = torch.where((torch.rand(n, device=d) < 0.2)[:, None, None, None], g, x)
    c = 1 + (torch.rand(n, 1, 1, 1, device=d) - 0.5) * 0.4
    b = (torch.rand(n, 1, 1, 1, device=d) - 0.5) * 50
    mean = x.mean((1, 2, 3), keepdim=True)
    x = (x - mean) * c + mean + b
    blurred = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="replicate"), _BLUR[d], groups=3)
    x = torch.where((torch.rand(n, device=d) < 0.1)[:, None, None, None], blurred, x)
    return x.clamp(0, 255)


def val_eval(model: nn.Module, X: np.ndarray, val_idx: np.ndarray, pos: np.ndarray, neg: np.ndarray,
             thr: float, dev: torch.device) -> dict:
    """TAR at FAR 0.1% on held-out people (model selection), plus rates at the production threshold."""
    e = torch_embed(model, X, dev, val_idx)
    row = {int(g): k for k, g in enumerate(val_idx)}

    def scores(P: np.ndarray) -> np.ndarray:
        a = np.array([row[int(i)] for i in P[:, 0]])
        b = np.array([row[int(i)] for i in P[:, 1]])
        return np.sum(e[a] * e[b], axis=1)

    sp, sn = scores(pos), scores(neg)
    t = float(np.quantile(sn, 1 - 1e-3))
    return dict(tar_at_far_1e3=float(np.mean(sp > t)), tar_at_prod=float(np.mean(sp >= thr)),
                far_at_prod=float(np.mean(sn >= thr)))


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, idx: np.ndarray, lab: np.ndarray):
        self.X, self.idx, self.lab = X, idx, lab

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        return torch.from_numpy(np.array(self.X[self.idx[i]])), int(self.lab[i])


def _log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def train_main() -> None:
    data = Path(os.environ["KOMUTE_DATA"])
    t0 = float(os.environ.get("KOMUTE_T0", time.time()))
    cfg = json.loads((data / "cfg.json").read_text(encoding="utf-8"))
    cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if cuda else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    if cuda:
        torch.cuda.set_device(local)
        torch.backends.cudnn.benchmark = True
    dev = torch.device(f"cuda:{local}" if cuda else "cpu")
    torch.manual_seed(cfg["seed"] + rank)

    meta = np.load(data / "meta.npz")
    X = np.load(data / "X.npy", mmap_mode="r")
    train_idx, train_lab = meta["train_idx"], meta["train_lab"]
    val_idx, pos, neg = meta["val_idx"], meta["pos"], meta["neg"]
    n_cls = int(meta["n_classes"])
    w600k = str(data / "w600k_r50.onnx")
    teacher = convert(onnx.load(w600k)).to(dev).eval().requires_grad_(False)
    student = convert(onnx.load(w600k)).to(dev)

    # Head starts at each training person's mean original embedding (sharded over ranks, then summed).
    shard = np.arange(rank, len(train_idx), world)
    emb = torch.from_numpy(torch_embed(teacher, X, dev, train_idx[shard])).to(dev, torch.float64)
    sums = torch.zeros(n_cls, 512, dtype=torch.float64, device=dev)
    sums.index_add_(0, torch.from_numpy(train_lab[shard]).to(dev), emb)
    dist.all_reduce(sums)
    head = ArcHead(F.normalize(sums, dim=1).float(), cfg["margin"], cfg["scale"])
    del emb, sums

    params = list(student.parameters())
    n_frozen = int(len(params) * cfg["freeze_fraction"])
    for p in params[:n_frozen]:
        p.requires_grad_(False)
    trainable = [p for p in params if p.requires_grad]
    _log(rank, f"[train] {world} process(es); weight tensors {len(params)}, frozen {n_frozen}, trainable "
               f"{len(trainable)} ({sum(p.numel() for p in trainable) / 1e6:.1f} M values); {n_cls} classes")

    net = Net(student, head).to(dev)
    ddp = nn.parallel.DistributedDataParallel(net, device_ids=[local] if cuda else None, broadcast_buffers=False)
    per_gpu = max(1, cfg["batch"] // world)
    ds = TrainSet(X, train_idx, train_lab)
    sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True,
                                                              seed=cfg["seed"], drop_last=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=per_gpu, sampler=sampler, drop_last=True,
                                         num_workers=cfg["workers"], pin_memory=cuda,
                                         persistent_workers=cfg["workers"] > 0)
    steps = len(loader) if not cfg["max_steps_per_epoch"] else min(len(loader), cfg["max_steps_per_epoch"])
    total = max(1, steps * cfg["epochs"])
    opt = torch.optim.SGD([{"params": trainable, "lr": cfg["lr_backbone"]},
                           {"params": head.parameters(), "lr": cfg["lr_head"]}],
                          momentum=0.9, weight_decay=cfg["weight_decay"])
    warm = max(1, cfg["warmup_steps"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    except (AttributeError, TypeError):  # torch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=cuda)

    thr = cfg["prod_threshold"]
    history, best, best_epoch, epoch_secs = [], -1.0, None, None
    if rank == 0:
        history.append(dict(epoch=0, **val_eval(student, X, val_idx, pos, neg, thr, dev)))
        _log(rank, f"[train] original model on validation: {history[0]}")
    dist.barrier()

    for ep in range(1, cfg["epochs"] + 1):
        stop = torch.zeros(1, device=dev)
        if rank == 0 and epoch_secs and time.time() - t0 + 1.2 * epoch_secs > cfg["time_budget_h"] * 3600:
            stop += 1
        dist.broadcast(stop, 0)
        if stop.item():
            _log(rank, f"[train] stopping before epoch {ep}: time budget")
            break
        t_ep = time.time()
        sampler.set_epoch(ep)
        set_train(net)
        for step, (xb, yb) in enumerate(loader):
            if step >= steps:
                break
            x = norm(augment(prep(xb, dev)))
            yb = yb.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=cuda):
                e_s, logits = ddp(x, yb)
                with torch.no_grad():
                    e_t = teacher(x).float()
            loss_ce = F.cross_entropy(logits, yb)
            loss_d = (1 - F.cosine_similarity(e_s, e_t, dim=1)).mean()
            loss = loss_ce + cfg["distill"] * loss_d
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss is not finite at epoch {ep} step {step}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if step % 100 == 0:
                _log(rank, f"[train] epoch {ep} step {step}/{steps} loss {loss.item():.3f} (arcface "
                           f"{loss_ce.item():.3f}, distill {loss_d.item():.4f}) lr {sched.get_last_lr()[0]:.2e} "
                           f"[{(time.time() - t0) / 60:.0f} min]")
        dist.barrier()
        if rank == 0:
            v = val_eval(student, X, val_idx, pos, neg, thr, dev)
            epoch_secs = time.time() - t_ep
            history.append(dict(epoch=ep, **v))
            _log(rank, f"[train] epoch {ep} done in {epoch_secs / 60:.1f} min: {v}")
            if v["tar_at_far_1e3"] > best:
                best, best_epoch = v["tar_at_far_1e3"], ep
                torch.save({k: t.detach().cpu() for k, t in student.state_dict().items()}, data / "best.pt")
            (data / "train_result.json").write_text(json.dumps(dict(best_epoch=best_epoch, history=history)),
                                                    encoding="utf-8")
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    train_main()
