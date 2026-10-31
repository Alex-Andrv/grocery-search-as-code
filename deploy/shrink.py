# -*- coding: utf-8 -*-
"""Ужать артефакты для доставки на VM (туннель с пода ~2 МБ/с).

  shrink.py delta <merged> <base> <out.safetensors> [rank]  — дельта merged−base: LoRA-слои → SVD ранга rank (A, B),
                                                              прочие изменённые тензоры целиком; печатает ошибку восстановления
  shrink.py apply <base> <delta.safetensors> <out_dir>      — base + дельта → полные веса (на VM)
  shrink.py fp16 <dir>                                      — model.safetensors → fp16 на месте
"""
import json, pathlib, shutil, sys, torch
from safetensors.torch import load_file, save_file


def shards(d):
    d = pathlib.Path(d)
    out = {}
    for f in sorted(d.glob("*.safetensors")):
        out.update(load_file(f))
    return out


def delta(merged, base, out, rank=32):
    M, B = shards(merged), shards(base)
    D, err = {}, []
    for k, w in M.items():
        b = B.get(k)
        if b is None:
            D[k + ".full"] = w.to(torch.bfloat16); continue
        d = w.float().cuda() - b.float().cuda()
        if d.abs().max() == 0:
            continue
        if d.ndim == 2 and min(d.shape) > rank * 2:
            U, S, Vh = torch.linalg.svd(d, full_matrices=False)
            A, Bm = (U[:, :rank] * S[:rank]), Vh[:rank]
            err.append(float((A @ Bm - d).norm() / d.norm()))
            D[k + ".A"], D[k + ".B"] = A.to(torch.bfloat16).contiguous().cpu(), Bm.to(torch.bfloat16).contiguous().cpu()
        else:
            D[k + ".full"] = w.to(torch.bfloat16)
    save_file(D, out)
    print(f"тензоров с дельтой: {len(D)}, макс. отн. ошибка SVD: {max(err) if err else 0:.2e}, "
          f"размер: {pathlib.Path(out).stat().st_size / 1e6:.0f} МБ")


def apply(base, dpath, out):
    W, D = shards(base), load_file(dpath)
    for k in {k.rsplit(".", 1)[0] for k in D}:
        if k + ".full" in D:
            W[k] = D[k + ".full"]
        else:
            W[k] = (W[k].float() + D[k + ".A"].float() @ D[k + ".B"].float()).to(W[k].dtype)
    o = pathlib.Path(out); o.mkdir(parents=True, exist_ok=True)
    for f in pathlib.Path(base).iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
            shutil.copy(f, o / f.name)
    save_file({k: v.contiguous() for k, v in W.items()}, o / "model.safetensors", metadata={"format": "pt"})


def fp16(d):
    p = pathlib.Path(d) / "model.safetensors"
    W = load_file(p)
    save_file({k: (v.half() if v.is_floating_point() else v) for k, v in W.items()}, p, metadata={"format": "pt"})


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "delta":
        delta(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 32)
    elif cmd == "apply":
        apply(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        fp16(sys.argv[2])
