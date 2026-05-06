from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from kut_v13_3_minimal import (
    ARCH_TAG,
    KUTv13Config,
    create_model,
    causal_grad,
    state_section,
)

CACHE_VERSION = "v13_3_minimal_tokens_v2_windowed"
GEOMETRIC_PARAM_KEYS = ("alpha_raw", "beta_raw", "vacuum_sq_raw")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    donor_model_path: str = "D:/models/qwen3-0.6b-base"
    init_from_checkpoint: str = ""
    resume_from: str = ""
    allow_arch_mismatch: bool = False

    output_dir: str = "C:/Users/12090/Desktop/kum_runs"
    run_name: str = ""

    dataset_name: str = "finemath-4plus"
    data_path: str = ""
    text_field: str = "text"
    cache_dir: str = "./data_cache"
    seq_len: int = 512
    stride: int = 256
    max_train_tokens: int = 100_000_000
    max_eval_tokens: int = 500_000
    data_skip_tokens: int = 0          # Skip first N tokens before taking this run's train window.
    eval_split: str = "validation"

    epochs: int = 1
    batch_size: int = 1
    grad_accum_steps: int = 4
    lr_network: float = 3e-4
    lr_geometric: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_grad_norm: float = 1.0
    freeze_embedding: bool = False
    use_bfloat16: bool = True

    log_every: int = 10
    eval_every: int = 2000
    eval_log_every: int = 2000
    save_every: int = 2000
    keep_last_n_checkpoints: int = 3
    stop_on_nonfinite: bool = True
    seed: int = 42
    num_workers: int = 0
    max_gpu_temp: int = 85
    check_gpu_every: int = 50
    quick_gen_tokens: int = 40


DATASET_REGISTRY = {
    "wikitext-2": {"hf_path": "wikitext", "hf_name": "wikitext-2-raw-v1", "splits": {"train": "train", "validation": "validation"}},
    "wikitext-103": {"hf_path": "wikitext", "hf_name": "wikitext-103-raw-v1", "splits": {"train": "train", "validation": "validation"}},
    "openwebmath": {"hf_path": "open-web-math/open-web-math", "hf_name": None, "splits": {"train": "train"}, "streaming": True},
    "finemath-4plus": {"hf_path": "HuggingFaceTB/finemath", "hf_name": "finemath-4plus", "splits": {"train": "train"}, "streaming": True},
    "local_jsonl": {"local": True, "format": "jsonl"},
    "local_txt": {"local": True, "format": "txt"},
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    return torch.device("cuda")


def parse_value(raw: str, current: Any) -> Any:
    if isinstance(current, bool):
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def parse_cli(obj):
    for arg in sys.argv[1:]:
        if "=" not in arg:
            continue
        k, v = arg.split("=", 1)
        if hasattr(obj, k):
            setattr(obj, k, parse_value(v, getattr(obj, k)))
            print(f"  Override: {k} = {getattr(obj, k)}")
    return obj


def load_pt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def tof(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().item())
    return float(x) if x is not None else 0.0


def ppl(ce: float) -> float:
    return math.exp(min(float(ce), 20.0))


def vram_mb() -> float:
    return torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def preserve_fp32(model: torch.nn.Module):
    for name, p in model.named_parameters():
        if any(k in name for k in GEOMETRIC_PARAM_KEYS) and p.dtype != torch.float32:
            p.data = p.data.float()


def cosine_lr(step: int, warmup: int, total: int, base: float) -> float:
    if step <= warmup:
        return base * step / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    prog = min(max(prog, 0.0), 1.0)
    return base * 0.5 * (1.0 + math.cos(math.pi * prog))


def update_lr(optimizer, step: int, warmup: int, total: int):
    lrs = {}
    for i, group in enumerate(optimizer.param_groups):
        lr = cosine_lr(step, warmup, total, group["base_lr"])
        group["lr"] = lr
        lrs[group.get("name", f"group{i}")] = lr
    return lrs


def tail_take_tokens(tokens: torch.Tensor, max_tokens: int) -> torch.Tensor:
    if max_tokens <= 0 or len(tokens) <= max_tokens:
        return tokens
    return tokens[-max_tokens:]


def head_take_tokens(tokens: torch.Tensor, max_tokens: int) -> torch.Tensor:
    """Take the first max_tokens while preserving stream/window order."""
    if max_tokens <= 0 or len(tokens) <= max_tokens:
        return tokens
    return tokens[:max_tokens]


def split_train_eval_window(tokens: torch.Tensor, cfg: TrainConfig, reserve_eval: bool = True):
    """Return non-overlapping train/eval windows after data_skip_tokens.

    Semantics for no-eval-split corpora:
      raw stream -> skip N -> train window -> eval window.

    This guarantees each run can train on fresh data by increasing data_skip_tokens,
    and guarantees eval tokens are not a subset of train tokens.
    """
    if cfg.data_skip_tokens > 0:
        if len(tokens) <= cfg.data_skip_tokens + cfg.seq_len:
            raise ValueError(
                f"data_skip_tokens={cfg.data_skip_tokens:,} leaves too few tokens "
                f"from loaded window len={len(tokens):,}"
            )
        print(f"  Skipping first {cfg.data_skip_tokens:,} tokens -> new train/eval window")
        tokens = tokens[cfg.data_skip_tokens:]

    train_cap = cfg.max_train_tokens if cfg.max_train_tokens > 0 else len(tokens)
    eval_cap = cfg.max_eval_tokens if cfg.max_eval_tokens > 0 else max(cfg.seq_len, len(tokens) // 20)

    if not reserve_eval:
        train_t = head_take_tokens(tokens, train_cap).contiguous()
        if len(train_t) < cfg.seq_len:
            raise ValueError(f"Train window too small: {len(train_t):,} tokens for seq_len={cfg.seq_len}")
        return train_t, torch.empty(0, dtype=torch.long)

    train_end = min(train_cap, len(tokens))
    eval_end = min(train_end + eval_cap, len(tokens))
    if eval_end - train_end < cfg.seq_len:
        eval_size = min(eval_cap, max(cfg.seq_len, len(tokens) // 20))
        if len(tokens) <= eval_size + cfg.seq_len:
            raise ValueError(
                f"Too few tokens ({len(tokens):,}) to create non-overlapping train/eval "
                f"windows with seq_len={cfg.seq_len}"
            )
        train_end = min(train_cap, len(tokens) - eval_size)
        eval_end = train_end + eval_size

    train_t = tokens[:train_end].contiguous()
    eval_t = tokens[train_end:eval_end].contiguous()
    if len(train_t) < cfg.seq_len or len(eval_t) < cfg.seq_len:
        raise ValueError(
            f"Non-overlapping split too small: train={len(train_t):,}, "
            f"eval={len(eval_t):,}, seq_len={cfg.seq_len}"
        )
    return train_t, eval_t


def has_nonfinite_parameters(model: torch.nn.Module) -> bool:
    for p in model.parameters():
        if not torch.isfinite(p).all():
            return True
    return False


def has_nonfinite_gradients(model: torch.nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return True
    return False


def move_optimizer_to_device(optimizer, device: torch.device):
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device=device)


def cache_identity(cfg: TrainConfig, tokenizer) -> str:
    tok_name = getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)
    local_meta = None
    if cfg.data_path:
        p = Path(cfg.data_path)
        if p.exists():
            st = p.stat()
            local_meta = {"path": str(p.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    payload = {
        "cache_version": CACHE_VERSION,
        "dataset_name": cfg.dataset_name,
        "data_path": cfg.data_path,
        "text_field": cfg.text_field,
        "seq_len": cfg.seq_len,
        "stride": cfg.stride,
        "max_train_tokens": cfg.max_train_tokens,
        "max_eval_tokens": cfg.max_eval_tokens,
        "data_skip_tokens": cfg.data_skip_tokens,
        "eval_split": cfg.eval_split,
        "tokenizer": tok_name,
        "local_meta": local_meta,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def get_hbar_oper(model) -> float:
    d = getattr(model, "_last_diagnostics", {})
    if "hbar_eff_oper" in d:
        return float(d["hbar_eff_oper"])
    h = getattr(model, "_last_hbar_eff_oper", None)
    return tof(h) if h is not None else float("nan")


def check_gpu_health(cfg: TrainConfig):
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return
        temp = int(result.stdout.strip().splitlines()[0])
        if temp > cfg.max_gpu_temp:
            print(f"\n⚠️ GPU {temp}°C > {cfg.max_gpu_temp}°C. Pausing 60s...")
            time.sleep(60)
    except Exception:
        pass


def nonfinite_stop(run_dir: Path, model, optimizer, tcfg: TrainConfig, mcfg: KUTv13Config,
                   step: int, best_eval: float, tok_seen: int, reason: str, extra: Dict[str, Any]):
    meta = {
        "reason": reason,
        "step": int(step),
        "tokens_seen": int(tok_seen),
        "best_eval_loss": float(best_eval),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        **extra,
    }
    with open(run_dir / "nonfinite_stop.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    save_ckpt(run_dir / f"nonfinite_stop_step{step}.pt", model, optimizer, step, mcfg, tcfg, best_eval, tok_seen)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChunkDataset(Dataset):
    def __init__(self, ids: torch.Tensor, seq_len: int, stride: int):
        self.ids = ids.contiguous()
        self.seq_len = seq_len
        self.stride = stride
        self.n = max(len(range(0, max(len(ids) - seq_len + 1, 0), stride)), 0)
        if self.n == 0:
            raise ValueError(f"Too few tokens ({len(ids)}) for seq_len={seq_len}")

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = i * self.stride
        ids = self.ids[s:s + self.seq_len]
        return {"input_ids": ids, "labels": ids.clone()}


def collate(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


def _tok_texts(texts, tokenizer, batch_size=2048):
    eos = tokenizer.eos_token_id
    chunks, total = [], 0
    for i in range(0, len(texts), batch_size):
        b = [t for t in texts[i:i + batch_size] if t and str(t).strip()]
        if not b:
            continue
        enc = tokenizer(b, add_special_tokens=False, padding=False, truncation=False)
        flat = []
        for ids in enc["input_ids"]:
            flat.extend(ids)
            if eos is not None:
                flat.append(eos)
        if flat:
            chunks.append(torch.tensor(flat, dtype=torch.long))
            total += len(flat)
        if (i // batch_size) % 50 == 0:
            print(f"    tokenized {min(i + batch_size, len(texts)):,}/{len(texts):,} -> {total:,} tokens")
    return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.long)


def _tok_stream(ds_iter, tokenizer, max_tok: int, text_field="text"):
    eos = tokenizer.eos_token_id
    chunks, total, batch = [], 0, []
    for item in ds_iter:
        text = item.get(text_field, "") if isinstance(item, dict) else ""
        if not text or not str(text).strip():
            continue
        batch.append(text)
        if len(batch) >= 256:
            enc = tokenizer(batch, add_special_tokens=False, padding=False, truncation=False)
            flat = []
            for ids in enc["input_ids"]:
                flat.extend(ids)
                if eos is not None:
                    flat.append(eos)
            if flat:
                chunks.append(torch.tensor(flat, dtype=torch.long))
                total += len(flat)
            batch = []
            if total % 1_000_000 < 256 * 50:
                print(f"    streaming: {total:,} tokens")
            if max_tok > 0 and total >= max_tok:
                break
    if batch:
        enc = tokenizer(batch, add_special_tokens=False, padding=False, truncation=False)
        flat = []
        for ids in enc["input_ids"]:
            flat.extend(ids)
            if eos is not None:
                flat.append(eos)
        if flat:
            chunks.append(torch.tensor(flat, dtype=torch.long))
    if not chunks:
        return torch.empty(0, dtype=torch.long)
    r = torch.cat(chunks)
    return head_take_tokens(r, max_tok) if max_tok > 0 else r


def load_tokens(tokenizer, cfg: TrainConfig):
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    reg = DATASET_REGISTRY.get(cfg.dataset_name)
    if reg is None:
        raise ValueError(f"Unknown dataset: {cfg.dataset_name}")

    ident = cache_identity(cfg, tokenizer)
    train_cache = cache_dir / f"{CACHE_VERSION}_{cfg.dataset_name}_{ident}_train.pt"
    eval_cache = cache_dir / f"{CACHE_VERSION}_{cfg.dataset_name}_{ident}_eval.pt"
    if train_cache.exists() and eval_cache.exists():
        print(f"  Loading cached tokens: {train_cache.name}")
        return {"train": load_pt(train_cache), "eval": load_pt(eval_cache)}

    if reg.get("local"):
        if not cfg.data_path:
            raise ValueError(f"dataset_name={cfg.dataset_name} requires data_path")
        if reg["format"] == "jsonl":
            texts = []
            with open(cfg.data_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        texts.append(json.loads(line).get(cfg.text_field, ""))
                    except Exception:
                        pass
            raw_tokens = _tok_texts([t for t in texts if t], tokenizer)
        else:
            with open(cfg.data_path, encoding="utf-8") as f:
                text = f.read()
            docs = [c.strip() for c in text.split("\n\n") if c.strip()]
            raw_tokens = _tok_texts(docs if len(docs) > 1 else [text], tokenizer)
        train_t, eval_t = split_train_eval_window(raw_tokens, cfg, reserve_eval=True)
    else:
        from datasets import load_dataset as hf_load
        hp, hn = reg["hf_path"], reg.get("hf_name")
        streaming = reg.get("streaming", False)
        text_field = reg.get("text_field", cfg.text_field)
        train_split = reg["splits"]["train"]
        eval_split = reg["splits"].get(cfg.eval_split, reg["splits"].get("validation", reg["splits"].get("test", "")))

        if streaming:
            need_eval_from_train = not bool(eval_split)
            fetch_total = cfg.max_train_tokens
            if need_eval_from_train:
                fetch_total = cfg.max_train_tokens + cfg.max_eval_tokens
            if cfg.data_skip_tokens > 0:
                fetch_total += cfg.data_skip_tokens
            print(f"  Streaming train window: skip={cfg.data_skip_tokens:,}, fetch={fetch_total:,}")
            raw_train = _tok_stream(
                hf_load(hp, hn, split=train_split, streaming=True, trust_remote_code=True),
                tokenizer,
                fetch_total,
                text_field,
            )
            if eval_split:
                train_t, _ = split_train_eval_window(raw_train, cfg, reserve_eval=False)
                eval_t = _tok_stream(
                    hf_load(hp, hn, split=eval_split, streaming=True, trust_remote_code=True),
                    tokenizer,
                    cfg.max_eval_tokens,
                    text_field,
                ).contiguous()
            else:
                train_t, eval_t = split_train_eval_window(raw_train, cfg, reserve_eval=True)
        else:
            ds = hf_load(hp, hn, split=train_split, trust_remote_code=True)
            raw_train = _tok_texts(ds[text_field], tokenizer)
            if eval_split:
                train_t, _ = split_train_eval_window(raw_train, cfg, reserve_eval=False)
                eds = hf_load(hp, hn, split=eval_split, trust_remote_code=True)
                eval_t = head_take_tokens(_tok_texts(eds[text_field], tokenizer), cfg.max_eval_tokens).contiguous()
            else:
                train_t, eval_t = split_train_eval_window(raw_train, cfg, reserve_eval=True)

    torch.save(train_t, train_cache)
    torch.save(eval_t, eval_cache)
    print(f"  Cached train={len(train_t):,} -> {train_cache.name}")
    print(f"  Cached eval={len(eval_t):,} -> {eval_cache.name}")
    print("  Data windows are non-overlapping; use data_skip_tokens to advance to fresh data.")
    return {"train": train_t, "eval": eval_t}


# ---------------------------------------------------------------------------
# Readout probe for v13.3
# ---------------------------------------------------------------------------

def install_readout_probe(model):
    """Attach a v13-compatible probe without changing the objective.

    The model already logs RMS-level readout diagnostics. The probe adds target-token
    flow-vs-state contribution and section stasis metrics for audit.
    """
    readout = model.readout

    def wrapped(self, h, z_a, z_b, section_amp, hbar):
        h_re = self.proj_re(h)
        h_im = self.proj_im(h)
        psi_a, psi_b = state_section(h_re, h_im, z_a, z_b, section_amp)
        state_logits = F.linear(psi_a, self.phi_re) + F.linear(psi_b, self.phi_im)
        flow_a = causal_grad(psi_a)
        flow_b = causal_grad(psi_b)
        flow_norm = torch.sqrt(flow_a.pow(2) + flow_b.pow(2) + 1e-8).mean(-1, keepdim=True)
        flow_a_n = flow_a / (1.0 + flow_norm)
        flow_b_n = flow_b / (1.0 + flow_norm)
        flow_logits = F.linear(flow_a_n, self.xi_re) + F.linear(flow_b_n, self.xi_im)
        logits = (state_logits + flow_logits) / hbar.to(h.dtype).clamp_min(1e-6)
        diag = {
            "state_logit_rms": state_logits.detach().float().pow(2).mean().sqrt(),
            "flow_logit_rms": flow_logits.detach().float().pow(2).mean().sqrt(),
            "flow_logit_share": flow_logits.detach().float().pow(2).mean().sqrt() / (state_logits.detach().float().pow(2).mean().sqrt() + flow_logits.detach().float().pow(2).mean().sqrt() + 1e-8),
            "flow_section_speed": torch.sqrt(flow_a_n.detach().float().pow(2) + flow_b_n.detach().float().pow(2) + 1e-8).mean(),
        }
        with torch.no_grad():
            model._last_probe = {
                "psi_a": psi_a.detach(),
                "psi_b": psi_b.detach(),
                "flow_a": flow_a_n.detach(),
                "flow_b": flow_b_n.detach(),
                **diag,
                "state_norm_mean": torch.sqrt(psi_a.detach().float().pow(2) + psi_b.detach().float().pow(2) + 1e-8).mean(),
                "flow_norm_mean": torch.sqrt(flow_a_n.detach().float().pow(2) + flow_b_n.detach().float().pow(2) + 1e-8).mean(),
            }
        return logits, psi_a, psi_b, diag

    readout.forward = MethodType(wrapped, readout)


def compute_probe_metrics(model, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    probe = getattr(model, "_last_probe", None)
    zero = logits.new_zeros(())
    out = {
        "top1_prob_mean": zero,
        "logit_stasis_mean": zero,
        "tangent_speed_mean": zero,
        "tangent_stasis_mean": zero,
        "state_logit_rms": zero,
        "flow_logit_rms": zero,
        "flow_logit_share": zero,
        "state_norm_mean": zero,
        "flow_norm_mean": zero,
        "flow_norm_share": zero,
        "target_state_contrib": zero,
        "target_flow_contrib": zero,
        "target_flow_share": zero,
    }
    if probe is None:
        return out

    with torch.no_grad():
        shift_logits = logits[:, :-1].detach()
        shift_labels = labels[:, 1:]
        valid = shift_labels.ne(-100)
        out["state_logit_rms"] = probe["state_logit_rms"]
        out["flow_logit_rms"] = probe["flow_logit_rms"]
        out["flow_logit_share"] = probe["flow_logit_share"]
        out["state_norm_mean"] = probe["state_norm_mean"]
        out["flow_norm_mean"] = probe["flow_norm_mean"]
        out["flow_norm_share"] = probe["flow_norm_mean"] / (probe["state_norm_mean"] + probe["flow_norm_mean"] + 1e-8)
        if shift_logits.shape[1] < 2 or not valid.any():
            return out

        pair_valid = valid[:, 1:] & valid[:, :-1]
        l1 = torch.topk(shift_logits.float(), k=1, dim=-1).values[..., 0]
        logZ = torch.logsumexp(shift_logits.float(), dim=-1)
        p1 = torch.exp(l1 - logZ).clamp(0.0, 1.0)
        out["top1_prob_mean"] = p1.masked_select(valid).mean()

        if pair_valid.any():
            logit_cos = F.cosine_similarity(shift_logits[:, 1:].float(), shift_logits[:, :-1].float(), dim=-1, eps=1e-8).clamp(-1.0, 1.0)
            out["logit_stasis_mean"] = (0.5 * (1.0 + logit_cos)).masked_select(pair_valid).mean()

        psi_a = probe["psi_a"][:, :-1]
        psi_b = probe["psi_b"][:, :-1]
        flow_a = probe["flow_a"][:, :-1]
        flow_b = probe["flow_b"][:, :-1]
        psi = torch.cat([psi_a, psi_b], dim=-1)
        if psi.shape[1] >= 2 and pair_valid.any():
            psi_prev = psi[:, :-1, :]
            psi_next = psi[:, 1:, :]
            dpsi = psi_next - psi_prev
            prev_norm_sq = psi_prev.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-8)
            proj_coeff = (dpsi * psi_prev).sum(dim=-1, keepdim=True) / prev_norm_sq
            radial_comp = proj_coeff * psi_prev
            tang_sq = (dpsi.pow(2).sum(dim=-1) - radial_comp.pow(2).sum(dim=-1)).clamp_min(0.0)
            denom = torch.sqrt(psi_prev.pow(2).sum(dim=-1) + 1e-8) + torch.sqrt(psi_next.pow(2).sum(dim=-1) + 1e-8)
            tangent_speed = (torch.sqrt(tang_sq + 1e-8) / denom.clamp_min(1e-8)).clamp(0.0, 1.0)
            out["tangent_speed_mean"] = tangent_speed.masked_select(pair_valid).mean()
            out["tangent_stasis_mean"] = (1.0 - tangent_speed).masked_select(pair_valid).mean()

        safe = shift_labels.masked_fill(~valid, 0)
        phi_re = model.readout.phi_re[safe].to(psi_a.dtype)
        phi_im = model.readout.phi_im[safe].to(psi_a.dtype)
        xi_re = model.readout.xi_re[safe].to(psi_a.dtype)
        xi_im = model.readout.xi_im[safe].to(psi_a.dtype)
        state_target = (psi_a * phi_re + psi_b * phi_im).sum(dim=-1)
        flow_target = (flow_a * xi_re + flow_b * xi_im).sum(dim=-1)
        if valid.any():
            st = state_target.masked_select(valid).abs().mean()
            fl = flow_target.masked_select(valid).abs().mean()
            out["target_state_contrib"] = st
            out["target_flow_contrib"] = fl
            out["target_flow_share"] = fl / (st + fl + 1e-8)
    return out


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def _l2_grad_norm(params) -> float:
    total = None
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        sq = torch.sum(g * g)
        total = sq if total is None else total + sq
    if total is None:
        return 0.0
    return float(torch.sqrt(total).item())


def collect_train_observations(model) -> Dict[str, float]:
    geo_params, mlp_params, chart_params, attn_params, readout_params, norm_params = [], [], [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in GEOMETRIC_PARAM_KEYS):
            geo_params.append(p)
        if ".mlp." in name or name.endswith("mlp"):
            mlp_params.append(p)
        if any(k in name for k in ("to_chart", "from_chart", "init_chart")):
            chart_params.append(p)
        if any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj")):
            attn_params.append(p)
        if name.startswith("readout"):
            readout_params.append(p)
        if "norm" in name.lower():
            norm_params.append(p)

    alpha, beta, v_sq = model._abv()
    d = getattr(model, "_last_diagnostics", {})
    return {
        "grad_geometric": _l2_grad_norm(geo_params),
        "grad_mlp": _l2_grad_norm(mlp_params),
        "grad_chart": _l2_grad_norm(chart_params),
        "grad_attn": _l2_grad_norm(attn_params),
        "grad_readout": _l2_grad_norm(readout_params),
        "grad_norm_layers": _l2_grad_norm(norm_params),
        "alpha_mean": tof(F.softplus(model.alpha_raw).mean()),
        "beta_mean": tof(F.softplus(model.beta_raw).mean() + model.config.metric_floor),
        "vacuum_sq_mean": tof(F.softplus(model.vacuum_sq_raw).mean()),
        "hbar_eff_oper": get_hbar_oper(model),
        "omega_bar": tof(d.get("omega_bar", 0.0)),
        "anchor_loss": tof(d.get("anchor_loss", 0.0)),
        "ke_loss": tof(d.get("ke_loss", 0.0)),
        "prob_energy": tof(d.get("prob_energy", 0.0)),
        "flow_vitality": tof(d.get("flow_vitality", 0.0)),
    }


# ---------------------------------------------------------------------------
# Model setup / checkpoint
# ---------------------------------------------------------------------------

def build_groups(model, cfg: TrainConfig):
    geo, net, nodecay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in GEOMETRIC_PARAM_KEYS):
            geo.append(p)
        elif p.ndim < 2 or "norm" in name.lower():
            nodecay.append(p)
        else:
            net.append(p)
    groups = []
    if net:
        groups.append({"params": net, "lr": cfg.lr_network, "base_lr": cfg.lr_network, "weight_decay": cfg.weight_decay, "name": "network"})
    if nodecay:
        groups.append({"params": nodecay, "lr": cfg.lr_network, "base_lr": cfg.lr_network, "weight_decay": 0.0, "name": "nodecay"})
    if geo:
        groups.append({"params": geo, "lr": cfg.lr_geometric, "base_lr": cfg.lr_geometric, "weight_decay": 0.0, "name": "geometric"})
    return groups


def init_donor_embed(model, donor_path: str):
    if not donor_path:
        return
    from transformers import AutoModelForCausalLM
    print(f"  Loading donor embeddings from {donor_path}")
    donor = AutoModelForCausalLM.from_pretrained(donor_path, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
    de = donor.get_input_embeddings().weight.detach().cpu()
    te = model.embed_tokens.weight.data.cpu()
    nv, nd = min(de.shape[0], te.shape[0]), min(de.shape[1], te.shape[1])
    if de.shape != te.shape:
        print(f"  Warning: donor embedding shape {tuple(de.shape)} != model {tuple(te.shape)}; copying overlap {nv} x {nd}")
    te[:nv, :nd] = de[:nv, :nd]
    model.embed_tokens.weight.data.copy_(te.to(model.embed_tokens.weight.device, model.embed_tokens.weight.dtype))
    print(f"    Copied {nv:,} × {nd} embedding weights")
    del donor
    gc.collect()
    torch.cuda.empty_cache()


def save_ckpt(path: Path, model, optimizer, step: int, mcfg: KUTv13Config, tcfg: TrainConfig, best: float, tok_seen: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "arch_tag": ARCH_TAG,
        "step": step,
        "tokens_seen": tok_seen,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_cfg": asdict(mcfg),
        "train_cfg": asdict(tcfg),
        "best_eval_loss": best,
    }, path)


def maybe_prune_checkpoints(run_dir: Path, keep_last_n: int):
    ckpts = sorted(run_dir.glob("checkpoint_step*.pt"), key=os.path.getmtime)
    while len(ckpts) > keep_last_n:
        try:
            ckpts.pop(0).unlink()
        except OSError:
            pass


def load_model_cfg_from_checkpoint(path: str, fallback: KUTv13Config) -> KUTv13Config:
    if not path:
        return fallback
    p = Path(path)
    if not p.exists():
        return fallback
    try:
        ckpt = load_pt(p)
        cfg_dict = ckpt.get("model_cfg")
        if not isinstance(cfg_dict, dict):
            return fallback
        allowed = {f.name for f in fields(KUTv13Config)}
        filtered = {k: v for k, v in cfg_dict.items() if k in allowed}
        cfg = KUTv13Config(**filtered)
        cfg.__post_init__()
        print("  Loaded model_cfg from checkpoint")
        return cfg
    except Exception as e:
        print(f"  Warning: model_cfg restore failed: {e}")
        return fallback


# ---------------------------------------------------------------------------
# Eval / logs / generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, max_batches: Optional[int] = None):
    model.eval()
    sums: Dict[str, float] = {}
    n = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(**batch)
        ce = tof(getattr(model, "_last_ce_loss", 0.0))
        metrics = {
            "loss": tof(out.loss),
            "ce": ce,
            "ppl": ppl(ce),
            "geo_diag": tof(getattr(model, "_last_geo", 0.0)),
            "info_align": tof(getattr(model, "_last_information_alignment", 0.0)),
            "info_norm": tof(getattr(model, "_last_information_state_norm", 0.0)),
            "hbar": get_hbar_oper(model),
        }
        for k, v in getattr(model, "_last_diagnostics", {}).items():
            metrics[k] = tof(v)
        for k, v in compute_probe_metrics(model, out.logits, batch["labels"]).items():
            metrics[k] = tof(v)
        for k, v in metrics.items():
            sums[k] = sums.get(k, 0.0) + v
        n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in sums.items()}


def write_eval_log(path: Path, step: int, eval_metrics: Dict[str, float], obs: Dict[str, float], best_eval_loss: float, tokens_seen: int):
    payload = {"step": step, "tokens_seen": tokens_seen, "best_eval_loss": best_eval_loss, **eval_metrics, **obs}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def quick_generation_test(model, tokenizer, device, max_new_tokens: int = 40):
    prompts = [
        "Albert Einstein was a",
        "The capital of France is",
        "The derivative of x squared is",
        "Newton's second law states that",
        "The integral of sin(x) is",
        "1 + 1 =",
    ]
    model.eval()
    print("\n  Generation test (greedy, no cache)")
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        generated = ids.clone()
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = model(input_ids=generated, labels=None, use_cache=False).logits
            next_token = logits[0, -1].argmax()
            generated = torch.cat([generated, next_token.view(1, 1)], dim=-1)
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"    '{text[:140]}'")
    model.train()


# ---------------------------------------------------------------------------
# Main train loop
# ---------------------------------------------------------------------------

def main():
    tcfg = parse_cli(TrainConfig())
    mcfg = parse_cli(KUTv13Config())
    mcfg.__post_init__()
    if tcfg.resume_from:
        mcfg = load_model_cfg_from_checkpoint(tcfg.resume_from, mcfg)

    set_seed(tcfg.seed)
    device = get_device()
    torch.cuda.reset_peak_memory_stats(device)

    run_name = tcfg.run_name or f"kutv13_3_minimal_{tcfg.dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(tcfg.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_log_dir = run_dir / "eval_logs"
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"

    print("=" * 78)
    print("  KUT v13.3 Minimal Unified Kähler Transformer — Training")
    print(f"  Run:     {run_name}")
    print(f"  Dataset: {tcfg.dataset_name}")
    print(f"  Layers:  {mcfg.num_hidden_layers} | heads={mcfg.n_heads}/{mcfg.n_kv_heads} | rank={mcfg.complex_rank}")
    print("=" * 78)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tcfg.donor_model_path or "Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokens = load_tokens(tokenizer, tcfg)
    train_ds = ChunkDataset(tokens["train"], tcfg.seq_len, tcfg.stride)
    eval_ds = ChunkDataset(tokens["eval"], tcfg.seq_len, tcfg.stride)
    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True, pin_memory=True, collate_fn=collate,
                              drop_last=True, num_workers=tcfg.num_workers)
    eval_loader = DataLoader(eval_ds, batch_size=tcfg.batch_size, shuffle=False, pin_memory=True, collate_fn=collate,
                             drop_last=False, num_workers=tcfg.num_workers)

    model = create_model(mcfg, device=device)
    install_readout_probe(model)
    if tcfg.donor_model_path and not tcfg.init_from_checkpoint and not tcfg.resume_from:
        init_donor_embed(model, tcfg.donor_model_path)
    if tcfg.init_from_checkpoint and not tcfg.resume_from:
        st = load_pt(Path(tcfg.init_from_checkpoint))
        ck_arch = st.get("arch_tag") if isinstance(st, dict) else None
        if ck_arch and ck_arch != ARCH_TAG and not tcfg.allow_arch_mismatch:
            raise ValueError(f"checkpoint arch_tag={ck_arch!r} != expected {ARCH_TAG!r}")
        state = st.get("model_state_dict", st) if isinstance(st, dict) else st
        model.load_state_dict(state, strict=not tcfg.allow_arch_mismatch)
    if tcfg.freeze_embedding:
        model.embed_tokens.weight.requires_grad_(False)
    if tcfg.use_bfloat16:
        model = model.to(dtype=torch.bfloat16)
        preserve_fp32(model)

    nt = sum(p.numel() for p in model.parameters())
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ng = sum(p.numel() for n, p in model.named_parameters() if any(k in n for k in GEOMETRIC_PARAM_KEYS))
    print(f"\n  Params: {nt:,} total | {ntr:,} trainable | {ng:,} geometric scalars")
    print(f"  Tokens: {len(tokens['train']):,} train | {len(tokens['eval']):,} eval")
    print(f"  Batch: {tcfg.batch_size} × {tcfg.grad_accum_steps} = {tcfg.batch_size * tcfg.grad_accum_steps}")
    print(f"  VRAM after load: {vram_mb():.0f} MB\n")

    optimizer = torch.optim.AdamW(build_groups(model, tcfg), betas=(0.9, 0.95), eps=1e-8)
    global_step, best_eval, tok_seen = 0, float("inf"), 0

    if tcfg.resume_from:
        ckpt = load_pt(Path(tcfg.resume_from))
        ck_arch = ckpt.get("arch_tag")
        if ck_arch != ARCH_TAG and not tcfg.allow_arch_mismatch:
            raise ValueError(f"checkpoint arch_tag={ck_arch!r} != expected {ARCH_TAG!r}")
        if ck_arch != ARCH_TAG and tcfg.allow_arch_mismatch:
            print(f"  Warning: checkpoint arch_tag={ck_arch!r} != expected {ARCH_TAG!r}; force-loading")
        model.load_state_dict(ckpt["model_state_dict"], strict=not tcfg.allow_arch_mismatch)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            move_optimizer_to_device(optimizer, device)
        global_step = int(ckpt.get("step", 0))
        best_eval = float(ckpt.get("best_eval_loss", float("inf")))
        tok_seen = int(ckpt.get("tokens_seen", 0))
        print(f"  Resumed from step {global_step} | best_eval={best_eval:.4f} | tokens_seen={tok_seen:,}")

    micro_per_epoch = len(train_loader)
    opt_steps_per_epoch = math.ceil(micro_per_epoch / tcfg.grad_accum_steps)
    total_opt_steps = opt_steps_per_epoch * tcfg.epochs
    tok_per_step = tcfg.batch_size * tcfg.grad_accum_steps * tcfg.seq_len
    print(f"  Steps/epoch: {opt_steps_per_epoch:,} | Total: {total_opt_steps:,} | Tok/step: {tok_per_step:,}\n")

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"arch_tag": ARCH_TAG, "model_cfg": asdict(mcfg), "train_cfg": asdict(tcfg)}, f, indent=2)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    micro = 0
    accum = 0
    ce_sum = 0.0
    n_log = 0
    last_obs: Dict[str, float] = {}
    log_history: List[Dict[str, Any]] = []

    for epoch in range(tcfg.epochs):
        for batch_idx, batch in enumerate(train_loader):
            micro += 1
            accum += 1
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            if loss is None:
                raise RuntimeError("loss is None")
            if tcfg.stop_on_nonfinite and not torch.isfinite(loss):
                print(f"  ⚠ Non-finite loss at micro {micro}")
                nonfinite_stop(run_dir, model, optimizer, tcfg, mcfg, global_step, best_eval, tok_seen, "nonfinite_forward", {"batch_index": batch_idx})
                return

            (loss / tcfg.grad_accum_steps).backward()
            ce_val = tof(getattr(model, "_last_ce_loss", 0.0))
            ce_sum += ce_val
            n_log += 1

            is_last_batch = (batch_idx + 1 == len(train_loader))
            if accum < tcfg.grad_accum_steps and not is_last_batch:
                continue

            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], tcfg.max_grad_norm)
            if tcfg.stop_on_nonfinite and has_nonfinite_gradients(model):
                print(f"  ⚠ Non-finite gradients at step {global_step + 1}")
                nonfinite_stop(run_dir, model, optimizer, tcfg, mcfg, global_step, best_eval, tok_seen, "nonfinite_gradient", {"batch_index": batch_idx})
                return

            # Collect gradient norms BEFORE step/zero_grad (otherwise set_to_none=True erases them)
            _cached_grad_obs = collect_train_observations(model) if ((global_step + 1) % tcfg.log_every == 0) else None

            next_step = global_step + 1
            lrs = update_lr(optimizer, next_step, tcfg.warmup_steps, total_opt_steps)
            optimizer.step()
            if tcfg.stop_on_nonfinite and has_nonfinite_parameters(model):
                print(f"  ⚠ Non-finite parameter detected after step {next_step}")
                nonfinite_stop(run_dir, model, optimizer, tcfg, mcfg, next_step, best_eval, tok_seen, "nonfinite_parameter", {"batch_index": batch_idx})
                return
            optimizer.zero_grad(set_to_none=True)
            preserve_fp32(model)
            global_step = next_step
            tok_seen += tcfg.batch_size * accum * tcfg.seq_len
            accum = 0

            if global_step % tcfg.log_every == 0:
                obs = _cached_grad_obs if _cached_grad_obs is not None else collect_train_observations(model)
                probe = compute_probe_metrics(model, out.logits, batch["labels"])
                d = getattr(model, "_last_diagnostics", {})
                last_obs = obs
                ce_avg = ce_sum / max(n_log, 1)
                ce_sum, n_log = 0.0, 0
                elapsed = time.time() - t0
                tps = tok_seen / max(elapsed, 1)
                eta = fmt_time((total_opt_steps - global_step) / max(global_step / max(elapsed, 1), 1e-6))
                log = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "tokens_seen": tok_seen,
                    "loss": tof(loss),
                    "ce_loss": ce_val,
                    "train_ppl": ppl(ce_avg),
                    "geo_diag": tof(getattr(model, "_last_geo", 0.0)),
                    "info_align": tof(getattr(model, "_last_information_alignment", 0.0)),
                    "info_norm": tof(getattr(model, "_last_information_state_norm", 0.0)),
                    **{k: tof(v) for k, v in d.items()},
                    **{k: tof(v) for k, v in probe.items()},
                    **obs,
                    "lr_network": lrs.get("network", 0.0),
                    "lr_geometric": lrs.get("geometric", 0.0),
                    "vram_mb": vram_mb(),
                    "elapsed": elapsed,
                    "tokens_per_sec": tps,
                }
                log_history.append(log)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log) + "\n")
                print(
                    f"[E{epoch+1}] {global_step:5d}/{total_opt_steps} | "
                    f"CE {ce_avg:.4f} PPL {ppl(ce_avg):.2f} | loss {tof(loss):.4f} | "
                    f"Anc {tof(d.get('anchor_loss',0)):.4f} KE {tof(d.get('ke_loss',0)):.4f} ProbE {tof(d.get('prob_energy',0)):.4f} | "
                    f"ω̄ {obs.get('omega_bar',0):.4f} ℏ {obs.get('hbar_eff_oper',0):.4f} | "
                    f"FlowShare {tof(probe.get('flow_logit_share',0)):.3f} TargetFlow {tof(probe.get('target_flow_share',0)):.3f} | "
                    f"logitSt {tof(probe.get('logit_stasis_mean',0)):.3f} tanSp {tof(probe.get('tangent_speed_mean',0)):.3f} | "
                    f"α {obs.get('alpha_mean',0):.5f} β {obs.get('beta_mean',0):.4f} v² {obs.get('vacuum_sq_mean',0):.4f} | "
                    f"|∇geo| {obs.get('grad_geometric',0):.2e} |∇chart| {obs.get('grad_chart',0):.2e} |∇attn| {obs.get('grad_attn',0):.2e} |∇read| {obs.get('grad_readout',0):.2e} | "
                    f"lr {lrs.get('network',0):.1e}/{lrs.get('geometric',0):.1e} | {tps:.0f} t/s | {vram_mb():.0f}MB | ETA {eta}"
                )

            if global_step % tcfg.eval_every == 0:
                em = evaluate(model, eval_loader, device)
                print(
                    f"  >>> EVAL {global_step}: loss={em.get('loss',0):.4f} CE={em.get('ce',0):.4f} PPL={em.get('ppl',0):.2f} | "
                    f"Anc={em.get('anchor_loss',0):.4f} KE={em.get('ke_loss',0):.4f} | "
                    f"FlowShare={em.get('flow_logit_share',0):.3f} TargetFlow={em.get('target_flow_share',0):.3f} | "
                    f"logitSt={em.get('logit_stasis_mean',0):.3f} tanSp={em.get('tangent_speed_mean',0):.3f}"
                )
                if not all(math.isfinite(float(v)) for v in em.values()):
                    print(f"  ⚠ Non-finite eval metric at step {global_step}")
                    if tcfg.stop_on_nonfinite:
                        nonfinite_stop(run_dir, model, optimizer, tcfg, mcfg, global_step, best_eval, tok_seen, "nonfinite_eval", {})
                        return
                if em.get("loss", float("inf")) < best_eval:
                    best_eval = em["loss"]
                    torch.save({
                        "arch_tag": ARCH_TAG,
                        "step": global_step,
                        "tokens_seen": tok_seen,
                        "model_state_dict": model.state_dict(),
                        "model_cfg": asdict(mcfg),
                        "best_eval_loss": best_eval,
                    }, run_dir / "best_weights.pt")
                    print(f"  >>> New best: {best_eval:.4f}")
                if global_step % max(tcfg.eval_log_every, 1) == 0:
                    write_eval_log(eval_log_dir / f"eval_step{global_step}.json", global_step, em, last_obs, best_eval, tok_seen)

            if global_step % tcfg.save_every == 0:
                cp = run_dir / f"checkpoint_step{global_step}.pt"
                save_ckpt(cp, model, optimizer, global_step, mcfg, tcfg, best_eval, tok_seen)
                maybe_prune_checkpoints(run_dir, tcfg.keep_last_n_checkpoints)

            if tcfg.check_gpu_every > 0 and global_step % tcfg.check_gpu_every == 0:
                check_gpu_health(tcfg)

    final = evaluate(model, eval_loader, device)
    print(
        f"\nFinal: loss={final.get('loss',0):.4f} CE={final.get('ce',0):.4f} PPL={final.get('ppl',0):.2f} | "
        f"FlowShare={final.get('flow_logit_share',0):.3f} TargetFlow={final.get('target_flow_share',0):.3f} | "
        f"logitSt={final.get('logit_stasis_mean',0):.3f} tanSp={final.get('tangent_speed_mean',0):.3f}"
    )
    with open(run_dir / "eval_final.json", "w", encoding="utf-8") as f:
        json.dump({"step": global_step, "tokens_seen": tok_seen, **final, **last_obs}, f, indent=2)
    with open(run_dir / "log_history.json", "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2)
    save_ckpt(run_dir / "final_checkpoint.pt", model, optimizer, global_step, mcfg, tcfg, best_eval, tok_seen)
    quick_generation_test(model, tokenizer, device, max_new_tokens=tcfg.quick_gen_tokens)
    print(f"\n✅ Done. Results in {run_dir}")


if __name__ == "__main__":
    main()
