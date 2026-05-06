"""
KUT Generation Diagnostic — 信息流诊断
=======================================

Training-time info flow:
  Input:  "Albert Einstein was a physicist who"
  Target: "lbert Einstein was a physicist who ..."
  → Model predicts next token at EVERY position simultaneously
  → Loss computed at ALL positions
  → Field (s, θ) evolves through layers, seeing full causal context

Generation-time info flow (standard):
  Step 1: model("Albert Einstein was a") → take logits[-1] → "physicist"
  Step 2: model("Albert Einstein was a physicist") → take logits[-1] → ???
  → Model reprocesses entire sequence from scratch
  → Only last position's logits used

This script diagnoses WHERE the breakdown happens:
  Test 1: Teacher-forced prediction (as in training) — does the model predict correctly?
  Test 2: Autoregressive with linear head — is the backbone OK but readout broken?
  Test 3: Per-position logit analysis — are consecutive positions too similar?
  Test 4: Standard autoregressive generation
  Test 5: v11.1 field state reset test
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer


HBAR_STAR = math.e
OMEGA_K_STAR = math.pi / math.e


# ---------------------------------------------------------------------------
# Try importing v11.1, v12.3, and v13.3
# ---------------------------------------------------------------------------
V11_AVAILABLE = False
V12_AVAILABLE = False
V13_AVAILABLE = False

try:
    from kum_v11_1_onepass_cleaned import KUMV7Config, KUMV7ForCausalLM
    V11_AVAILABLE = True
except ImportError:
    pass

try:
    from kum_v12_3_kahler_spectral_fixed import KUTv123Config, KUTv123System
    V12_AVAILABLE = True
except ImportError:
    pass

try:
    from kut_v13_3_minimal import KUTv13Config, KUTv13System, causal_grad, state_section
    V13_AVAILABLE = True
except ImportError:
    pass


def token_entropy(logits):
    p = F.softmax(logits, dim=-1)
    lp = F.log_softmax(logits, dim=-1)
    return -float((p * lp).sum().item())


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model_cfg = ckpt.get("model_cfg", {})
    state = ckpt.get("model_state_dict", ckpt)

    # Detect architecture from state dict keys
    keys = set(state.keys())
    is_v13 = any("layers.0.self_attn" in k for k in keys) and any("init_chart" in k for k in keys) and any("alpha_raw" in k for k in keys)
    is_v12 = any("encoder_layers" in k or "cognition_layers" in k for k in keys)
    is_v11 = any("model.layers." in k for k in keys)
    detected = "v13.3" if is_v13 else "v12.3" if is_v12 else "v11.1" if is_v11 else "unknown"
    print(f"  Detected: {detected} (from state_dict keys)")

    if is_v13 and V13_AVAILABLE:
        allowed = {f.name for f in KUTv13Config.__dataclass_fields__.values()}
        filtered = {k: v for k, v in model_cfg.items() if k in allowed}
        cfg = KUTv13Config(**filtered)
        cfg.__post_init__()
        model = KUTv13System(cfg)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  WARNING: {len(missing)} missing keys")
        if unexpected:
            print(f"  WARNING: {len(unexpected)} unexpected keys")
        model = model.to(device=device, dtype=torch.bfloat16)
        model.eval()
        print(f"  Loaded v13.3 model, step {ckpt.get('step', '?')}")
        return model, "v13.3"

    if is_v12 and V12_AVAILABLE:
        allowed = {f.name for f in KUTv123Config.__dataclass_fields__.values()}
        filtered = {k: v for k, v in model_cfg.items() if k in allowed}
        cfg = KUTv123Config(**filtered)
        cfg.__post_init__()
        model = KUTv123System(cfg)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  WARNING: {len(missing)} missing keys")
        if unexpected:
            print(f"  WARNING: {len(unexpected)} unexpected keys")
        model = model.to(device=device, dtype=torch.bfloat16)
        model.eval()
        print(f"  Loaded v12.3 model, step {ckpt.get('step', '?')}")
        return model, "v12.3"

    if is_v11 and V11_AVAILABLE:
        allowed = {f.name for f in KUMV7Config.__dataclass_fields__.values()}
        filtered = {k: v for k, v in model_cfg.items() if k in allowed}
        cfg = KUMV7Config(**filtered)
        cfg.__post_init__()
        model = KUMV7ForCausalLM(cfg)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  WARNING: {len(missing)} missing keys")
        if unexpected:
            print(f"  WARNING: {len(unexpected)} unexpected keys")
        model = model.to(device=device, dtype=torch.bfloat16)
        model.eval()
        print(f"  Loaded v11.1 model, step {ckpt.get('step', '?')}")
        return model, "v11.1"

    raise RuntimeError(f"Cannot detect architecture. is_v13={is_v13}, is_v12={is_v12}, is_v11={is_v11}")


def reset_field_states(model, arch):
    """Reset persistent field states (critical for autoregressive generation)."""
    if arch == "v11.1":
        for layer in model.model.layers:
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                if hasattr(attn, 'field_state'):
                    attn.field_state.shell_s = None
                    attn.field_state.theta = None
    elif arch == "v13.3":
        # v13.3 has no persistent field state — (s, θ) initialized from h each forward pass
        pass


# ===========================================================================
# Test 1: Teacher-forced prediction — 训练时的信息流
# ===========================================================================
@torch.no_grad()
def test_teacher_forcing(model, arch, tokenizer, device):
    """Run model on a full sequence and check predictions at EACH position.
    This is EXACTLY how the model was trained. If predictions are good here
    but bad in autoregressive mode, the issue is in the generation loop."""

    print("\n" + "=" * 70)
    print("TEST 1: Teacher-Forced Prediction (training-time info flow)")
    print("=" * 70)

    texts = [
        "Albert Einstein was a theoretical physicist who developed the theory of relativity",
        "The capital of France is Paris, which is known for the Eiffel Tower",
        "In mathematics, the derivative of x squared is two times x",
        "Let f(x) = x^2 + 3x + 1. Then f'(x) = 2x + 3",
        "The integral of sin(x) dx equals negative cos(x) plus a constant",
        "If a triangle has sides 3, 4 and 5, then it is a right triangle",
    ]

    for text in texts:
        ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        T = ids.shape[1]

        reset_field_states(model, arch)
        out = model(input_ids=ids, use_cache=False)
        logits = out.logits[0]  # [T, V]

        print(f"\n  Input: \"{text}\"")
        print(f"  {'Pos':>4} | {'Input':>12} | {'Predicted':>12} | {'Target':>12} | {'Match':>5} | {'Top1_p':>7} | {'H(nats)':>7}")
        print("  " + "-" * 80)

        n_correct = 0
        for t in range(T - 1):
            input_tok = tokenizer.decode([ids[0, t].item()])
            pred_id = logits[t].argmax().item()
            pred_tok = tokenizer.decode([pred_id])
            target_id = ids[0, t + 1].item()
            target_tok = tokenizer.decode([target_id])
            match = pred_id == target_id
            if match:
                n_correct += 1
            top1_p = F.softmax(logits[t], dim=-1).max().item()
            H = token_entropy(logits[t])
            marker = "✓" if match else "✗"
            print(f"  {t:>4} | {repr(input_tok):>12} | {repr(pred_tok):>12} | {repr(target_tok):>12} | {marker:>5} | {top1_p:>7.4f} | {H:>7.2f}")

        acc = n_correct / (T - 1)
        print(f"  Accuracy: {n_correct}/{T-1} = {acc:.1%}")

        # Logit stasis: cosine similarity between consecutive positions
        cos_sims = []
        for t in range(T - 2):
            cs = F.cosine_similarity(logits[t].unsqueeze(0), logits[t+1].unsqueeze(0)).item()
            cos_sims.append(cs)
        if cos_sims:
            print(f"  Logit stasis (mean cos_sim): {sum(cos_sims)/len(cos_sims):.4f}")


# ===========================================================================
# Test 2: Linear head comparison — 是 backbone 还是 readout 的问题?
# ===========================================================================
@torch.no_grad()
def test_linear_head(model, arch, tokenizer, device):
    """Replace section readout with a standard linear head (h @ W_embed^T).
    If this generates but section readout doesn't, the problem is isolated to readout."""

    print("\n" + "=" * 70)
    print("TEST 2: Linear Head Comparison (bypass section readout)")
    print("=" * 70)

    prompts = [
        "Albert Einstein was a",
        "The capital of France is",
        "1 + 1 =",
        "The derivative of x squared is",
        "Proof. Let x be a real number such that",
    ]

    # Get embedding weights
    if arch == "v11.1":
        embed_weight = model.model.embed_tokens.weight  # [V, H]
    else:
        embed_weight = model.embed_tokens.weight

    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        gen = ids.clone()

        for _ in range(40):
            reset_field_states(model, arch)
            out = model(input_ids=gen, use_cache=False)

            # Get last hidden state BEFORE readout
            if arch == "v11.1":
                # Re-run to get hidden states
                with torch.no_grad():
                    if hasattr(model.model, 'embed_tokens'):
                        h = model.model.embed_tokens(gen)
                        for layer in model.model.layers:
                            layer_out = layer(h)
                            h = layer_out[0]
                        h = model.model.norm(h)
                    else:
                        h = out.logits  # fallback
                        break
            elif arch == "v13.3":
                # v13.3: run embed → layers → final_norm, skip readout
                h = model.embed_tokens(gen)
                alpha, beta, v_sq = model._abv()
                R = model.config.complex_rank
                chart_init = model.init_chart(h)
                s = torch.tanh(chart_init[..., :R].float())
                theta = math.pi * torch.tanh(chart_init[..., R:].float())
                b_sz, t_len = h.shape[:2]
                position_ids = torch.arange(t_len, device=device).unsqueeze(0).expand(b_sz, -1)
                # Compute hbar and omega_bar from bath
                omega_bath = torch.sqrt((2.0 * alpha * v_sq + beta) / v_sq.clamp_min(1e-8)).mean()
                hbar = h.new_tensor(math.pi) / omega_bath.detach().clamp_min(1e-6)
                for layer in model.layers:
                    lo = layer(h, s, theta, alpha, beta, v_sq, hbar, omega_bath.detach(), position_ids)
                    h, s, theta = lo["h"], lo["s"], lo["theta"]
                from kut_v13_3_minimal import kahler_bundle, chart_to_z
                z_a, z_b, _ = chart_to_z(s, theta, v_sq)
                geo_f = kahler_bundle(z_a, z_b, alpha, beta, v_sq)
                g_f = geo_f["g_ratio"].mean(-1, keepdim=True).to(h.dtype)
                h = model.final_norm(h, g_f)
            else:
                # For v12.3, get h from the last cognition layer
                h = model.embed_tokens(gen)
                alpha, beta, v_sq = model._abv()
                R = model.config.complex_rank
                chart_init = model.init_chart(h)
                s = torch.tanh(chart_init[..., :R].float())
                theta = math.pi * torch.tanh(chart_init[..., R:].float())
                for layer in model.encoder_layers:
                    h, s, theta = layer(h, s, theta, alpha, beta, v_sq)
                h = model.encoder_norm(h)
                s_d, theta_d = s.detach(), theta.detach()
                for i, layer in enumerate(model.cognition_layers):
                    is_last = (i == len(model.cognition_layers) - 1)
                    out_cog = layer(h, s_d if not is_last else s_d, theta_d if not is_last else theta_d,
                                     alpha, beta, v_sq, model.memory)
                    h, s_d, theta_d = out_cog["h"], out_cog["s"], out_cog["theta"]

            # Linear head: logits = h @ embed^T
            h_last = h[0, -1, :].float()
            linear_logits = h_last @ embed_weight.float().T
            next_id = linear_logits.argmax().item()
            gen = torch.cat([gen, torch.tensor([[next_id]], device=device)], dim=1)
            if next_id == tokenizer.eos_token_id:
                break

        text = tokenizer.decode(gen[0], skip_special_tokens=True)
        print(f"\n  Prompt: \"{prompt}\"")
        print(f"  Linear head: \"{text[:160]}\"")

        # Also show section readout result for comparison
        gen2 = ids.clone()
        for _ in range(40):
            reset_field_states(model, arch)
            out2 = model(input_ids=gen2, use_cache=False)
            next_id2 = out2.logits[0, -1].argmax().item()
            gen2 = torch.cat([gen2, torch.tensor([[next_id2]], device=device)], dim=1)
            if next_id2 == tokenizer.eos_token_id:
                break
        text2 = tokenizer.decode(gen2[0], skip_special_tokens=True)
        print(f"  Section head: \"{text2[:160]}\"")


# ===========================================================================
# Test 3: Per-position logit analysis — 连续位置有多相似?
# ===========================================================================
@torch.no_grad()
def test_position_analysis(model, arch, tokenizer, device):
    """Analyze how logits change across positions in a SINGLE forward pass."""

    print("\n" + "=" * 70)
    print("TEST 3: Per-Position Logit Analysis")
    print("=" * 70)

    texts = [
        "Albert Einstein was born in Germany and later moved to the United States where he worked",
        "The solution to the quadratic equation ax^2 + bx + c = 0 is given by the formula x = (-b +/- sqrt(b^2 - 4ac)) / 2a",
    ]

    for text in texts:
        ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

        reset_field_states(model, arch)
        out = model(input_ids=ids, use_cache=False)
        logits = out.logits[0].float()  # [T, V]

        print(f"\n  Input: \"{text}\"")
        print(f"\n  {'Pos':>4} | {'Token':>12} | {'argmax':>12} | {'Top1_p':>7} | {'H':>6} | {'cos(t,t-1)':>11} | {'logit_max':>10} | {'logit_std':>10}")
        print("  " + "-" * 100)

        for t in range(logits.shape[0]):
            tok = tokenizer.decode([ids[0, t].item()])
            pred = tokenizer.decode([logits[t].argmax().item()])
            p1 = F.softmax(logits[t], dim=-1).max().item()
            H = token_entropy(logits[t])
            lmax = logits[t].max().item()
            lstd = logits[t].std().item()

            if t > 0:
                cs = F.cosine_similarity(logits[t-1].unsqueeze(0), logits[t].unsqueeze(0)).item()
            else:
                cs = 0.0

            print(f"  {t:>4} | {repr(tok):>12} | {repr(pred):>12} | {p1:>7.4f} | {H:>6.2f} | {cs:>11.4f} | {lmax:>10.2f} | {lstd:>10.4f}")


# ===========================================================================
# Test 4: Autoregressive generation with field state reset
# ===========================================================================
@torch.no_grad()
def test_generation_with_reset(model, arch, tokenizer, device):
    """Standard autoregressive generation WITH field state reset between steps."""

    print("\n" + "=" * 70)
    print("TEST 4: Autoregressive Generation (with field state reset)")
    print("=" * 70)

    prompts = [
        # General (baseline)
        "Albert Einstein was a",
        "The capital of France is",
        # Math: calculus
        "The derivative of x squared is",
        "The integral of sin(x) dx is",
        # Math: algebra
        "If x + 3 = 7, then x =",
        "The solution to x^2 - 4 = 0 is",
        # Math: arithmetic
        "1 + 1 =",
        "The sum of 2 and 3 is",
        # Math: geometry
        "The area of a circle with radius r is",
        # Math: proof style
        "Theorem: For all integers n,",
        "Proof. Let x be a real number such that",
        # Math: notation
        "Consider the function f(x) =",
    ]

    strategies = [
        ("greedy", {}),
        ("nucleus_p0.9", {"top_p": 0.9}),
        ("nucleus_p0.95_rep1.2", {"top_p": 0.95, "rep_penalty": 1.2}),
    ]

    for prompt in prompts:
        print(f"\n  Prompt: \"{prompt}\"")
        for name, params in strategies:
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            gen = ids.clone()
            prompt_len = ids.shape[1]

            for step in range(60):
                reset_field_states(model, arch)  # RESET before each call
                out = model(input_ids=gen, use_cache=False)
                logits = out.logits[0, -1, :].float()

                # Repetition penalty
                if "rep_penalty" in params:
                    rp = params["rep_penalty"]
                    for prev_id in gen[0, prompt_len:].tolist():
                        if logits[prev_id] > 0:
                            logits[prev_id] /= rp
                        else:
                            logits[prev_id] *= rp

                if name.startswith("greedy"):
                    next_id = logits.argmax().item()
                else:
                    top_p = params.get("top_p", 0.9)
                    probs = F.softmax(logits, dim=-1)
                    sorted_p, sorted_i = torch.sort(probs, descending=True)
                    cum = sorted_p.cumsum(dim=-1)
                    mask = cum - sorted_p > top_p
                    sorted_p[mask] = 0.0
                    sorted_p = sorted_p / sorted_p.sum()
                    idx = torch.multinomial(sorted_p, 1)
                    next_id = sorted_i[idx].item()

                gen = torch.cat([gen, torch.tensor([[next_id]], device=device)], dim=1)
                if next_id == tokenizer.eos_token_id:
                    break

            text = tokenizer.decode(gen[0, prompt_len:], skip_special_tokens=True)
            gen_ids = gen[0, prompt_len:].tolist()
            rep1 = sum(1 for i in range(1, len(gen_ids)) if gen_ids[i] == gen_ids[i-1]) / max(len(gen_ids)-1, 1)
            print(f"    [{name:>25}] rep1={rep1:.3f} | \"{text[:120]}\"")


# ===========================================================================
# Test 5: Autoregressive vs single-pass comparison
# ===========================================================================
@torch.no_grad()
def test_autoregressive_divergence(model, arch, tokenizer, device):
    """Compare: does the model give the SAME prediction when processing
    'A B C' in one pass vs processing 'A B C D' where D was generated?
    If the predictions diverge after the first generated token, the model
    is sensitive to its own output."""

    print("\n" + "=" * 70)
    print("TEST 5: Autoregressive Divergence Analysis")
    print("=" * 70)

    test5_prompts = [
        "Albert Einstein was a theoretical physicist who developed",
        "The solution to the equation x^2 + 2x + 1 = 0 is",
    ]

    for prompt in test5_prompts:
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        T = ids.shape[1]

        print(f"\n  Prompt: \"{prompt}\" ({T} tokens)")

        # Single-pass: get predictions at all positions
        reset_field_states(model, arch)
        out = model(input_ids=ids, use_cache=False)
        single_pass_preds = out.logits[0].argmax(dim=-1)  # [T]

        # Autoregressive: generate from the first token
        print(f"  {'Step':>4} | {'SP_pred':>15} | {'AR_pred':>15} | {'Match':>5} | {'SP_p1':>7} | {'AR_p1':>7}")
        print("  " + "-" * 75)

        gen = ids[:, :1].clone()  # Start with just the first token
        n_match = 0

        for t in range(min(T - 1, 20)):
            # Single-pass prediction at position t
            sp_id = single_pass_preds[t].item()
            sp_tok = tokenizer.decode([sp_id])
            sp_p1 = F.softmax(out.logits[0, t], dim=-1).max().item()

            # Feed the CORRECT next token (teacher forcing)
            gen = ids[:, :t+2].clone()
            reset_field_states(model, arch)
            out_ar = model(input_ids=gen, use_cache=False)
            ar_id = out_ar.logits[0, t].argmax().item()
            ar_tok = tokenizer.decode([ar_id])
            ar_p1 = F.softmax(out_ar.logits[0, t], dim=-1).max().item()

            match = sp_id == ar_id
            if match:
                n_match += 1
            marker = "✓" if match else "✗"
            print(f"  {t:>4} | {repr(sp_tok):>15} | {repr(ar_tok):>15} | {marker:>5} | {sp_p1:>7.4f} | {ar_p1:>7.4f}")

        print(f"\n  Agreement: {n_match}/{min(T-1, 20)}")
        print("  If agreement is low, the model gives different predictions")
        print("  when processing the same prefix with vs without future context.")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KUT Generation Diagnostic")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=r"D:\models\qwen3-0.6b-base")
    parser.add_argument("--output_dir", type=str, default="./gen_diagnostic_output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.checkpoint_path}")
    model, arch = load_model(args.checkpoint_path, device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nKUT Generation Diagnostic — {arch}")
    print(f"Device: {device}")

    # Run all tests
    test_teacher_forcing(model, arch, tokenizer, device)
    test_position_analysis(model, arch, tokenizer, device)
    test_autoregressive_divergence(model, arch, tokenizer, device)
    test_generation_with_reset(model, arch, tokenizer, device)

    # Linear head test only if we can access embeddings
    try:
        test_linear_head(model, arch, tokenizer, device)
    except Exception as e:
        print(f"\n  Linear head test failed: {e}")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print("\nKey questions answered:")
    print("  Test 1: Does the model predict correctly in teacher forcing? (training-time)")
    print("  Test 2: Does a linear head generate better? (readout vs backbone)")
    print("  Test 3: Are consecutive positions too similar? (logit stasis)")
    print("  Test 4: Does field state reset + sampling help? (generation strategy)")
    print("  Test 5: Does single-pass vs incremental give same results? (consistency)")
