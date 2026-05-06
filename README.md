# CDIP Corpus — Kähler Unified Transformer (KUT)

A unified normative framework for conditional geometric quantization of human-AI interaction dynamics, with the Kähler Unified Transformer (KUT) architecture and KUM model family.

## Core Idea

Every component of a language model — attention, residual, MLP, normalization, readout — derives from a single Kähler potential:

$$K(z, \bar{z}) = \frac{\alpha}{2}|z|^4 + \beta|z|^2$$

Setting α = 0 recovers a standard Transformer. Training increases α 13×, demonstrating that cross-entropy gradient descent actively discovers curved geometry.

## Repository Structure

| File | Description |
|------|-------------|
| `kut_v13_3_minimal.py` | KUT v13.3 architecture (647 lines) |
| `train_kut_v13_3_minimal_windowed.py` | Training script with non-overlapping data windows |
| `gen_diagnostic.py` | Generation diagnostic (5 tests, math prompts) |
| `Kähler Unified Transformer v2...` | Architecture paper (LaTeX) |

## Key Results (KUM-Lumina-0.6B, 578M params)

- **Generation**: Coherent English and mathematical text, rep1 = 0.000
- **PPL**: 96 (WikiText-103, 100M tokens) → 43.4 (FineMath, 450M total)
- **α growth**: 0.049 → 0.654 (13× increase, CE-driven)
- **Bohr–Sommerfeld**: ℏω = π maintained to 4 decimal places
- **Section > Linear**: Geometric readout is necessary; linear head degenerates
- **Causal safety**: 25/25 SP/AR agreement, zero future leakage
- **Zero hand-tuned physics parameters**: loss = CE + anchor only

## Quick Start

```bash
# Generate from a trained checkpoint
python gen_diagnostic.py --checkpoint_path path/to/best_weights.pt

# Train on WikiText-103
python train_kut_v13_3_minimal_windowed.py \
    dataset_name=wikitext-103 \
    batch_size=4 \
    lr_network=3e-4 \
    lr_geometric=1e-3 \
    max_train_tokens=100000000 \
    donor_model_path=path/to/qwen3-0.6b-base \
    output_dir=./runs

# Train on FineMath (new data window)
python train_kut_v13_3_minimal_windowed.py \
    dataset_name=finemath-4plus \
    batch_size=4 \
    grad_accum_steps=2 \
    lr_network=3e-4 \
    lr_geometric=1e-3 \
    max_train_tokens=100000000 \
    data_skip_tokens=100000000 \
    init_from_checkpoint=path/to/best_weights.pt \
    donor_model_path=path/to/qwen3-0.6b-base \
    output_dir=./runs
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (with CUDA)
- transformers
- Qwen3-0.6B-Base (donor model for weight initialization)

## Architecture: Four-Fold Unity
K(z, z̄) = (α/2)|z|⁴ + β|z|²
├── Geometry:    K → g (metric) → Γ (connection) → κ (curvature)
├── Probability: K → ℏ = π/ω (temperature) → √(g/g_vac) (section weight)
├── Physics:     K → SSB + Calabi forces → field evolution
└── Information:  K → Ψ = √(g/g_vac)·R(θ)·proj(h) → logits = Re⟨Φ,Ψ⟩/ℏ
## Citation

```bibtex
@article{lin2026kut,
  title={K{\"a}hler Unified Transformer: A Language Model Architecture Derived from a Single Geometric Potential},
  author={Lin, Jian},
  year={2026},
  note={The Wallfacer Project}
}
```

## License

MIT
