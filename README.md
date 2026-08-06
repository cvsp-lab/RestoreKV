<h1 align="center">RestoreKV</h1>

<p align="center">
  <b>Recovering Full-Cache Behavior Under Aggressive Query-Agnostic KV Cache Eviction</b>
</p>

<p align="center">
  <a href="https://sites.google.com/view/changwoobaek00/%ED%99%88">Changwoo Baek</a><sup>1</sup> &nbsp;·&nbsp;
  <a href="https://ansl-lab.github.io/professor/">Seungjun Shin</a><sup>2†</sup> &nbsp;·&nbsp;
  <a href="https://www.pnu-cvsp.com/prof">Kyeongbo Kong</a><sup>1†</sup>
  <br>
  <sup>1</sup>Pusan National University &nbsp;·&nbsp; <sup>2</sup>Sookmyung Women's University
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2608.01247"><img src="https://img.shields.io/badge/arXiv-2608.01247-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://paper.pnu-cvsp.com/RestoreKV/"><img src="https://img.shields.io/badge/%F0%9F%8C%8E_Project-Page-4a90d9" alt="Project Page"></a>
  <a href="https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/restorekv_press.py"><img src="https://img.shields.io/badge/Inference-KVPress-76b900?logo=nvidia&logoColor=white" alt="KVPress"></a>
  <a href="https://huggingface.co/collections/higokri/restorekv"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Weights-HuggingFace-FFD21E?labelColor=555" alt="Weights"></a>
  <a href="https://huggingface.co/spaces/nvidia/kvpress-leaderboard"><img src="https://img.shields.io/badge/KVPress_Leaderboard-%F0%9F%A5%871st-gold?labelColor=555" alt="Leaderboard"></a>
</p>

---

> **🏆 #1 on the [KVPress Leaderboard](https://huggingface.co/spaces/nvidia/kvpress-leaderboard).**

**RestoreKV** complements selection-based query-agnostic KV cache eviction with **learned restoration** under
the same total KV budget. After context prefill, a few restore tokens attend to the full KV cache in a single
**LoRA-adapted pass**, generating a compact, context-conditioned **restore cache**. The base importance scorer and
eviction rule remain unchanged, and the adapters are disabled for all subsequent queries and decoding. RestoreKV is
trained through parameter-efficient **self-distillation** from the frozen full-cache model, optimizing only **0.4%** of
the parameters and requiring no task-specific tuning.

- Improves **59 of 60** paired, budget-matched settings across five base eviction methods (Qwen3-4B).
- At a **5%** budget, raises KVzip from **38.2 → 73.2** on RULER-4K.
- Applied to KVzip+, reaches **86.4** RULER accuracy at **16×** compression on the KVPress Benchmark, while adding
  **<0.5%** one-time cache-construction overhead in a 32K-context evaluation.

## 🔗 Resources

| | |
|---|---|
| 📄 **Paper** | [arXiv:2608.01247](https://arxiv.org/abs/2608.01247) |
| 🌎 **Project page** | [paper.pnu-cvsp.com/RestoreKV](https://paper.pnu-cvsp.com/RestoreKV/) |
| ⚙️ **Inference code** | [`restorekv_press.py` in NVIDIA/KVPress](https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/restorekv_press.py) |
| 🤗 **Weights** | [huggingface.co/collections/higokri/restorekv](https://huggingface.co/collections/higokri/restorekv) |
| 🏆 **Leaderboard** | [KVPress Leaderboard](https://huggingface.co/spaces/nvidia/kvpress-leaderboard) — **1st place** |

## 🚀 Usage

Inference is available through **[NVIDIA KVPress](https://github.com/NVIDIA/kvpress)** via `RestoreKVPress`, with
pretrained restore adapters hosted on the [Hugging Face collection](https://huggingface.co/collections/higokri/restorekv).

```python
from kvpress import RestoreKVPress

# Wrap any base eviction press with a learned, budget-matched restore pass.
press = RestoreKVPress(...)
```

See [`kvpress/presses/restorekv_press.py`](https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/restorekv_press.py)
for the full inference implementation and arguments.

## 🗺️ Release Plan

- [x] Inference code (integrated into [NVIDIA/KVPress](https://github.com/NVIDIA/kvpress))
- [x] Pretrained restore adapters ([Hugging Face](https://huggingface.co/collections/higokri/restorekv))
- [ ] Full training code — **coming soon**

> **Note.** The full training / self-distillation code is currently being prepared and will be released here.
> In the meantime, **inference is fully available** through KVPress with the released weights.

## 📚 Citation

```bibtex
@article{baek2026restorekv,
  title   = {RestoreKV: Recovering Full-Cache Behavior Under Aggressive Query-Agnostic KV Cache Eviction},
  author  = {Baek, Changwoo and Shin, Seungjun and Kong, Kyeongbo},
  journal = {arXiv preprint arXiv:2608.01247},
  year    = {2026}
}
```

## 🙏 Acknowledgements

RestoreKV builds on [KVzip](https://github.com/snu-mllab/KVzip) for query-agnostic context-reconstruction eviction,
and its inference is implemented on top of [NVIDIA KVPress](https://github.com/NVIDIA/kvpress). RestoreKV is part of the
[BTS — Busan Token-pruning Series](https://higokri.github.io/BTS/) from the
[PNU-CVSP](https://www.pnu-cvsp.com/) lab.
