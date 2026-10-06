import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import ModelKVzip
from model.learnable_restore import (
    LearnableRestoreTokens,
    RestoreConfig,
    RestoreLoRALinear,
    install_restore_lora,
    lora_enabled,
)
from model.wrapper import chunk_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Train KVzip learnable restore tokens")
    parser.add_argument("--model", type=str, default="qwen3-4b")
    parser.add_argument("--restore-mode", choices=["opt1", "opt2"], default="opt1")
    parser.add_argument("--num-restore-tokens", type=int, default=32)
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--ratio-min", type=float, default=None)
    parser.add_argument("--ratio-max", type=float, default=None)
    parser.add_argument("--level", choices=["pair", "pair-uniform", "snapkv", "h2o", "random"], default="pair")
    parser.add_argument("--budget-mode", choices=["same-ratio", "budget-matched"],
                        default="budget-matched")
    parser.add_argument("--kl-mode", choices=["forward", "reverse", "symmetric"], default="symmetric")
    parser.add_argument("--reverse-kl-weight", type=float, default=1.0)
    parser.add_argument("--distill-alpha", type=float, default=1.0,
                        help="Alpha blending for total loss: alpha*distill + (1-alpha)*lm. "
                             "1.0 = pure distill (default). 0.0 = pure LM. 0.5 = mix.")
    parser.add_argument("--context-len", type=int, default=None)
    parser.add_argument("--context-len-min", type=int, default=512)
    parser.add_argument("--context-len-max", type=int, default=8192)
    parser.add_argument("--context-len-sampling", choices=["log-uniform", "uniform"], default="log-uniform")
    parser.add_argument("--target-len", type=int, default=512)
    parser.add_argument("--window-sampling", choices=["random", "stride"], default="random")
    parser.add_argument("--windows-per-doc", type=int, default=1)
    parser.add_argument("--window-stride", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-shuffle-buffer", type=int, default=10000)
    parser.add_argument("--val-holdout-stride", type=int, default=20)
    parser.add_argument("--val-holdout-remainder", type=int, default=0)
    parser.add_argument("--prefill-chunk-size", type=int, default=16000)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--opt2-query-gate-init", type=float, default=-5.0)
    parser.add_argument("--use-ratio-conditioning", action="store_true")
    parser.add_argument("--ratio-condition-type", choices=["mlp", "film", "moe"], default="mlp")
    parser.add_argument("--ratio-condition-hidden-size", type=int, default=None)
    parser.add_argument("--moe-num-experts", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--restore-lr", type=float, default=None)
    parser.add_argument("--lora-lr", type=float, default=None)
    parser.add_argument("--layer-query-lr", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--lr-scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--val-windows", type=int, default=4)
    parser.add_argument("--val-skip-windows", type=int, default=0)
    parser.add_argument("--val-seed", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--val-ratios", type=str, default=None)
    parser.add_argument("--val-target-len", type=int, default=None,
                        help="Target length for validation windows. If None, uses --target-len. "
                             "Set to a fixed value (e.g. 64) for cross-run comparable val_distill.")
    parser.add_argument("--recontext-target", action="store_true",
                        help="Use the last N tokens of context as target (recontext mode), "
                             "instead of the next-token chunk from the document.")
    parser.add_argument("--recontext-len", type=int, default=2000,
                        help="Target length (suffix of context) when --recontext-target is set.")
    parser.add_argument("--ratio-curriculum", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/learnable_restore")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-name", type=str, default="sample-10BT")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default=None)
    parser.add_argument("--data-source", choices=["fineweb", "teacher-responses"],
                        default="fineweb")
    parser.add_argument("--teacher-responses-path", type=str, default=None,
                        help="jsonl produced by experiments/generate_teacher_responses.py")
    # Attention / hidden-state matching aux losses (Zweiger-style attn output match).
    # Both default 0.0 → no-op, no hooks registered, existing training path bit-exact.
    parser.add_argument("--attn-out-match-alpha", type=float, default=0.0,
                        help="weight on MSE between teacher and student self_attn module outputs "
                             "at response positions (per-layer averaged). 0 disables.")
    parser.add_argument("--hidden-state-match-alpha", type=float, default=0.0,
                        help="weight on MSE between teacher and student post-MLP layer outputs "
                             "at response positions (per-layer averaged). 0 disables.")
    parser.add_argument("--attn-match-layers", type=str, default="",
                        help="comma-separated layer indices to match. Empty=all. "
                             "Example: '8,16,24' or 'quarters' (=L/4,L/2,3L/4,L-1).")
    args = parser.parse_args()
    argv = sys.argv[1:]
    args.context_len_was_explicit = any(
        arg == "--context-len" or arg.startswith("--context-len=")
        for arg in argv
    )
    args.lr_was_explicit = any(arg == "--lr" or arg.startswith("--lr=") for arg in argv)
    args.detailed_lr_was_explicit = any(
        arg == flag or arg.startswith(f"{flag}=")
        for flag in ("--restore-lr", "--lora-lr", "--layer-query-lr")
        for arg in argv
    )
    return args


def parse_ratio_curriculum(spec: Optional[str]) -> Optional[List[Tuple[int, float, float]]]:
    if not spec:
        return None

    curriculum = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        try:
            step_str, range_str = item.split(":", 1)
            min_str, max_str = range_str.split(",", 1)
            start_step = int(step_str)
            ratio_min = float(min_str)
            ratio_max = float(max_str)
        except ValueError as exc:
            raise ValueError(
                "--ratio-curriculum must look like "
                "'0:0.3,0.5;200:0.2,0.5;500:0.1,0.5'"
            ) from exc
        if start_step < 0:
            raise ValueError("--ratio-curriculum start steps must be non-negative")
        if ratio_min > ratio_max:
            raise ValueError("--ratio-curriculum ranges must satisfy min <= max")
        curriculum.append((start_step, ratio_min, ratio_max))

    if not curriculum:
        return None
    return sorted(curriculum, key=lambda item: item[0])


def ratio_range_for_step(args, step: Optional[int] = None) -> Tuple[float, float]:
    if step is not None and getattr(args, "parsed_ratio_curriculum", None):
        ratio_min, ratio_max = args.parsed_ratio_curriculum[0][1:]
        for start_step, current_min, current_max in args.parsed_ratio_curriculum:
            if step < start_step:
                break
            ratio_min, ratio_max = current_min, current_max
        return ratio_min, ratio_max

    if args.ratio_min is None and args.ratio_max is None:
        return args.ratio, args.ratio
    ratio_min = args.ratio if args.ratio_min is None else args.ratio_min
    ratio_max = args.ratio if args.ratio_max is None else args.ratio_max
    if ratio_min > ratio_max:
        raise ValueError("--ratio-min must be <= --ratio-max")
    return ratio_min, ratio_max


def sample_ratio(args, step: Optional[int] = None) -> float:
    ratio_min, ratio_max = ratio_range_for_step(args, step=step)
    if ratio_min == ratio_max:
        return ratio_min
    return random.uniform(ratio_min, ratio_max)


def budget_adjusted_prune_ratio(ratio: float, student_kv, restore_tokens, args) -> float:
    if getattr(args, "budget_mode", "budget-matched") != "budget-matched":
        return ratio
    ctx_len = getattr(student_kv, "ctx_len", 0) or 0
    if ctx_len <= 0:
        return ratio
    restore_overhead = restore_tokens.num_tokens / ctx_len
    return max(0.0, ratio - restore_overhead)


def fixed_eval_ratio(args) -> float:
    if args.val_ratio is not None:
        return args.val_ratio
    ratio_min, ratio_max = ratio_range_for_step(args)
    return (ratio_min + ratio_max) / 2


def ratio_column_name(ratio: float) -> str:
    return f"val_loss_{ratio:g}"


def ratio_range_grid(ratio_min: float, ratio_max: float, step: float = 0.1) -> List[float]:
    if ratio_min > ratio_max:
        raise ValueError("--ratio-min must be <= --ratio-max")

    ratios = [ratio_min]
    start = math.ceil((ratio_min - 1e-9) / step) * step
    current = start
    while current <= ratio_max + 1e-9:
        rounded = round(current, 10)
        if ratio_min - 1e-9 <= rounded <= ratio_max + 1e-9:
            ratios.append(rounded)
        current += step
    ratios.append(ratio_max)

    deduped = []
    for ratio in sorted(ratios):
        rounded = round(ratio, 10)
        if not deduped or abs(rounded - deduped[-1]) > 1e-9:
            deduped.append(rounded)
    return deduped


def parse_val_ratios(args) -> List[float]:
    if args.val_ratios:
        ratios = [float(item.strip()) for item in args.val_ratios.split(",") if item.strip()]
        if not ratios:
            raise ValueError("--val-ratios did not contain any ratios")
        return ratios
    if args.val_ratio is not None:
        return [args.val_ratio]
    if args.ratio_min is not None or args.ratio_max is not None:
        ratio_min, ratio_max = ratio_range_for_step(args)
        return ratio_range_grid(ratio_min, ratio_max)
    if args.ratio_curriculum:
        return [0.1, 0.2, 0.3, 0.4, 0.5]
    return [args.ratio]


def resolve_context_lengths(args):
    if args.context_len_was_explicit:
        if args.context_len is None:
            raise ValueError("--context-len requires an integer value")
        args.context_len_min = args.context_len
        args.context_len_max = args.context_len
    if args.context_len_min <= 0 or args.context_len_max <= 0:
        raise ValueError("--context-len-min/max must be positive")
    if args.context_len_min > args.context_len_max:
        raise ValueError("--context-len-min must be <= --context-len-max")
    if args.target_len <= 1:
        raise ValueError("--target-len must be > 1 for next-token KL")
    if args.val_holdout_stride <= 1:
        raise ValueError("--val-holdout-stride must be > 1")
    if not 0 <= args.val_holdout_remainder < args.val_holdout_stride:
        raise ValueError("--val-holdout-remainder must satisfy 0 <= remainder < stride")


def sample_context_len(args, rng: random.Random, max_available: int) -> Optional[int]:
    if max_available < args.context_len_min:
        return None
    low = args.context_len_min
    high = min(args.context_len_max, max_available)
    if low == high:
        return low
    if args.context_len_sampling == "uniform":
        return rng.randint(low, high)

    sampled = round(math.exp(rng.uniform(math.log(low), math.log(high))))
    return max(low, min(high, sampled))


def should_use_row_for_split(row_idx: int, args, validation: bool) -> bool:
    if validation:
        return row_idx % args.val_holdout_stride == args.val_holdout_remainder
    return row_idx % args.val_holdout_stride != args.val_holdout_remainder


def should_partition_split(active_split: str, args) -> bool:
    return active_split == args.split and (args.val_split is None or args.val_split == args.split)


def _window_specs(num_tokens: int, args, rng: random.Random) -> Iterator[Tuple[int, int]]:
    max_available = num_tokens - args.target_len
    if max_available < args.context_len_min:
        return

    if args.window_sampling == "random":
        for _ in range(args.windows_per_doc):
            ctx_len = sample_context_len(args, rng, max_available)
            if ctx_len is None:
                continue
            total_len = ctx_len + args.target_len
            yield rng.randint(0, num_tokens - total_len), ctx_len
        return

    produced = 0
    max_start_for_min = num_tokens - args.target_len - args.context_len_min
    for start in range(0, max_start_for_min + 1, args.window_stride):
        if produced >= args.windows_per_doc:
            break
        ctx_len = sample_context_len(args, rng, num_tokens - start - args.target_len)
        if ctx_len is None:
            continue
        yield start, ctx_len
        produced += 1


def freeze_base_model(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad_(False)


def enable_trainable_restore_params(restore_tokens: LearnableRestoreTokens,
                                    lora_wrappers: Iterator[RestoreLoRALinear]):
    for param in restore_tokens.parameters():
        param.requires_grad_(True)
    for wrapper in lora_wrappers:
        wrapper.lora_a.requires_grad_(True)
        wrapper.lora_b.requires_grad_(True)


def trainable_parameters(model: torch.nn.Module, restore_tokens: LearnableRestoreTokens):
    for param in restore_tokens.parameters():
        if param.requires_grad:
            yield param
    for param in model.parameters():
        if param.requires_grad:
            yield param


def resolve_param_group_lrs(args) -> Tuple[float, float, float]:
    if (
        args.restore_lr is None
        and args.lora_lr is None
        and args.layer_query_lr is None
        and getattr(args, "lr_was_explicit", False)
        and not getattr(args, "detailed_lr_was_explicit", False)
    ):
        return args.lr, args.lr, args.lr
    restore_lr = 1e-4 if args.restore_lr is None else args.restore_lr
    lora_lr = 1e-4 if args.lora_lr is None else args.lora_lr
    layer_query_lr = 1e-5 if args.layer_query_lr is None else args.layer_query_lr
    return restore_lr, lora_lr, layer_query_lr


def make_optimizer_param_groups(
    args,
    restore_tokens: LearnableRestoreTokens,
    lora_wrappers: Iterator[RestoreLoRALinear],
) -> List[Dict]:
    restore_lr, lora_lr, layer_query_lr = resolve_param_group_lrs(args)
    param_groups = []

    restore_embedding_params = []
    if restore_tokens.embeddings.requires_grad:
        restore_embedding_params.append(restore_tokens.embeddings)
    if restore_tokens.ratio_mlp is not None:
        restore_embedding_params.extend(
            param for param in restore_tokens.ratio_mlp.parameters() if param.requires_grad
        )
    if restore_embedding_params:
        param_groups.append({
            "name": "restore_embeddings",
            "params": restore_embedding_params,
            "lr": restore_lr,
        })

    if restore_tokens.layer_queries is not None:
        layer_query_params = []
        if restore_tokens.layer_queries.requires_grad:
            layer_query_params.append(restore_tokens.layer_queries)
        if (
            restore_tokens.layer_query_gate_logits is not None
            and restore_tokens.layer_query_gate_logits.requires_grad
        ):
            layer_query_params.append(restore_tokens.layer_query_gate_logits)
        if layer_query_params:
            param_groups.append({
                "name": "layer_queries",
                "params": layer_query_params,
                "lr": layer_query_lr,
            })

    lora_params = []
    for wrapper in lora_wrappers:
        if wrapper.lora_a.requires_grad:
            lora_params.append(wrapper.lora_a)
        if wrapper.lora_b.requires_grad:
            lora_params.append(wrapper.lora_b)
    if lora_params:
        param_groups.append({
            "name": "restore_lora",
            "params": lora_params,
            "lr": lora_lr,
        })

    if not param_groups:
        raise RuntimeError("No trainable restore parameters found")
    return param_groups


def make_lr_scheduler(args, optimizer: torch.optim.Optimizer):
    if args.lr_scheduler == "none":
        return None
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0")

    def lr_lambda(current_step: int) -> float:
        step = current_step + 1
        if args.warmup_steps > 0 and step <= args.warmup_steps:
            return step / args.warmup_steps
        decay_steps = max(1, args.max_steps - args.warmup_steps)
        progress = min(1.0, max(0.0, (step - args.warmup_steps) / decay_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def iter_token_windows(
    tokenizer,
    args,
    split: Optional[str] = None,
    seed: Optional[int] = None,
    validation: bool = False,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    rng = random.Random(args.seed if seed is None else seed)
    dataset_seed = args.seed if seed is None else seed
    active_split = args.split if split is None else split
    dataset = load_dataset(
        args.dataset,
        name=args.dataset_name,
        split=active_split,
        streaming=True,
    )
    if should_partition_split(active_split, args):
        dataset = dataset.filter(
            lambda _, idx: should_use_row_for_split(idx, args, validation),
            with_indices=True,
        )
    if args.dataset_shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=dataset_seed, buffer_size=args.dataset_shuffle_buffer)
    for row in dataset:
        text = row.get("text", "")
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        for start, ctx_len in _window_specs(len(ids), args, rng):
            total_len = ctx_len + args.target_len
            window = torch.tensor(ids[start:start + total_len], dtype=torch.long).unsqueeze(0)
            yield window[:, :ctx_len], window[:, ctx_len:]


def collect_validation_windows(tokenizer, args) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    if args.val_windows <= 0:
        return []

    val_split = args.split if args.val_split is None else args.val_split
    val_seed = args.val_seed if args.val_seed is not None else args.seed + 100000
    if val_split == args.split:
        print(
            "Validation uses holdout partition "
            f"{args.val_holdout_remainder}/{args.val_holdout_stride} from split '{args.split}'."
        )

    original_target_len = args.target_len
    val_tl = args.val_target_len if args.val_target_len is not None else original_target_len
    if val_tl != original_target_len:
        print(f"Validation target_len overridden: {original_target_len} -> {val_tl}")
    args.target_len = val_tl
    try:
        windows = iter_token_windows(
            tokenizer,
            args,
            split=val_split,
            seed=val_seed,
            validation=True,
        )
        for _ in tqdm(range(args.val_skip_windows), desc="Skip validation windows"):
            next(windows)
        result = [next(windows) for _ in tqdm(range(args.val_windows), desc="Collect validation")]
    finally:
        args.target_len = original_target_len
    return result


def prefill_context(model_kv: ModelKVzip, ctx_ids: torch.Tensor, do_score: bool, args):
    prefill_ids = torch.cat([model_kv.sys_prompt_ids, ctx_ids], dim=1)
    evict_range = (model_kv.sys_prompt_ids.shape[1], prefill_ids.shape[1])
    kv = model_kv._init_kv(evict_range=evict_range)
    kv.ctx_ids = ctx_ids
    kv.prefill_ids = prefill_ids

    sm = getattr(args, "score_method", "kvzip") if do_score else "kvzip"
    # H2O (KVzip-paper counterpart): score DURING prefill from causal self-attention.
    # Preallocate a running-max buffer, use a small chunk to bound the attention
    # tensor, then slice the sys-prompt region off after prefill (see wrapper.prefill).
    _h2o = (do_score and sm == "h2o")
    with torch.no_grad():
        if _h2o:
            full_len = prefill_ids.shape[1]
            kv.get_score = True
            kv.score_mode = "h2o"
            kv.score = [
                torch.zeros((1, kv.n_heads_kv, full_len), dtype=kv.dtype, device=kv.device)
                for _ in range(kv.n_layers)
            ]
            _pf_chunk = int(os.environ.get("H2O_PREFILL_CHUNK", "1024"))
        else:
            _pf_chunk = args.prefill_chunk_size
        for input_ids in chunk_fn(prefill_ids, _pf_chunk):
            model_kv.model.model(input_ids, past_key_values=kv)
        if _h2o:
            kv.get_score = False
            kv.score_mode = "kvzip"
            sys_len = model_kv.sys_prompt_ids.shape[1]
            kv.score = [s[:, :, sys_len:].contiguous() for s in kv.score]
        elif do_score:
            if sm == "random":
                # Random pruning: assign uniform-random importance -> random eviction.
                # No reconstruction needed; KV cache is already built by the prefill above.
                ctx_len = ctx_ids.shape[1]
                kv.get_score = False
                kv.score = [torch.rand((1, kv.n_heads_kv, ctx_len), device=kv.device,
                                       dtype=torch.float32) for _ in range(kv.n_layers)]
            elif sm == "snapkv":
                model_kv.scoring_snapkv(kv, ctx_ids)
            elif sm == "kvzip_plus":
                # KVzip+ (Jégou & Jeblick 2026): scoring formula uses ||W_O v_i|| / ||h_j||
                kv.score_mode = "kvzip_plus"
                model_kv.scoring(kv, ctx_ids, load_score=False)
                kv.score_mode = "kvzip"
            elif sm == "contrastkv":
                model_kv.scoring_contrastkv(kv, ctx_ids, load_score=False)
            else:
                model_kv.scoring(kv, ctx_ids, load_score=False)
    return kv


def target_logits(model_kv: ModelKVzip, target_ids: torch.Tensor, kv):
    outputs = model_kv.model(target_ids, past_key_values=kv)
    return outputs.logits


def forward_restore_tokens(
    model_kv: ModelKVzip,
    kv,
    restore_tokens: LearnableRestoreTokens,
    lora_wrappers,
    ratio: Optional[float] = None,
):
    # V1 ablation: SKIP_RESTORE=1 → LoRA-only training (no restore tokens appended).
    if os.environ.get("SKIP_RESTORE", "0") == "1":
        return
    restore_embeds = restore_tokens(
        batch_size=1,
        device=model_kv.device,
        ratio=ratio,
    ).to(dtype=model_kv.dtype)
    used_len = restore_tokens.num_tokens
    restore_start = kv._seen_tokens
    kv.mark_restore_tokens(
        start=restore_start,
        length=used_len,
        query_override=restore_tokens if restore_tokens.layer_queries is not None else None,
    )
    import os as _os
    _ps_env = _os.environ.get("RESTORE_POS_SHIFT", "0")
    _pos_shift = int(_ps_env) if _ps_env.lstrip("-").isdigit() else 0
    kv.restore_query_active = restore_tokens.layer_queries is not None
    # Optional: restrict restore attention to (sys + will-be-evicted-ctx) via mask
    _use_mask = os.environ.get("RESTORE_ATTN_MASK", "0") == "1" and ratio is not None
    _mask_sys = os.environ.get("RESTORE_ATTN_MASK_SYS", "0") == "1"
    if _use_mask or _mask_sys:
        if _use_mask:
            _future_valid = kv.compute_future_valid(ratio, "pair")
            _invert = os.environ.get("RESTORE_ATTN_MASK_INVERT", "0") == "1"
            kv.restore_visible_mask = _future_valid if _invert else ~_future_valid
        else:
            kv.restore_visible_mask = None
        kv.restore_mask_sys = _mask_sys
        kv.restore_forward_active = True
    try:
        with lora_enabled(lora_wrappers):
            if _pos_shift != 0:
                _seen = kv._seen_tokens
                _pos_ids = torch.arange(
                    _seen + _pos_shift,
                    _seen + _pos_shift + used_len,
                    device=model_kv.device,
                ).unsqueeze(0)
                _ = model_kv.model(
                    inputs_embeds=restore_embeds,
                    past_key_values=kv,
                    position_ids=_pos_ids,
                    use_cache=True,
                )
            else:
                model_kv.forward_embeds(restore_embeds, kv, update_cache=True)
    finally:
        if _use_mask or _mask_sys:
            kv.restore_forward_active = False
            kv.restore_visible_mask = None
            kv.restore_mask_sys = False
    kv.restore_query_active = False


def iter_teacher_response_samples(
    args,
    seed: Optional[int] = None,
    validation: bool = False,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    if not args.teacher_responses_path:
        raise ValueError("--teacher-responses-path is required for data-source teacher-responses")
    path = args.teacher_responses_path
    base_seed = args.seed if seed is None else seed

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if should_use_row_for_split(idx, args, validation):
                rows.append(line)

    rng = random.Random(base_seed)

    while True:
        if not validation:
            rng.shuffle(rows)
        for line in rows:
            row = json.loads(line)
            response_ids = row.get("response_ids")
            ctx_text = row.get("ctx")
            if not response_ids or not ctx_text:
                continue
            ctx_ids = _sft_tokenizer.encode(
                ctx_text, add_special_tokens=False, return_tensors="pt")
            response_tensor = torch.tensor(response_ids, dtype=torch.long).unsqueeze(0)
            # Optional un-prunable question (LongAlpaca format). Empty tensor for Tulu jsonl.
            q_text = row.get("question", "") or ""
            if q_text:
                question_tensor = _sft_tokenizer.encode(
                    q_text, add_special_tokens=False, return_tensors="pt")
            else:
                question_tensor = torch.empty((1, 0), dtype=torch.long)
            yield ctx_ids, question_tensor, response_tensor
        if validation:
            return


_sft_tokenizer = None


def set_sft_tokenizer(tokenizer):
    global _sft_tokenizer
    _sft_tokenizer = tokenizer


def collect_validation_response_samples(args) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    if args.val_windows <= 0:
        return []
    val_seed = args.val_seed if args.val_seed is not None else args.seed + 100000
    it = iter_teacher_response_samples(args, seed=val_seed, validation=True)
    for _ in tqdm(range(args.val_skip_windows), desc="Skip validation samples"):
        next(it, None)
    result = []
    for _ in tqdm(range(args.val_windows), desc="Collect validation"):
        item = next(it, None)
        if item is None:
            break
        result.append(item)
    return result


def kl_loss_response(student_resp: torch.Tensor, teacher_resp: torch.Tensor,
                    args) -> torch.Tensor:
    # fp32 + log-space KL: avoids softmax->0 underflow that caused NaN in bf16
    student = student_resp.float().clamp(-50.0, 50.0)
    teacher = teacher_resp.float().clamp(-50.0, 50.0)
    log_student = F.log_softmax(student, dim=-1)
    log_teacher = F.log_softmax(teacher, dim=-1)
    # F.kl_div(input, target, log_target=True, reduction='batchmean')
    #   = mean over batch of sum(exp(target) * (target - input))
    #   = forward KL: KL(p_teacher || p_student)
    forward_kl = F.kl_div(log_student, log_teacher,
                          log_target=True, reduction="batchmean")
    if args.kl_mode == "forward":
        return forward_kl
    reverse_kl = F.kl_div(log_teacher, log_student,
                          log_target=True, reduction="batchmean")
    if args.kl_mode == "reverse":
        return reverse_kl
    return forward_kl + args.reverse_kl_weight * reverse_kl


def lm_loss_response(student_resp: torch.Tensor, response_ids: torch.Tensor) -> torch.Tensor:
    logits = student_resp.float()
    labels = response_ids
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
    )


# ---- Activation matching (attn output / hidden state) helpers ----

def _decoder_layers(model_kv: ModelKVzip):
    """Return list of decoder layer modules for hook attachment.
    Works for Qwen2/Qwen3 ForCausalLM where layers live at model.model.layers."""
    return model_kv.model.model.layers


def parse_match_layers(spec: str, num_layers: int) -> List[int]:
    spec = (spec or "").strip()
    if not spec:
        return list(range(num_layers))
    if spec == "quarters":
        return sorted({max(0, num_layers // 4), num_layers // 2,
                       max(0, 3 * num_layers // 4), num_layers - 1})
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        i = int(tok)
        if i < 0 or i >= num_layers:
            raise ValueError(f"--attn-match-layers index {i} out of range [0,{num_layers})")
        out.append(i)
    return sorted(set(out))


def _extract_tensor(output):
    """HF self_attn / decoder layer module outputs vary across versions:
    sometimes a Tensor, sometimes a tuple where [0] is the hidden state."""
    if isinstance(output, tuple):
        return output[0]
    return output


def register_match_hooks(model_kv: ModelKVzip,
                         layer_indices: List[int],
                         capture_attn: bool,
                         capture_hidden: bool):
    """Returns (handles, attn_store, hidden_store).
    Stores fill during the next forward pass on the model."""
    layers = _decoder_layers(model_kv)
    attn_store: Dict[int, torch.Tensor] = {}
    hidden_store: Dict[int, torch.Tensor] = {}
    handles = []
    for li in layer_indices:
        layer = layers[li]
        if capture_attn:
            def make_attn_hook(idx):
                def hook(_module, _inputs, output):
                    attn_store[idx] = _extract_tensor(output)
                return hook
            handles.append(layer.self_attn.register_forward_hook(make_attn_hook(li)))
        if capture_hidden:
            def make_hidden_hook(idx):
                def hook(_module, _inputs, output):
                    hidden_store[idx] = _extract_tensor(output)
                return hook
            handles.append(layer.register_forward_hook(make_hidden_hook(li)))
    return handles, attn_store, hidden_store


def _remove_handles(handles):
    for h in handles:
        h.remove()


def activation_match_loss(student_store: Dict[int, torch.Tensor],
                          teacher_store: Dict[int, torch.Tensor],
                          resp_slice: slice) -> torch.Tensor:
    """Mean MSE across captured layers on response positions only.
    Both stores must share the same key set."""
    if not student_store:
        return torch.zeros((), device="cuda")
    total = None
    for li in student_store:
        s = student_store[li][:, resp_slice].float()
        t = teacher_store[li][:, resp_slice].float()
        loss = F.mse_loss(s, t)
        total = loss if total is None else total + loss
    return total / max(1, len(student_store))


def restore_loss_sft(
    model_kv: ModelKVzip,
    restore_tokens: LearnableRestoreTokens,
    lora_wrappers,
    ctx_ids: torch.Tensor,
    response_ids: torch.Tensor,
    ratio: float,
    args,
    question_ids: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Cache layout under SFT teacher-response mode:
        [sys][ctx (prunable)][restore_tokens (kept)][question][postfix][response]
    Only `response` positions contribute to the loss; question is un-prunable.
    """
    postfix_ids = model_kv.postfix_ids
    if question_ids is None or question_ids.shape[1] == 0:
        target_ids = torch.cat([postfix_ids, response_ids], dim=1)
        Q = 0
    else:
        target_ids = torch.cat([question_ids, postfix_ids, response_ids], dim=1)
        Q = question_ids.shape[1]
    P = postfix_ids.shape[1]
    R = response_ids.shape[1]
    resp_slice = slice(Q + P - 1, Q + P - 1 + R)
    alpha = args.distill_alpha

    # Aux activation-matching losses (default off, no impact on existing runs).
    attn_alpha = float(getattr(args, "attn_out_match_alpha", 0.0))
    hidden_alpha = float(getattr(args, "hidden_state_match_alpha", 0.0))
    match_active = attn_alpha > 0.0 or hidden_alpha > 0.0
    teacher_needed = alpha > 0.0 or match_active

    student_kv = prefill_context(model_kv, ctx_ids, do_score=True, args=args)
    forward_restore_tokens(model_kv, student_kv, restore_tokens, lora_wrappers,
                           ratio=ratio)
    student_kv.prune(budget_adjusted_prune_ratio(ratio, student_kv, restore_tokens, args), args.level)

    if match_active:
        layer_idx = args.parsed_match_layers
        s_handles, s_attn, s_hidden = register_match_hooks(
            model_kv, layer_idx,
            capture_attn=attn_alpha > 0.0,
            capture_hidden=hidden_alpha > 0.0,
        )
        student_logits = target_logits(model_kv, target_ids, student_kv)
        _remove_handles(s_handles)
    else:
        student_logits = target_logits(model_kv, target_ids, student_kv)
        s_attn = s_hidden = {}

    student_resp = student_logits[:, resp_slice]
    lm = lm_loss_response(student_resp, response_ids)

    attn_match = torch.zeros((), device=student_resp.device, dtype=torch.float32)
    hidden_match = torch.zeros((), device=student_resp.device, dtype=torch.float32)

    if teacher_needed:
        teacher_kv = prefill_context(model_kv, ctx_ids, do_score=False, args=args)
        if match_active:
            t_handles, t_attn, t_hidden = register_match_hooks(
                model_kv, args.parsed_match_layers,
                capture_attn=attn_alpha > 0.0,
                capture_hidden=hidden_alpha > 0.0,
            )
            with torch.no_grad():
                teacher_logits = target_logits(model_kv, target_ids, teacher_kv).detach()
            _remove_handles(t_handles)
            # Detach captured teacher tensors so no grad flows.
            t_attn = {k: v.detach() for k, v in t_attn.items()}
            t_hidden = {k: v.detach() for k, v in t_hidden.items()}
        else:
            with torch.no_grad():
                teacher_logits = target_logits(model_kv, target_ids, teacher_kv).detach()

        if alpha > 0.0:
            teacher_resp = teacher_logits[:, resp_slice]
            distill = kl_loss_response(student_resp, teacher_resp, args)
            del teacher_resp
        else:
            distill = torch.zeros((), device=student_resp.device, dtype=student_resp.dtype)

        if attn_alpha > 0.0:
            attn_match = activation_match_loss(s_attn, t_attn, resp_slice)
        if hidden_alpha > 0.0:
            hidden_match = activation_match_loss(s_hidden, t_hidden, resp_slice)
        del teacher_kv, teacher_logits
        if match_active:
            del t_attn, t_hidden
    else:
        distill = torch.zeros((), device=student_resp.device, dtype=student_resp.dtype)

    total = (alpha * distill + (1.0 - alpha) * lm
             + attn_alpha * attn_match
             + hidden_alpha * hidden_match)
    losses = {
        "distill": distill, "lm": lm, "total": total,
        "attn_match": attn_match, "hidden_match": hidden_match,
    }
    del student_kv, student_logits, student_resp
    if match_active:
        del s_attn, s_hidden
    return losses


def kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, args) -> torch.Tensor:
    student = student_logits[:, :-1].float()
    teacher = teacher_logits[:, :-1].float()
    forward_kl = F.kl_div(
        F.log_softmax(student, dim=-1),
        F.softmax(teacher, dim=-1),
        reduction="none",
    ).sum(dim=-1).mean()
    if args.kl_mode == "forward":
        return forward_kl

    reverse_kl = F.kl_div(
        F.log_softmax(teacher, dim=-1),
        F.softmax(student, dim=-1),
        reduction="none",
    ).sum(dim=-1).mean()
    if args.kl_mode == "reverse":
        return reverse_kl
    return forward_kl + args.reverse_kl_weight * reverse_kl


def lm_loss(student_logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    logits = student_logits[:, :-1].float()
    labels = target_ids[:, 1:]
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
    )


def restore_loss(
    model_kv: ModelKVzip,
    restore_tokens: LearnableRestoreTokens,
    lora_wrappers,
    ctx_ids: torch.Tensor,
    target_ids: torch.Tensor,
    ratio: float,
    args,
) -> Dict[str, torch.Tensor]:
    teacher_kv = prefill_context(model_kv, ctx_ids, do_score=False, args=args)
    with torch.no_grad():
        teacher_logits = target_logits(model_kv, target_ids, teacher_kv).detach()

    student_kv = prefill_context(model_kv, ctx_ids, do_score=True, args=args)
    forward_restore_tokens(model_kv, student_kv, restore_tokens, lora_wrappers,
                           ratio=ratio)
    student_kv.prune(budget_adjusted_prune_ratio(ratio, student_kv, restore_tokens, args), args.level)
    student_logits = target_logits(model_kv, target_ids, student_kv)

    distill = kl_loss(student_logits, teacher_logits, args)
    lm = lm_loss(student_logits, target_ids)
    alpha = args.distill_alpha
    total = alpha * distill + (1.0 - alpha) * lm
    losses = {"distill": distill, "lm": lm, "total": total}
    del teacher_kv, student_kv, teacher_logits, student_logits
    return losses


@torch.no_grad()
def validation_loss(
    model_kv: ModelKVzip,
    restore_tokens: LearnableRestoreTokens,
    lora_wrappers,
    val_windows: List[Tuple[torch.Tensor, torch.Tensor]],
    args,
) -> Dict[float, float]:
    if not val_windows:
        return {}

    val_losses = {}
    is_sft = args.data_source == "teacher-responses"
    for ratio in args.parsed_val_ratios:
        distill_losses = []
        lm_losses = []
        for sample in val_windows:
            if is_sft:
                ctx_ids, question_ids, target_ids = sample
                question_ids = question_ids.to(model_kv.device)
            else:
                ctx_ids, target_ids = sample
                question_ids = None
            ctx_ids = ctx_ids.to(model_kv.device)
            target_ids = target_ids.to(model_kv.device)
            if not is_sft and args.recontext_target:
                n_tail = min(args.recontext_len, ctx_ids.shape[1])
                target_ids = ctx_ids[:, -n_tail:]
            if is_sft:
                loss_dict = restore_loss_sft(
                    model_kv, restore_tokens, lora_wrappers,
                    ctx_ids, target_ids, ratio, args,
                    question_ids=question_ids,
                )
            else:
                loss_dict = restore_loss(
                    model_kv, restore_tokens, lora_wrappers,
                    ctx_ids, target_ids, ratio, args,
                )
            distill_losses.append(loss_dict["distill"].item())
            lm_losses.append(loss_dict["lm"].item())
            del loss_dict
        val_losses[ratio] = {
            "distill": sum(distill_losses) / len(distill_losses),
            "lm": sum(lm_losses) / len(lm_losses),
        }
    return val_losses


def history_path(args) -> str:
    return os.path.join(args.output_dir, "loss_history.csv")


def append_loss_history(args, step: int, train_loss: float, train_distill_loss: float,
                        train_lm_loss: float, ratio: float, context_len: int,
                        val_losses,
                        attn_match_loss: float = 0.0,
                        hidden_match_loss: float = 0.0):
    path = history_path(args)
    write_header = not os.path.exists(path)
    val_distill_cols = [f"val_distill_loss_{r:g}" for r in args.parsed_val_ratios]
    val_lm_cols = [f"val_lm_loss_{r:g}" for r in args.parsed_val_ratios]
    fieldnames = [
        "step", "train_loss", "train_distill_loss", "train_lm_loss",
        "train_attn_match_loss", "train_hidden_match_loss",
        "val_distill_loss", "val_lm_loss",
        *val_distill_cols, *val_lm_cols,
        "ratio", "context_len",
    ]
    if not write_header:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
            existing_fieldnames = rows[0].keys() if rows else []
        if list(existing_fieldnames) != fieldnames:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for existing_row in rows:
                    writer.writerow({
                        key: existing_row.get(key, "")
                        for key in fieldnames
                    })
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        avg_distill = None
        avg_lm = None
        if val_losses:
            distill_vals = [v["distill"] for v in val_losses.values()]
            lm_vals = [v["lm"] for v in val_losses.values()]
            avg_distill = sum(distill_vals) / len(distill_vals)
            avg_lm = sum(lm_vals) / len(lm_vals)
        row = {
            "step": step,
            "train_loss": train_loss,
            "train_distill_loss": train_distill_loss,
            "train_lm_loss": train_lm_loss,
            "train_attn_match_loss": attn_match_loss,
            "train_hidden_match_loss": hidden_match_loss,
            "val_distill_loss": "" if avg_distill is None else avg_distill,
            "val_lm_loss": "" if avg_lm is None else avg_lm,
            "ratio": ratio,
            "context_len": context_len,
        }
        for val_ratio in args.parsed_val_ratios:
            entry = val_losses.get(val_ratio) if val_losses else None
            row[f"val_distill_loss_{val_ratio:g}"] = "" if entry is None else entry["distill"]
            row[f"val_lm_loss_{val_ratio:g}"] = "" if entry is None else entry["lm"]
        writer.writerow({
            key: row[key]
            for key in fieldnames
        })


def plot_loss_history(args):
    path = history_path(args)
    if not os.path.exists(path):
        return

    mpl_config_dir = "/tmp/matplotlib"
    os.makedirs(mpl_config_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, train_losses = [], []
    distill_losses, lm_losses = [], []
    val_steps, val_distill, val_lm = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            step = int(row["step"])
            steps.append(step)
            train_losses.append(float(row["train_loss"]))
            distill_losses.append(float(row["train_distill_loss"]) if row.get("train_distill_loss") else float("nan"))
            lm_losses.append(float(row["train_lm_loss"]) if row.get("train_lm_loss") else float("nan"))
            vd = row.get("val_distill_loss") or row.get("val_loss")
            vl = row.get("val_lm_loss")
            if vd:
                val_steps.append(step)
                val_distill.append(float(vd))
                val_lm.append(float(vl) if vl else float("nan"))

    plt.figure(figsize=(8, 5))
    plt.plot(steps, train_losses, label="total", linewidth=1.5)
    plt.plot(steps, distill_losses, label="distill (KL)", alpha=0.7, linewidth=1)
    plt.plot(steps, lm_losses, label="lm (CE)", alpha=0.7, linewidth=1)
    if val_steps:
        plt.plot(val_steps, val_distill, marker="o", label="val distill")
        plt.plot(val_steps, val_lm, marker="s", label="val lm")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Learnable restore loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "loss_curve.png"), dpi=150)
    plt.close()

    if val_steps:
        plt.figure(figsize=(8, 5))
        plt.plot(val_steps, val_distill, marker="o", label="val distill (KL)")
        plt.plot(val_steps, val_lm, marker="s", label="val lm (CE)")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Learnable restore validation loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "val_loss_curve.png"), dpi=150)
        plt.close()


def checkpoint_state(
    args,
    step: int,
    loss: float,
    model_kv: ModelKVzip,
    restore_tokens: LearnableRestoreTokens,
) -> Dict:
    trainable_model_state = {
        name: param.detach().cpu()
        for name, param in model_kv.model.named_parameters()
        if param.requires_grad
    }
    return {
        "step": step,
        "loss": loss,
        "model": args.model,
        "restore": restore_tokens.checkpoint_payload(vars(args)),
        "trainable_model_state": trainable_model_state,
    }


def save_checkpoint(args, step, loss, model_kv, restore_tokens):
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f"checkpoint_step{step}.pt")
    torch.save(checkpoint_state(args, step, loss, model_kv, restore_tokens), path)
    return path


def apply_sft_defaults(args):
    if args.data_source != "teacher-responses":
        return
    if not args.teacher_responses_path:
        raise ValueError("--teacher-responses-path is required for data-source teacher-responses")

    # Defaults applied ONLY when user didn't pass the flag on CLI.
    argv = sys.argv[1:]
    def is_explicit(*flags):
        return any(a == f or a.startswith(f + "=") for a in argv for f in flags)

    defaults = [
        ("restore_mode",        "opt1",  ["--restore-mode"]),
        ("num_restore_tokens",  8,       ["--num-restore-tokens"]),
        ("lora_rank",           8,       ["--lora-rank"]),
        ("lora_alpha",          16,      ["--lora-alpha"]),
        ("lora_dropout",        0.0,     ["--lora-dropout"]),
    ]
    for key, value, flags in defaults:
        if is_explicit(*flags):
            print(f"[sft] keep CLI {key}={getattr(args, key)}")
            continue
        current = getattr(args, key)
        if current != value:
            print(f"[sft] default {key}: {current} -> {value}")
            setattr(args, key, value)

    if args.ratio_min is None:
        args.ratio_min = 0.1
        print(f"[sft] ratio_min default -> 0.1")
    if args.ratio_max is None:
        args.ratio_max = 0.5
        print(f"[sft] ratio_max default -> 0.5")


def main():
    args = parse_args()
    apply_sft_defaults(args)
    resolve_context_lengths(args)
    # If level=snapkv: use SnapKV scoring during prefill, but pair-style prune
    # downstream. Save the score method then rewrite args.level to "pair" so all
    # the existing prune(args.level) calls keep working.
    # KVzip+ activated via env KVZIP_PLUS=1 (Jégou & Jeblick 2026)
    import os as _os
    _kvzip_plus = _os.environ.get("KVZIP_PLUS", "") in ("1","true","yes")
    _contrastkv = _os.environ.get("CONTRASTKV", "") in ("1","true","yes")
    if args.level == "snapkv":
        args.score_method = "snapkv"
        args.level = "pair"
    elif args.level == "h2o":
        args.score_method = "h2o"
        args.level = "pair"
    elif args.level == "random":
        args.score_method = "random"
        args.level = "pair"
    elif _contrastkv:
        args.score_method = "contrastkv"
    elif _kvzip_plus:
        args.score_method = "kvzip_plus"
    else:
        args.score_method = "kvzip"
    args.parsed_ratio_curriculum = parse_ratio_curriculum(args.ratio_curriculum)
    args.parsed_val_ratios = parse_val_ratios(args)
    # parsed_match_layers requires model.config; deferred until after model load.
    args.parsed_match_layers = []
    if args.val_every is None:
        args.val_every = args.save_every
    if args.val_every <= 0:
        args.val_windows = 0
    os.makedirs(args.output_dir, exist_ok=True)
    print(
        "Context length sampling: "
        f"{args.context_len_sampling}, range={args.context_len_min}-{args.context_len_max}, "
        f"target_len={args.target_len}"
    )

    model_kv = ModelKVzip(args.model, kv_type="evict")
    freeze_base_model(model_kv.model)

    # Resolve match layers now that model.config is available.
    if args.attn_out_match_alpha > 0.0 or args.hidden_state_match_alpha > 0.0:
        args.parsed_match_layers = parse_match_layers(
            args.attn_match_layers, model_kv.config.num_hidden_layers
        )
        print(f"[match] attn_alpha={args.attn_out_match_alpha} "
              f"hidden_alpha={args.hidden_state_match_alpha} "
              f"layers={args.parsed_match_layers}")
    else:
        args.parsed_match_layers = []

    restore_config = RestoreConfig(
        mode=args.restore_mode,
        num_restore_tokens=args.num_restore_tokens,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        opt2_query_gate_init=args.opt2_query_gate_init,
        use_ratio_conditioning=args.use_ratio_conditioning,
        ratio_condition_type=args.ratio_condition_type,
        ratio_condition_hidden_size=args.ratio_condition_hidden_size,
        moe_num_experts=args.moe_num_experts,
    )
    restore_tokens = LearnableRestoreTokens(
        model_kv.config,
        restore_config=restore_config,
        dtype=model_kv.dtype,
    ).to(model_kv.device)

    # Optional: initialize restore embeddings via env var
    _ri = os.environ.get("RESTORE_INIT_MODE", "")
    if _ri == "boundary":
        boundary_strs = ["<|im_end|>", "\n\n", "\n", ".", ",", "!", "?", ";"]
        n = restore_tokens.num_tokens
        ids = []
        for s in boundary_strs:
            enc = model_kv.tokenizer.encode(s, add_special_tokens=False)
            if enc: ids.append(enc[0])
        while len(ids) < n: ids.append(ids[len(ids) % len(boundary_strs)])
        ids = ids[:n]
        embed_w = model_kv.model.model.embed_tokens.weight
        with torch.no_grad():
            restore_tokens.embeddings.data = embed_w[ids].clone().to(
                restore_tokens.embeddings.dtype)
        print(f"[sft] init restore embeddings from boundary tokens: {boundary_strs[:n]}")
        print(f"      token ids: {ids}")
    elif _ri == "random_vocab":
        import torch as _torch
        n = restore_tokens.num_tokens
        vocab_size = model_kv.model.model.embed_tokens.weight.shape[0]
        _g = _torch.Generator(device="cpu").manual_seed(0)
        ids = _torch.randint(0, vocab_size, (n,), generator=_g).tolist()
        embed_w = model_kv.model.model.embed_tokens.weight
        with torch.no_grad():
            restore_tokens.embeddings.data = embed_w[ids].clone().to(
                restore_tokens.embeddings.dtype)
        print(f"[sft] init restore embeddings from {n} random vocab ids: {ids}")
    elif _ri == "newline":
        n = restore_tokens.num_tokens
        enc = model_kv.tokenizer.encode("\n", add_special_tokens=False)
        nl_id = enc[0]
        ids = [nl_id] * n
        embed_w = model_kv.model.model.embed_tokens.weight
        with torch.no_grad():
            restore_tokens.embeddings.data = embed_w[ids].clone().to(
                restore_tokens.embeddings.dtype)
        print(f"[sft] init restore embeddings from {n} newline tokens (id={nl_id})")
    elif _ri == "prompt":
        n = restore_tokens.num_tokens
        prompt = "Given the previous context, the relevant information is:"
        enc = model_kv.tokenizer.encode(prompt, add_special_tokens=False)
        # take first n; if shorter, repeat from start
        if len(enc) < n:
            enc = (enc * ((n // len(enc)) + 1))[:n]
        else:
            enc = enc[:n]
        embed_w = model_kv.model.model.embed_tokens.weight
        with torch.no_grad():
            restore_tokens.embeddings.data = embed_w[enc].clone().to(
                restore_tokens.embeddings.dtype)
        decoded = [model_kv.tokenizer.decode([i]) for i in enc]
        print(f"[sft] init restore embeddings from prompt tokens: {decoded}")
        print(f"      token ids: {enc}")
    if args.lora_rank > 0:
        lora_wrappers = install_restore_lora(
            model_kv.model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
    else:
        print("[sft] lora_rank=0 → skipping LoRA install (pure embedding-only restore)")
        lora_wrappers = []
    freeze_base_model(model_kv.model)
    enable_trainable_restore_params(restore_tokens, lora_wrappers)

    param_groups = make_optimizer_param_groups(args, restore_tokens, lora_wrappers)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = make_lr_scheduler(args, optimizer)
    clip_params = [param for group in param_groups for param in group["params"]]

    if args.data_source == "teacher-responses":
        set_sft_tokenizer(model_kv.tokenizer)
        val_windows = collect_validation_response_samples(args)
    else:
        val_windows = collect_validation_windows(model_kv.tokenizer, args)
    if val_windows:
        val_ratio_str = ",".join(f"{ratio:g}" for ratio in args.parsed_val_ratios)
        print(
            f"Prepared {len(val_windows)} validation windows "
            f"(ratios={val_ratio_str})"
        )

    if args.data_source == "teacher-responses":
        windows = iter_teacher_response_samples(args, seed=args.seed, validation=False)
    else:
        windows = iter_token_windows(model_kv.tokenizer, args, split=args.split, seed=args.seed)
    progress = tqdm(range(1, args.max_steps + 1), desc="train restore")
    for step in progress:
        if args.data_source == "teacher-responses":
            ctx_ids, question_ids, target_ids = next(windows)
            question_ids = question_ids.to(model_kv.device)
        else:
            ctx_ids, target_ids = next(windows)
            question_ids = None
        context_len = ctx_ids.shape[1]
        ctx_ids = ctx_ids.to(model_kv.device)
        target_ids = target_ids.to(model_kv.device)
        if args.data_source != "teacher-responses" and args.recontext_target:
            n_tail = min(args.recontext_len, ctx_ids.shape[1])
            target_ids = ctx_ids[:, -n_tail:]

        ratio = sample_ratio(args, step=step)
        if args.data_source == "teacher-responses":
            loss_dict = restore_loss_sft(
                model_kv, restore_tokens, lora_wrappers,
                ctx_ids, target_ids, ratio, args,
                question_ids=question_ids,
            )
        else:
            loss_dict = restore_loss(
                model_kv, restore_tokens, lora_wrappers,
                ctx_ids, target_ids, ratio, args,
            )
        total_loss = loss_dict["total"]
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(clip_params, args.grad_clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_value = total_loss.item()
        distill_value = loss_dict["distill"].item()
        lm_value = loss_dict["lm"].item()
        attn_match_value = float(loss_dict.get("attn_match", torch.zeros(())).item()) \
            if "attn_match" in loss_dict else 0.0
        hidden_match_value = float(loss_dict.get("hidden_match", torch.zeros(())).item()) \
            if "hidden_match" in loss_dict else 0.0
        del loss_dict, total_loss
        if step % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        val_losses = {}
        should_validate = (
            val_windows
            and (step % args.val_every == 0 or step == 1 or step == args.max_steps)
        )
        if should_validate:
            val_losses = validation_loss(model_kv, restore_tokens, lora_wrappers, val_windows, args)

        append_loss_history(args, step, loss_value, distill_value, lm_value,
                            ratio, context_len, val_losses,
                            attn_match_loss=attn_match_value,
                            hidden_match_loss=hidden_match_value)
        if val_losses or step % args.save_every == 0 or step == args.max_steps:
            plot_loss_history(args)

        if step % args.log_every == 0 or step == 1:
            postfix = {
                "total": f"{loss_value:.4f}",
                "distill": f"{distill_value:.4f}",
                "lm": f"{lm_value:.4f}",
                "ratio": f"{ratio:.3f}",
                "ctx": context_len,
            }
            if args.attn_out_match_alpha > 0.0:
                postfix["a_mse"] = f"{attn_match_value:.4f}"
            if args.hidden_state_match_alpha > 0.0:
                postfix["h_mse"] = f"{hidden_match_value:.4f}"
            if val_losses:
                distill_vals = [v["distill"] for v in val_losses.values()]
                lm_vals = [v["lm"] for v in val_losses.values()]
                postfix["val_dis"] = f"{sum(distill_vals)/len(distill_vals):.4f}"
                postfix["val_lm"] = f"{sum(lm_vals)/len(lm_vals):.4f}"
            progress.set_postfix(**postfix)
        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(args, step, loss_value, model_kv, restore_tokens)


if __name__ == "__main__":
    main()
