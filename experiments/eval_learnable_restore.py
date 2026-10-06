import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import DataWrapper, load_dataset_all
from eval import set_ratios
from model import ModelKVzip
from model.learnable_restore import install_restore_lora, load_restore_tokens, lora_enabled
from utils import Evaluator, TimeStamp, save_result, set_gen_length


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate KVzip with learnable restore tokens")
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-d", "--data", type=str, required=True)
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--restore-checkpoint", type=str, required=True)
    parser.add_argument("--opt2-query-gate-init", type=float, default=-5.0)
    parser.add_argument(
        "--level",
        type=str,
        default="pair",
        choices=["pair", "pair-uniform", "snapkv", "h2o"],
        help="KVzip pruning level for the context KV scores",
    )
    parser.add_argument(
        "--budget-mode",
        type=str,
        default="same-ratio",
        choices=["same-ratio", "budget-matched"],
        help=(
            "same-ratio prunes context KV at the requested ratio. "
            "budget-matched subtracts always-retained restore-token KV from the context budget."
        ),
    )
    parser.add_argument("--tag", type=str, default="restore")
    parser.add_argument(
        "--oracle-restore",
        action="store_true",
        help=(
            "For each ratio/task, compare question-token KL for context-only pruning "
            "versus learnable restore and generate only the lower-KL branch."
        ),
    )
    return parser.parse_args()


def _restore_lora_config(checkpoint):
    restore_payload = checkpoint["restore"] if "restore" in checkpoint else checkpoint
    return restore_payload["restore_config"]


def _placeholder_restore_ids(model_kv: ModelKVzip, length: int) -> torch.Tensor:
    token_id = model_kv.tokenizer.pad_token_id
    if token_id is None:
        eos_token_id = model_kv.tokenizer.eos_token_id
        token_id = eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id
    if token_id is None:
        token_id = 0
    return torch.full(
        (1, length),
        int(token_id),
        dtype=torch.long,
        device=model_kv.device,
    )


def restore_budget_adjusted_ratio(target_ratio: float, ctx_len: int, restore_len: int):
    if ctx_len <= 0:
        raise ValueError("ctx_len must be positive for restore budget matching")
    restore_overhead = restore_len / ctx_len
    return max(0.0, target_ratio - restore_overhead), restore_overhead


def context_ratio_true_for_prune_ratio(kv, ratio: float, level: str):
    if "uniform" in level:
        valid, _ = kv._threshold_uniform(kv.score, ratio)
    else:
        valid, _ = kv._threshold(kv.score, ratio)

    return 1 - (valid == False).float().mean().item()


def restore_budget_adjusted_ratio_like_kvzip(
    kv,
    target_ratio: float,
    restore_len: int,
    level: str,
):
    prune_ratio, restore_overhead = restore_budget_adjusted_ratio(
        target_ratio,
        kv.ctx_len,
        restore_len,
    )
    if restore_len <= 0:
        baseline_context_ratio = context_ratio_true_for_prune_ratio(kv, target_ratio, level)
        return prune_ratio, restore_overhead, baseline_context_ratio

    baseline_context_ratio = context_ratio_true_for_prune_ratio(kv, target_ratio, level)
    target_context_ratio = max(0.0, baseline_context_ratio - restore_overhead)

    lo, hi = 0.0, 1.0
    best_ratio = prune_ratio
    best_error = float("inf")
    for _ in range(24):
        mid = (lo + hi) / 2
        mid_context_ratio = context_ratio_true_for_prune_ratio(kv, mid, level)
        error = abs(mid_context_ratio - target_context_ratio)
        if error < best_error:
            best_ratio = mid
            best_error = error
        if mid_context_ratio < target_context_ratio:
            lo = mid
        else:
            hi = mid

    for candidate in (lo, hi, prune_ratio, target_ratio):
        candidate = max(0.0, min(1.0, candidate))
        candidate_context_ratio = context_ratio_true_for_prune_ratio(kv, candidate, level)
        error = abs(candidate_context_ratio - target_context_ratio)
        if error < best_error:
            best_ratio = candidate
            best_error = error

    return best_ratio, restore_overhead, baseline_context_ratio


def make_ratio_metadata(
    args,
    ratio: float,
    prune_ratio: float,
    context_ratio_true: float,
    restore_overhead: float,
    restore_len: int,
    ctx_len: int,
):
    if args.budget_mode != "budget-matched":
        return None
    return {
        "budget_mode": args.budget_mode,
        "context_prune_ratio": round(prune_ratio, 6),
        "context_ratio_true": round(context_ratio_true, 6),
        "restore_overhead_ratio": round(restore_overhead, 6),
        "restore_tokens": restore_len,
        "ctx_len": ctx_len,
    }


def oracle_metadata(
    base_metadata,
    *,
    selected_restore: bool,
    restore_infeasible: bool = False,
    fallback_reason: str | None = None,
    baseline_question_kld: float | None = None,
    restore_question_kld: float | None = None,
    budget_match_target_ratio_true: float | None = None,
    budget_error: float | None = None,
    candidate_restore_overhead: float | None = None,
):
    metadata = {} if base_metadata is None else dict(base_metadata)
    metadata.update({
        "oracle_restore": True,
        "oracle_selected_restore": bool(selected_restore),
        "oracle_restore_infeasible": bool(restore_infeasible),
        "fallback_reason": fallback_reason,
        "baseline_question_kld": baseline_question_kld,
        "restore_question_kld": restore_question_kld,
        "budget_error": budget_error,
        "candidate_restore_overhead_ratio": (
            round(candidate_restore_overhead, 6)
            if candidate_restore_overhead is not None
            else None
        ),
    })
    if budget_match_target_ratio_true is not None:
        metadata["budget_match_target_ratio_true"] = round(
            budget_match_target_ratio_true, 6)
    return metadata


def make_ratio_info(ratio: float, ratio_true: float, thres: float, metadata=None):
    ratio_info = [ratio, round(ratio_true, 4), round(thres, 4)]
    if metadata is not None:
        ratio_info.append(metadata)
    return ratio_info


@torch.inference_mode()
def append_restore_tokens(model_kv: ModelKVzip, kv, restore_tokens, lora_wrappers, ratio=None):
    import os as _os
    # V1 ablation: SKIP_RESTORE=1 → LoRA-only inference (no restore tokens appended).
    if _os.environ.get("SKIP_RESTORE", "0") == "1":
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

    # Optional: shift restore tokens' RoPE position via RESTORE_POS_SHIFT env var
    # (e.g., -8 → restore K vectors use RoPE phases [ctx_len-8 .. ctx_len-1] overlapping
    # with last 8 ctx positions, instead of default [ctx_len .. ctx_len+7]).
    _ps_env = _os.environ.get("RESTORE_POS_SHIFT", "0")
    _pos_shift = int(_ps_env) if _ps_env.lstrip("-").isdigit() else 0

    try:
        kv.restore_query_active = restore_tokens.layer_queries is not None
        _use_mask = _os.environ.get("RESTORE_ATTN_MASK", "0") == "1" and ratio is not None
        _mask_sys = _os.environ.get("RESTORE_ATTN_MASK_SYS", "0") == "1"
        if _use_mask or _mask_sys:
            if _use_mask:
                _future_valid = kv.compute_future_valid(ratio, "pair")
                _invert = _os.environ.get("RESTORE_ATTN_MASK_INVERT", "0") == "1"
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
    finally:
        kv.restore_query_active = False

    seen_after = kv._seen_tokens
    if seen_after - restore_start != used_len:
        raise RuntimeError(
            f"Restore append length mismatch: expected {used_len}, "
            f"got {seen_after - restore_start}"
        )
    if kv.restore_token_len != used_len:
        raise RuntimeError(
            f"KV restore_token_len mismatch: expected {used_len}, "
            f"got {kv.restore_token_len}"
        )

    # The K/V cache contains learned embedding tokens, but generate() still uses input_ids
    # length to decide which suffix is the new query. Add placeholders for length only.
    if kv.prefill_ids is not None:
        restore_ids = _placeholder_restore_ids(model_kv, used_len)
        kv.prefill_ids = torch.cat([kv.prefill_ids, restore_ids], dim=1)


def remove_restore_tokens(kv, seen_token_prev: int, prefill_len_prev: int):
    kv.slice(seen_token_prev)
    kv.restore_token_start = None
    kv.restore_token_len = 0
    kv.restore_query_override = None
    kv.restore_query_active = False
    if kv.prefill_ids is not None:
        kv.prefill_ids = kv.prefill_ids[:, :prefill_len_prev]


@torch.inference_mode()
def question_kld(model_kv: ModelKVzip, q_ids: torch.Tensor, full_prob: torch.Tensor, kv):
    candidate_prob = model_kv._prob(q_ids, kv)
    full_prob = full_prob.to(device=candidate_prob.device, dtype=candidate_prob.dtype)
    if candidate_prob.shape != full_prob.shape:
        raise RuntimeError(
            "Question probability shape mismatch: "
            f"full={tuple(full_prob.shape)}, candidate={tuple(candidate_prob.shape)}"
        )
    kld = F.kl_div(candidate_prob.clamp_min(1e-12).log(), full_prob, reduction="none")
    return kld.sum(dim=-1).mean().item()


@torch.inference_mode()
def oracle_eval_task(
    args,
    model_kv: ModelKVzip,
    eval,
    task: str,
    q_ids: torch.Tensor,
    full_question_prob: torch.Tensor,
    kv,
    ratio: float,
    baseline_prune_ratio: float,
    baseline_thres: float,
    baseline_context_ratio_true: float,
    restore_tokens,
    lora_wrappers,
):
    baseline_question_kld = question_kld(model_kv, q_ids, full_question_prob, kv)

    context_seen_tokens = kv._seen_tokens
    context_prefill_len = kv.prefill_ids.shape[1] if kv.prefill_ids is not None else 0
    baseline_metadata = make_ratio_metadata(
        args,
        ratio,
        baseline_prune_ratio,
        baseline_context_ratio_true,
        0.0,
        0,
        kv.ctx_len,
    )

    if args.budget_mode == "budget-matched":
        candidate_restore_overhead = restore_tokens.num_tokens / kv.ctx_len
        restore_budget_feasible = (
            candidate_restore_overhead <= baseline_context_ratio_true
        )
    else:
        candidate_restore_overhead = 0.0
        restore_budget_feasible = True

    if not restore_budget_feasible:
        value = eval.generation(kv, task)
        metadata = oracle_metadata(
            baseline_metadata,
            selected_restore=False,
            restore_infeasible=True,
            fallback_reason="restore_overhead_exceeds_budget",
            baseline_question_kld=baseline_question_kld,
            restore_question_kld=None,
            budget_match_target_ratio_true=baseline_context_ratio_true,
            budget_error=0.0,
            candidate_restore_overhead=candidate_restore_overhead,
        )
        ratio_info = make_ratio_info(
            ratio,
            baseline_context_ratio_true,
            baseline_thres,
            metadata,
        )
        return ratio_info, value

    kv.pruned = False
    append_restore_tokens(model_kv, kv, restore_tokens, lora_wrappers, ratio=ratio)
    print(
        f"# restore tokens appended: start={kv.restore_token_start}, "
        f"len={kv.restore_token_len}, seen={kv._seen_tokens}"
    )

    if args.budget_mode == "budget-matched":
        restore_prune_ratio, restore_overhead, restore_budget_target_ratio = (
            restore_budget_adjusted_ratio_like_kvzip(
                kv,
                ratio,
                kv.restore_token_len,
                args.level,
            ))
    else:
        restore_prune_ratio = ratio
        restore_overhead = 0.0
        restore_budget_target_ratio = None

    restore_thres, restore_context_ratio_true = kv.prune(restore_prune_ratio, args.level)
    restore_question_kld = question_kld(model_kv, q_ids, full_question_prob, kv)
    oracle_selected_restore = restore_question_kld < baseline_question_kld

    restore_metadata = make_ratio_metadata(
        args,
        ratio,
        restore_prune_ratio,
        restore_context_ratio_true,
        restore_overhead,
        kv.restore_token_len,
        kv.ctx_len,
    )
    total_restore_ratio_true = restore_context_ratio_true + restore_overhead
    budget_error = None
    if restore_budget_target_ratio is not None:
        budget_error = abs(total_restore_ratio_true - baseline_context_ratio_true)
        budget_tolerance = max(1e-3, 1.0 / max(1, kv.ctx_len))
        if budget_error > budget_tolerance:
            raise AssertionError(
                "Restore budget matching exceeded tolerance: "
                f"baseline={baseline_context_ratio_true:.6f}, "
                f"restore_total={total_restore_ratio_true:.6f}, "
                f"error={budget_error:.6f}, tolerance={budget_tolerance:.6f}"
            )
    restore_metadata = oracle_metadata(
        restore_metadata,
        selected_restore=oracle_selected_restore,
        baseline_question_kld=baseline_question_kld,
        restore_question_kld=restore_question_kld,
        budget_match_target_ratio_true=restore_budget_target_ratio,
        budget_error=budget_error,
    )

    if oracle_selected_restore:
        value = eval.generation(kv, task)
        ratio_info = make_ratio_info(
            ratio,
            total_restore_ratio_true,
            restore_thres,
            restore_metadata,
        )
    else:
        remove_restore_tokens(kv, context_seen_tokens, context_prefill_len)
        kv.prune(baseline_prune_ratio, args.level)
        value = eval.generation(kv, task)
        baseline_metadata = oracle_metadata(
            baseline_metadata,
            selected_restore=oracle_selected_restore,
            baseline_question_kld=baseline_question_kld,
            restore_question_kld=restore_question_kld,
            budget_match_target_ratio_true=baseline_context_ratio_true,
            budget_error=0.0,
        )
        ratio_info = make_ratio_info(
            ratio,
            baseline_context_ratio_true,
            baseline_thres,
            baseline_metadata,
        )

    remove_restore_tokens(kv, context_seen_tokens, context_prefill_len)
    return ratio_info, value


def load_restore_components(model_kv: ModelKVzip, checkpoint_path: str):
    restore_tokens, checkpoint = load_restore_tokens(
        checkpoint_path,
        model_kv.config,
        map_location=model_kv.device,
    )
    restore_tokens = restore_tokens.to(device=model_kv.device, dtype=model_kv.dtype)
    restore_tokens.eval()

    lora_config = _restore_lora_config(checkpoint)
    if lora_config["lora_rank"] > 0:
        lora_wrappers = install_restore_lora(
            model_kv.model,
            rank=lora_config["lora_rank"],
            alpha=lora_config["lora_alpha"],
            dropout=lora_config.get("lora_dropout", 0.0),
        )
    else:
        print("[eval] lora_rank=0 → skipping LoRA install (pure embedding-only restore)")
        lora_wrappers = []
    model_kv.model.load_state_dict(checkpoint.get("trainable_model_state", {}), strict=False)
    model_kv.model.eval()

    if any(wrapper.enabled for wrapper in lora_wrappers):
        raise RuntimeError("Restore LoRA wrappers should be disabled outside restore prefill")

    print(
        "Loaded restore checkpoint:",
        checkpoint_path,
        f"mode={restore_tokens.restore_config.mode}",
        f"tokens={restore_tokens.num_tokens}",
        f"ratio_conditioning={restore_tokens.restore_config.use_ratio_conditioning}",
        f"lora_rank={lora_config['lora_rank']}",
    )
    return restore_tokens, lora_wrappers


def main():
    args = parse_args()

    model_kv = ModelKVzip(args.model, kv_type="retain")
    restore_tokens, lora_wrappers = load_restore_components(model_kv, args.restore_checkpoint)

    n_data = max(100, args.idx + args.num)
    dataset = load_dataset_all(args.data, model_kv.tokenizer, n_data=n_data)
    dataset = DataWrapper(args.data, dataset, model_kv)
    set_gen_length(args.data, model_kv)

    tt = TimeStamp(True)
    eval_indices = list(range(args.idx, min(args.idx + args.num, len(dataset))))
    print("=" * 80, f"\nStart restore evaluation with {len(eval_indices)} samples")

    # SnapKV: query-agnostic last-window scoring during prefill; downstream prune
    # uses standard pair-threshold logic.
    # kvzip+ (Jégou & Jeblick 2026): activate via env `KVZIP_PLUS=1` — SFT ckpt still
    # loaded normally, only the prefill scoring formula changes to KVzip+.
    import os as _os
    _kvzip_plus = _os.environ.get("KVZIP_PLUS", "") in ("1","true","yes")
    _contrastkv = _os.environ.get("CONTRASTKV", "") in ("1","true","yes")
    if args.level == "snapkv":
        score_method = "snapkv"
        args.level = "pair"
    elif args.level == "h2o":
        score_method = "h2o"
        args.level = "pair"
    elif _contrastkv:
        score_method = "contrastkv"
    elif _kvzip_plus:
        score_method = "kvzip_plus"
    else:
        score_method = "kvzip"

    import os as _os
    for data_idx in eval_indices:
        # skip if output already exists (idempotent re-runs at larger --num)
        _folder_tag = f"_{args.tag}" if args.tag else ""
        _out_path = f"./results/{args.data}/{data_idx}_{args.model}{_folder_tag}/output-{args.level}.json"
        if _os.path.exists(_out_path):
            print(f"[skip] {_out_path} exists")
            continue
        _USE_LORA_PREFILL = _os.environ.get("LORA_PREFILL", "0") == "1"
        if _USE_LORA_PREFILL:
            with lora_enabled(lora_wrappers):
                kv = dataset.prefill_context(data_idx, load_score=False,
                                             score_method=score_method)
        else:
            kv = dataset.prefill_context(data_idx, load_score=False,
                                         score_method=score_method)
        inputs, info = dataset.generate_answer(data_idx, kv)

        eval = Evaluator(model_kv, inputs, info)
        outputs = defaultdict(list)
        full_question_probs = {}
        if args.oracle_restore:
            for task in info.keys():
                full_question_probs[task] = model_kv._prob(inputs[task]["q"], kv, device="cpu")

        use_ratio_conditioning = restore_tokens.restore_config.use_ratio_conditioning
        _use_mask_eval = _os.environ.get("RESTORE_ATTN_MASK", "0") == "1"
        # Mask requires per-ratio re-append (mask depends on ratio); route as ratio-cond
        _reappend_per_ratio = use_ratio_conditioning or _use_mask_eval
        context_seen_tokens = kv._seen_tokens
        context_prefill_len = kv.prefill_ids.shape[1] if kv.prefill_ids is not None else 0

        if not args.oracle_restore and not _reappend_per_ratio:
            append_restore_tokens(model_kv, kv, restore_tokens, lora_wrappers)
            print(
                f"# restore tokens appended: start={kv.restore_token_start}, "
                f"len={kv.restore_token_len}, seen={kv._seen_tokens}"
            )

        for ratio in set_ratios(args.model):
            if not args.oracle_restore and _reappend_per_ratio:
                kv.pruned = False
                append_restore_tokens(model_kv, kv, restore_tokens, lora_wrappers, ratio=ratio)
                print(
                    f"# restore tokens appended: start={kv.restore_token_start}, "
                    f"len={kv.restore_token_len}, seen={kv._seen_tokens}, "
                    f"ratio_condition={ratio:.4f}"
                )

            if args.budget_mode == "budget-matched":
                prune_ratio, restore_overhead, baseline_context_ratio = (
                    restore_budget_adjusted_ratio_like_kvzip(
                        kv,
                        ratio,
                        restore_tokens.num_tokens
                        if use_ratio_conditioning else kv.restore_token_len,
                        args.level,
                    ))
            else:
                prune_ratio = ratio
                restore_overhead = 0.0
                baseline_context_ratio = None

            active_prune_ratio = ratio if args.oracle_restore else prune_ratio
            thres, context_ratio_true = kv.prune(active_prune_ratio, args.level)
            total_ratio_true = context_ratio_true + restore_overhead
            ratio_metadata = make_ratio_metadata(
                args,
                ratio,
                active_prune_ratio,
                context_ratio_true,
                0.0 if args.oracle_restore else restore_overhead,
                kv.restore_token_len,
                kv.ctx_len,
            )
            if ratio_metadata is not None and baseline_context_ratio is not None:
                ratio_metadata["budget_match_target_ratio_true"] = round(baseline_context_ratio,
                                                                          6)
            ratio_info = make_ratio_info(ratio, context_ratio_true, thres)
            if args.budget_mode == "budget-matched":
                ratio_info = make_ratio_info(ratio, total_ratio_true, thres, ratio_metadata)
                print(
                    "# budget matched:",
                    f"target_total_ratio={ratio:.4f}",
                    f"baseline_ratio_true={baseline_context_ratio:.4f}",
                    f"context_prune_ratio={prune_ratio:.4f}",
                    f"restore_overhead={restore_overhead:.4f}",
                    f"total_ratio_true={total_ratio_true:.4f}",
                )

            if args.oracle_restore:
                for task in info.keys():
                    thres, context_ratio_true = kv.prune(active_prune_ratio, args.level)
                    task_ratio_info, value = oracle_eval_task(
                        args=args,
                        model_kv=model_kv,
                        eval=eval,
                        task=task,
                        q_ids=inputs[task]["q"],
                        full_question_prob=full_question_probs[task],
                        kv=kv,
                        ratio=ratio,
                        baseline_prune_ratio=active_prune_ratio,
                        baseline_thres=thres,
                        baseline_context_ratio_true=context_ratio_true,
                        restore_tokens=restore_tokens,
                        lora_wrappers=lora_wrappers,
                    )
                    outputs[task].append([task_ratio_info, value])
            else:
                try:
                    results = eval(kv, generate=True)

                    for fmt, value in results.items():
                        outputs[fmt].append([ratio_info, value])
                finally:
                    if use_ratio_conditioning:
                        remove_restore_tokens(kv, context_seen_tokens, context_prefill_len)

        save_result(args, args.data, outputs, data_idx)

        if any(wrapper.enabled for wrapper in lora_wrappers):
            raise RuntimeError("Restore LoRA wrappers leaked into eval/generation")

        tt(f"{args.data}-{data_idx}")
        del kv, inputs, info, eval
    print("Finished.")


if __name__ == "__main__":
    main()
