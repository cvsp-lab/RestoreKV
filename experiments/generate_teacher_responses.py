"""Pre-generate teacher (full-KV) greedy responses for learnable-restore SFT training.

Output jsonl, one row per sample:
    {"ctx": "<user prompt>", "ctx_len": int, "response_ids": [int, ...], "resp_len": int}

The training loop re-encodes ctx and re-uses response_ids verbatim.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterator, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import ModelKVzip


def parse_args():
    p = argparse.ArgumentParser(description="Generate teacher greedy responses")
    p.add_argument("--model", type=str, default="qwen3-4b")
    p.add_argument("--dataset", type=str, default="allenai/tulu-3-sft-mixture")
    p.add_argument("--dataset-name", type=str, default=None)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--text-field", type=str, default="messages",
                   help="Either 'messages' (chat list) or a string field name.")
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--min-prompt-tokens", type=int, default=512)
    p.add_argument("--max-prompt-tokens", type=int, default=16000)
    p.add_argument("--shuffle-buffer", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--report-every", type=int, default=50)
    p.add_argument("--source-filter", type=str, default="",
                   help="comma-separated substrings; keep row only if row['source'] contains any. Empty = no filter.")
    return p.parse_args()


def extract_user_prompt(row, text_field: str) -> Optional[str]:
    value = row.get(text_field)
    if isinstance(value, list) and value:
        for msg in value:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content
        return None
    if isinstance(value, str) and value.strip():
        return value
    for fallback in ("prompt", "instruction", "input", "question", "text"):
        v = row.get(fallback)
        if isinstance(v, str) and v.strip():
            return v
    return None


def iter_dataset(args) -> Iterator[dict]:
    ds = load_dataset(
        args.dataset,
        name=args.dataset_name,
        split=args.split,
        streaming=args.streaming,
    )
    if args.streaming and args.shuffle_buffer > 0:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    elif not args.streaming:
        ds = ds.shuffle(seed=args.seed)
    filters = [s.strip() for s in (args.source_filter or "").split(",") if s.strip()]
    for row in ds:
        if filters:
            src = str(row.get("source", ""))
            if not any(f in src for f in filters):
                continue
        yield row


@torch.inference_mode()
def generate_response(model_kv: ModelKVzip, ctx_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    full_input = torch.cat(
        [model_kv.sys_prompt_ids, ctx_ids, model_kv.postfix_ids],
        dim=1,
    )
    output = model_kv.model.generate(
        full_input,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=None,
        max_new_tokens=max_new_tokens,
        pad_token_id=model_kv.tokenizer.pad_token_id or model_kv.tokenizer.eos_token_id,
    )
    response_ids = output[:, full_input.shape[1]:]
    return response_ids


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_kv = ModelKVzip(args.model, kv_type="evict")
    model_kv.model.eval()

    n_written = 0
    n_seen = 0
    n_skip_short = 0
    n_skip_long = 0
    n_skip_empty = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        progress = tqdm(total=args.num_samples, desc="generate")
        for row in iter_dataset(args):
            if n_written >= args.num_samples:
                break
            n_seen += 1

            user_prompt = extract_user_prompt(row, args.text_field)
            if not user_prompt:
                n_skip_empty += 1
                continue

            ctx_ids = model_kv.encode(user_prompt)
            ctx_len = ctx_ids.shape[1]
            if ctx_len < args.min_prompt_tokens:
                n_skip_short += 1
                continue
            if ctx_len > args.max_prompt_tokens:
                n_skip_long += 1
                continue

            response_ids = generate_response(model_kv, ctx_ids, args.max_new_tokens)
            resp_list = response_ids[0].tolist()

            fout.write(json.dumps({
                "ctx": user_prompt,
                "ctx_len": ctx_len,
                "response_ids": resp_list,
                "resp_len": len(resp_list),
            }, ensure_ascii=False) + "\n")
            fout.flush()
            n_written += 1
            progress.update(1)

            if n_written % args.report_every == 0:
                progress.set_postfix(
                    seen=n_seen,
                    short=n_skip_short,
                    long=n_skip_long,
                    empty=n_skip_empty,
                )

        progress.close()

    print(
        f"Wrote {n_written} rows to {output_path}\n"
        f"  seen={n_seen} short={n_skip_short} long={n_skip_long} empty={n_skip_empty}"
    )


if __name__ == "__main__":
    main()
