"""Self-study Q+A generation on PG-19 (Project Gutenberg books).

PG-19 books are huge (~1M tokens each). We:
  - Take N books (default 50)
  - Slice each book into non-overlapping 3K-token chunks
  - Randomly sample K chunks per book (default 10)
  - Generate 5 teacher Q+A per chunk

Output JSONL schema matches generate_selfstudy.py / train_learnable_restore.py SFT loop.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import ModelKVzip
from experiments.generate_selfstudy import (
    QGEN_PROMPTS, model_generate, build_qgen_prompt, build_qa_prompt, clean_question,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="qwen3-4b")
    p.add_argument("--book-start", type=int, default=0)
    p.add_argument("--book-end", type=int, default=50)
    p.add_argument("--chunk-tokens", type=int, default=3072)
    p.add_argument("--chunks-per-book", type=int, default=10)
    p.add_argument("--max-q-tokens", type=int, default=64)
    p.add_argument("--max-a-tokens", type=int, default=512)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_kv = ModelKVzip(args.model, kv_type="evict")
    model_kv.model.eval()
    tok = model_kv.tokenizer

    print(f"loading PG-19 (streaming) — taking books {args.book_start}..{args.book_end - 1}")
    ds = load_dataset("deepmind/pg19", split="train",
                      streaming=True, trust_remote_code=True)
    print(f"chunk_tokens={args.chunk_tokens}, chunks_per_book={args.chunks_per_book}, "
          f"{len(QGEN_PROMPTS)} Q-types per chunk")

    n_written = n_skip_bad_q = 0
    total_chunks = (args.book_end - args.book_start) * args.chunks_per_book
    pbar = tqdm(total=total_chunks * len(QGEN_PROMPTS), desc="qa")

    with open(out_path, "w", encoding="utf-8") as fout:
        for book_idx, row in enumerate(ds):
            if book_idx < args.book_start:
                continue
            if book_idx >= args.book_end:
                break
            text = (row.get("text") or "").strip()
            if not text:
                continue
            full_ids = tok.encode(text, add_special_tokens=False)
            n_chunks_avail = max(1, len(full_ids) // args.chunk_tokens)
            chunk_ranges = [(i * args.chunk_tokens,
                             min((i + 1) * args.chunk_tokens, len(full_ids)))
                            for i in range(n_chunks_avail)]
            k = min(args.chunks_per_book, len(chunk_ranges))
            sel = torch.randperm(len(chunk_ranges), generator=rng)[:k].tolist()
            sel.sort()
            chosen_ranges = [chunk_ranges[i] for i in sel]

            for c_idx, (s, e) in enumerate(chosen_ranges):
                chunk_ids_list = full_ids[s:e]
                chunk_text = tok.decode(chunk_ids_list, skip_special_tokens=True)
                if not chunk_text.strip():
                    pbar.update(len(QGEN_PROMPTS))
                    continue
                chunk_ids_re = tok.encode(chunk_text, add_special_tokens=False)
                chunk_ids = torch.tensor(chunk_ids_re, dtype=torch.long).unsqueeze(0).cuda()

                for q_type, q_prompt in QGEN_PROMPTS:
                    q_full = build_qgen_prompt(model_kv, chunk_ids, q_prompt)
                    q_ids_out = model_generate(model_kv, q_full, args.max_q_tokens)
                    question_text = clean_question(
                        tok.decode(q_ids_out, skip_special_tokens=True))
                    if not question_text or len(question_text) < 10:
                        n_skip_bad_q += 1
                        pbar.update(1)
                        continue

                    a_full = build_qa_prompt(model_kv, chunk_ids, question_text)
                    a_ids_out = model_generate(model_kv, a_full, args.max_a_tokens)
                    if len(a_ids_out) < 4:
                        pbar.update(1)
                        continue

                    q_enc_ids = tok.encode(question_text, add_special_tokens=False)
                    fout.write(json.dumps({
                        "book_idx": book_idx,
                        "chunk_idx": c_idx,
                        "chunk_range": [s, e],
                        "q_type": q_type,
                        "ctx": chunk_text,
                        "ctx_len": len(chunk_ids_re),
                        "question": question_text,
                        "question_len": len(q_enc_ids),
                        "response_ids": a_ids_out,
                        "resp_len": len(a_ids_out),
                    }, ensure_ascii=False) + "\n")
                    fout.flush()
                    n_written += 1
                    pbar.update(1)

    pbar.close()
    print(f"Wrote {n_written} rows to {out_path}  (bad_q skipped={n_skip_bad_q})")


if __name__ == "__main__":
    main()
