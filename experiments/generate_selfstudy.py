"""Self-study data synthesis (Cartridges-style).
For each LongAlpaca paper, have the teacher (full-KV model) generate N diverse
question types, then for each Q have the teacher answer (greedy).
Output jsonl is fully compatible with the train_learnable_restore.py SFT loop:
   {"ctx": str, "ctx_len": int, "question": str, "question_len": int,
    "response_ids": [int,...], "resp_len": int}
"""
import argparse
import json
import os
import sys
import re
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import ModelKVzip


# 5 diverse question prompts; teacher fills in.
QGEN_PROMPTS = [
    ("factual",
     "Based on the paper above, write ONE specific factual retrieval question "
     "whose answer is a small concrete fact (a name, number, or short phrase) found in the paper. "
     "Output only the question, nothing else."),
    ("summary",
     "Based on the paper above, write ONE high-level question asking for the paper's "
     "main contribution or finding. Output only the question."),
    ("multihop",
     "Based on the paper above, write ONE multi-hop reasoning question that requires "
     "combining information from at least two different sections. Output only the question."),
    ("method",
     "Based on the paper above, write ONE concrete question about a specific method, "
     "algorithm, equation, or experimental setup. Output only the question."),
    ("compare",
     "Based on the paper above, write ONE comparison question that contrasts two "
     "concepts, methods, or results discussed in the paper. Output only the question."),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="qwen2.5-7b")
    p.add_argument("--paper-start", type=int, default=0)
    p.add_argument("--paper-end", type=int, default=500)
    p.add_argument("--min-ctx-tokens", type=int, default=2048)
    p.add_argument("--max-ctx-tokens", type=int, default=16000)
    p.add_argument("--max-q-tokens", type=int, default=64)
    p.add_argument("--max-a-tokens", type=int, default=512)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def strip_paper_extract(instruction: str) -> str:
    """LongAlpaca instruction = '... paper text ... Now the paper ends. Question: ...'
    We just want the paper body."""
    markers = ["Now the paper ends.", "Question:", "Now,", "Now you", "My question is"]
    best = -1
    for m in markers:
        i = instruction.rfind(m)
        if i > best:
            best = i
    return instruction[:best].rstrip() if best >= 0 else instruction


@torch.inference_mode()
def model_generate(model_kv, full_input_ids, max_new_tokens):
    out = model_kv.model.generate(
        full_input_ids,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=None,
        max_new_tokens=max_new_tokens,
        pad_token_id=model_kv.tokenizer.pad_token_id or model_kv.tokenizer.eos_token_id,
    )
    new_ids = out[:, full_input_ids.shape[1]:]
    return new_ids[0].tolist()


def build_qgen_prompt(model_kv, paper_ids, prompt_text):
    """Compose [sys + paper + qgen prompt + postfix]."""
    qgen_ids = model_kv.encode(f"\n\n{prompt_text}")
    full = torch.cat(
        [model_kv.sys_prompt_ids, paper_ids, qgen_ids, model_kv.postfix_ids],
        dim=1,
    )
    return full


def build_qa_prompt(model_kv, paper_ids, question_text):
    """Compose [sys + paper + question + postfix] for answer generation."""
    q_ids = model_kv.encode(f"\n\n{question_text}")
    full = torch.cat(
        [model_kv.sys_prompt_ids, paper_ids, q_ids, model_kv.postfix_ids],
        dim=1,
    )
    return full


def clean_question(text: str) -> str:
    """Strip leading numbering/quotes/etc from generated question."""
    text = text.strip()
    # remove leading number/dot/space
    text = re.sub(r'^["\'\d\.\):\-\s]+', '', text).strip()
    # cut at newline (just take first line)
    if "\n" in text:
        text = text.split("\n", 1)[0].strip()
    return text


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_kv = ModelKVzip(args.model, kv_type="evict")
    model_kv.model.eval()
    tok = model_kv.tokenizer

    ds = load_dataset("Yukang/LongAlpaca-12k", split="train")
    n_papers = min(args.paper_end, len(ds))
    print(f"papers {args.paper_start}..{n_papers}  output: {out_path}")

    n_written = n_seen = 0
    n_skip_short = n_skip_long = n_skip_bad_q = 0

    pbar = tqdm(total=(n_papers - args.paper_start) * len(QGEN_PROMPTS), desc="qa")
    with open(out_path, "w", encoding="utf-8") as fout:
        for paper_idx in range(args.paper_start, n_papers):
            row = ds[paper_idx]
            paper_text = strip_paper_extract(row.get("instruction", ""))
            if not paper_text.strip():
                continue
            n_seen += 1

            paper_ids_list = tok.encode(paper_text, add_special_tokens=False)
            if len(paper_ids_list) < args.min_ctx_tokens:
                n_skip_short += 1
                pbar.update(len(QGEN_PROMPTS))
                continue
            if len(paper_ids_list) > args.max_ctx_tokens:
                n_skip_long += 1
                pbar.update(len(QGEN_PROMPTS))
                continue
            paper_ids = torch.tensor(paper_ids_list, dtype=torch.long).unsqueeze(0).cuda()

            for q_type, q_prompt in QGEN_PROMPTS:
                # 1. generate question
                q_full = build_qgen_prompt(model_kv, paper_ids, q_prompt)
                q_ids_out = model_generate(model_kv, q_full, args.max_q_tokens)
                question_text = clean_question(tok.decode(q_ids_out, skip_special_tokens=True))
                if not question_text or len(question_text) < 10:
                    n_skip_bad_q += 1
                    pbar.update(1)
                    continue

                # 2. generate answer
                a_full = build_qa_prompt(model_kv, paper_ids, question_text)
                a_ids_out = model_generate(model_kv, a_full, args.max_a_tokens)
                if len(a_ids_out) < 4:
                    pbar.update(1)
                    continue

                q_enc_ids = tok.encode(question_text, add_special_tokens=False)
                fout.write(json.dumps({
                    "paper_idx": paper_idx,
                    "q_type": q_type,
                    "ctx": paper_text,
                    "ctx_len": len(paper_ids_list),
                    "question": question_text,
                    "question_len": len(q_enc_ids),
                    "response_ids": a_ids_out,
                    "resp_len": len(a_ids_out),
                }, ensure_ascii=False) + "\n")
                fout.flush()
                n_written += 1
                pbar.update(1)

    pbar.close()
    print(f"Wrote {n_written} rows to {out_path}")
    print(f"  papers seen={n_seen}  short_ctx={n_skip_short}  long_ctx={n_skip_long}  bad_q={n_skip_bad_q}")


if __name__ == "__main__":
    main()
