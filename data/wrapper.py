import torch
from typing import List, Tuple, Union, Optional
from collections import defaultdict

from attention.kvcache import RetainCache, EvictCache
from model import ModelKVzip


def get_query(task, q=None, options=None):
    if task == "repeat":
        query = f"Repeat the previous context exactly."
    elif task == "qa":
        if options is not None:
            labels = "ABCDE"
            num_options = len(options)
            opts_str = "\n".join(f"{labels[i]}. {opt}" for i, opt in enumerate(options))
            option_list = "A, B, C, D, or E" if num_options > 4 else "A, B, C, or D"
            query = (
                f"{q}\n\n{opts_str}\n\n"
                f"Please think very briefly and then respond with **only** the letter "
                f"({option_list}) of the correct option. If you are not sure, still make a guess."
            )
        elif q is None:
            query = f"Q: Answer the question based on the previous context."
        elif q.startswith("\n\n"):
            # LongBench-standard prompt (already formatted with instruction + Answer:)
            query = q
        else:
            query = f"Q: {q}"
    elif task == "reason":
        query = f"Reason and answer the question. You must say the answer in the last sentence beginning with 'The answer is'. Q: {q}"
    elif task == "summarize":
        query = f"Please summarize the previous context."
    else:
        raise ValueError(f"Invalid task: {task}")

    return query


def parse_mcq_letter(text):
    """Parse letter answer (A-E) from model output. Returns index 0-4 or -1 (no parse)."""
    import re
    text = text.strip()
    if text and text[0].upper() in "ABCDE":
        return ord(text[0].upper()) - ord('A')
    match = re.search(r'\(([A-E])\)', text)
    if match:
        return ord(match.group(1)) - ord('A')
    match = re.search(r'\b([A-E])\b', text)
    if match:
        return ord(match.group(1)) - ord('A')
    return -1


class DataWrapper():

    def __init__(self, dataname, dataset, model: ModelKVzip):
        self.name, self.dataset, self.model = dataname, dataset, model
        model.set_chat_template(dataname)

    def __len__(self):
        return len(self.dataset)

    def prefill_context(self, idx: int, load_score=False,
                        score_method: str = "kvzip") -> Union[RetainCache, EvictCache]:
        """ Prefill and scoring KV importance.

        score_method: "kvzip" (default) or "snapkv".
        """
        data = self.dataset[idx]
        ctx_ids = self.model.encode(data['context'])

        kv = self.model.prefill(ctx_ids, load_score=load_score,
                                score_method=score_method)

        print(f"# prefill {self.model.name} {self.name}-{idx}:", end=" ")
        print(f"{len(ctx_ids[0])} tokens, KV cache {kv._mem()} GB, {kv.key_cache[0].dtype}")
        return kv

    def _prepare_query(self, data, kv, inputs: dict, task: str):
        """ Generate answers of each task for evaluation.
            For each task, we store (query, answer, grount_truth) in inputs
        """
        import os as _os
        _kvpress_compat = (_os.environ.get("KVPRESS_COMPAT", "0") in ("1", "true", "yes")
                           and "answer_prefix" in data)
        if task in ["qa", "reason"]:
            print("# Generated output | Ground truth")
            is_mcq = "options" in data
            for i, (q, gt) in enumerate(zip(data['question'], data['answers'])):
                if _kvpress_compat:
                    # kvpress protocol: raw question + generation suffix + answer_prefix,
                    # no "Q: " wrapper; per-task max_new_tokens.
                    q_ids = self.model.apply_template_kvpress(q, data["answer_prefix"][i])
                    if "max_new_tokens" in data:
                        self.model.gen_kwargs["max_new_tokens"] = int(data["max_new_tokens"][i])
                else:
                    opts = data["options"][i] if is_mcq else None
                    q = get_query(task, q, options=opts)
                    q_ids = self.model.apply_template(q)

                a = self.model.generate(q_ids, kv=kv)

                a_ids = self.model.encode(a)
                gt_ids = self.model.encode(gt)

                tag = f"qa-{i}" if i > 0 else "qa"
                inputs[tag] = {"q": q_ids, "a": a_ids, "gt": gt_ids}
                if _kvpress_compat and "max_new_tokens" in data:
                    # Evaluator re-generates per ratio — carry the per-task gen length
                    inputs[tag]["max_new_tokens"] = int(data["max_new_tokens"][i])
                inputs["eval_task"].append(tag)

                print(f"[QA {i}] {a} | {gt}")

        else:
            q = get_query(task)
            q_ids = self.model.apply_template(q)

            if task == "repeat":
                a_ids = kv.ctx_ids
            else:
                a = self.model.generate(q_ids, kv=kv)
                a_ids = self.model.encode(a)

            gt_ids = a_ids  # no ground truth
            inputs[task] = {"q": q_ids, "a": a_ids, "gt": gt_ids}
            if a_ids.shape[-1] < 512:
                inputs["eval_task"].append(task)

    @torch.inference_mode()
    def generate_answer(self, idx: int, kv: Union[RetainCache, EvictCache]):
        """ Prepare inputs, answers, and prediction probabilities (with full KV cache) for evaluation.
        """
        data = self.dataset[idx]

        eval_task = ["qa"]

        inputs = defaultdict(list)
        for task in eval_task:
            self._prepare_query(data, kv, inputs, task)

        info = defaultdict(dict)
        for fmt in inputs["eval_task"]:
            input_ids = torch.cat([inputs[fmt][k] for k in ["q", "a"]], dim=1)
            info[fmt]["prob"] = self.model._prob(input_ids, kv, device="cpu")

        return inputs, info
