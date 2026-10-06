import os
import torch
from collections import defaultdict
from datasets import load_dataset
from results.metric import evaluate_answer
from eval import set_ratios


def parse_answer(name):
    answers = []
    subtasks = []
    if "many_shot" in name:
        answers = []
        samples = load_dataset('Jang-Hyun/SCBench-preprocessed',
                           data_files=f"{name}.parquet",
                           split='train')
        for data in samples:
            d = []
            for q, gt in zip(data["prompts"][1:], data["ground_truth"]):
                # parse options, e.g., "(A) xxx" from gt = A
                cand = [sol for sol in q.split('\n') if f'({gt})' in sol]
                if len(cand) != 1:
                    print(f"Error: {q} {gt}")
                d.append(cand[0].strip())

            answers.append(d)

    elif "repoqa" in name:
        answers = []
        samples = load_dataset('Jang-Hyun/SCBench-preprocessed',
                           data_files=f"{name}.parquet",
                           split='train')
        for data in samples:
            d = defaultdict(list)
            d["lang"] = data["lang"]
            d["repo"] = data["repo"]
            d["func_name"] = data["func_name"]
            d["ground_truth"] = data["ground_truth"]
            answers.append(d)

            if "task" in data:
                subtasks.append(data["task"])

    elif "summary_with_needles" in name:
        answers = []
        subtasks = []
        samples = load_dataset('Jang-Hyun/SCBench-preprocessed',
                           data_files=f"{name}.parquet",
                           split='train')
        for data in samples:
            d = defaultdict(list)
            subtasks.append(data["task"])
            answers.append(data["ground_truth"])

    elif name == "quality" or name.startswith("longhealth"):
        from data.load import load_quality, load_longhealth
        ds = load_quality(100) if name == "quality" else load_longhealth(name, 100)
        # MCQ: pack (answers_text, gold_labels) per sample as (refs, gold_idx_list)
        for d in ds:
            answers.append({
                "answers": d["answers"],
                "gold_labels": d["gold_labels"],
            })

    elif name.startswith("ruler"):
        from data.load import load_ruler
        ds = load_ruler(name, 100)
        for d in ds:
            # answers per sample = list-of-list (each qa has a list of acceptable refs)
            answers.append(d["answer_refs"])

    elif name == "qasper":
        from data.load import load_qasper
        ds = load_qasper(100)
        for d in ds:
            answers.append(d["answer_refs"])

    return answers, subtasks


def mean(l):
    return sum(l) / len(l)


def avg_list_of_list(l):
    l = [vals for vals in l if vals]
    if not l:
        return float("nan")
    score = mean([mean(vals) for vals in l])
    return score


def mean_or_nan(vals):
    if not vals:
        return float("nan")
    return mean(vals)


def result_sample_idx(path):
    return int(os.path.basename(os.path.dirname(path)).split("_", 1)[0])


def set_ratios(model_name):
    if "duo" == model_name:
        ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    else:
        ratios = [1.0, 0.8, 0.4, 0.2, 0.1, 0.05]
    return ratios


if __name__ == "__main__":
    import argparse
    import os
    import glob
    import json
    from model.load import get_model_id

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="llama3-8b")
    parser.add_argument("-d", "--data", type=str, default="squad")
    parser.add_argument("-s", "--level", type=str, default="pair")
    parser.add_argument("--task", type=str, default="qa")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--idx", type=int, default=0, help="first sample index to parse")
    parser.add_argument("--num", type=int, default=None, help="number of samples to parse")
    args = parser.parse_args()

    ratios = set_ratios(args.model)

    cur_path = "./results"
    answers_supp, subtasks = parse_answer(args.data)

    folder_tag = f"_{args.tag}" if args.tag else ""
    folder_list = glob.glob(
        os.path.join(cur_path, f"{args.data}/*_{args.model}{folder_tag}/output-{args.level}.json")
    )
    folder_list = sorted(
        folder_list,
        key=result_sample_idx,
    )
    if args.num is None:
        end_idx = None
    else:
        end_idx = args.idx + args.num
    folder_list = [
        path for path in folder_list if result_sample_idx(path) >= args.idx
        and (end_idx is None or result_sample_idx(path) < end_idx)
    ]
    sample_ids = [result_sample_idx(path) for path in folder_list]
    
    print(f"\nEvaluate {args.data} on {len(folder_list)} samples, {args.model}")
    print(f"level: {args.level}")
    if args.tag:
        print(f"tag: {args.tag}")
    if args.num is None:
        print(f"sample range: idx >= {args.idx}")
    else:
        print(f"sample range: {args.idx} <= idx < {end_idx}")
    if sample_ids:
        print(f"sample ids: {sample_ids[0]}..{sample_ids[-1]} ({len(sample_ids)} files)")
    else:
        print("sample ids: none")

    eval_list_ratio = {r: [] for r in ratios}
    kv_ratio_stats = {r: [] for r in ratios}
    context_ratio_stats = {r: [] for r in ratios}
    restore_overhead_stats = {r: [] for r in ratios}
    live_kv_token_stats = {r: [] for r in ratios}
    ctx_len_stats = {r: [] for r in ratios}
    for i, file in enumerate(folder_list):
        with open(file, "r") as f:
            data = json.load(f)

        preds = defaultdict(list)
        answers = []
        task_names = [k for k in list(data.keys()) if k.startswith(args.task)]

        # parse generated responses from json files
        for fmt in task_names:
            for output_per_ratio in data[fmt]:
                info, text = output_per_ratio
                ratio_ = info[0]
                preds[ratio_].append(text["pruned"])
                if ratio_ in kv_ratio_stats and len(info) > 1:
                    kv_ratio_stats[ratio_].append(info[1])

                metadata = info[3] if len(info) > 3 and isinstance(info[3], dict) else {}
                if metadata and ratio_ in context_ratio_stats:
                    context_ratio = metadata.get("context_ratio_true")
                    restore_overhead = metadata.get("restore_overhead_ratio")
                    restore_tokens = metadata.get("restore_tokens")
                    ctx_len = metadata.get("ctx_len")

                    if context_ratio is not None:
                        context_ratio_stats[ratio_].append(context_ratio)
                    if restore_overhead is not None:
                        restore_overhead_stats[ratio_].append(restore_overhead)
                    if ctx_len is not None:
                        ctx_len_stats[ratio_].append(ctx_len)
                    if (
                            context_ratio is not None
                            and restore_tokens is not None
                            and ctx_len is not None):
                        live_kv_token_stats[ratio_].append(context_ratio * ctx_len +
                                                           restore_tokens)

            if len(preds[1.0]) < len(preds[ratios[-1]]):  # add full cache results
                preds[1.0].append(text["full__"])
                kv_ratio_stats[1.0].append(1.0)
            answers.append(text["answer"])

        # for some tasks, evaluation require additional information (e.g., code language in repoqa)
        gold_labels = None
        if answers_supp:
            answers = answers_supp[i]
            # MCQ: parse_answer returns dict {answers, gold_labels}
            if isinstance(answers, dict) and "gold_labels" in answers:
                gold_labels = answers["gold_labels"]
                answers = answers["answers"]
        subtask = None
        if subtasks:
            subtask = subtasks[i]

        # evaluate answers across compression ratios
        for r in ratios:
            if not preds.get(r):  # skip ratios with no data (e.g., Ours only ran 0.2/0.1/0.05)
                continue
            perf = evaluate_answer(
                preds[r], answers, args.data, args.task,
                subtask=subtask, gold_labels=gold_labels,
            )
            eval_list_ratio[r].append(perf)

    print(
        "ratio avg_performance avg_kv_ratio avg_context_ratio "
        "avg_restore_overhead avg_live_kv_tokens avg_ctx_len"
    )
    for r in ratios:
        print(
            f"{r:.2f}  "
            f"{avg_list_of_list(eval_list_ratio[r])*100:.2f}  "
            f"{mean_or_nan(kv_ratio_stats[r]):.4f}  "
            f"{mean_or_nan(context_ratio_stats[r]):.4f}  "
            f"{mean_or_nan(restore_overhead_stats[r]):.4f}  "
            f"{mean_or_nan(live_kv_token_stats[r]):.2f}  "
            f"{mean_or_nan(ctx_len_stats[r]):.2f}"
        )
