from collections import defaultdict
import json
import os


def load_eval_manifest(path: str, dataset_name: str):
    rows = []
    seen = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["dataset"] != dataset_name:
                continue
            key = (row["dataset"], row.get("hf_split"), int(row["sample_idx"]))
            if key in seen:
                raise ValueError(f"Duplicate manifest row: {key}")
            seen.add(key)
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows for dataset {dataset_name} in manifest {path}")
    splits = {row.get("hf_split") for row in rows}
    if len(splits) > 1:
        raise ValueError(f"eval.py supports one hf_split per run; got {sorted(splits)}")
    return rows, next(iter(splits))


def set_ratios(model_name):
    import os
    env = os.environ.get("KVZIP_EVAL_RATIOS")
    if env:
        return [float(x) for x in env.split(",") if x.strip()]
    return [0.8, 0.4, 0.2, 0.1, 0.05]


if __name__ == "__main__":
    from args import args
    from data import load_dataset_all, DataWrapper
    from model import ModelKVzip
    from utils import Evaluator, TimeStamp, set_gen_length, save_result

    args.kv_type = "retain"  # RetainCache enables efficient evaluation across multiple compression ratios with a single prefilling.
    model = ModelKVzip(args.model, kv_type=args.kv_type)

    manifest_rows = None
    manifest_split = None
    if args.manifest:
        manifest_rows, manifest_split = load_eval_manifest(args.manifest, args.data)
    n_data = max(100, args.idx + args.num)
    if manifest_rows is not None:
        n_data = max(int(row["sample_idx"]) for row in manifest_rows) + 1
    dataset = load_dataset_all(args.data, model.tokenizer, n_data=n_data,
                               split=manifest_split)  # list of data
    dataset = DataWrapper(args.data, dataset, model)
    set_gen_length(args.data, model)

    tt = TimeStamp(True)
    if manifest_rows is None:
        eval_indices = list(range(args.idx, min(args.idx + args.num, len(dataset))))
    else:
        eval_indices = [int(row["sample_idx"]) for row in manifest_rows]
    print("=" * 80, f"\nStart evaluation with {len(eval_indices)} samples")

    # SnapKV uses query-agnostic last-window scoring during prefill; the prune
    # step then uses the standard pair-threshold logic over the SnapKV scores.
    # kvzip_plus (Jégou & Jeblick 2026, arxiv 2601.07891): activate via env `KVZIP_PLUS=1`
    # with --level pair (output goes to output-pair.json for compat with existing scorers).
    import os as _os
    _kvzip_plus = _os.environ.get("KVZIP_PLUS", "") in ("1","true","yes")
    _contrastkv = _os.environ.get("CONTRASTKV", "") in ("1","true","yes")
    if args.level == "snapkv":
        score_method = "snapkv"
    elif args.level == "h2o":
        score_method = "h2o"
    elif _contrastkv:
        score_method = "contrastkv"
    elif _kvzip_plus:
        score_method = "kvzip_plus"
    else:
        score_method = "kvzip"
    prune_level = "pair" if args.level in ("snapkv", "h2o") else args.level

    for data_idx in eval_indices:
        # skip if output already exists (idempotent re-runs at larger --num)
        _folder_tag = f"_{args.tag}" if args.tag else ""
        _out_path = f"./results/{args.data}/{data_idx}_{args.model}{_folder_tag}/output-{args.level}.json"
        if os.path.exists(_out_path):
            print(f"[skip] {_out_path} exists")
            continue
        kv = dataset.prefill_context(data_idx, load_score=args.level == "head",
                                     score_method=score_method)
        inputs, info = dataset.generate_answer(data_idx, kv)
        eval = Evaluator(model, inputs, info)

        outputs = defaultdict(list)
        for ratio in set_ratios(args.model):
            thres, ratio_true = kv.prune(ratio, prune_level)
            results = eval(kv, generate=True)  # generation

            for fmt, v in results.items():
                outputs[fmt].append([[ratio, round(ratio_true, 4), round(thres, 4)], v])

        save_result(args, args.data, outputs, data_idx)

        tt(f"{args.data}-{data_idx}")
        del kv, inputs, info, eval
    print("Finished.")
