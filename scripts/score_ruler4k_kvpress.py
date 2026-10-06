"""Score RULER outputs exactly like kvpress evaluation/benchmarks/ruler/calculate_metrics.py.

- per-task grouping; qa_* -> string_match_part (any-ref), others -> string_match_all
  (per-ref recall avg), x100 round(2)
- control chars [\\x00-\\x1f] stripped from predictions (kvpress does this)
- final = macro average over the 13 task scores (kvpress leaderboard convention)

Usage:
  python -B scripts/score_ruler4k_kvpress.py -m qwen3-4b -t kvzip_pair_kvcompat [-d ruler_4096] [-n 6500]
"""
import argparse, glob, json, os, re, sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True)
parser.add_argument("-t", "--tag", required=True)
parser.add_argument("-d", "--data", default="ruler_4096")
parser.add_argument("-n", "--num", type=int, default=6500)
args = parser.parse_args()

from data.load import load_dataset_all
data = load_dataset_all(args.data, None, n_data=args.num)

# per grouped-sample: parallel lists of (task, refs) per question
sample_meta = []
for d in data:
    refs_l = d.get("answer_refs") or []
    tasks = d.get("task") or ["qa"] * len(refs_l)
    sample_meta.append(list(zip(tasks, [[str(x) for x in r] for r in refs_l])))

_np = re.compile(r"[\x00-\x1f]")

# {ratio: {task: [(pred, refs), ...]}}
per_ratio = defaultdict(lambda: defaultdict(list))
dirs = sorted(glob.glob(f"results/{args.data}/*_{args.model}_{args.tag}/output-pair.json"))
n_used = 0
for f in dirs:
    idx = int(os.path.basename(os.path.dirname(f)).split("_", 1)[0])
    if idx >= len(sample_meta):
        continue
    try:
        out = json.load(open(f))
    except Exception:
        continue
    n_used += 1
    for tt, ent in out.items():
        if not tt.startswith("qa"):
            continue
        qi = 0 if tt == "qa" else int(tt.split("-", 1)[1])
        if qi >= len(sample_meta[idx]):
            continue
        task, refs = sample_meta[idx][qi]
        for info, val in ent:
            r = round(float(info[0]), 4)
            pred = _np.sub("", str(val.get("pruned", "")).strip()).strip()
            per_ratio[r][task].append((pred, refs))

if not per_ratio:
    sys.exit(f"No outputs found for results/{args.data}/*_{args.model}_{args.tag}/")

def string_match_part(pairs):
    return round(sum(max(1.0 if r.lower() in p.lower() else 0.0 for r in refs)
                     for p, refs in pairs) / len(pairs) * 100, 2)

def string_match_all(pairs):
    return round(sum(sum(1.0 if r.lower() in p.lower() else 0.0 for r in refs) / len(refs)
                     for p, refs in pairs) / len(pairs) * 100, 2)

print(f"# {args.model} / {args.tag} / {args.data} — {n_used} sample dirs")
for r in sorted(per_ratio, reverse=True):
    by_task = per_ratio[r]
    scores = {}
    for task in sorted(by_task):
        fn = string_match_part if task.split("_")[0] == "qa" else string_match_all
        scores[task] = fn(by_task[task])
    macro = sum(scores.values()) / len(scores)
    n_q = sum(len(v) for v in by_task.values())
    print(f"\n== ratio {r:g} (n={n_q}, {len(scores)} tasks, macro {macro:.4f}) ==")
    for task, s in scores.items():
        print(f"  {task:24s} {s:7.2f}  (n={len(by_task[task])})")
