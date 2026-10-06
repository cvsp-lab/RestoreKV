import json
import os

from datasets import load_dataset


def load_dataset_all(name, tokenizer, n_data=100, split=None):
    """Load an evaluation dataset.

    Each example has the format
        {context: str, question: List[str], answers: List[str], ...}
    where a single context is reused across all of its questions
    (query-agnostic compression).

    Supported datasets: "quality", "qasper", "longhealth" (or "longhealthN"),
    "scbench_*" (SCBench sub-tasks, incl. tiny/short/mid length tags),
    "ruler_{4096,8192,16384}" (and ruler360/ruler1300 aliases).
    """
    if name == "quality":
        dataset = load_quality(n_data)
    elif name == "qasper":
        dataset = load_qasper(n_data)
    elif name.startswith("longhealth"):
        dataset = load_longhealth(name, n_data)
    elif "scbench" in name:
        dataset = load_scbench(name)
    elif name.startswith("ruler360"):
        # Alias: force n_data=360 (~28 samples x 13 tasks). Map to ruler_8192 config.
        alias = name.replace("ruler360", "ruler_8192") if name != "ruler360" else "ruler_8192"
        dataset = load_ruler(alias, 360)
    elif name.startswith("ruler1300"):
        # Alias: force n_data=1300 (100 samples x 13 tasks). Map to ruler_8192 config.
        alias = name.replace("ruler1300", "ruler_8192") if name != "ruler1300" else "ruler_8192"
        dataset = load_ruler(alias, 1300)
    elif name.startswith("ruler"):
        dataset = load_ruler(name, n_data)
    else:
        raise ValueError(f"Invalid dataset: {name}")

    print(f"\n{name} loaded, #data: {len(dataset)}")
    return dataset


def _find_data_file(filename):
    """Locate a dataset file next to this module (data/)."""
    path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"{filename} not found at {path}")


def load_quality(n_data=100):
    """QuALITY: long fiction MCQ (4 options). One article -> many questions.

    Each item in our wrapper format:
      context: article text
      question: list of question strings (options appended inline for the model)
      answers: list of correct option-text strings
      options: list-of-list of option texts
      gold_labels: list of 0-indexed correct option indices
    """
    path = _find_data_file("QuALITY.v1.0.1.htmlstripped.dev")
    # Merge JSONL entries that share the same (article_id, article_text) so each
    # unique article carries its full question set. The raw dev file splits each
    # article into multiple entries with disjoint question subsets.
    merged_order = []
    merged = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            article = row.get("article", "").strip()
            if not article:
                continue
            key = (row.get("article_id"), article)
            if key not in merged:
                merged[key] = {"article": article, "questions": []}
                merged_order.append(key)
            merged[key]["questions"].extend(row.get("questions", []))

    dataset = []
    for key in merged_order:
        article = merged[key]["article"]
        questions, answers, options_list, gold_labels = [], [], [], []
        for q in merged[key]["questions"]:
            q_text = q.get("question", "").strip()
            opts = q.get("options", [])
            gold_1based = q.get("gold_label")
            if gold_1based is None or not opts:
                continue
            gold_idx = int(gold_1based) - 1
            if gold_idx < 0 or gold_idx >= len(opts):
                continue
            questions.append(q_text)
            answers.append(opts[gold_idx])
            options_list.append(opts)
            gold_labels.append(gold_idx)

        if not questions:
            continue
        dataset.append({
            "context": article,
            "question": questions,
            "answers": answers,
            "options": options_list,
            "gold_labels": gold_labels,
        })
        if len(dataset) >= n_data:
            break
    return dataset


def load_longhealth(name, n_data=100):
    """LongHealth: medical 5-choice MCQ. One patient per article (longhealth)
    or N patients grouped (longhealthN, e.g. longhealth5).

    Returns wrapper format with options/gold_labels for MCQ scoring.
    """
    path = _find_data_file("longhealth_benchmark_v5.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if name == "longhealth":
        patients_per_article = 1
    else:
        patients_per_article = int(name[len("longhealth"):])

    sorted_ids = sorted(raw.keys())
    num_articles = len(sorted_ids) // patients_per_article

    dataset = []
    for article_idx in range(min(num_articles, n_data)):
        start = article_idx * patients_per_article
        end = start + patients_per_article
        group_ids = sorted_ids[start:end]

        article_parts = []
        questions, answers, options_list, gold_labels = [], [], [], []

        for pid in group_ids:
            patient = raw[pid]
            for note_id, note_text in patient["texts"].items():
                article_parts.append(f"<{note_id}>\n{note_text}\n</{note_id}>")

            patient_info = (
                f"ID {pid}, Name: {patient['name']}, "
                f"Birthday: {patient['birthday']}, "
                f"Diagnosis: {patient['diagnosis']}"
            )

            for q in patient["questions"]:
                q_text = q["question"].strip()
                # actual keys: answer_a..answer_e
                opts = [q.get(f"answer_{c}", "") for c in ["a", "b", "c", "d", "e"]]
                opts = [o for o in opts if o]  # drop empty
                if not opts:
                    continue
                correct = q.get("correct", "").strip()
                gold_idx = next(
                    (i for i, o in enumerate(opts) if o.strip() == correct), 0
                )
                if patients_per_article > 1:
                    q_with_patient = f"About patient {patient_info}: {q_text}"
                else:
                    q_with_patient = q_text
                # raw question — options/letter prompt added in get_query()
                questions.append(q_with_patient)
                answers.append(correct)
                options_list.append(opts)
                gold_labels.append(gold_idx)

        if not questions:
            continue
        context = "\n\n".join(article_parts).strip()
        dataset.append({
            "context": context,
            "question": questions,
            "answers": answers,
            "options": options_list,
            "gold_labels": gold_labels,
        })
    return dataset


def load_qasper(n_data=100):
    """QASPER (Dasigi et al.) dev split: scientific paper long-context QA.
    Each paper -> abstract + full text as ctx, multiple (question, multi-annotator refs) pairs.
    Scoring: F1 over best-matching annotator reference (include + token-F1 hybrid).
    """
    path = _find_data_file("qasper-dev-v0.3.json")
    with open(path) as f:
        raw = json.load(f)

    dataset = []
    for paper_id, paper in raw.items():
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")

        # reconstruct full text
        parts = [abstract] if abstract else []
        for sec in paper.get("full_text", []):
            name = sec.get("section_name") or ""
            if name:
                parts.append(f"## {name}")
            parts.extend(sec.get("paragraphs", []))
        ctx = "\n\n".join(p for p in parts if p).strip()
        if not ctx:
            continue

        questions, answers_str, answer_refs = [], [], []
        for qa in paper.get("qas", []):
            q_text = qa.get("question", "").strip()
            if not q_text:
                continue
            refs = []
            for annot in qa.get("answers", []):
                ans = annot.get("answer", {})
                if ans.get("unanswerable"):
                    refs.append("Unanswerable")
                elif ans.get("yes_no") is not None:
                    refs.append("Yes" if ans["yes_no"] else "No")
                elif ans.get("extractive_spans"):
                    refs.append(", ".join(ans["extractive_spans"]))
                elif ans.get("free_form_answer"):
                    refs.append(ans["free_form_answer"].strip())
            refs = [r for r in refs if r]
            if not refs:
                continue
            questions.append(q_text)
            answers_str.append(" | ".join(refs))   # for wrapper.encode compat
            answer_refs.append(refs)

        if not questions:
            continue

        dataset.append({
            "context": ctx,
            "question": questions,
            "answers": answers_str,
            "answer_refs": answer_refs,
            "title": title,
            "paper_id": paper_id,
        })
        if len(dataset) >= n_data:
            break
    return dataset


def load_scbench(name):
    """SCBench sub-task loader.

    `name` is e.g. "scbench_kv", "scbench_vt", "scbench_mf", "scbench_repoqa",
    "scbench_qa_eng", "scbench_choice_eng", "scbench_prefix_suffix",
    "scbench_summary", "scbench_many_shot", plus length tags
    "scbench_kv_tiny/short/mid" (~8k/20k/60k tokens).

    Preprocessed to the wrapper format and hosted at
    Jang-Hyun/SCBench-preprocessed (subsampled so context <125K LLaMA3 tokens).
    """
    check_scbench_name(name)
    samples = load_dataset('Jang-Hyun/SCBench-preprocessed',
                           data_files=f"{name}.parquet",
                           split='train')

    dataset = []
    for data in samples:
        d = {}
        d["context"] = data["prompts"][0]
        d["question"] = data["prompts"][1:]  # only the first question matters now
        d["answers"] = []
        for gt in data["ground_truth"]:
            if isinstance(gt, list):
                gt = ", ".join(gt)
            else:
                gt = str(gt)
            d["answers"].append(gt)

        dataset.append(d)

    return dataset


def check_scbench_name(name):
    name = name.split("scbench_")[1]
    possible_tags = [
        "many_shot",
        "mf",
        "repoqa",
        "choice_eng",
        "prefix_suffix",
        "summary",
        "qa_eng",
        "vt",
        "kv",
        "summary_with_needles",
        "repoqa_and_kv",
    ]
    if "tiny" in name:
        name = name.split("_tiny")[0]
    elif "short" in name:
        name = name.split("_short")[0]
    elif "mid" in name:
        name = name.split("_mid")[0]

    assert name in possible_tags, "SCBench data name not exist!"


def load_ruler(name, n_data=100):
    """RULER long-context benchmark (simonjegou/ruler HF dataset).

    name: 'ruler_4096' / 'ruler_8192' / 'ruler_16384' — picks the HF config.

    The HF dump is sorted by task (500 per task, 13 tasks). To get
    task-balanced evaluations from small n_data, we stratify: take roughly
    ceil(n_data / num_tasks) per task in stable order. Final list has up to
    n_data items, with tasks interleaved.

    Set KVPRESS_COMPAT=1 to reproduce the full kvpress RULER protocol: the
    question is kept raw and `answer_prefix` is recorded separately (the wrapper
    places it after the chat-template generation suffix), together with kvpress
    per-task max_new_tokens. Otherwise `answer_prefix` (e.g. "Answer:") is
    appended to the question unless RULER_NO_ANSWER_PREFIX=1.
    """
    parts = name.split("_", 1)
    config_name = parts[1] if len(parts) > 1 else "4096"
    if config_name not in ("4096", "8192", "16384"):
        raise ValueError(f"Unsupported RULER config '{config_name}'. "
                         f"Use ruler_4096 / ruler_8192 / ruler_16384.")

    raw = load_dataset("simonjegou/ruler", config_name, split="test")

    # bucket by task in original order
    buckets = {}
    task_order = []
    for row in raw:
        t = row["task"]
        if t not in buckets:
            buckets[t] = []
            task_order.append(t)
        buckets[t].append(row)

    n_tasks = len(task_order)
    per_task = max(1, (n_data + n_tasks - 1) // n_tasks)
    # take per_task from each bucket, interleave round-robin so task diversity
    # appears even when downstream truncates to small n_data
    picked = []
    for j in range(per_task):
        for t in task_order:
            if j < len(buckets[t]):
                picked.append(buckets[t][j])
            if len(picked) >= n_data:
                break
        if len(picked) >= n_data:
            break

    import os as _os
    _compat = _os.environ.get("KVPRESS_COMPAT", "0") in ("1", "true", "yes")
    _use_prefix = (not _compat) and \
        _os.environ.get("RULER_NO_ANSWER_PREFIX", "0") not in ("1", "true", "yes")

    # kvpress evaluation/benchmarks/ruler MAX_NEW_TOKENS (by task category)
    _KVPRESS_MAX_NEW_TOKENS = {"niah": 128, "vt": 30, "cwe": 120, "fwe": 50, "qa": 32}

    groups = {}
    order = []
    for row in picked:
        ctx = row["context"]
        ctx_key = hash(ctx)
        if ctx_key not in groups:
            groups[ctx_key] = {
                "context": ctx,
                "question": [],
                "answers": [],
                "answer_refs": [],
                "task": [],
            }
            if _compat:
                groups[ctx_key]["answer_prefix"] = []
                groups[ctx_key]["max_new_tokens"] = []
            order.append(ctx_key)
        q_text = row["question"]
        ap = str(row.get("answer_prefix", "") or "")
        if _use_prefix and ap.strip():
            q_text = q_text.rstrip() + "\n" + ap.strip()
        groups[ctx_key]["question"].append(q_text)
        refs = list(row["answer"])
        groups[ctx_key]["answer_refs"].append(refs)
        groups[ctx_key]["answers"].append(" | ".join(str(r) for r in refs))
        groups[ctx_key]["task"].append(row["task"])
        if _compat:
            groups[ctx_key]["answer_prefix"].append(ap)
            mnt = row.get("max_new_tokens") or \
                _KVPRESS_MAX_NEW_TOKENS.get(str(row["task"]).split("_")[0], 128)
            groups[ctx_key]["max_new_tokens"].append(int(mnt))

    return [groups[k] for k in order]
