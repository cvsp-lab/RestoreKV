# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Code adapted from https://github.com/microsoft/MInference/tree/main/scbench

import re
import string
from collections import Counter, defaultdict
from rouge import Rouge
from results.repo_qa_utils import compute_score as compute_repoqa_score


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def replace_num(text):
        word_to_number = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9"
        }

        pattern = re.compile(r'\b(' + '|'.join(word_to_number.keys()) + r')\b')
        text = pattern.sub(lambda x: word_to_number[x.group()], text)

        return text

    return replace_num(white_space_fix(remove_articles(remove_punc(lower(s)))))


def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]

    # similar outcome to the above, but much slower (source: scbench)
    # import evaluate
    # ROUGE_SCORER = evaluate.load("rouge")
    # score = ROUGE_SCORER.compute(predictions=[prediction],
    #                              references=[ground_truth],
    #                              use_aggregator=False)
    # return score["rougeLsum"][0]


def f1_score(pred, ref, normalize=True):
    if normalize:
        pred, ref = normalize_answer(pred), normalize_answer(ref)
    prediction_tokens = pred.split()
    ground_truth_tokens = ref.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def include_score(pred, ref, normalize=True):
    if normalize:
        pred, ref = normalize_answer(pred), normalize_answer(ref)
    return ref in pred


def include_score_multi(pred, ref, normalize=True):
    # scbench_vt
    refs = ref.split(", ")
    if normalize:
        pred = normalize_answer(pred)
        refs = [normalize_answer(r) for r in refs]

    scores = [ref in pred for ref in refs]
    score = sum(scores) / len(scores)
    return score


def include_score_gsm(pred, ref, normalize=True):
    ref = ref.strip().split("#### ")[-1]
    if normalize:
        pred, ref = normalize_answer(pred), normalize_answer(ref)
    return ref in pred


def include_score_manyshot(pred, ref, normalize=True):
    if "(" in pred and "(" in ref:
        pred = pred.split("(")[1].split(")")[0]  # (A) xx => A
        ref = ref.split("(")[1].split(")")[0]
        val = pred == ref
    else:
        if ref[0] == "(":
            ref = ref.split(")")[1].strip()  # (A) xx => xx
        if normalize:
            pred, ref = normalize_answer(pred), normalize_answer(ref)
        val = ref in pred
    return val


def exact_match_score(pred, ref, normalize=True):
    if normalize:
        pred, ref = normalize_answer(pred), normalize_answer(ref)
    return pred == ref


def qasper_score(pred, refs, normalize=True):
    """QASPER: token-F1 vs best-matching annotator reference (max over refs).
    refs is a list of reference strings; we take max(f1_score(pred, ref)) across refs,
    matching the standard QASPER eval (per-annotator F1, max aggregation).
    """
    if isinstance(refs, str):
        refs = [refs]
    if not refs:
        return 0.0
    return max(f1_score(pred, r, normalize=normalize) for r in refs)


def ruler_score(pred, refs, normalize=True):
    """RULER: recall over a list of reference strings.
    refs may be passed as a list (preferred) or as a JSON-like list string.
    Returns fraction of refs that appear in pred (after normalize).
    """
    if isinstance(refs, str):
        # tolerate stringified list (e.g. "['a','b']")
        try:
            import ast
            parsed = ast.literal_eval(refs)
            if isinstance(parsed, (list, tuple)):
                refs = list(parsed)
            else:
                refs = [str(parsed)]
        except Exception:
            refs = [refs]
    if not refs:
        return 0.0
    if normalize:
        pred_n = normalize_answer(pred)
        refs_n = [normalize_answer(r) for r in refs]
    else:
        pred_n, refs_n = pred.lower(), [r.lower() for r in refs]
    hits = sum(1 for r in refs_n if r and r in pred_n)
    return hits / len(refs_n)


def mcq_letter_score(pred, gold_label):
    """MCQ (QuALITY / LongHealth): parse predicted letter, compare to gold_label index.
    Returns 1.0 if pred letter == gold_label (0-indexed), else 0.0.
    Unparseable predictions count as wrong.
    """
    from data.wrapper import parse_mcq_letter
    pred_idx = parse_mcq_letter(pred)
    if pred_idx == -1:
        return 0.0
    return float(pred_idx == gold_label)


def repoqa_score(preds, refs, subtask=None):
    needle_by_repo = defaultdict(list)
    for name, gt in zip(refs["func_name"], refs["ground_truth"]):
        needle_by_repo[refs["repo"]].append({"needle": gt, "name": name})

    pred_list = []

    for idx in range(len(preds)):
        if subtask is not None:
            if not "repoqa" in subtask[idx]:
                continue

        result = {}
        result["prediction"] = preds[idx]
        if preds[idx].endswith("</s>"):
            result["prediction"] = preds[idx][:-4]
        if len(result["prediction"].strip()) == 0:
            continue

        result["lang"] = refs["lang"]
        result["repo"] = refs["repo"]
        result["func_name"] = refs["func_name"][idx]
        result["ground_truth"] = refs["ground_truth"][idx]
        pred_list.append(result)

    acc = compute_repoqa_score(pred_list, None, needle_by_repo)
    acc = acc["scores"]["all"][0.8]["pass@1"]
    return acc


def evaluate_answer(preds, refs, dataname, format, similarity=False, subtask=None, gold_labels=None):
    score = []
    if "repoqa" in dataname and not similarity:
        if "repoqa_and_kv" in dataname:
            for i, (pred, ref) in enumerate(zip(preds, refs["ground_truth"])):
                if pred.endswith("</s>"):
                    pred = pred[:-4]
                if len(pred.strip()) == 0:
                    score.append(0.0)
                    continue
                if "kv" in subtask[i]:
                    score.append(include_score(pred, ref))
                    print("include_score..", end="\r")

            score_repoqa = repoqa_score(preds, refs, subtask)
            score = [sum(score) / len(score), score_repoqa]
            print("repoqa_score..", end="\r")
        else:
            score.append(repoqa_score(preds, refs))
            print("repoqa_score..", end="\r")

    else:
        for i, (pred, ref) in enumerate(zip(preds, refs)):
            if pred.endswith("</s>"):
                pred = pred[:-4]

            if len(pred.strip()) == 0:
                score.append(0.0)
                continue

            if subtask is not None:
                dataname = subtask[i]

            if similarity:
                score.append(f1_score(pred, ref))
            elif format != "qa":
                score.append(rouge_score(pred, ref))
                print("rouge_score..", end="\r")
            else:
                if "_vt" in dataname:
                    score.append(include_score_multi(pred, ref, normalize=False))
                    print("include_score_multi..", end="\r")

                elif "_mf" in dataname:
                    score.append(exact_match_score(pred, ref, normalize=False))
                    print("exact_match_score..", end="\r")

                elif "_many_shot" in dataname:
                    score.append(include_score_manyshot(pred, ref))
                    print("include_score_manyshot..", end="\r")

                elif "summary" in dataname:
                    score.append(rouge_score(pred, ref))
                    print("rouge_score..", end="\r")

                elif "qa_eng" in dataname:
                    score.append(max(f1_score(pred, ref), include_score(pred, ref)))
                    print("f1_score..", end="\r")

                elif "choice_eng" in dataname:
                    pred = pred.split("\n")[0]  # cutoff explanation
                    score.append(include_score(pred, ref))
                    print("include_score..", end="\r")

                elif "gsm" in dataname:
                    pred = pred.strip().lower().split("the answer is ")[-1]
                    score.append(include_score_gsm(pred, ref, normalize=False))
                    print("include_score_gsm..", end="\r")

                elif dataname == "quality" or dataname.startswith("longhealth"):
                    if gold_labels is None or i >= len(gold_labels):
                        raise ValueError(
                            f"MCQ dataset '{dataname}' requires gold_labels in evaluate_answer (got None)"
                        )
                    score.append(mcq_letter_score(pred, gold_labels[i]))
                    print("mcq_letter_score..", end="\r")

                elif dataname.startswith("ruler"):
                    # ref is a list of reference strings (recall over needles)
                    score.append(ruler_score(pred, ref))
                    print("ruler_score..", end="\r")

                elif dataname == "qasper":
                    # ref is a list of annotator answers; max F1 over annotators
                    score.append(qasper_score(pred, ref))
                    print("qasper_score..", end="\r")

                elif dataname.startswith("longbench"):
                    # LongBench QA (e.g. longbench:qasper): the stored gold is
                    # "ref1 | ref2 | ..."; take max token-F1 over the refs.
                    if isinstance(ref, str):
                        refs = [r.strip() for r in ref.split(" | ")] if " | " in ref else [ref]
                    else:
                        refs = ref
                    score.append(qasper_score(pred, refs))
                    print("longbench_qa_f1..", end="\r")

                else:
                    score.append(include_score(pred, ref))
                    print("include_score..", end="\r")
    return score


if __name__ == "__main__":
    text = "hello one two threes . "
    print(normalize_answer(text))
