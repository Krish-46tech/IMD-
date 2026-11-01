"""Train a local step-error model from OpenAI PRM800K and validate on ProcessBench."""
from __future__ import annotations

import hashlib
import heapq
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from research.data_split import split_for
from research.ponr_model import PonrModel
from train_detector import prefix_scores, summarize_scores

PRMBENCH = ROOT / "research/data/processbench.json"
MODEL_OUT = ROOT / "research/data/ponr_model.json"
RESULTS_OUT = ROOT / "research/results_prm800k_model.json"
OFFICIAL_SHA256 = "1110237feeb51d1bc200cb37b8f965cfdc1036eac7d506094049366fe7dc1089"
TOKEN_PATTERN = r"(?u)\b\w\w+\b"


def problem_key(problem: str) -> str:
    return " ".join(problem.lower().split())


def chosen_examples(record: dict):
    problem = record["question"]["problem"]
    prefix = []
    for labeled_step in record["label"]["steps"]:
        completions = labeled_step.get("completions") or []
        chosen = labeled_step.get("chosen_completion")
        if isinstance(chosen, int) and 0 <= chosen < len(completions):
            candidate_list = [completions[chosen]]
        elif chosen is None and not labeled_step.get("human_completion"):
            # At an unresolved branch, keep every real human-rated candidate,
            # then stop this trajectory because no single continuation is chosen.
            candidate_list = completions
        else:
            break
        stop_after_candidates = chosen is None
        saw_error = False
        for candidate in candidate_list:
            rating = candidate.get("rating")
            if rating not in (-1, 0, 1) or candidate.get("flagged") is True:
                continue
            text = candidate.get("text", "").strip()
            if not text:
                continue
            yield f"Problem: {problem}\nReasoning so far:\n" + "\n".join(prefix + [text]), int(rating == -1)
            saw_error |= rating == -1
        if stop_after_candidates or saw_error:
            break
        if isinstance(chosen, int) and 0 <= chosen < len(completions):
            selected = completions[chosen]
            if selected.get("rating") in (0, 1) and selected.get("text", "").strip():
                prefix.append(selected["text"].strip())


def load_source(path: Path, excluded_problems: set[str], max_per_problem: int = 3):
    selected = defaultdict(list)
    valid_records = skipped_overlap = total_records = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            total_records += 1
            record = json.loads(line)
            if (record.get("is_quality_control_question") or record.get("is_initial_screening_question")
                    or record.get("label", {}).get("finish_reason") not in {"solution", "found_error"}):
                continue
            problem = record["question"]["problem"]
            if problem_key(problem) in excluded_problems:
                skipped_overlap += 1
                continue
            sample_text = json.dumps(record["question"].get("pre_generated_steps", []), ensure_ascii=False)
            sample_hash = hashlib.sha256((problem + sample_text).encode()).hexdigest()
            bucket = selected[problem_key(problem)]
            if any(existing_hash == sample_hash for existing_hash, _ in bucket):
                continue
            bucket.append((sample_hash, record))
            bucket.sort(key=lambda item: item[0])
            del bucket[max_per_problem:]
            valid_records += 1
    rows = [record for bucket in selected.values() for _, record in bucket]
    return rows, {"source_rows": total_records, "eligible_rows": valid_records,
                  "selected_solutions": len(rows), "unique_problems": len(selected),
                  "exact_problem_overlap_excluded": skipped_overlap,
                  "max_solutions_per_problem": max_per_problem}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python research/train_prm800k.py /path/to/phase2_train.jsonl")
    source = Path(sys.argv[1])
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if sha != OFFICIAL_SHA256:
        raise SystemExit("PRM800K file checksum does not match the official OpenAI phase2_train LFS object.")

    benchmark_rows = json.loads(PRMBENCH.read_text(encoding="utf-8"))
    excluded = {problem_key(row["problem"]) for row in benchmark_rows}
    records, source_metadata = load_source(source, excluded)
    train_docs, train_labels = [], []
    for record in records:
        for doc, label in chosen_examples(record):
            train_docs.append(doc)
            train_labels.append(label)
    if len(set(train_labels)) != 2:
        raise SystemExit("Processed PRM800K examples must contain both error and non-error labels.")

    validation_rows = [row for row in benchmark_rows if split_for(row["problem"]) == "validation"]
    test_rows = [row for row in benchmark_rows if split_for(row["problem"]) == "test"]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000,
                                 sublinear_tf=True, token_pattern=TOKEN_PATTERN)
    x_train = vectorizer.fit_transform(train_docs)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=17)
    model.fit(x_train, train_labels)

    validation_scores = prefix_scores(validation_rows, vectorizer, model)
    thresholds = [round(i / 100, 2) for i in range(5, 96)]
    validation_results = [(summarize_scores(validation_rows, validation_scores, threshold), threshold)
                          for threshold in thresholds]
    validation_metrics, selected_threshold = max(validation_results, key=lambda item: item[0]["processbench_f1"])

    output_model = {
        "model_type": "tfidf_logistic_regression",
        "source_dataset": "OpenAI PRM800K phase2_train",
        "source_license": "MIT (official repository)",
        "source_sha256": sha,
        "training_method": "up to three deterministic representative solutions per unique PRM800K problem; all exact prompt overlaps with ProcessBench excluded; earliest -1 label is the first error; ProcessBench validation selects threshold",
        "feature_config": {"analyzer": "word", "ngram_range": [1, 2], "min_df": 2,
                           "max_features": 60000, "sublinear_tf": True,
                           "token_pattern": TOKEN_PATTERN,
                           "input": "problem plus reasoning prefix through current step"},
        "selected_threshold": selected_threshold,
        "vocabulary": {term: int(index) for term, index in vectorizer.vocabulary_.items()},
        "idf": vectorizer.idf_.tolist(),
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
    }
    MODEL_OUT.write_text(json.dumps(output_model, separators=(",", ":")), encoding="utf-8")

    test_scores = prefix_scores(test_rows, vectorizer, model)
    test_result = summarize_scores(test_rows, test_scores, selected_threshold)
    results = {"model": "TF-IDF word unigram/bigram logistic regression trained on PRM800K",
               "source_sha256": sha, "source_metadata": source_metadata,
               "training_examples": len(train_docs), "training_error_examples": sum(train_labels),
               "validation_selected_threshold": selected_threshold,
               "validation_metrics": validation_metrics,
               "heldout_processbench_test": test_result,
               "note": "All exact problem-text overlaps with the ProcessBench snapshot were excluded from PRM training. Threshold was selected on the fixed ProcessBench validation split, not the test split."}
    RESULTS_OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved model: {MODEL_OUT} ({MODEL_OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
