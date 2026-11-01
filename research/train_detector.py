"""Train the step-error logistic model on ProcessBench with problem-held-out evaluation."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from server import analyze
from research.data_split import problem_key, split_for

DATA = ROOT / "research/data/processbench.json"
MODEL = ROOT / "research/data/ponr_model.json"
RESULTS = ROOT / "research/results_learned_detector.json"
TOKEN_PATTERN = r"(?u)\b\w\w+\b"


def examples(rows: list[dict]):
    docs, labels, groups = [], [], []
    for row in rows:
        gold = int(row["label"])
        for index, step in enumerate(row["steps"]):
            if gold >= 0 and index > gold:
                break  # Later steps depend on the first error and are ambiguous.
            prefix = "\n".join(row["steps"][:index + 1])
            docs.append(f"Problem: {row['problem']}\nReasoning so far:\n{prefix}")
            labels.append(int(gold == index))
            groups.append(split_for(row["problem"]))
    return docs, labels, groups


def prefix_scores(rows: list[dict], vectorizer, model) -> list[list[float]]:
    documents, lengths = [], []
    for row in rows:
        docs = [f"Problem: {row['problem']}\nReasoning so far:\n" + "\n".join(row["steps"][:i + 1])
                for i in range(len(row["steps"]))]
        documents.extend(docs)
        lengths.append(len(docs))
    probabilities = model.predict_proba(vectorizer.transform(documents))[:, 1]
    results, cursor = [], 0
    for length in lengths:
        results.append(probabilities[cursor:cursor + length].tolist())
        cursor += length
    return results


def summarize_scores(rows: list[dict], scores: list[list[float]], threshold: float) -> dict:
    error_matches, clean_matches, error_alerted, distances = [], [], [], []
    for row, probabilities in zip(rows, scores):
        predicted = next((i for i, p in enumerate(probabilities) if p >= threshold), -1)
        gold = int(row["label"])
        if gold == -1:
            clean_matches.append(predicted == -1)
        else:
            error_matches.append(predicted == gold)
            error_alerted.append(predicted >= 0)
            if predicted >= 0:
                distances.append(abs(predicted - gold))
    error_acc = sum(error_matches) / len(error_matches) if error_matches else 0
    clean_acc = sum(clean_matches) / len(clean_matches) if clean_matches else 0
    return {
        "count": len(rows), "incorrect_traces": len(error_matches), "correct_traces": len(clean_matches),
        "error_localization_accuracy": round(error_acc, 4),
        "correct_trace_accuracy": round(clean_acc, 4),
        "processbench_f1": round(2 * error_acc * clean_acc / (error_acc + clean_acc), 4) if error_acc + clean_acc else 0,
        "error_detection_rate": round(sum(error_alerted) / len(error_alerted), 4) if error_alerted else 0,
        "mean_step_distance_when_alerted": round(sum(distances) / len(distances), 4) if distances else None,
    }


def trace_predictions(rows: list[dict], vectorizer, model, threshold: float) -> dict:
    return summarize_scores(rows, prefix_scores(rows, vectorizer, model), threshold)


def heuristic_predictions(rows: list[dict], threshold: float) -> dict:
    errors, correct, alerts, distances = [], [], [], []
    for row in rows:
        prediction = analyze({"question": row["problem"], "steps": row["steps"],
                              "threshold": threshold, "detector": "heuristic"})["predicted_first_error"]
        predicted = prediction - 1 if prediction is not None else -1
        gold = int(row["label"])
        if gold == -1:
            correct.append(predicted == -1)
        else:
            errors.append(predicted == gold)
            alerts.append(predicted >= 0)
            if predicted >= 0:
                distances.append(abs(predicted - gold))
    error_acc = sum(errors) / len(errors) if errors else 0
    correct_acc = sum(correct) / len(correct) if correct else 0
    return {"count": len(rows), "incorrect_traces": len(errors), "correct_traces": len(correct),
            "error_localization_accuracy": round(error_acc, 4),
            "correct_trace_accuracy": round(correct_acc, 4),
            "processbench_f1": round(2 * error_acc * correct_acc / (error_acc + correct_acc), 4) if error_acc + correct_acc else 0,
            "error_detection_rate": round(sum(alerts) / len(alerts), 4) if alerts else 0,
            "mean_step_distance_when_alerted": round(sum(distances) / len(distances), 4) if distances else None}


def main() -> None:
    rows = json.loads(DATA.read_text(encoding="utf-8"))
    by_split = defaultdict(list)
    for row in rows:
        by_split[split_for(row["problem"])].append(row)
    train_rows = by_split["train"]
    validation = by_split["validation"]
    test = by_split["test"]
    thresholds = [round(i / 100, 2) for i in range(5, 96)]
    configurations = [
        {"name": "word_1_2", "analyzer": "word", "ngram_range": [1, 2], "min_df": 2, "max_features": 40000},
        {"name": "word_1_3", "analyzer": "word", "ngram_range": [1, 3], "min_df": 2, "max_features": 60000},
        {"name": "char_wb_3_5", "analyzer": "char_wb", "ngram_range": [3, 5], "min_df": 3, "max_features": 60000},
    ]
    fold_rows = defaultdict(list)
    for row in train_rows:
        fold_rows[int(problem_key(row["problem"])[:8], 16) % 3].append(row)
    candidate_results = []
    best = None
    for config in configurations:
        vectorizer_args = {k: config[k] for k in ("analyzer", "min_df", "max_features")}
        vectorizer_args["ngram_range"] = tuple(config["ngram_range"])
        if config["analyzer"] == "word":
            vectorizer_args["token_pattern"] = TOKEN_PATTERN
        for c_value in (0.3, 1.0, 3.0):
            out_of_fold_scores = {}
            for fold in range(3):
                fold_train = [row for fold_id, fold_data in fold_rows.items() if fold_id != fold for row in fold_data]
                fold_valid = fold_rows[fold]
                fold_docs, fold_labels, _ = examples(fold_train)
                vectorizer = TfidfVectorizer(**vectorizer_args, sublinear_tf=True)
                x_fold = vectorizer.fit_transform(fold_docs)
                y_fold = [int(label) for label in fold_labels]
                fold_model = LogisticRegression(max_iter=1000, class_weight="balanced", C=c_value, random_state=17)
                fold_model.fit(x_fold, y_fold)
                for row, scores in zip(fold_valid, prefix_scores(fold_valid, vectorizer, fold_model)):
                    out_of_fold_scores[row["id"]] = scores
            oof_scores = [out_of_fold_scores[row["id"]] for row in train_rows]
            oof_validation = [(summarize_scores(train_rows, oof_scores, t), t) for t in thresholds]
            oof_metrics, oof_threshold = max(oof_validation, key=lambda item: item[0]["processbench_f1"])
            candidate = {"name": config["name"], "C": c_value, "oof_threshold": oof_threshold,
                         "oof": oof_metrics, "config": config}
            candidate_results.append({"name": config["name"], "C": c_value,
                                      "oof_threshold": oof_threshold, "oof": oof_metrics})
            if best is None or oof_metrics["processbench_f1"] > best["oof"]["processbench_f1"]:
                best = candidate
    selected_config = best["config"]
    selected_vectorizer_args = {k: selected_config[k] for k in ("analyzer", "min_df", "max_features")}
    selected_vectorizer_args["ngram_range"] = tuple(selected_config["ngram_range"])
    if selected_config["analyzer"] == "word":
        selected_vectorizer_args["token_pattern"] = TOKEN_PATTERN
    docs, labels, _ = examples(train_rows)
    vectorizer = TfidfVectorizer(**selected_vectorizer_args, sublinear_tf=True)
    x_train = vectorizer.fit_transform(docs)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", C=best["C"], random_state=17)
    model.fit(x_train, [int(label) for label in labels])
    validation_scores = prefix_scores(validation, vectorizer, model)
    validation_results = [(summarize_scores(validation, validation_scores, t), t) for t in thresholds]
    validation_metrics, selected_threshold = max(validation_results, key=lambda item: item[0]["processbench_f1"])
    test_result = trace_predictions(test, vectorizer, model, selected_threshold)
    baseline_validation = [(heuristic_predictions(validation, t), t) for t in thresholds]
    _, baseline_threshold = max(baseline_validation, key=lambda item: item[0]["processbench_f1"])
    baseline_test = heuristic_predictions(by_split["test"], baseline_threshold)

    model_json = {
        "model_type": "tfidf_logistic_regression",
        "source_dataset": "Qwen/ProcessBench",
        "source_license": "Apache-2.0",
        "training_method": "problem-level SHA-256 split: 70% train, 10% validation, 20% test; 3-fold problem-grouped cross-validation within train selects representation and C; later steps after first error excluded",
        "feature_config": {**{k: selected_config[k] for k in ("analyzer", "ngram_range", "min_df", "max_features")},
                           "sublinear_tf": True, "token_pattern": TOKEN_PATTERN if selected_config["analyzer"] == "word" else None,
                           "input": "problem plus reasoning prefix through current step"},
        "selected_threshold": selected_threshold,
        "vocabulary": {term: int(index) for term, index in vectorizer.vocabulary_.items()},
        "idf": vectorizer.idf_.tolist(),
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
    }
    MODEL.write_text(json.dumps(model_json, separators=(",", ":")), encoding="utf-8")
    results = {
        "model": f"TF-IDF {selected_config['name']} logistic regression",
        "data_source": "Qwen/ProcessBench human-annotated labels",
        "split_counts": {name: len(records) for name, records in by_split.items()},
        "validation_selected_threshold": selected_threshold,
        "validation_selected_C": best["C"],
        "cross_validation_model_selection": candidate_results,
        "cross_validation_selected_model": {"name": best["name"], "C": best["C"],
                                             "out_of_fold_f1": best["oof"]["processbench_f1"]},
        "validation_threshold_metrics": validation_metrics,
        "heldout_test": test_result,
        "heuristic_baseline_validation_threshold": baseline_threshold,
        "heuristic_baseline_heldout_test": baseline_test,
        "note": "Held-out problems selected by a deterministic SHA-256 split; both thresholds selected only on validation problems.",
    }
    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved model: {MODEL} ({MODEL.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
