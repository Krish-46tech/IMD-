"""Point of No Return API and static-file server; local inference uses Python standard library only."""
from __future__ import annotations

import json
import math
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from research.data_split import split_for
from research.ponr_model import PonrModel

ROOT = Path(__file__).parent
PORT = int(os.getenv("PORT", "8000"))
PROCESSBENCH_FILE = ROOT / "research" / "data" / "processbench.json"
MODEL_FILE = ROOT / "research" / "data" / "ponr_model.json"
TRAINING_RESULTS_FILE = ROOT / "research" / "results_learned_detector.json"
PROCESSBENCH_SPLITS = {"gsm8k", "math", "olympiadbench", "omnimath"}
_PONR_MODEL = None
WORD = re.compile(r"[a-zA-Z0-9]+(?:\.[0-9]+)?")
NEGATIONS = {"not", "never", "cannot", "can't", "incorrect", "false", "wrong", "impossible"}
CONTRAST = {"but", "however", "instead", "therefore", "thus", "so"}


def words(text: str) -> set[str]:
    return {w.lower() for w in WORD.findall(text)}


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-max(-20, min(20, x))))


def analyze(payload: dict) -> dict:
    global _PONR_MODEL
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    steps = payload.get("steps", [])
    if not isinstance(steps, list) or not steps:
        raise ValueError("Provide at least one reasoning step.")
    if any(not isinstance(s, str) or not s.strip() for s in steps):
        raise ValueError("Each step must contain text.")
    detector = payload.get("detector", "learned")
    if detector not in {"learned", "heuristic"}:
        raise ValueError("Detector must be 'learned' or 'heuristic'.")
    if detector == "learned":
        if not MODEL_FILE.is_file():
            raise ValueError("Trained detector model is missing. Run python research/train_detector.py.")
        if _PONR_MODEL is None:
            _PONR_MODEL = PonrModel(MODEL_FILE)
    features = []
    risks = []
    prior = " ".join(steps[:1])
    prior_numbers = [float(n) for n in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", prior)]
    for i, step in enumerate(steps):
        toks = words(step)
        prev = words(steps[i - 1]) if i else set()
        overlap = len(toks & prev) / max(1, len(toks | prev)) if i else 1.0
        negation = int(bool(toks & NEGATIONS))
        contrast = int(bool(toks & CONTRAST))
        nums = [float(n) for n in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", step)]
        number_shift = int(bool(nums and prior_numbers and nums[0] != prior_numbers[-1]))
        if detector == "learned":
            prefix = "\n".join(steps[:i + 1])
            p = _PONR_MODEL.risk(f"Problem: {payload.get('question', '')}\nReasoning so far:\n{prefix}")
        else:
            # Fixed v1 baseline; visible cues are not proof of error.
            raw = -2.15 + 1.2 * (1 - overlap) + .28 * negation + .23 * contrast + .42 * number_shift
            p = sigmoid(raw)
        features.append({"step": i + 1, "text": step, "similarity_to_previous": round(overlap, 3),
                         "negation_cue": negation, "contrast_cue": contrast,
                         "numeric_shift_cue": number_shift, "risk": round(p, 4)})
        risks.append(p)
        prior_numbers.extend(nums)
    default_threshold = _PONR_MODEL.selected_threshold if detector == "learned" else .3
    try:
        threshold = float(payload.get("threshold", default_threshold))
    except (TypeError, ValueError) as exc:
        raise ValueError("Threshold must be a number from 0 to 1.") from exc
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be a number from 0 to 1.")
    first = next((i for i, p in enumerate(risks) if p >= threshold), None)
    # Optional reference label lets a user score the detector on annotated cases.
    gold = payload.get("gold_first_error")
    metrics = None
    if gold is not None:
        try:
            gold = int(gold)
        except (TypeError, ValueError) as exc:
            raise ValueError("gold_first_error must be a one-indexed step number.") from exc
        if not 1 <= gold <= len(steps):
            raise ValueError("gold_first_error must refer to a step in the trace.")
        predicted = (first + 1) if first is not None else None
        metrics = {"gold_first_error": gold, "predicted_first_error": predicted,
                   "exact_match": predicted == gold,
                   "absolute_step_error": abs(predicted - gold) if predicted and gold else (0 if predicted == gold else None)}
    return {"features": features, "threshold": threshold, "detector": detector,
            "predicted_first_error": (first + 1) if first is not None else None,
            "status": "candidate found" if first is not None else "no step crossed threshold",
            "metrics": metrics,
            "caveat": ("Learned model score is not a calibrated probability and does not establish correctness or irrecoverability."
                       if detector == "learned" else "Heuristic risk cues only; this does not establish factual correctness or irrecoverability.")}


def selected_threshold(detector: str) -> float:
    global _PONR_MODEL
    if detector == "learned":
        if _PONR_MODEL is None and MODEL_FILE.is_file():
            _PONR_MODEL = PonrModel(MODEL_FILE)
        if _PONR_MODEL is None:
            raise ValueError("Trained detector model is missing. Run python research/train_detector.py.")
        return _PONR_MODEL.selected_threshold
    if detector == "heuristic":
        if not TRAINING_RESULTS_FILE.is_file():
            raise ValueError("Validation-selected heuristic threshold is missing. Run python research/train_detector.py.")
        return float(json.loads(TRAINING_RESULTS_FILE.read_text(encoding="utf-8"))["heuristic_baseline_validation_threshold"])
    raise ValueError("Detector must be 'learned' or 'heuristic'.")


def evaluate_processbench(split: str, detector: str = "learned") -> dict:
    if split not in PROCESSBENCH_SPLITS:
        raise ValueError("Choose a ProcessBench split: gsm8k, math, olympiadbench, or omnimath.")
    if not PROCESSBENCH_FILE.is_file():
        raise ValueError("ProcessBench data is not installed. Run python research/download_processbench.py.")
    threshold = selected_threshold(detector)
    rows = json.loads(PROCESSBENCH_FILE.read_text(encoding="utf-8"))
    rows = [row for row in rows if row["id"].split("-", 1)[0] == split and split_for(row["problem"]) == "test"]
    if not rows:
        raise ValueError(f"No ProcessBench records found for {split}.")
    errors = []
    correct = []
    exact = 0
    detection = 0
    distances = []
    for row in rows:
        prediction = analyze({"question": row["problem"], "steps": row["steps"],
                              "threshold": threshold, "detector": detector})["predicted_first_error"]
        predicted_index = prediction - 1 if prediction is not None else -1
        label = int(row["label"])
        match = predicted_index == label
        if label == -1:
            correct.append(match)
        else:
            errors.append(match)
            detection += predicted_index >= 0
            if predicted_index >= 0:
                distances.append(abs(predicted_index - label))
            exact += match
    error_accuracy = sum(errors) / len(errors) if errors else 0.0
    correct_accuracy = sum(correct) / len(correct) if correct else 0.0
    balanced_f1 = (2 * error_accuracy * correct_accuracy / (error_accuracy + correct_accuracy)
                   if error_accuracy + correct_accuracy else 0.0)
    return {
        "dataset": "ProcessBench",
        "split": split,
        "detector": detector,
        "threshold": threshold,
        "count": len(rows),
        "incorrect_traces": len(errors),
        "correct_traces": len(correct),
        "error_localization_accuracy": round(error_accuracy, 4),
        "correct_trace_accuracy": round(correct_accuracy, 4),
        "balanced_f1": round(balanced_f1, 4),
        "first_error_exact_match": round(exact / len(errors), 4) if errors else None,
        "error_detection_rate": round(detection / len(errors), 4) if errors else None,
        "mean_step_distance_when_alerted": round(sum(distances) / len(distances), 4) if distances else None,
        "source": "Qwen/ProcessBench (ACL 2025), human-annotated first-error labels; labels are zero-indexed, -1 means no error.",
        "license": "Apache-2.0",
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, b"", "text/plain")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            return self._send(200, b'{"ok":true,"mode":"local-heuristic"}', "application/json")
        if path == "/api/benchmark/sample":
            if not PROCESSBENCH_FILE.is_file():
                return self._send(503, b'{"error":"ProcessBench is not installed."}', "application/json")
            query = parse_qs(urlparse(self.path).query)
            split = query.get("split", ["gsm8k"])[0]
            sample_id = query.get("id", [None])[0]
            if split not in PROCESSBENCH_SPLITS:
                return self._send(400, b'{"error":"Unknown ProcessBench split."}', "application/json")
            rows = json.loads(PROCESSBENCH_FILE.read_text(encoding="utf-8"))
            sample = next((row for row in rows if row["id"].split("-", 1)[0] == split and split_for(row["problem"]) == "test"
                           and (sample_id is None or row["id"] == sample_id)), None)
            if sample is None:
                return self._send(404, b'{"error":"No held-out test sample found for that split and ID."}', "application/json")
            return self._send(200, json.dumps(sample, ensure_ascii=False).encode(), "application/json")
        if path == "/api/benchmark/info":
            if not PROCESSBENCH_FILE.is_file():
                return self._send(503, b'{"error":"ProcessBench is not installed. Run python research/download_processbench.py."}', "application/json")
            rows = json.loads(PROCESSBENCH_FILE.read_text(encoding="utf-8"))
            counts = {split: sum(row["id"].split("-", 1)[0] == split and split_for(row["problem"]) == "test" for row in rows) for split in sorted(PROCESSBENCH_SPLITS)}
            all_counts = {split: sum(row["id"].split("-", 1)[0] == split for row in rows) for split in sorted(PROCESSBENCH_SPLITS)}
            global _PONR_MODEL
            if _PONR_MODEL is None and MODEL_FILE.is_file():
                _PONR_MODEL = PonrModel(MODEL_FILE)
            result_metadata = json.loads(TRAINING_RESULTS_FILE.read_text(encoding="utf-8")) if TRAINING_RESULTS_FILE.is_file() else {}
            return self._send(200, json.dumps({"dataset": "ProcessBench", "source": "Qwen/ProcessBench", "license": "Apache-2.0", "counts": counts,
                                               "all_counts": all_counts, "default_detector": "learned" if MODEL_FILE.is_file() else "heuristic",
                                               "selected_thresholds": {"learned": _PONR_MODEL.selected_threshold if _PONR_MODEL else None,
                                                                        "heuristic": result_metadata.get("heuristic_baseline_validation_threshold")}}).encode(), "application/json")
        file = "index.html" if path == "/" else path.lstrip("/")
        target = (ROOT / "frontend" / file).resolve()
        if not target.is_relative_to((ROOT / "frontend").resolve()) or not target.is_file():
            return self._send(404, b"Not found", "text/plain")
        mime = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8"}.get(target.suffix, "application/octet-stream")
        self._send(200, target.read_bytes(), mime)

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in {"/api/analyze", "/api/benchmark"}:
            return self._send(404, b'{"error":"Not found"}', "application/json")
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            if path == "/api/benchmark":
                if not isinstance(data, dict):
                    raise ValueError("Request body must be a JSON object.")
                result = evaluate_processbench(str(data.get("split", "gsm8k")), str(data.get("detector", "learned")))
            else:
                result = analyze(data)
            body, status = json.dumps(result).encode(), 200
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            body, status = json.dumps({"error": str(e)}).encode(), 400
        self._send(status, body, "application/json")

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    print(f"Point of No Return v1 running at http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
