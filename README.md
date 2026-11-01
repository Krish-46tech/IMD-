# Point of No Return — v1

A small research prototype for inspecting ordered reasoning traces and flagging the earliest step whose heuristic risk score crosses a user-selected threshold. The UI and backend are served from one local Python process.

## Run locally

Requires Python 3.10 or newer. No package installation is required.

```sh
cd "/Users/g.o.a.t/Downloads/IMD Project"
python server.py
```

Open <http://localhost:8000>. Stop the server with Ctrl+C. Set a different port with `PORT=8001 python server.py`. Since the backend serves the frontend, there is no separate frontend command or CORS setup required. `frontend/` contains the UI files.

## Use

Enter the problem and reasoning steps, one step per line. Optionally select the independently judged first incorrect step, then analyze. The UI shows per-step model scores, heuristic cues, the first threshold crossing, and optional single-trace exact-match/step-distance feedback. Export the full result as JSON. “Load ProcessBench test sample” loads a real human-labeled trace from the held-out test set.

## API

`GET /api/health` reports server availability. `POST /api/analyze` accepts JSON such as:

```json
{"question":"Compute the average speed.","steps":["Distance is 60 km.","Time is 2 hours.","60 / 2 = 20."],"threshold":0.3,"gold_first_error":3}
```

`gold_first_error` is optional and one-indexed. The API returns feature values, scores, the earliest crossing, and optional per-trace metrics.

`GET /api/benchmark/info` returns the locally installed dataset sizes, held-out test counts, and validation-selected thresholds. `POST /api/benchmark` accepts `{"split":"gsm8k","detector":"learned"}` and evaluates the selected detector on held-out problems only.

## VS Code extension

The `vscode-extension/` folder contains a companion extension that monitors workspace changes, keeps local recovery checkpoints, and can run a configured build/test command to offer a guarded rollback and AI retry prompt after a failure. It can watch edits from different tools inside the workspace, but VS Code does not expose a universal API to identify or control every AI assistant; the retry prompt is copied for the user to hand off. The extension also retains the original reasoning-trace heuristic. Open the folder in VS Code and press F5 to launch the Extension Development Host; its README has workflow and packaging details.

## Limits

The original fixed heuristic is a weak comparison baseline. On 668 problem-held-out ProcessBench cases, the trained model reached 33.6% F1 and 23.7% exact first-error localization; this is not reliable for consequential decisions. Its scores are not calibrated probabilities or correctness judgments. Both detectors can flag valid steps or miss errors; the system has no general ground-truth verifier, does not inspect hidden model activations, and cannot determine whether a mistake is irrecoverable. Results are a limited math-domain evaluation.

## Real benchmark evaluation

The project ships the public [Qwen ProcessBench](https://huggingface.co/datasets/Qwen/ProcessBench) dataset (3,400 human-annotated traces; Apache-2.0) and a locally runnable TF-IDF logistic step-error classifier trained from its real labels. The backend exposes both the trained model and original fixed heuristic. The benchmark panel evaluates only the problem-grouped held-out test portion; thresholds are selected on validation problems and fixed before test scoring. The backend maps ProcessBench's zero-indexed labels (`-1` means no error) to the app's one-indexed display and reports first-error exact match, correct-trace accuracy, and the benchmark's harmonic-mean score. The original benchmark defines the task and labels in its [ACL 2025 paper](https://aclanthology.org/2025.acl-long.50/).

The saved model runs with the Python standard library. To retrain it from the checked-in data:

```sh
python -m pip install -r requirements-training.txt
python research/train_detector.py
```

Training uses a deterministic problem-level split (70% train, 10% validation, 20% held-out test), excludes steps after the first labeled error, and selects thresholds on validation data only. Results are in `research/results_learned_detector.json` and `research/results_processbench_test.json`.

Reproduce held-out results for both detectors without opening the UI:

```sh
python research/evaluate_processbench.py --detector both
```

Thresholds are selected on validation problems and fixed in this evaluation. The recorded train/validation/test results are in `research/results_learned_detector.json`; per-dataset held-out results are in `research/results_processbench_test.json`.

To refresh the checked-in dataset from Hugging Face:

```sh
python research/download_processbench.py
```

The data source, label mapping, benchmark metrics, and caveats are documented in `research/EVALUATION.md`.
