# ProcessBench training and evaluation

## Real data source

This project uses the official [Qwen/ProcessBench dataset](https://huggingface.co/datasets/Qwen/ProcessBench), published with the ACL 2025 paper [ProcessBench: Identifying Process Errors in Mathematical Reasoning](https://aclanthology.org/2025.acl-long.50/). The checked-in snapshot contains 3,400 records across GSM8K (400), MATH (1,000), OlympiadBench (1,000), and OmniMath (1,000). The dataset page lists an Apache-2.0 license.

Records contain the problem, ordered solution steps, generator, final-answer correctness, and an expert-annotated earliest-error label. Labels use zero-based indices; `-1` means the full solution is correct. The UI presents one-based step numbers and converts labels in the backend.

The checked-in file is a direct export from the public Hugging Face dataset viewer with original IDs and fields. Refresh with `python research/download_processbench.py` (internet required). Snapshot date: 2026-11-01. SHA-256: `4c3fdfde298fab7a5440f5c64f1cd6b3d87196b31672404bbb1cd0bb6f9665a3`.

## Trained detector

The backend uses a TF-IDF word unigram/bigram logistic regression classifier. For each reasoning step it reads the problem plus the trace prefix up to that step and estimates whether this is the earliest erroneous step. Training examples include known-correct prefixes before the first labeled error, the first erroneous prefix itself, and every prefix from correct traces. Steps after the first labeled error are excluded because their correctness can depend on the wrong premise.

Problems are assigned as a group using a deterministic SHA-256 hash of normalized problem text: 70% train, 10% validation, 20% test. This keeps different model-generated traces for the same prompt together. Model weights are learned on train only. Separate probability thresholds for the trained model and fixed heuristic are selected by ProcessBench F1 on validation only, then frozen for the test results. The app benchmark endpoint evaluates only held-out test problems and does not accept a user-selected test threshold. The manual trace analyzer still lets the user change its threshold.

To retrain, install `requirements-training.txt` and run `python research/train_detector.py`. The resulting model uses a pure-Python inference module, so the normal server has no third-party runtime dependency.

## Metrics

- **Error localization accuracy:** exact first-error location on traces with an error.
- **Correct-trace accuracy:** returns “no error” for traces labeled `-1`.
- **ProcessBench F1:** harmonic mean of those two accuracies.
- **Error detection rate:** error-containing traces on which it returns any candidate.
- **Mean step distance:** absolute index distance among error-containing traces that receive an alert.

## Held-out test results

| ProcessBench subset | Test traces | Trained exact error location | Trained correct-trace accuracy | Trained F1 | Heuristic exact error location | Heuristic correct-trace accuracy | Heuristic F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | 89 | 26.0% | 64.1% | 37.0% | 12.0% | 53.9% | 19.6% |
| MATH | 204 | 19.9% | 65.8% | 30.5% | 21.4% | 34.3% | 26.3% |
| OlympiadBench | 165 | 19.3% | 50.0% | 27.8% | 22.0% | 19.6% | 20.8% |
| OmniMath | 210 | 29.1% | 51.9% | 37.3% | 24.7% | 25.0% | 24.8% |
| **All held-out problems** | **668** | **23.7%** | **58.2%** | **33.6%** | **21.7%** | **31.8%** | **25.8%** |

The learned model improves balanced accuracy over the fixed heuristic on this internal held-out split, but 23.7% exact error localization remains weak. These figures should not be compared as a new official ProcessBench leaderboard result: the project trains on one partition of the public benchmark and tests on its own problem-grouped holdout. Per-category scores and split metadata are in `research/results_processbench_test.json` and `research/results_learned_detector.json`.

## Alternative model experiments

Additional approaches were checked against the same fixed problem-grouped ProcessBench test partition. Model settings and thresholds were selected using training cross-validation and/or the validation partition; test scores were used only for final comparison.

| Training approach | Held-out test F1 | Error localization | Correct-trace accuracy |
|---|---:|---:|---:|
| ProcessBench TF-IDF word unigram/bigram (deployed) | **33.6%** | **23.7%** | **58.2%** |
| PRM800K labels, exact ProcessBench problem overlaps removed | 27.0% | 22.3% | 34.1% |
| all-MiniLM-L6-v2 embeddings with logistic classifier | 24.2% | 17.0% | 42.3% |

The external training source experiment used the official [OpenAI PRM800K release](https://github.com/openai/prm800k) (MIT license); the pretrained encoder is [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Apache-2.0). Neither candidate improved on the deployed model. They are not included in runtime dependencies or the deployed model artifact. Their experiment reports are retained in `research/results_prm800k_model.json` and `research/results_minilm_experiment.json`. These comparisons are exploratory and have not been independently replicated.

## Limits

This model is a modest text classifier trained and tested on one mathematical reasoning dataset. Exact first-error localization is only 23.7% on this held-out set, so it is not reliable for consequential decisions. It does not verify arbitrary factual claims, is not probability-calibrated, and provides no evidence about non-math domains or irrecoverability. Any stronger “point of no return” claim requires a separate recovery study. Further modeling and broader independent evaluation are needed before treating outputs as reliable.
