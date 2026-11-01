# Project Design (Milestone 2): Point of No Return

## Objectives

The proposed research objective is to identify the earliest reasoning step that precedes or introduces an error in a model-generated solution. The v1 implementation operationalizes this as **first-error candidate localization** over an ordered, user-supplied trace. It supports inspection and collection of labeled examples, but does not claim that it can determine correctness or irrecoverability.

## Methodology in v1

The user supplies a problem statement and a sequence of steps. For each step, the backend calculates a small feature vector and a fixed score. It returns the earliest step whose score meets the selected threshold. A user may optionally supply a one-indexed gold first-error label; the interface reports exact match and absolute step distance for that trace. This per-trace feedback is not aggregate FED.

### Algorithms and techniques

1. **Adjacent lexical similarity:** lowercase alphanumeric token sets are compared with Jaccard similarity, intersection divided by union. Low overlap is treated as a weak discontinuity cue.
2. **Language markers:** presence of a negation word and of a contrast/conclusion marker are represented as binary features. This is a deliberately simple lexical baseline.
3. **Numeric shift:** numeric literals are extracted with a regular expression; a change from prior observed numeric values produces a binary cue. It does not evaluate equations, units, or whether a change is justified.
4. **Fixed logistic score:** for step *t*, the baseline computes `sigmoid(-2.15 + 1.2 × (1 − similarity) + 0.28 × negation + 0.23 × contrast + 0.42 × numeric_shift)`. The earliest score at or above the selected threshold is reported. These scores are not calibrated probabilities.
5. **Human reference label:** optional gold labels enable exact-match and absolute index distance for an individual trace. JSON export preserves the input, features, scores, threshold, prediction, and feedback for later dataset work.

The backend now includes a TF-IDF logistic classifier trained on the real ProcessBench step labels, and retains the fixed heuristic as a comparison baseline. It reads the problem and each prefix through a candidate step. It does not generate traces or provide an automatic correctness oracle.

## Research evaluation plan

The project includes Qwen's ProcessBench, a public human-annotated benchmark with 3,400 traces. ProcessBench labels the earliest error at a zero-indexed step or uses -1 when all steps are correct. The backend trains with problem-grouped train, validation, and held-out partitions, and the UI calls the learned model or fixed heuristic on held-out problems.

The integration reports exact first-error localization, correct-trace accuracy, ProcessBench's harmonic-mean score, detection rate, and distance when flagged. On 668 problem-held-out cases, the learned model reaches 33.6% F1 versus 25.8% for the fixed heuristic, with thresholds selected on validation data. This is a modest improvement, but 23.7% exact localization is still too weak for reliable use. Full results and the reproducible protocol are in `research/EVALUATION.md`.

## Relevance of algorithms and techniques

The lexical, marker, and numeric features remain an inexpensive, inspectable baseline. The learned text classifier adds supervised signal from human ProcessBench labels while remaining small enough for local inference. A future stronger system should be compared on independent data under the same grouped evaluation protocol. Any “point of no return” claim would need an operational definition and evidence that the error persists despite subsequent reasoning or correction; a first wrong step by itself does not establish irrecoverability.

## Scope, assumptions, and limitations

ProcessBench is focused on mathematical reasoning and its published generators and labels. It does not establish performance in other domains or show whether a wrong step can be recovered. The learned classifier remains weak on the held-out set; its scores are not calibrated probabilities. The application processes manually supplied text and the public benchmark; it does not inspect hidden states.

## Running the prototype

See [README.md](README.md) for the local command, browser URL, API schema, benchmark workflow, and dataset refresh command. It uses Python’s standard library and serves both UI and API from `server.py`.
