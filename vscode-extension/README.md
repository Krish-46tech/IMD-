# Point of No Return — VS Code extension

PONR watches workspace files and groups edits into local checkpoints. When a configured build or test command fails, it can offer to restore the most recent checkpoint and copy a retry instruction with the failure output for the AI coding assistant. The extension also retains the original reasoning-trace analyzer.

## What it can and cannot observe

VS Code exposes workspace edits and file changes, but there is no single API that tells an extension which AI product made a change or lets it control every assistant. PONR therefore watches workspace changes from any source, including Codex, Copilot, Claude Code, terminals, and manual edits. A user-chosen build/test command supplies an objective failure signal. After rollback, PONR copies a retry prompt that can be pasted into the assistant in use. A VS Code Chat view may be opened when one is available, but PONR does not silently submit prompts or claim to control external tools.

## Recovery workflow

1. Open a project folder in VS Code. PONR starts local monitoring and creates a baseline of eligible text files.
2. After the workspace is quiet for one second (configurable), PONR stores before/after content for the changed files and runs the configured check.
3. Set a project check such as `npm test`, `pytest`, or `python -m compileall .` as `ponr.verifyCommand`. Use **PONR: Run Verification and Offer Recovery** to enter it the first time; later checks run automatically when `ponr.autoVerify` is enabled.
4. When an automatic check fails, PONR restores the latest checkpoint by default (`ponr.autoRecover: true`) and copies a retry prompt with the failure output. It leaves files untouched if they no longer match the checkpoint. Disable `ponr.autoRecover` if you prefer to review and confirm each rollback.
5. Paste the prompt into the coding assistant. The assistant can then retry from the restored workspace state.

The extension keeps the newest 20 checkpoints per workspace. It skips dependency/build folders, binary files, files larger than 2 MB, and a change burst larger than 20 MB. A workspace scan is capped at 20,000 files / 80 MB of eligible text. Rollback is confirmed and only restores files whose current contents still match the checkpoint. These safeguards reduce accidental loss, but they do not replace source control; commit or back up important work.

Checkpoints contain source text and stay on the local machine in VS Code's extension storage. Use **PONR: Delete Workspace Checkpoints** to remove them. No code or prompt is uploaded by the extension.

## Commands

- **PONR: Run Verification and Offer Recovery** — run the configured build/test command and, if it fails, offer a rollback and retry prompt.
- **PONR: Roll Back Latest Checkpoint** — confirm and restore one checkpoint without running a check.
- **PONR: Copy AI Retry Prompt** — copy the latest failure prompt again.
- **PONR: Start Workspace Monitoring** — restart the current workspace watcher.
- **PONR: Delete Workspace Checkpoints** — remove stored checkpoints for this workspace.
- **PONR: Analyze Reasoning Trace** — keep using the original trace heuristic on selected text or the current document.
- **PONR: Set Risk Threshold** and **PONR: Clear Analysis Highlights** — configure or clear trace-analysis decorations.

Example workspace settings:

```json
{
  "ponr.verifyCommand": "npm test",
  "ponr.autoVerify": true,
  "ponr.autoRecover": true,
  "ponr.changeDebounceMs": 1000,
  "ponr.riskThreshold": 0.3
}
```

Automatic verification can run builds/tests after each quiet edit burst; set `ponr.autoVerify` to `false` to run checks only on demand. The assistant prompt is copied for you to paste; it is not silently submitted to an AI tool.

## Try it in VS Code

1. Open the `vscode-extension` folder in VS Code.
2. Press **F5** to launch an Extension Development Host.
3. Open a test project and make a few edits, then wait for the checkpoint to be recorded (shown in the PONR status-bar item and output channel).
4. Run **PONR: Run Verification and Offer Recovery** and enter a check command.

To analyze a reasoning trace, open a `.txt` or `.md` file with one step per non-empty line and run **PONR: Analyze Reasoning Trace**. `examples/example_trace.md` contains a real `gsm8k-0` trace from Qwen ProcessBench. Select a portion first to analyze only that selection.

## Package as a VSIX

Install Visual Studio Code. To install the packaged build from the project root, run:

```sh
code --install-extension deliverables/point-of-no-return-0.3.0.vsix
```

To build a fresh package from this directory, install Node.js/npm and `@vscode/vsce`, then run `vsce package`.

## Interpretation limits

PONR does not determine whether code is correct or whether an AI caused a change. A failing build/test is a useful signal, not proof that the latest edit caused the failure. The rollback prompt is a handoff for an assistant; the assistant's retry must still be reviewed and verified. The reasoning-trace analyzer remains a weak heuristic: scores are not calibrated probabilities, and “irrecoverability” is not inferred.
