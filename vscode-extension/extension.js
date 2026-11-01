const vscode = require('vscode');
const path = require('path');
const crypto = require('crypto');
const { spawn } = require('child_process');
const {
  MAX_CHECKPOINT_BYTES, scanWorkspace, changedFiles, encodedSize,
  loadCheckpoints, saveCheckpoints, restoreFiles
} = require('./recovery');

const WORD = /[a-zA-Z0-9]+(?:\.[0-9]+)?/g;
const NEGATIONS = new Set(['not', 'never', 'cannot', "can't", 'incorrect', 'false', 'wrong', 'impossible']);
const CONTRAST = new Set(['but', 'however', 'instead', 'therefore', 'thus', 'so']);

function tokenize(text) { return new Set((text.match(WORD) || []).map(word => word.toLowerCase())); }
function sigmoid(value) { const x = Math.max(-20, Math.min(20, value)); return 1 / (1 + Math.exp(-x)); }
function numbers(text) { return [...text.matchAll(/(?<![\w.])-?\d+(?:\.\d+)?/g)].map(match => Number(match[0])); }

function scoreSteps(stepLines) {
  const priorNumbers = [];
  return stepLines.map((item, index) => {
    const tokens = tokenize(item.text);
    const previous = index ? tokenize(stepLines[index - 1].text) : new Set();
    let similarity = 1;
    if (index) {
      let intersection = 0;
      for (const token of tokens) if (previous.has(token)) intersection++;
      similarity = intersection / Math.max(1, new Set([...tokens, ...previous]).size);
    }
    const negation = [...tokens].some(token => NEGATIONS.has(token)) ? 1 : 0;
    const contrast = [...tokens].some(token => CONTRAST.has(token)) ? 1 : 0;
    const currentNumbers = numbers(item.text);
    const numericShift = currentNumbers.length && priorNumbers.length && currentNumbers[0] !== priorNumbers[priorNumbers.length - 1] ? 1 : 0;
    const raw = -2.15 + 1.2 * (1 - similarity) + 0.28 * negation + 0.23 * contrast + 0.42 * numericShift;
    priorNumbers.push(...currentNumbers);
    return { ...item, similarity, negation, contrast, numericShift, risk: sigmoid(raw) };
  });
}

function runCommand(command, cwd, timeoutMs = 120000) {
  return new Promise(resolve => {
    const child = spawn(command, { cwd, shell: true, windowsHide: true, env: process.env });
    let output = '';
    let timedOut = false;
    const append = chunk => { output += chunk.toString(); if (output.length > 24000) output = output.slice(-24000); };
    child.stdout.on('data', append);
    child.stderr.on('data', append);
    const timer = setTimeout(() => { timedOut = true; child.kill(); }, timeoutMs);
    child.on('error', error => { clearTimeout(timer); resolve({ code: -1, output: `${output}\n${error.message}`, timedOut }); });
    child.on('close', code => { clearTimeout(timer); resolve({ code, output, timedOut }); });
  });
}

function activate(context) {
  const output = vscode.window.createOutputChannel('Point of No Return');
  const diagnostics = vscode.languages.createDiagnosticCollection('ponr');
  const decoration = vscode.window.createTextEditorDecorationType({
    isWholeLine: true,
    backgroundColor: new vscode.ThemeColor('editorWarning.background'),
    border: '1px solid',
    borderColor: new vscode.ThemeColor('editorWarning.foreground')
  });
  const status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 80);
  status.text = '$(eye) PONR';
  status.tooltip = 'Workspace change monitoring is starting';
  status.command = 'ponr.checkAndRecover';
  status.show();

  let checkpoints = [];
  let baselines = new Map();
  let timer = null;
  let scanBusy = false;
  let scanAgain = false;
  let suppress = false;
  let monitorEnabled = true;
  let failureContext = null;
  const watchers = [];
  const workspaceRoots = () => (vscode.workspace.workspaceFolders || []).map(folder => folder.uri.fsPath);
  const storagePath = () => {
    const identity = workspaceRoots().join('\n') || 'no-workspace';
    const digest = crypto.createHash('sha256').update(identity).digest('hex').slice(0, 20);
    return path.join(context.globalStorageUri.fsPath, 'checkpoints', digest);
  };
  const folderFor = root => (vscode.workspace.workspaceFolders || []).find(folder => folder.uri.fsPath === root);
  const refreshStatus = () => {
    status.text = `$(eye) PONR${checkpoints.length ? ` · ${checkpoints.length}` : ''}`;
    status.tooltip = `Monitoring workspace changes · ${checkpoints.length} local recovery checkpoint(s). Click to run the configured check.`;
  };

  async function scanAll() {
    const result = new Map();
    const docs = vscode.workspace.textDocuments;
    for (const root of workspaceRoots()) result.set(root, await scanWorkspace(root, docs));
    return result;
  }

  async function startMonitoring() {
    const roots = workspaceRoots();
    if (!roots.length) {
      status.tooltip = 'Open a workspace folder to monitor coding changes.';
      return;
    }
    checkpoints = await loadCheckpoints(storagePath());
    baselines = await scanAll();
    monitorEnabled = true;
    for (const watcher of watchers) watcher.dispose();
    watchers.length = 0;
    for (const folder of vscode.workspace.workspaceFolders || []) {
      const watcher = vscode.workspace.createFileSystemWatcher(new vscode.RelativePattern(folder, '**/*'));
      watchers.push(watcher);
      context.subscriptions.push(watcher);
      watcher.onDidCreate(scheduleScan);
      watcher.onDidChange(scheduleScan);
      watcher.onDidDelete(scheduleScan);
    }
    refreshStatus();
    output.appendLine(`Monitoring ${roots.length} workspace folder(s). Local checkpoints are stored under VS Code extension storage.`);
  }

  async function recordChangeBurst({ autoVerify = false } = {}) {
    if (!monitorEnabled || suppress) return;
    if (scanBusy) { scanAgain = true; return; }
    scanBusy = true;
    try {
      const current = await scanAll();
      const changes = changedFiles(baselines, current);
      baselines = current;
      if (changes.length) {
        const size = encodedSize(changes);
        if (size > MAX_CHECKPOINT_BYTES) {
          output.appendLine(`Change burst not checkpointed: ${Math.ceil(size / 1024 / 1024)} MB exceeds the 20 MB checkpoint limit.`);
          vscode.window.showWarningMessage('PONR saw a large change burst but skipped its checkpoint because it exceeds the 20 MB limit.');
        } else {
          const checkpoint = {
            id: crypto.randomUUID(),
            createdAt: new Date().toISOString(),
            files: changes
          };
          checkpoints.push(checkpoint);
          checkpoints = checkpoints.slice(-20);
          await saveCheckpoints(storagePath(), checkpoints);
          refreshStatus();
          output.appendLine(`Checkpoint ${checkpoint.id.slice(0, 8)} · ${changes.length} changed file(s) · ${Math.ceil(size / 1024)} KB`);
          const config = vscode.workspace.getConfiguration('ponr');
          const command = config.get('verifyCommand', '').trim();
          if (autoVerify && command && config.get('autoVerify', false)) await verifyAndOfferRecovery(command, checkpoint, true);
        }
      }
    } catch (error) {
      output.appendLine(`Checkpoint scan failed: ${error.message}`);
      vscode.window.showWarningMessage(`PONR could not checkpoint this change burst: ${error.message}`);
    } finally {
      scanBusy = false;
      if (scanAgain) { scanAgain = false; scheduleScan(); }
    }
  }

  function scheduleScan() {
    if (!monitorEnabled || suppress) return;
    if (timer) clearTimeout(timer);
    status.text = '$(sync~spin) PONR · Change detected';
    const delay = vscode.workspace.getConfiguration('ponr').get('changeDebounceMs', 1000);
    timer = setTimeout(() => { timer = null; recordChangeBurst({ autoVerify: true }); }, delay);
  }

  async function rollbackLast({ confirm = true, announce = true } = {}) {
    const checkpoint = checkpoints.at(-1);
    if (!checkpoint) {
      vscode.window.showInformationMessage('PONR has no recovery checkpoint to roll back.');
      return { restored: [], conflicts: [] };
    }
    const changedNames = checkpoint.files.map(file => file.path);
    if (confirm) {
      const answer = await vscode.window.showWarningMessage(
        `Roll back the latest checkpoint affecting ${changedNames.length} file(s)?`,
        { modal: true, detail: `${changedNames.slice(0, 8).join('\n')}${changedNames.length > 8 ? `\n…and ${changedNames.length - 8} more` : ''}\n\nPONR only restores files that still match this checkpoint, to avoid overwriting newer edits.` },
        'Roll Back'
      );
      if (answer !== 'Roll Back') return { restored: [], conflicts: [], cancelled: true };
    }
    suppress = true;
    try {
      const result = await restoreFiles(checkpoint.files);
      checkpoints.pop();
      await saveCheckpoints(storagePath(), checkpoints);
      baselines = await scanAll();
      refreshStatus();
      if (!announce) {
        // The caller reports the automatic recovery result and retry prompt.
      } else if (result.conflicts.length) {
        vscode.window.showWarningMessage(`Restored ${result.restored.length} file(s). Left ${result.conflicts.length} newer or externally changed file(s) untouched.`);
      } else {
        vscode.window.showInformationMessage(`Rolled back ${result.restored.length} file(s) to the prior checkpoint.`);
      }
      return result;
    } finally {
      suppress = false;
    }
  }

  function retryPrompt(checkResult, command, changedNames) {
    const failure = (checkResult.output || (checkResult.timedOut ? 'The check timed out.' : `Process exited with code ${checkResult.code}.`)).trim().slice(-8000);
    return [
      'The previous code change failed verification. Please redo the change carefully from the current workspace state.',
      `Verification command: ${command}`,
      `Verification exit code: ${checkResult.timedOut ? 'timed out' : checkResult.code}`,
      `Files rolled back: ${changedNames.join(', ') || 'none'}`,
      '', 'Failure output:', failure || '(no output)',
      '', 'Please inspect the relevant code, make a corrected implementation, and run the verification command again. Explain any remaining failure instead of claiming it passed.'
    ].join('\n');
  }

  async function prepareRetryPrompt(checkResult, command, checkpoint) {
    const changedNames = checkpoint ? checkpoint.files.map(file => file.path) : [];
    let rollbackResult = { restored: [], conflicts: [] };
    if (checkpoint) rollbackResult = await rollbackLast({ confirm: false });
    const prompt = retryPrompt(checkResult, command, rollbackResult.restored);
    await vscode.env.clipboard.writeText(prompt);
    failureContext = prompt;
    const action = await vscode.window.showInformationMessage(
      `Verification failed. Rolled back ${rollbackResult.restored.length} file(s); AI retry instructions were copied to the clipboard.`,
      'Open Chat', 'Show Prompt'
    );
    if (action === 'Show Prompt') {
      const document = await vscode.workspace.openTextDocument({ language: 'markdown', content: prompt });
      await vscode.window.showTextDocument(document, { preview: true });
    } else if (action === 'Open Chat') {
      const available = await vscode.commands.getCommands(true);
      const commandId = available.includes('workbench.action.chat.open') ? 'workbench.action.chat.open' : null;
      if (commandId) await vscode.commands.executeCommand(commandId);
      else vscode.window.showInformationMessage('No VS Code Chat view is available. The retry prompt is on the clipboard; paste it into your coding assistant.');
    }
    if (rollbackResult.conflicts.length) {
      output.appendLine(`Rollback left ${rollbackResult.conflicts.length} file(s) untouched because they changed after the checkpoint.`);
    }
    return { changedNames, rollbackResult };
  }

  async function verifyAndOfferRecovery(command, checkpoint, automatic = false) {
    const roots = workspaceRoots();
    if (!roots.length) return;
    const result = await runCommand(command, roots[0]);
    output.appendLine(`\n${automatic ? 'Automatic ' : ''}verification: ${command}\nExit code: ${result.timedOut ? 'timeout' : result.code}\n${result.output}`);
    if (result.code === 0 && !result.timedOut) {
      failureContext = null;
      if (!automatic) vscode.window.showInformationMessage('Verification passed. No rollback was needed.');
      return;
    }
    const config = vscode.workspace.getConfiguration('ponr');
    if (automatic && config.get('autoRecover', true) && checkpoint) {
      const rollback = await rollbackLast({ confirm: false, announce: false });
      const prompt = retryPrompt(result, command, rollback.restored);
      await vscode.env.clipboard.writeText(prompt);
      failureContext = prompt;
      output.appendLine(`Automatic recovery restored ${rollback.restored.length} file(s); ${rollback.conflicts.length} newer edit(s) were left untouched. Retry prompt copied to clipboard.`);
      const action = await vscode.window.showWarningMessage(
        `Check failed. PONR rolled back ${rollback.restored.length} file(s) and copied an AI retry prompt.`,
        'Show Retry Prompt'
      );
      if (action === 'Show Retry Prompt') {
        const document = await vscode.workspace.openTextDocument({ language: 'markdown', content: prompt });
        await vscode.window.showTextDocument(document, { preview: true });
      }
      return;
    }
    const detail = `${automatic ? 'The configured check ran after a workspace change.' : 'The configured check was run on request.'}\n${checkpoint ? `The latest checkpoint includes ${checkpoint.files.length} changed file(s).` : 'No automatic checkpoint is available.'}\n\nA failing check is a signal to review; it does not prove that this edit caused the failure.`;
    const choice = await vscode.window.showWarningMessage(
      `Verification failed${result.timedOut ? ' (timed out)' : ` (exit ${result.code})`}.`,
      { modal: true, detail },
      'Roll Back and Prepare Retry Prompt', 'Keep Changes'
    );
    if (choice === 'Roll Back and Prepare Retry Prompt') await prepareRetryPrompt(result, command, checkpoint);
  }

  context.subscriptions.push(output, diagnostics, decoration, status);
  context.subscriptions.push(vscode.workspace.onDidChangeTextDocument(scheduleScan));
  context.subscriptions.push(vscode.workspace.onDidSaveTextDocument(scheduleScan));
  context.subscriptions.push(vscode.workspace.onDidChangeWorkspaceFolders(() => startMonitoring().catch(error => output.appendLine(error.message))));

  context.subscriptions.push(vscode.commands.registerCommand('ponr.analyzeTrace', async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return vscode.window.showWarningMessage('Open a text or Markdown reasoning trace first.');
    const selected = !editor.selection.isEmpty;
    const startLine = selected ? editor.selection.start.line : 0;
    const source = selected ? editor.document.getText(editor.selection) : editor.document.getText();
    const stepLines = source.split(/\r?\n/).map((text, offset) => ({ text: text.trim(), line: startLine + offset })).filter(item => item.text.length > 0);
    if (!stepLines.length) return vscode.window.showWarningMessage('No reasoning steps found. Put one step on each non-empty line.');
    const threshold = vscode.workspace.getConfiguration('ponr').get('riskThreshold', 0.3);
    const scored = scoreSteps(stepLines);
    const candidate = scored.find(step => step.risk >= threshold);
    diagnostics.delete(editor.document.uri);
    editor.setDecorations(decoration, []);
    output.clear();
    output.appendLine(`Point of No Return · ${editor.document.fileName}`);
    output.appendLine(`Threshold: ${threshold.toFixed(2)} · ${scored.length} non-empty steps`);
    output.appendLine('Scores are heuristic cues, not correctness probabilities.');
    output.appendLine('');
    scored.forEach((step, index) => {
      const flags = [step.negation && 'negation', step.contrast && 'contrast', step.numericShift && 'numeric shift'].filter(Boolean);
      output.appendLine(`Step ${index + 1} (line ${step.line + 1})  risk=${step.risk.toFixed(3)}  similarity=${step.similarity.toFixed(3)}  cues=${flags.join(', ') || 'none'}`);
      output.appendLine(`  ${step.text}`);
    });
    if (candidate) {
      const range = new vscode.Range(candidate.line, 0, candidate.line, editor.document.lineAt(candidate.line).text.length);
      const diagnostic = new vscode.Diagnostic(range, `Earliest PONR candidate (heuristic risk ${candidate.risk.toFixed(2)} ≥ ${threshold.toFixed(2)}). Review this step; the score does not establish that it is incorrect.`, vscode.DiagnosticSeverity.Warning);
      diagnostic.source = 'Point of No Return';
      diagnostics.set(editor.document.uri, [diagnostic]);
      editor.setDecorations(decoration, [range]);
      output.appendLine(`\nEarliest candidate: analysis step ${scored.indexOf(candidate) + 1}, document line ${candidate.line + 1}.`);
      output.show(true);
      vscode.window.showInformationMessage(`PONR candidate: line ${candidate.line + 1} (risk ${candidate.risk.toFixed(2)}).`);
    } else {
      output.appendLine('\nNo step crossed the threshold.');
      output.show(true);
      vscode.window.showInformationMessage('PONR found no step above the selected threshold.');
    }
  }));

  context.subscriptions.push(vscode.commands.registerCommand('ponr.checkAndRecover', async () => {
    const roots = workspaceRoots();
    if (!roots.length) return vscode.window.showWarningMessage('Open a workspace folder first.');
    const config = vscode.workspace.getConfiguration('ponr');
    const configured = config.get('verifyCommand', '');
    const command = await vscode.window.showInputBox({
      title: 'Verify current code changes',
      prompt: 'Enter a build or test command. It will run in the workspace folder.',
      value: configured,
      ignoreFocusOut: true
    });
    if (!command?.trim()) return;
    await config.update('verifyCommand', command.trim(), vscode.ConfigurationTarget.Workspace);
    await recordChangeBurst({ autoVerify: false });
    output.show(true);
    await verifyAndOfferRecovery(command.trim(), checkpoints.at(-1), false);
  }));

  context.subscriptions.push(vscode.commands.registerCommand('ponr.rollbackLastCheckpoint', () => rollbackLast()));
  context.subscriptions.push(vscode.commands.registerCommand('ponr.startMonitoring', async () => {
    await startMonitoring();
    vscode.window.showInformationMessage('PONR is monitoring workspace file changes and recording local recovery checkpoints.');
  }));
  context.subscriptions.push(vscode.commands.registerCommand('ponr.clearCheckpoints', async () => {
    const answer = await vscode.window.showWarningMessage('Delete all PONR recovery checkpoints for this workspace?', { modal: true }, 'Delete Checkpoints');
    if (answer !== 'Delete Checkpoints') return;
    checkpoints = [];
    await saveCheckpoints(storagePath(), checkpoints);
    refreshStatus();
    vscode.window.showInformationMessage('PONR recovery checkpoints deleted.');
  }));
  context.subscriptions.push(vscode.commands.registerCommand('ponr.copyRetryPrompt', async () => {
    if (!failureContext) return vscode.window.showInformationMessage('Run a failing verification check first to create a retry prompt.');
    await vscode.env.clipboard.writeText(failureContext);
    vscode.window.showInformationMessage('AI retry prompt copied to the clipboard.');
  }));
  context.subscriptions.push(vscode.commands.registerCommand('ponr.setThreshold', async () => {
    const current = vscode.workspace.getConfiguration('ponr').get('riskThreshold', 0.3);
    const value = await vscode.window.showInputBox({ title: 'PONR risk threshold', prompt: 'Choose a threshold between 0.10 and 0.90. Lower values flag more steps.', value: String(current), validateInput: raw => {
      const n = Number(raw);
      return Number.isFinite(n) && n >= 0.1 && n <= 0.9 ? null : 'Enter a number from 0.10 to 0.90.';
    } });
    if (value !== undefined) {
      await vscode.workspace.getConfiguration('ponr').update('riskThreshold', Number(value), vscode.ConfigurationTarget.Workspace);
      vscode.window.showInformationMessage(`PONR threshold set to ${Number(value).toFixed(2)}.`);
    }
  }));
  context.subscriptions.push(vscode.commands.registerCommand('ponr.clearAnalysis', () => {
    diagnostics.clear();
    for (const editor of vscode.window.visibleTextEditors) editor.setDecorations(decoration, []);
    output.clear();
    vscode.window.showInformationMessage('PONR highlights cleared.');
  }));

  context.subscriptions.push({ dispose: () => { if (timer) clearTimeout(timer); for (const watcher of watchers) watcher.dispose(); } });
  startMonitoring().catch(error => {
    status.tooltip = `Monitoring could not start: ${error.message}`;
    output.appendLine(`Monitoring could not start: ${error.message}`);
    vscode.window.showWarningMessage(`PONR could not start workspace monitoring: ${error.message}`);
  });
}

function deactivate() {}
module.exports = { activate, deactivate };
