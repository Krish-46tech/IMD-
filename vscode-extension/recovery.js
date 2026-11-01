const fs = require('fs/promises');
const path = require('path');

const EXCLUDED = new Set([
  '.git', '.hg', '.svn', 'node_modules', 'vendor', 'dist', 'build',
  'coverage', '.next', '.nuxt', '.venv', 'venv', '__pycache__',
  '.idea', '.vscode-test', 'target', 'Pods'
]);
const MAX_FILE_BYTES = 2 * 1024 * 1024;
const MAX_WORKSPACE_BYTES = 80 * 1024 * 1024;
const MAX_FILES = 20000;
const MAX_CHECKPOINT_BYTES = 20 * 1024 * 1024;
const MAX_CHECKPOINTS = 20;

async function scanWorkspace(root, openDocuments) {
  const files = new Map();
  let bytes = 0;
  async function visit(dir) {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.name.startsWith('.') && EXCLUDED.has(entry.name)) continue;
      if (EXCLUDED.has(entry.name)) continue;
      const absolute = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        await visit(absolute);
      } else if (entry.isFile()) {
        const stat = await fs.stat(absolute);
        if (stat.size > MAX_FILE_BYTES) continue;
        bytes += stat.size;
        if (bytes > MAX_WORKSPACE_BYTES || files.size >= MAX_FILES) {
          throw new Error('Workspace is too large for local automatic checkpoints (limit: 20,000 files / 80 MB of text files).');
        }
        const data = await fs.readFile(absolute);
        if (data.includes(0)) continue;
        files.set(path.relative(root, absolute), data.toString('utf8'));
      }
    }
  }
  await visit(root);
  for (const doc of openDocuments) {
    if (doc.uri.scheme !== 'file') continue;
    const relative = path.relative(root, doc.uri.fsPath);
    if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) continue;
    if (doc.isDirty) files.set(relative, doc.getText());
  }
  return files;
}

function changedFiles(beforeByRoot, afterByRoot) {
  const changes = [];
  for (const [root, before] of beforeByRoot) {
    const after = afterByRoot.get(root) || new Map();
    const paths = new Set([...before.keys(), ...after.keys()]);
    for (const relativePath of paths) {
      const oldText = before.has(relativePath) ? before.get(relativePath) : null;
      const newText = after.has(relativePath) ? after.get(relativePath) : null;
      if (oldText !== newText) changes.push({ root, path: relativePath, before: oldText, after: newText });
    }
  }
  return changes;
}

function encodedSize(files) {
  return files.reduce((size, file) => size + Buffer.byteLength(file.before || '') + Buffer.byteLength(file.after || ''), 0);
}

async function loadCheckpoints(directory) {
  try {
    const value = JSON.parse(await fs.readFile(path.join(directory, 'checkpoints.json'), 'utf8'));
    return Array.isArray(value) ? value.slice(-MAX_CHECKPOINTS) : [];
  } catch (error) {
    if (error.code === 'ENOENT') return [];
    throw error;
  }
}

async function saveCheckpoints(directory, checkpoints) {
  await fs.mkdir(directory, { recursive: true });
  const target = path.join(directory, 'checkpoints.json');
  const temporary = `${target}.tmp`;
  await fs.writeFile(temporary, JSON.stringify(checkpoints.slice(-MAX_CHECKPOINTS)), 'utf8');
  await fs.rename(temporary, target);
}

function withinRoot(root, relativePath) {
  const target = path.resolve(root, relativePath);
  const relative = path.relative(root, target);
  if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new Error(`Invalid checkpoint path: ${relativePath}`);
  }
  return target;
}

async function restoreFiles(files) {
  const restored = [];
  const conflicts = [];
  for (const file of files) {
    const target = withinRoot(file.root, file.path);
    let current = null;
    try {
      const buffer = await fs.readFile(target);
      if (!buffer.includes(0)) current = buffer.toString('utf8');
      else current = '__binary__';
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
    if (current !== file.after) {
      conflicts.push(file.path);
      continue;
    }
    if (file.before === null) {
      await fs.unlink(target);
    } else {
      await fs.mkdir(path.dirname(target), { recursive: true });
      await fs.writeFile(target, file.before, 'utf8');
    }
    restored.push(file.path);
  }
  return { restored, conflicts };
}

module.exports = {
  MAX_CHECKPOINTS,
  MAX_CHECKPOINT_BYTES,
  scanWorkspace,
  changedFiles,
  encodedSize,
  loadCheckpoints,
  saveCheckpoints,
  restoreFiles
};
