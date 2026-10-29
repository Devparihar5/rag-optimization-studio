/* ── RAG Optimization Studio — app.js ──────────────────────── */
'use strict';

// ── Tab navigation ──────────────────────────────────────────
const NAV_ITEMS   = document.querySelectorAll('.nav-item');
const TAB_CONTENT = document.querySelectorAll('.tab-content');
const PAGE_TITLE  = document.getElementById('page-title');

const TAB_TITLES = {
  overview:    'Overview',
  experiments: 'Experiments',
  datasets:    'Datasets',
  benchmarks:  'Benchmarks',
  models:      'Models',
};

function switchTab(tabName) {
  NAV_ITEMS.forEach(el => el.classList.toggle('active', el.dataset.tab === tabName));
  TAB_CONTENT.forEach(el =>
    el.classList.toggle('active', el.id === `tab-${tabName}`)
  );
  PAGE_TITLE.textContent = TAB_TITLES[tabName] ?? tabName;

  // Refresh data when switching
  if (tabName === 'overview')    loadOverview();
  if (tabName === 'experiments') loadExperiments();
  if (tabName === 'datasets')    loadDatasets();
  if (tabName === 'benchmarks')  loadBenchmarks();
}

NAV_ITEMS.forEach(item => {
  item.addEventListener('click', e => {
    e.preventDefault();
    switchTab(item.dataset.tab);
  });
});

// ── API helpers ─────────────────────────────────────────────
async function api(method, path, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const res = await fetch(`/api${path}`, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? res.statusText);
  }
  if (res.status === 204) return null;
  return res.json();
}

function html(strings, ...vals) {
  return strings.reduce((acc, str, i) =>
    acc + str + (vals[i] !== undefined ? escHtml(String(vals[i])) : ''), '');
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
          .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function fmtDate(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleString(undefined, {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  });
}

function setHtml(id, markup) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = markup;
}

function showStatus(id, msg, type) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = msg;
  el.className = `form-status ${type}`;
}

// ── Overview ─────────────────────────────────────────────────
async function loadOverview() {
  try {
    const data = await api('GET', '/overview');
    document.getElementById('stat-datasets').textContent    = data.datasets;
    document.getElementById('stat-benchmarks').textContent  = data.benchmarks;
    document.getElementById('stat-experiments').textContent = data.experiments;
    document.getElementById('stat-running').textContent     = data.running;
  } catch {
    /* silently fail — server may be starting */
  }

  // Also populate recent experiments
  try {
    const exps = await api('GET', '/experiments');
    const recent = exps.slice(-5).reverse();
    const container = document.getElementById('recent-experiments');
    if (!recent.length) {
      container.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">🧪</div>
          <p>No experiments yet. Create one to get started.</p>
        </div>`;
      return;
    }
    container.innerHTML = `
      <table class="data-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Method</th>
            <th>Status</th>
            <th>Created</th>
          </tr>
        </thead>
        <tbody>
          ${recent.map(e => `
            <tr>
              <td class="name-cell">${escHtml(e.name)}</td>
              <td>${escHtml(e.config.retrieval_method)}</td>
              <td><span class="pill ${statusPill(e.status)}">${escHtml(e.status)}</span></td>
              <td>${fmtDate(e.created_at)}</td>
            </tr>`).join('')}
        </tbody>
      </table>`;
  } catch { /* ignore */ }
}

function statusPill(status) {
  return { pending: 'pill-yellow', running: 'pill-blue', done: 'pill-green', failed: 'pill-gray' }[status] ?? 'pill-gray';
}

// ── Experiments ──────────────────────────────────────────────
async function loadExperiments() {
  try {
    const exps = await api('GET', '/experiments');
    const container = document.getElementById('experiments-list');
    if (!exps.length) {
      container.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">🧪</div>
          <p>No experiments yet. Click <strong>+ New experiment</strong> to start.</p>
        </div>`;
      return;
    }
    container.innerHTML = `
      <table class="data-table">
        <thead>
          <tr>
            <th>Select</th>
            <th>Name</th>
            <th>Chunking</th>
            <th>Retrieval</th>
            <th>Top-k</th>
            <th>Rerank</th>
            <th>Status</th>
            <th>Created</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          ${exps.map(e => `
            <tr>
              <td><input type="checkbox" data-select-exp="${escHtml(e.id)}" aria-label="Select ${escHtml(e.name)} for comparison" ${e.status !== 'done' ? 'disabled' : ''}></td>
              <td class="name-cell">${escHtml(e.name)}</td>
              <td>${escHtml(e.config.chunking_strategy)}</td>
              <td><span class="pill pill-purple">${escHtml(e.config.retrieval_method)}</span></td>
              <td>${e.config.top_k}</td>
              <td>${e.config.reranking ? '<span class="pill pill-green">on</span>' : '<span class="pill pill-gray">off</span>'}</td>
              <td><span class="pill ${statusPill(e.status)}">${escHtml(e.status)}</span></td>
              <td>${fmtDate(e.created_at)}</td>
              <td>
                <button class="btn btn-ghost" data-run-exp="${escHtml(e.id)}" ${e.status === 'running' ? 'disabled' : ''}>Run</button>
                <button class="btn btn-ghost" data-results-exp="${escHtml(e.id)}">Results</button>
                <button class="btn btn-danger" data-delete-exp="${escHtml(e.id)}" title="Delete">🗑</button>
              </td>
            </tr>`).join('')}
        </tbody>
      </table>`;

    const compareButton = document.getElementById('btn-compare-experiments');
    const updateCompareButton = () => {
      const selected = container.querySelectorAll('[data-select-exp]:checked');
      if (compareButton) compareButton.disabled = selected.length < 2;
    };
    container.querySelectorAll('[data-select-exp]').forEach(input => input.addEventListener('change', updateCompareButton));
    updateCompareButton();
    container.querySelectorAll('[data-run-exp]').forEach(btn => btn.addEventListener('click', async () => {
      btn.disabled = true;
      btn.textContent = 'Starting…';
      try {
        await api('POST', `/experiments/${btn.dataset.runExp}/run`);
        showStatus('form-status', 'Experiment started. Results will appear when execution finishes.', 'success');
        loadExperiments();
        pollExperiment(btn.dataset.runExp, 0);
      } catch (err) {
        btn.disabled = false;
        btn.textContent = 'Run';
        alert(err.message);
      }
    }));
    container.querySelectorAll('[data-results-exp]').forEach(btn => btn.addEventListener('click', () => openExperimentResults(btn.dataset.resultsExp)));
    // Delete buttons
    container.querySelectorAll('[data-delete-exp]').forEach(btn => {
      btn.addEventListener('click', async () => {
        if (!confirm('Delete this experiment?')) return;
        try {
          await api('DELETE', `/experiments/${btn.dataset.deleteExp}`);
          loadExperiments();
          loadOverview();
        } catch (err) { alert(err.message); }
      });
    });
  } catch (err) {
    setHtml('experiments-list', `<div class="empty-state"><p style="color:var(--red)">Error: ${escHtml(err.message)}</p></div>`);
  }
}

// Experiment drawer
const experimentDrawer   = document.getElementById('experiment-drawer');
const drawerBackdrop     = document.getElementById('drawer-backdrop');
const experimentForm     = document.getElementById('experiment-form');

function openDrawer() {
  experimentDrawer.classList.remove('hidden');
  drawerBackdrop.classList.remove('hidden');
  document.body.style.overflow = 'hidden';
}

function closeDrawer() {
  experimentDrawer.classList.add('hidden');
  drawerBackdrop.classList.add('hidden');
  document.body.style.overflow = '';
  experimentForm.reset();
  const status = document.getElementById('form-status');
  if (status) { status.textContent = ''; status.className = 'form-status'; }
}

document.getElementById('btn-new-experiment').addEventListener('click', openDrawer);
document.getElementById('close-drawer').addEventListener('click', closeDrawer);
document.getElementById('cancel-experiment').addEventListener('click', closeDrawer);
drawerBackdrop.addEventListener('click', closeDrawer);

experimentForm.addEventListener('submit', async e => {
  e.preventDefault();
  const fd = new FormData(experimentForm);

  const payload = {
    name:        fd.get('name'),
    dataset_id:  fd.get('dataset_id'),
    benchmark_id: fd.get('benchmark_id') || null,
    config: {
      chunking_strategy: fd.get('chunking_strategy'),
      chunk_size:        parseInt(fd.get('chunk_size'), 10),
      overlap:           parseInt(fd.get('overlap'), 10),
      retrieval_method:  fd.get('retrieval_method'),
      top_k:             parseInt(fd.get('top_k'), 10),
      reranking:         fd.get('reranking') === 'on',
      query_rewrite:     fd.get('query_rewrite') === 'on',
      embedding_model:   fd.get('embedding_model'),
      generation_model:  fd.get('generation_model'),
      notes:             fd.get('notes') || null,
    },
  };

  try {
    await api('POST', '/experiments', payload);
    showStatus('form-status', '✓ Experiment saved!', 'success');
    loadExperiments();
    loadOverview();
    setTimeout(closeDrawer, 900);
  } catch (err) {
    showStatus('form-status', `✗ ${err.message}`, 'error');
  }
});

// ── Datasets ─────────────────────────────────────────────────
async function loadDatasets() {
  try {
    const datasets = await api('GET', '/datasets');
    const container = document.getElementById('datasets-list');
    if (!datasets.length) {
      container.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">🗄️</div>
          <p>No datasets yet. Click <strong>+ New dataset</strong> to add one.</p>
        </div>`;
      return;
    }
    container.innerHTML = `
      <table class="data-table">
        <thead>
          <tr><th>Name</th><th>Type</th><th>Description</th><th>Docs</th><th>Created</th><th></th></tr>
        </thead>
        <tbody>
          ${datasets.map(d => `
            <tr>
              <td class="name-cell">${escHtml(d.name)}</td>
              <td><span class="pill pill-blue">${escHtml(d.source_type)}</span></td>
              <td>${escHtml(d.description || '—')}</td>
              <td>${d.document_count}</td>
              <td>${fmtDate(d.created_at)}</td>
              <td>
                <button class="btn btn-ghost" data-upload-ds="${escHtml(d.id)}" title="Upload or view documents">📄 Documents</button>
                <button class="btn btn-danger" data-delete-ds="${escHtml(d.id)}" title="Delete">🗑</button>
              </td>
            </tr>`).join('')}
        </tbody>
      </table>`;

    container.querySelectorAll('[data-delete-ds]').forEach(btn => {
      btn.addEventListener('click', async () => {
        if (!confirm('Delete this dataset?')) return;
        try {
          await api('DELETE', `/datasets/${btn.dataset.deleteDs}`);
          loadDatasets();
          loadOverview();
        } catch (err) { alert(err.message); }
      });
    });
  } catch (err) {
    setHtml('datasets-list', `<div class="empty-state"><p style="color:var(--red)">Error: ${escHtml(err.message)}</p></div>`);
  }
}

// Dataset modal
const datasetModal = document.getElementById('dataset-modal');
const datasetForm  = document.getElementById('dataset-form');

function openDatasetModal()  { datasetModal.classList.remove('hidden'); }
function closeDatasetModal() {
  datasetModal.classList.add('hidden');
  datasetForm.reset();
  const s = document.getElementById('dataset-status');
  if (s) { s.textContent = ''; s.className = 'form-status'; }
}

document.getElementById('btn-new-dataset').addEventListener('click', openDatasetModal);
document.getElementById('close-dataset-modal').addEventListener('click', closeDatasetModal);
document.getElementById('cancel-dataset').addEventListener('click', closeDatasetModal);
datasetModal.addEventListener('click', e => { if (e.target === datasetModal) closeDatasetModal(); });

datasetForm.addEventListener('submit', async e => {
  e.preventDefault();
  const fd = new FormData(datasetForm);
  const payload = {
    name:        fd.get('name'),
    description: fd.get('description') ?? '',
    source_type: fd.get('source_type'),
  };
  try {
    await api('POST', '/datasets', payload);
    showStatus('dataset-status', '✓ Dataset created!', 'success');
    loadDatasets();
    loadOverview();
    setTimeout(closeDatasetModal, 800);
  } catch (err) {
    showStatus('dataset-status', `✗ ${err.message}`, 'error');
  }
});

// ── Benchmarks ───────────────────────────────────────────────
async function loadBenchmarks() {
  try {
    const benchmarks = await api('GET', '/benchmarks');
    const container = document.getElementById('benchmarks-list');
    if (!benchmarks.length) {
      container.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">🎯</div>
          <p>No benchmarks yet. Click <strong>+ New benchmark</strong> to add one.</p>
        </div>`;
      return;
    }
    container.innerHTML = `
      <table class="data-table">
        <thead>
          <tr><th>Name</th><th>Dataset ID</th><th>Questions</th><th>Created</th><th>Actions</th></tr>
        </thead>
        <tbody>
          ${benchmarks.map(b => `
            <tr>
              <td class="name-cell">${escHtml(b.name)}</td>
              <td>${escHtml(b.dataset_id)}</td>
              <td>${b.question_count}</td>
              <td>${fmtDate(b.created_at)}</td>
              <td><button class="btn btn-ghost" data-edit-bench="${escHtml(b.id)}">View / Edit</button></td>
            </tr>`).join('')}
        </tbody>
      </table>`;
    container.querySelectorAll('[data-edit-bench]').forEach(btn => btn.addEventListener('click', () => editBenchmark(btn.dataset.editBench)));
  } catch (err) {
    setHtml('benchmarks-list', `<div class="empty-state"><p style="color:var(--red)">Error: ${escHtml(err.message)}</p></div>`);
  }
}

// Benchmark editor with question CRUD and CSV/JSON import.
const benchmarkModal = document.getElementById('benchmark-modal');
const benchmarkForm  = document.getElementById('benchmark-form');
const benchmarkQuestions = document.getElementById('benchmark-questions');
const benchmarkImport = document.getElementById('benchmark-import-file');
let editingBenchmarkId = null;

function questionRow(question = {}) {
  const row = document.createElement('div');
  row.className = 'benchmark-question-row';
  row.innerHTML = \`
    <div class="benchmark-question-heading"><strong>Question</strong><button type="button" class="btn btn-danger remove-question">Remove</button></div>
    <label class="form-label">Question text <span class="required">*</span><textarea class="form-input" data-question rows="2" required maxlength="5000"></textarea></label>
    <label class="form-label">Expected answer <span class="optional">(optional)</span><textarea class="form-input" data-answer rows="2" maxlength="10000"></textarea></label>
    <label class="form-label">Relevant passages <span class="optional">(one per line, optional)</span><textarea class="form-input" data-passages rows="2" maxlength="20000"></textarea></label>\`;
  row.querySelector('[data-question]').value = question.question || '';
  row.querySelector('[data-answer]').value = question.answer || '';
  row.querySelector('[data-passages]').value = (question.relevant_passages || []).join('\n');
  row.querySelector('.remove-question').addEventListener('click', () => {
    row.remove();
    renumberQuestions();
  });
  benchmarkQuestions.appendChild(row);
  renumberQuestions();
}

function renumberQuestions() {
  benchmarkQuestions.querySelectorAll('.benchmark-question-row').forEach((row, index) => {
    row.querySelector('strong').textContent = \`Question \${index + 1}\`;
  });
}

function readBenchmarkQuestions() {
  const rows = [...benchmarkQuestions.querySelectorAll('.benchmark-question-row')];
  const questions = [];
  for (let index = 0; index < rows.length; index++) {
    const row = rows[index];
    const question = row.querySelector('[data-question]').value.trim();
    if (!question) throw new Error(\`Question \${index + 1}: question text is required.\`);
    questions.push({
      question,
      answer: row.querySelector('[data-answer]').value.trim() || null,
      relevant_passages: row.querySelector('[data-passages]').value.split('\n').map(value => value.trim()).filter(Boolean),
    });
  }
  if (!questions.length) throw new Error('Add at least one benchmark question.');
  return questions;
}

function openBenchmarkModal() {
  editingBenchmarkId = null;
  benchmarkForm.reset();
  benchmarkQuestions.replaceChildren();
  questionRow();
  document.querySelector('#benchmark-modal .drawer-header h2').textContent = 'New benchmark';
  document.getElementById('save-benchmark').textContent = 'Create benchmark';
  benchmarkModal.classList.remove('hidden');
}

async function editBenchmark(id) {
  try {
    const benchmark = await api('GET', \`/benchmarks/\${id}\`);
    editingBenchmarkId = id;
    benchmarkForm.reset();
    benchmarkQuestions.replaceChildren();
    benchmarkForm.elements.name.value = benchmark.name;
    await loadBenchmarkDatasetOptions(benchmark.dataset_id);
    (benchmark.questions || []).forEach(questionRow);
    document.querySelector('#benchmark-modal .drawer-header h2').textContent = 'View / edit benchmark';
    document.getElementById('save-benchmark').textContent = 'Save changes';
    benchmarkModal.classList.remove('hidden');
  } catch (err) { alert(err.message); }
}

async function loadBenchmarkDatasetOptions(selectedId) {
  const select = benchmarkForm.elements.dataset_id;
  const datasets = await api('GET', '/datasets');
  select.replaceChildren(new Option('Choose a dataset…', ''));
  datasets.forEach(dataset => select.add(new Option(\`\${dataset.name} (\${dataset.id.slice(0, 8)})\`, dataset.id)));
  select.value = selectedId || '';
}

function closeBenchmarkModal() {
  benchmarkModal.classList.add('hidden');
  benchmarkForm.reset();
  benchmarkQuestions.replaceChildren();
  editingBenchmarkId = null;
  const s = document.getElementById('benchmark-status');
  if (s) { s.textContent = ''; s.className = 'form-status'; }
  const importStatus = document.getElementById('benchmark-import-status');
  if (importStatus) { importStatus.textContent = ''; importStatus.className = 'form-status'; }
}

document.getElementById('btn-new-benchmark').addEventListener('click', openBenchmarkModal);
document.getElementById('close-benchmark-modal').addEventListener('click', closeBenchmarkModal);
document.getElementById('cancel-benchmark').addEventListener('click', closeBenchmarkModal);
document.getElementById('add-benchmark-question').addEventListener('click', () => questionRow());
benchmarkModal.addEventListener('click', e => { if (e.target === benchmarkModal) closeBenchmarkModal(); });

function parseCsv(text) {
  const rows = [];
  let row = [], field = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === '"' && quoted && text[i + 1] === '"') { field += '"'; i++; }
    else if (ch === '"') quoted = !quoted;
    else if (ch === ',' && !quoted) { row.push(field); field = ''; }
    else if ((ch === '\n' || ch === '\r') && !quoted) {
      if (ch === '\r' && text[i + 1] === '\n') i++;
      row.push(field); field = '';
      if (row.some(cell => cell.trim())) rows.push(row);
      row = [];
    } else field += ch;
  }
  if (field || row.length) { row.push(field); if (row.some(cell => cell.trim())) rows.push(row); }
  if (quoted) throw new Error('CSV import: unterminated quoted field.');
  if (!rows.length) throw new Error('CSV import: file is empty.');
  const header = rows.shift().map(value => value.trim().toLowerCase());
  const questionColumn = header.indexOf('question');
  if (questionColumn < 0) throw new Error('CSV import: missing required "question" column.');
  return rows.map((values, index) => {
    if (!values[questionColumn]?.trim()) throw new Error(\`CSV row \${index + 2}, field question: value is required.\`);
    const get = name => { const column = header.indexOf(name); return column >= 0 ? (values[column] || '').trim() : ''; };
    const passages = get('relevant_passages') || get('relevant_passage');
    return { question: values[questionColumn].trim(), answer: get('answer') || get('expected_answer') || null,
      relevant_passages: passages ? passages.split('|').map(value => value.trim()).filter(Boolean) : [] };
  });
}

benchmarkImport.addEventListener('change', async () => {
  const file = benchmarkImport.files?.[0];
  if (!file) return;
  const status = document.getElementById('benchmark-import-status');
  try {
    const text = await file.text();
    let questions;
    if (file.name.toLowerCase().endsWith('.json')) {
      const parsed = JSON.parse(text);
      questions = Array.isArray(parsed) ? parsed : parsed.questions;
      if (!Array.isArray(questions)) throw new Error('JSON import: expected an array or an object with a questions array.');
      questions = questions.map((item, index) => {
        if (!item || typeof item.question !== 'string' || !item.question.trim())
          throw new Error(\`JSON row \${index + 1}, field question: value is required.\`);
        return { question: item.question.trim(), answer: item.answer || item.expected_answer || null,
          relevant_passages: Array.isArray(item.relevant_passages) ? item.relevant_passages : [] };
      });
    } else questions = parseCsv(text);
    if (!questions.length) throw new Error('Import file has no questions.');
    benchmarkQuestions.replaceChildren();
    questions.forEach(questionRow);
    status.textContent = \`Imported \${questions.length} questions. Review them before saving.\`;
    status.className = 'form-status success';
  } catch (err) {
    status.textContent = err.message;
    status.className = 'form-status error';
  } finally { benchmarkImport.value = ''; }
});

benchmarkForm.addEventListener('submit', async e => {
  e.preventDefault();
  const saveButton = document.getElementById('save-benchmark');
  saveButton.disabled = true;
  try {
    const fd = new FormData(benchmarkForm);
    const payload = { name: fd.get('name'), dataset_id: fd.get('dataset_id'), questions: readBenchmarkQuestions() };
    if (!payload.dataset_id) throw new Error('Choose a dataset.');
    if (editingBenchmarkId) await api('PUT', \`/benchmarks/\${editingBenchmarkId}\`, payload);
    else await api('POST', '/benchmarks', payload);
    showStatus('benchmark-status', editingBenchmarkId ? '✓ Benchmark updated!' : '✓ Benchmark created!', 'success');
    loadBenchmarks();
    loadOverview();
    setTimeout(closeBenchmarkModal, 700);
  } catch (err) {
    showStatus('benchmark-status', \`✗ \${err.message}\`, 'error');
  } finally { saveButton.disabled = false; }
});

// Results detail and comparison.
const resultsModal = document.getElementById('results-modal');
document.getElementById('close-results-modal').addEventListener('click', () => resultsModal.classList.add('hidden'));
resultsModal.addEventListener('click', e => { if (e.target === resultsModal) resultsModal.classList.add('hidden'); });

function metricLabel(value, digits = 3) {
  return value === null || value === undefined ? 'Unavailable' :
    (typeof value === 'number' ? value.toFixed(digits) : String(value));
}

async function openExperimentResults(id) {
  resultsModal.classList.remove('hidden');
  document.getElementById('results-title').textContent = 'Experiment results';
  const content = document.getElementById('results-content');
  content.textContent = 'Loading results…';
  try {
    const response = await api('GET', \`/experiments/\${id}/results\`);
    if (response.status === 'running' || response.status === 'pending') {
      content.innerHTML = \`<p>Experiment status: <strong>\${escHtml(response.status)}</strong>. Results are not available yet.</p>\`;
      return;
    }
    if (response.status === 'failed') {
      content.innerHTML = \`<p class="error-text">Run failed: \${escHtml(response.error || 'No error details available.')}</p>\`;
      return;
    }
    const result = response.results || {};
    const metrics = result.metrics || response.summary || {};
    const cards = ['precision_at_k','recall_at_k','mrr','ndcg_at_k','avg_latency_ms','total_tokens','cost']
      .map(key => \`<div class="result-metric"><span>\${escHtml(key.replaceAll('_',' '))}</span><strong>\${metricLabel(metrics[key], key.includes('latency') ? 1 : 3)}</strong></div>\`).join('');
    const cases = (result.cases || []).map((item, index) => \`
      <article class="result-case">
        <h3>Q\${index + 1}. \${escHtml(item.question)}</h3>
        <p><strong>Answer</strong><br>\${escHtml(item.answer || 'No answer returned.')}</p>
        <p><strong>Expected answer</strong><br>\${escHtml(item.expected_answer || 'Not provided')}</p>
        <p><strong>Latency</strong>: \${metricLabel(item.latency_ms, 1)} ms · <strong>Cost</strong>: \${metricLabel(item.cost)}</p>
        <details><summary>Retrieved evidence (\${(item.retrieved || []).length})</summary>
          \${(item.retrieved || []).map(doc => \`<div class="evidence-item"><strong>\${escHtml(doc.filename)} · chunk \${doc.chunk_index}</strong><p>\${escHtml(doc.text)}</p></div>\`).join('') || '<p>No evidence retrieved.</p>'}
        </details>
      </article>\`).join('');
    content.innerHTML = \`<div class="results-metrics">\${cards}</div>
      <p class="selector-help">Retrieval metrics are averaged only over questions with relevant passages. Cost is unavailable unless the provider reports a usable price.</p>
      \${cases || '<p>No question results were recorded.</p>'}\`;
  } catch (err) { content.innerHTML = \`<p class="error-text">\${escHtml(err.message)}</p>\`; }
}

async function pollExperiment(id, attempt) {
  if (attempt > 90) { loadExperiments(); return; }
  try {
    const result = await api('GET', \`/experiments/\${id}/results\`);
    if (result.status === 'running' || result.status === 'pending') {
      setTimeout(() => pollExperiment(id, attempt + 1), 1000);
    } else {
      loadExperiments();
      loadOverview();
      if (result.status === 'done') openExperimentResults(id);
      else if (result.status === 'failed') alert(\`Experiment failed: \${result.error || 'Unknown error'}\`);
    }
  } catch { loadExperiments(); }
}

document.getElementById('btn-compare-experiments').addEventListener('click', async () => {
  const ids = [...document.querySelectorAll('[data-select-exp]:checked')].map(input => input.dataset.selectExp);
  if (ids.length < 2) return;
  const content = document.getElementById('results-content');
  resultsModal.classList.remove('hidden');
  document.getElementById('results-title').textContent = 'Compare experiments';
  content.textContent = 'Comparing selected experiments…';
  try {
    const comparison = await api('POST', '/experiments/compare', { experiment_ids: ids });
    const metricKeys = [...new Set(comparison.experiments.flatMap(item => Object.keys(item.summary || {})))];
    const headers = comparison.experiments.map(item => \`<th>\${escHtml(item.name)}</th>\`).join('');
    const rows = metricKeys.map(key => \`<tr><th>\${escHtml(key.replaceAll('_',' '))}</th>\${comparison.experiments.map(item => \`<td>\${metricLabel(item.summary[key])}</td>\`).join('')}</tr>\`).join('');
    const diffs = Object.entries(comparison.configuration_differences || {}).map(([key, values]) =>
      \`<li><strong>\${escHtml(key)}</strong>: \${Object.entries(values).map(([id, value]) => {
        const name = comparison.experiments.find(item => item.id === id)?.name || id;
        return \`\${escHtml(name)} = \${escHtml(JSON.stringify(value))}\`;
      }).join('; ')}</li>\`).join('');
    content.innerHTML = \`<div class="table-scroll"><table class="data-table"><thead><tr><th>Metric</th>\${headers}</tr></thead><tbody>\${rows}</tbody></table></div>
      <h3>Configuration differences</h3>\${diffs ? \`<ul>\${diffs}</ul>\` : '<p>Selected configurations are identical.</p>'}\`;
  } catch (err) { content.innerHTML = \`<p class="error-text">\${escHtml(err.message)}</p>\`; }
});

// ── Keyboard shortcuts ────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    closeDrawer();
    closeDatasetModal();
    closeBenchmarkModal();
  }
});

// ── Init ──────────────────────────────────────────────────────
loadOverview();
