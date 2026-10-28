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
            <th>Compare</th><th>Name</th><th>Chunking</th><th>Retrieval</th><th>Top-k</th>
            <th>Rerank</th><th>Status</th><th>Created</th><th>Actions</th>
          </tr>
        </thead>
        <tbody>
          ${exps.map(e => `
            <tr>
              <td><input type="checkbox" data-compare-exp="${escHtml(e.id)}" aria-label="Select ${escHtml(e.name)} for comparison" ${e.status !== 'done' ? 'disabled' : ''} /></td>
              <td class="name-cell">${escHtml(e.name)}</td>
              <td>${escHtml(e.config.chunking_strategy)}</td>
              <td><span class="pill pill-purple">${escHtml(e.config.retrieval_method)}</span></td>
              <td>${e.config.top_k}</td>
              <td><span class="pill pill-gray">off</span></td>
              <td><span class="pill ${statusPill(e.status)}">${escHtml(e.status)}</span></td>
              <td>${fmtDate(e.created_at)}</td>
              <td class="action-cell">
                ${e.status === 'pending' || e.status === 'failed' ? '<button class="btn btn-primary" data-run-exp="' + escHtml(e.id) + '">Run</button>' : ''}
                ${e.status === 'done' || e.status === 'failed' ? '<button class="btn btn-ghost" data-results-exp="' + escHtml(e.id) + '">Results</button>' : ''}
                ${e.status === 'failed' && e.error_message ? '<small class="selector-help">' + escHtml(e.error_message) + '</small>' : ''}
                <button class="btn btn-danger" data-delete-exp="${escHtml(e.id)}" title="Delete">🗑</button>
              </td>
            </tr>`).join('')}
        </tbody>
      </table>`;

    container.querySelectorAll('[data-run-exp]').forEach(btn => {
      btn.addEventListener('click', async () => {
        btn.disabled = true;
        showStatus('experiment-action-status', 'Running experiment… This can take time depending on dataset size and provider.', '');
        try {
          await api('POST', `/experiments/${btn.dataset.runExp}/run`);
          showStatus('experiment-action-status', 'Experiment completed.', 'success');
          await loadExperiments();
          await loadOverview();
          await openExperimentResults(btn.dataset.runExp);
        } catch (err) {
          showStatus('experiment-action-status', `Experiment failed: ${err.message}`, 'error');
          await loadExperiments();
          await loadOverview();
        }
      });
    });
    container.querySelectorAll('[data-results-exp]').forEach(btn => {
      btn.addEventListener('click', () => openExperimentResults(btn.dataset.resultsExp));
    });
    container.querySelectorAll('[data-compare-exp]').forEach(input => input.addEventListener('change', updateCompareButton));

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
          <tr><th>Name</th><th>Dataset</th><th>Questions</th><th>Created</th><th>Actions</th></tr>
        </thead>
        <tbody>
          ${benchmarks.map(b => `
            <tr>
              <td class="name-cell">${escHtml(b.name)}</td>
              <td>${escHtml(b.dataset_id.slice(0, 8))}</td>
              <td>${b.question_count}</td>
              <td>${fmtDate(b.created_at)}</td>
              <td><button class="btn btn-ghost" data-edit-benchmark="${escHtml(b.id)}">Edit questions</button>
              <button class="btn btn-danger" data-delete-benchmark="${escHtml(b.id)}">Delete</button></td>
            </tr>`).join('')}
        </tbody>
      </table>`;
    container.querySelectorAll('[data-edit-benchmark]').forEach(btn => btn.addEventListener('click', () => editBenchmark(btn.dataset.editBenchmark)));
    container.querySelectorAll('[data-delete-benchmark]').forEach(btn => btn.addEventListener('click', async () => {
      if (!confirm('Delete this benchmark? Existing experiments will be unlinked.')) return;
      try { await api('DELETE', `/benchmarks/${btn.dataset.deleteBenchmark}`); await loadBenchmarks(); }
      catch (err) { alert(err.message); }
    }));
  } catch (err) {
    setHtml('benchmarks-list', `<div class="empty-state"><p style="color:var(--red)">Error: ${escHtml(err.message)}</p></div>`);
  }
}

// Benchmark question editor, JSON/CSV import, and edit flow.
const benchmarkModal = document.getElementById('benchmark-modal');
const benchmarkForm  = document.getElementById('benchmark-form');
const benchmarkQuestionEditor = document.getElementById('benchmark-questions-editor');
const benchmarkImportFile = document.getElementById('benchmark-import-file');
const benchmarkModalTitle = document.getElementById('benchmark-modal-title');
let editingBenchmarkId = null;

function addBenchmarkQuestion(question = {}) {
  const row = document.createElement('div');
  row.className = 'card benchmark-question-row';
  row.style.padding = '14px';
  row.innerHTML = '<div class="form-row">' +
    '<label class="form-label">Question <span class="required">*</span><textarea class="form-input" data-question rows="2" maxlength="5000" required></textarea></label>' +
    '<label class="form-label">Expected answer<textarea class="form-input" data-answer rows="2"></textarea></label></div>' +
    '<label class="form-label">Relevant passages (one per line)<textarea class="form-input" data-passages rows="2" placeholder="Paste exact relevant text from the dataset"></textarea></label>' +
    '<div class="form-actions"><button type="button" class="btn btn-danger" data-remove-question>Remove question</button></div>';
  row.querySelector('[data-question]').value = question.question || '';
  row.querySelector('[data-answer]').value = question.answer || question.expected_answer || '';
  const passages = question.relevant_passages || [];
  row.querySelector('[data-passages]').value = Array.isArray(passages) ? passages.join('\n') : String(passages);
  row.querySelector('[data-remove-question]').addEventListener('click', () => {
    row.remove();
    if (!benchmarkQuestionEditor.children.length) addBenchmarkQuestion();
  });
  benchmarkQuestionEditor.appendChild(row);
}

function readBenchmarkQuestions() {
  return [...benchmarkQuestionEditor.querySelectorAll('.benchmark-question-row')].map((row, index) => {
    const question = row.querySelector('[data-question]').value.trim();
    const answer = row.querySelector('[data-answer]').value.trim();
    const relevant_passages = row.querySelector('[data-passages]').value.split(/\r?\n/).map(value => value.trim()).filter(Boolean);
    if (!question) throw new Error('Question ' + (index + 1) + ' is empty.');
    return { question, answer: answer || null, relevant_passages };
  });
}

function parseCsv(text) {
  const rows = [];
  let row = [], field = '', quoted = false;
  for (let index = 0; index < text.length; index++) {
    const char = text[index];
    if (quoted && char === '"' && text[index + 1] === '"') { field += '"'; index++; }
    else if (char === '"') quoted = !quoted;
    else if (char === ',' && !quoted) { row.push(field); field = ''; }
    else if ((char === '\n' || char === '\r') && !quoted) {
      if (char === '\r' && text[index + 1] === '\n') index++;
      row.push(field); field = '';
      if (row.some(value => value.trim())) rows.push(row);
      row = [];
    } else field += char;
  }
  row.push(field);
  if (row.some(value => value.trim())) rows.push(row);
  if (quoted) throw new Error('CSV has an unclosed quoted field.');
  if (rows.length < 2) throw new Error('CSV must contain a header and at least one data row.');
  const headers = rows[0].map(value => value.trim().toLowerCase());
  if (!headers.includes('question')) throw new Error('CSV header must include question.');
  return rows.slice(1).map((values, index) => {
    const item = Object.fromEntries(headers.map((header, i) => [header, values[i] || '']));
    if (!item.question.trim()) throw new Error('CSV row ' + (index + 2) + ': question is empty.');
    return {
      question: item.question.trim(),
      answer: (item.answer || item.expected_answer || '').trim() || null,
      relevant_passages: (item.relevant_passages || '').split('||').map(value => value.trim()).filter(Boolean),
    };
  });
}

function openBenchmarkModal() {
  editingBenchmarkId = null;
  benchmarkForm.reset();
  benchmarkForm.querySelector('[name="dataset_id"]').disabled = false;
  benchmarkModalTitle.textContent = 'New benchmark';
  benchmarkQuestionEditor.replaceChildren();
  addBenchmarkQuestion();
  benchmarkModal.classList.remove('hidden');
}

async function editBenchmark(id) {
  try {
    const benchmark = await api('GET', '/benchmarks/' + id);
    editingBenchmarkId = id;
    benchmarkForm.reset();
    benchmarkForm.querySelector('[name="name"]').value = benchmark.name;
    const datasetSelect = benchmarkForm.querySelector('[name="dataset_id"]');
    datasetSelect.value = benchmark.dataset_id;
    datasetSelect.disabled = true;
    benchmarkModalTitle.textContent = 'Edit benchmark questions';
    benchmarkQuestionEditor.replaceChildren();
    (benchmark.questions || []).forEach(question => addBenchmarkQuestion(question));
    if (!benchmark.questions?.length) addBenchmarkQuestion();
    benchmarkModal.classList.remove('hidden');
  } catch (err) { alert(err.message); }
}

function closeBenchmarkModal() {
  benchmarkModal.classList.add('hidden');
  benchmarkForm.reset();
  benchmarkForm.querySelector('[name="dataset_id"]').disabled = false;
  benchmarkQuestionEditor.replaceChildren();
  editingBenchmarkId = null;
  const status = document.getElementById('benchmark-status');
  if (status) { status.textContent = ''; status.className = 'form-status'; }
}

document.getElementById('btn-new-benchmark').addEventListener('click', openBenchmarkModal);
document.getElementById('close-benchmark-modal').addEventListener('click', closeBenchmarkModal);
document.getElementById('cancel-benchmark').addEventListener('click', closeBenchmarkModal);
document.getElementById('add-benchmark-question').addEventListener('click', () => addBenchmarkQuestion());
benchmarkModal.addEventListener('click', e => { if (e.target === benchmarkModal) closeBenchmarkModal(); });

benchmarkImportFile.addEventListener('change', async () => {
  const file = benchmarkImportFile.files?.[0];
  if (!file) return;
  try {
    const text = await file.text();
    let questions;
    if (file.name.toLowerCase().endsWith('.json')) {
      const parsed = JSON.parse(text);
      questions = Array.isArray(parsed) ? parsed : parsed.questions;
      if (!Array.isArray(questions)) throw new Error('JSON must be an array of question objects or an object containing a questions array.');
      questions = questions.map((item, index) => {
        if (!item || typeof item.question !== 'string' || !item.question.trim()) throw new Error('JSON row ' + (index + 1) + ': question must be a non-empty string.');
        return { question: item.question.trim(), answer: item.answer ?? item.expected_answer ?? null, relevant_passages: item.relevant_passages ?? [] };
      });
    } else questions = parseCsv(text);
    if (!questions.length) throw new Error('Import contains no questions.');
    benchmarkQuestionEditor.replaceChildren();
    questions.forEach(question => addBenchmarkQuestion(question));
    showStatus('benchmark-status', 'Loaded ' + questions.length + ' questions. Review them and save the benchmark.', 'success');
  } catch (err) {
    showStatus('benchmark-status', 'Import error: ' + err.message, 'error');
  } finally {
    benchmarkImportFile.value = '';
  }
});

benchmarkForm.addEventListener('submit', async e => {
  e.preventDefault();
  try {
    const questions = readBenchmarkQuestions();
    if (!questions.length) throw new Error('Add at least one question.');
    const payload = {
      name: benchmarkForm.querySelector('[name="name"]').value.trim(),
      dataset_id: benchmarkForm.querySelector('[name="dataset_id"]').value,
      questions,
    };
    if (!payload.name || !payload.dataset_id) throw new Error('Name and dataset are required.');
    if (editingBenchmarkId) {
      await api('PUT', '/benchmarks/' + editingBenchmarkId, { name: payload.name, questions });
      showStatus('benchmark-status', '✓ Benchmark updated!', 'success');
    } else {
      await api('POST', '/benchmarks', payload);
      showStatus('benchmark-status', '✓ Benchmark created!', 'success');
    }
    await loadBenchmarks();
    await loadOverview();
    setTimeout(closeBenchmarkModal, 700);
  } catch (err) {
    showStatus('benchmark-status', '✗ ' + err.message, 'error');
  }
});

// Result detail and comparison views.
const experimentResultsModal = document.getElementById('experiment-results-modal');
const experimentCompareModal = document.getElementById('experiment-compare-modal');
const metricLabels = {
  precision_at_k: 'Precision@k', recall_at_k: 'Recall@k', mrr: 'MRR',
  ndcg_at_k: 'NDCG@k', average_latency_ms: 'Average latency (ms)',
  total_tokens: 'Total tokens', cost_usd: 'Cost (USD)',
};
function metricValue(value, key) {
  if (value === null || value === undefined) return 'Unavailable';
  if (key === 'average_latency_ms') return Number(value).toFixed(2);
  return typeof value === 'number' ? Number(value).toFixed(4).replace(/\.?0+$/, '') : escHtml(value);
}
async function openExperimentResults(id) {
  const title = document.getElementById('experiment-results-title');
  const content = document.getElementById('experiment-results-content');
  title.textContent = 'Experiment results';
  content.innerHTML = '<p class="selector-help">Loading results…</p>';
  experimentResultsModal.classList.remove('hidden');
  try {
    const response = await api('GET', '/experiments/' + id + '/results');
    if (!response.results) {
      content.innerHTML = '<p><strong>Status:</strong> ' + escHtml(response.status) + '</p><p>' + escHtml(response.error || 'Results are not available yet.') + '</p>';
      return;
    }
    const result = response.results;
    title.textContent = 'Results: ' + id.slice(0, 8);
    const summary = result.summary || {};
    const summaryRows = Object.entries(metricLabels).map(([key, label]) =>
      '<tr><th>' + escHtml(label) + '</th><td>' + metricValue(summary[key], key) + '</td></tr>'
    ).join('');
    const cases = (result.questions || []).map((item, index) =>
      '<div class="card" style="padding:14px;margin-top:12px;">' +
      '<h3>Question ' + (index + 1) + ': ' + escHtml(item.question) + '</h3>' +
      '<p><strong>Expected answer:</strong> ' + escHtml(item.expected_answer || 'Not provided') + '</p>' +
      '<p><strong>Generated answer:</strong> ' + escHtml(item.generated_answer || 'No answer returned') + '</p>' +
      '<p><strong>Latency:</strong> ' + metricValue(item.latency_ms, 'average_latency_ms') + ' ms · <strong>Cost:</strong> Unavailable</p>' +
      '<p><strong>Retrieval metrics:</strong> ' + Object.entries(item.metrics || {}).map(([key, value]) => escHtml(metricLabels[key] || key) + ': ' + metricValue(value, key)).join(' · ') + '</p>' +
      '<details><summary>Retrieved evidence (' + (item.retrieved_evidence || []).length + ')</summary>' +
      (item.retrieved_evidence || []).map(evidence =>
        '<div style="padding:8px;border-top:1px solid var(--border);"><p><strong>' + escHtml(evidence.filename) +
        ' · chunk ' + (evidence.chunk_index + 1) + ' · rank ' + evidence.rank + '</strong></p><p>' + escHtml(evidence.text) + '</p></div>'
      ).join('') + '</details></div>'
    ).join('');
    content.innerHTML = '<p>Chunks: ' + escHtml(result.chunk_count) + ' · Questions: ' + escHtml(summary.evaluated_questions) +
      ' · Questions with relevance labels: ' + escHtml(summary.questions_with_relevance_labels) + '</p>' +
      '<table class="data-table"><tbody>' + summaryRows + '</tbody></table>' +
      '<p class="selector-help" style="margin-top:8px;">' + escHtml(result.metric_definitions?.precision_at_k || '') +
      ' Cost is unavailable unless provider pricing is configured.</p>' + cases;
  } catch (err) {
    content.innerHTML = '<p class="form-status error">' + escHtml(err.message) + '</p>';
  }
}

function updateCompareButton() {
  const selected = document.querySelectorAll('[data-compare-exp]:checked').length;
  const button = document.getElementById('btn-compare-experiments');
  if (button) {
    button.textContent = 'Compare selected (' + selected + ')';
    button.disabled = selected < 2;
  }
}
document.getElementById('btn-compare-experiments').addEventListener('click', async () => {
  const experiment_ids = [...document.querySelectorAll('[data-compare-exp]:checked')].map(input => input.dataset.compareExp);
  const content = document.getElementById('experiment-compare-content');
  content.innerHTML = '<p class="selector-help">Comparing completed experiments…</p>';
  experimentCompareModal.classList.remove('hidden');
  try {
    const result = await api('POST', '/experiments/compare', { experiment_ids });
    const metrics = ['precision_at_k', 'recall_at_k', 'mrr', 'ndcg_at_k', 'average_latency_ms'];
    const rows = metrics.map(metric => '<tr><th>' + escHtml(metricLabels[metric]) + '</th>' +
      result.experiments.map(exp => '<td>' + metricValue(exp.summary?.[metric], metric) + '</td>').join('') + '</tr>').join('');
    content.innerHTML = '<p>Only runs on the same dataset and benchmark can be compared. Deltas below are relative to the first selected run.</p>' +
      '<table class="data-table"><thead><tr><th>Metric</th>' + result.experiments.map(exp => '<th>' + escHtml(exp.name) + '</th>').join('') +
      '</tr></thead><tbody>' + rows + '</tbody></table><h3 style="margin-top:16px;">Configuration differences</h3>' +
      '<pre style="white-space:pre-wrap;overflow-wrap:anywhere;">' + escHtml(JSON.stringify(result.configuration_differences, null, 2)) +
      '</pre><h3 style="margin-top:16px;">Metric deltas vs first run</h3><pre style="white-space:pre-wrap;overflow-wrap:anywhere;">' +
      escHtml(JSON.stringify(result.metric_deltas_vs_first, null, 2)) + '</pre>';
  } catch (err) {
    content.innerHTML = '<p class="form-status error">' + escHtml(err.message) + '</p>';
  }
});
document.getElementById('close-experiment-results').addEventListener('click', () => experimentResultsModal.classList.add('hidden'));
document.getElementById('close-experiment-compare').addEventListener('click', () => experimentCompareModal.classList.add('hidden'));
experimentResultsModal.addEventListener('click', e => { if (e.target === experimentResultsModal) experimentResultsModal.classList.add('hidden'); });
experimentCompareModal.addEventListener('click', e => { if (e.target === experimentCompareModal) experimentCompareModal.classList.add('hidden'); });

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
