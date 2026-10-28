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

// Benchmark modal
const benchmarkModal = document.getElementById('benchmark-modal');
const benchmarkForm  = document.getElementById('benchmark-form');

function openBenchmarkModal()  { benchmarkModal.classList.remove('hidden'); }
function closeBenchmarkModal() {
  benchmarkModal.classList.add('hidden');
  benchmarkForm.reset();
  const s = document.getElementById('benchmark-status');
  if (s) { s.textContent = ''; s.className = 'form-status'; }
}

document.getElementById('btn-new-benchmark').addEventListener('click', openBenchmarkModal);
document.getElementById('close-benchmark-modal').addEventListener('click', closeBenchmarkModal);
document.getElementById('cancel-benchmark').addEventListener('click', closeBenchmarkModal);
benchmarkModal.addEventListener('click', e => { if (e.target === benchmarkModal) closeBenchmarkModal(); });

benchmarkForm.addEventListener('submit', async e => {
  e.preventDefault();
  const fd = new FormData(benchmarkForm);
  const payload = { name: fd.get('name'), dataset_id: fd.get('dataset_id'), questions: [] };
  try {
    await api('POST', '/benchmarks', payload);
    showStatus('benchmark-status', '✓ Benchmark created!', 'success');
    loadBenchmarks();
    loadOverview();
    setTimeout(closeBenchmarkModal, 800);
  } catch (err) {
    showStatus('benchmark-status', `✗ ${err.message}`, 'error');
  }
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
