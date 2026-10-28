/* Dataset upload and document management. */
'use strict';

(() => {
  const list = document.getElementById('datasets-list');
  if (!list) return;

  const modal = document.createElement('div');
  modal.className = 'modal hidden document-modal';
  modal.id = 'documents-modal';
  modal.innerHTML = `
    <div class="modal-box">
      <div class="drawer-header">
        <div><h2 id="documents-title">Dataset documents</h2><small id="documents-subtitle"></small></div>
        <button class="btn-close" type="button" data-close-documents aria-label="Close">✕</button>
      </div>
      <div class="upload-dropzone" id="document-dropzone">
        <p><strong>Upload documents</strong></p>
        <p>PDF, Markdown, TXT, or HTML. Maximum 10 MB per file.</p>
        <input class="form-input" id="document-files" type="file" multiple accept=".pdf,.md,.txt,.html,.htm" style="margin-top:12px" />
        <button class="btn btn-primary" id="upload-documents" type="button" style="margin-top:12px">Upload selected files</button>
      </div>
      <div id="document-status" class="form-status" role="status" aria-live="polite" style="margin-top:12px"></div>
      <h3 style="margin-top:20px">Uploaded files</h3>
      <div class="document-list" id="document-list"><p class="document-empty">Loading documents…</p></div>
    </div>`;
  document.body.appendChild(modal);

  let activeDatasetId = null;
  let busy = false;

  async function request(path, options = {}) {
    const response = await fetch('/api' + path, options);
    const data = response.status === 204 ? null : await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || response.statusText || 'Request failed');
    return data;
  }

  function status(message, type) {
    const node = document.getElementById('document-status');
    node.textContent = message;
    node.className = 'form-status ' + type;
  }

  function prettyBytes(value) {
    if (value < 1024) return value + ' B';
    if (value < 1024 * 1024) return (value / 1024).toFixed(1) + ' KB';
    return (value / (1024 * 1024)).toFixed(1) + ' MB';
  }

  async function refreshDocuments() {
    const documents = await request('/datasets/' + encodeURIComponent(activeDatasetId) + '/documents');
    const target = document.getElementById('document-list');
    if (!documents.length) {
      target.innerHTML = '<p class="document-empty">No documents yet. Choose files above to add your first documents.</p>';
      return;
    }
    target.innerHTML = documents.map(doc => `
      <div class="document-row">
        <div class="document-meta"><strong>${escapeHtml(doc.filename)}</strong><small>${prettyBytes(doc.size_bytes)} · ${new Date(doc.created_at).toLocaleString()}</small></div>
        <button class="btn btn-danger" type="button" data-remove-document="${escapeHtml(doc.id)}" aria-label="Delete ${escapeHtml(doc.filename)}">Remove</button>
      </div>`).join('');
  }

  function escapeHtml(value) {
    return String(value).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
      .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
  }

  async function openDocuments(datasetId) {
    activeDatasetId = datasetId;
    modal.classList.remove('hidden');
    document.getElementById('document-files').value = '';
    document.getElementById('documents-subtitle').textContent = 'Dataset ID: ' + datasetId;
    status('', '');
    document.getElementById('document-list').innerHTML = '<p class="document-empty">Loading documents…</p>';
    try {
      const dataset = await request('/datasets/' + encodeURIComponent(datasetId));
      document.getElementById('documents-title').textContent = dataset.name + ' documents';
      await refreshDocuments();
    } catch (error) {
      status(error.message, 'error');
    }
  }

  function closeDocuments() {
    modal.classList.add('hidden');
    activeDatasetId = null;
  }

  list.addEventListener('click', event => {
    const button = event.target.closest('[data-upload-ds]');
    if (button) openDocuments(button.dataset.uploadDs);
  });

  modal.addEventListener('click', async event => {
    if (event.target === modal || event.target.closest('[data-close-documents]')) {
      closeDocuments();
      return;
    }
    const removeButton = event.target.closest('[data-remove-document]');
    if (removeButton && activeDatasetId) {
      if (!confirm('Remove this document from the dataset?')) return;
      try {
        await request('/datasets/' + encodeURIComponent(activeDatasetId) + '/documents/' + encodeURIComponent(removeButton.dataset.removeDocument), {method:'DELETE'});
        await refreshDocuments();
        document.dispatchEvent(new CustomEvent('ragstudio:documents-changed'));
      } catch (error) { status(error.message, 'error'); }
    }
  });

  document.getElementById('upload-documents').addEventListener('click', async () => {
    if (busy || !activeDatasetId) return;
    const input = document.getElementById('document-files');
    if (!input.files.length) { status('Choose one or more files first.', 'error'); return; }
    busy = true;
    const button = document.getElementById('upload-documents');
    button.disabled = true;
    const form = new FormData();
    for (const file of input.files) form.append('files', file);
    try {
      const result = await request('/datasets/' + encodeURIComponent(activeDatasetId) + '/documents', {method:'POST', body:form});
      const uploaded = result.uploaded.length;
      const failed = result.failed.length;
      status(`Uploaded ${uploaded} file(s).${failed ? ' Failed: ' + result.failed.map(item => item.filename + ': ' + item.error).join(' | ') : ''}`, failed ? 'error' : 'success');
      input.value = '';
      await refreshDocuments();
      document.dispatchEvent(new CustomEvent('ragstudio:documents-changed'));
    } catch (error) {
      status(error.message, 'error');
    } finally {
      busy = false;
      button.disabled = false;
    }
  });

  const dropzone = document.getElementById('document-dropzone');
  dropzone.addEventListener('dragover', event => { event.preventDefault(); dropzone.classList.add('dragover'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', event => {
    event.preventDefault();
    dropzone.classList.remove('dragover');
    const input = document.getElementById('document-files');
    if (event.dataTransfer && event.dataTransfer.files.length) input.files = event.dataTransfer.files;
  });

  document.addEventListener('ragstudio:documents-changed', () => {
    if (typeof loadDatasets === 'function') loadDatasets();
    if (typeof loadOverview === 'function') loadOverview();
  });
  document.addEventListener('keydown', event => { if (event.key === 'Escape') closeDocuments(); });
})();
