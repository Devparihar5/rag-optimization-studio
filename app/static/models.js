/* Provider setup, generation and embedding model discovery. */
'use strict';

(() => {
  const form = document.getElementById('model-provider-form');
  if (!form) return;

  const provider = document.getElementById('model-provider');
  const modelType = document.getElementById('model-type');
  const apiKey = document.getElementById('model-api-key');
  const apiKeyWrap = document.getElementById('model-api-key-wrap');
  const baseUrl = document.getElementById('model-base-url');
  const baseUrlWrap = document.getElementById('model-base-url-wrap');
  const discoverButton = document.getElementById('discover-models');
  const choiceWrap = document.getElementById('model-choice-wrap');
  const modelChoice = document.getElementById('discovered-model');
  const saveButton = document.getElementById('save-model');
  const status = document.getElementById('model-discovery-status');
  const generationList = document.getElementById('configured-models-list');
  const embeddingList = document.getElementById('configured-embedding-models-list');
  const experimentForm = document.getElementById('experiment-form');
  const experimentModel = experimentForm?.querySelector('[name="generation_model"]');
  const embeddingModel = experimentForm?.querySelector('[name="embedding_model"]');
  const experimentSubmit = experimentForm?.querySelector('[type="submit"]');
  const experimentHelp = document.getElementById('generation-model-help');
  const embeddingHelp = document.getElementById('embedding-model-help');
  const datasetSelect = experimentForm?.querySelector('[name="dataset_id"]');

  function updateExperimentSubmitState() {
    if (!experimentSubmit) return;
    experimentSubmit.disabled = !datasetSelect?.value || !experimentModel?.value ||
      !embeddingModel?.value || experimentModel.disabled || embeddingModel.disabled;
  }

  // Prevent a brief enabled-submit window while async selectors are loading.
  updateExperimentSubmitState();

  let discovered = [];
  let discoveredFor = '';

  function escapeHtml(value) {
    return String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  async function request(path, method = 'GET', body) {
    const options = { method, headers: { 'Content-Type': 'application/json' } };
    if (body !== undefined) options.body = JSON.stringify(body);
    const response = await fetch('/api' + path, options);
    if (response.status === 204) {
      if (!response.ok) throw new Error('Request failed');
      return null;
    }
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || 'Request failed');
    return payload;
  }

  function setStatus(message, type = '') {
    status.textContent = message;
    status.className = 'form-status' + (type ? ' ' + type : '');
  }

  function currentPayload() {
    const value = { provider: provider.value };
    if (provider.value === 'ollama') value.base_url = baseUrl.value.trim() || 'http://localhost:11434';
    else value.api_key = apiKey.value.trim();
    return value;
  }

  function invalidateDiscovery() {
    discovered = [];
    discoveredFor = '';
    choiceWrap.classList.add('hidden');
    modelChoice.replaceChildren(new Option('Verify provider first…', ''));
    saveButton.disabled = true;
  }

  function updateProviderFields() {
    const isLocal = provider.value === 'ollama';
    apiKeyWrap.classList.toggle('hidden', isLocal);
    baseUrlWrap.classList.toggle('hidden', !isLocal);
    apiKey.required = !isLocal;
    apiKey.disabled = isLocal;
    invalidateDiscovery();
    setStatus('');
  }

  provider.addEventListener('change', updateProviderFields);
  modelType.addEventListener('change', () => {
    invalidateDiscovery();
    setStatus('');
    discoverButton.textContent = modelType.value === 'embedding'
      ? 'Verify connection & load embedding models'
      : 'Verify connection & load generation models';
  });
  apiKey.addEventListener('input', invalidateDiscovery);
  baseUrl.addEventListener('input', invalidateDiscovery);

  discoverButton.addEventListener('click', async () => {
    invalidateDiscovery();
    discoverButton.disabled = true;
    discoverButton.textContent = 'Checking provider…';
    setStatus('Checking credentials and loading available models…');
    try {
      const payload = currentPayload();
      const kind = modelType.value;
      const endpoint = kind === 'embedding' ? '/embedding-models/discover' : '/models/discover';
      const result = await request(endpoint, 'POST', payload);
      discovered = result.models || [];
      discoveredFor = JSON.stringify({ ...payload, model_type: kind });
      if (!discovered.length) {
        setStatus('Connection succeeded, but no compatible ' + kind + ' models were found for this account.', 'error');
        return;
      }
      modelChoice.replaceChildren(new Option('Choose a model…', ''));
      discovered.forEach(model => {
        modelChoice.add(new Option(model.name === model.id ? model.id : model.name + ' (' + model.id + ')', model.id));
      });
      choiceWrap.classList.remove('hidden');
      setStatus('Verified. Found ' + discovered.length + ' available ' + kind + ' model(s).', 'success');
    } catch (error) {
      setStatus(error.message, 'error');
    } finally {
      discoverButton.disabled = false;
      discoverButton.textContent = modelType.value === 'embedding'
        ? 'Verify connection & load embedding models'
        : 'Verify connection & load generation models';
    }
  });

  modelChoice.addEventListener('change', () => {
    saveButton.disabled = !modelChoice.value || !discovered.some(item => item.id === modelChoice.value);
  });

  form.addEventListener('submit', async event => {
    event.preventDefault();
    const payload = currentPayload();
    const kind = modelType.value;
    if (!modelChoice.value || discoveredFor !== JSON.stringify({ ...payload, model_type: kind })) {
      setStatus('Verify the provider again and select one of its available models.', 'error');
      return;
    }
    saveButton.disabled = true;
    saveButton.textContent = 'Saving…';
    try {
      const endpoint = kind === 'embedding' ? '/embedding-models' : '/models';
      const result = await request(endpoint, 'POST', { ...payload, model_id: modelChoice.value });
      setStatus('Added ' + result.name + ' successfully.', 'success');
      apiKey.value = '';
      invalidateDiscovery();
      await loadConfiguredModels();
    } catch (error) {
      setStatus(error.message, 'error');
    } finally {
      saveButton.textContent = 'Add selected model';
      saveButton.disabled = true;
    }
  });

  async function renderModelList(path, target, kind) {
    const models = await request(path);
    if (!models.length) {
      target.innerHTML = '<div class="empty-state"><div class="empty-icon">🤖</div><p>No ' + kind + ' models configured yet.</p></div>';
      return models;
    }
    target.innerHTML = '<div class="configured-models">' + models.map(model => {
      const providerName = { openai: 'OpenAI', gemini: 'Google Gemini', ollama: 'Ollama · Local' }[model.provider] || model.provider;
      const credential = model.provider === 'ollama' ? 'Local endpoint' : 'API key saved securely';
      return '<article class="configured-model"><div class="configured-model-icon">🤖</div>' +
        '<div class="configured-model-info"><strong>' + escapeHtml(model.name) + '</strong>' +
        '<span>' + escapeHtml(providerName) + ' · ' + escapeHtml(model.model_id) + '</span>' +
        '<small>' + escapeHtml(credential) + '</small></div>' +
        '<button type="button" class="btn btn-danger" data-remove-model="' + escapeHtml(model.id) +
        '" data-kind="' + kind + '">Remove</button></article>';
    }).join('') + '</div>';
    target.querySelectorAll('[data-remove-model]').forEach(button => {
      button.addEventListener('click', async () => {
        if (!confirm('Remove this ' + kind + ' model?')) return;
        button.disabled = true;
        try {
          const endpoint = button.dataset.kind === 'embedding' ? '/embedding-models/' : '/models/';
          await request(endpoint + encodeURIComponent(button.dataset.removeModel), 'DELETE');
          await loadConfiguredModels();
        } catch (error) {
          alert(error.message);
          button.disabled = false;
        }
      });
    });
    return models;
  }

  async function loadConfiguredModels() {
    try {
      const [generations, embeddings] = await Promise.all([
        renderModelList('/models', generationList, 'generation'),
        renderModelList('/embedding-models', embeddingList, 'embedding'),
      ]);

      if (experimentModel) {
        const previous = experimentModel.value;
        experimentModel.replaceChildren();
        experimentModel.add(new Option(generations.length ? 'Choose a generation model…' : 'Configure a generation model first…', ''));
        generations.forEach(model => {
          const providerName = { openai: 'OpenAI', gemini: 'Gemini', ollama: 'Ollama' }[model.provider] || model.provider;
          experimentModel.add(new Option(providerName + ' · ' + model.name, model.id));
        });
        experimentModel.value = generations.some(model => model.id === previous) ? previous : '';
        experimentModel.disabled = generations.length === 0;
        if (experimentHelp) experimentHelp.textContent = generations.length
          ? 'Choose a configured generation model.'
          : 'Add a generation model on the Models page first.';
      }

      if (embeddingModel) {
        const previous = embeddingModel.value;
        embeddingModel.replaceChildren();
        embeddingModel.add(new Option(embeddings.length ? 'Choose an embedding model…' : 'Configure an embedding model first…', ''));
        embeddings.forEach(model => {
          const providerName = { openai: 'OpenAI', gemini: 'Gemini', ollama: 'Ollama' }[model.provider] || model.provider;
          embeddingModel.add(new Option(providerName + ' · ' + model.name, model.id));
        });
        embeddingModel.value = embeddings.some(model => model.id === previous) ? previous : '';
        embeddingModel.disabled = embeddings.length === 0;
        if (embeddingHelp) embeddingHelp.textContent = embeddings.length
          ? 'Choose a configured embedding model.'
          : 'Add an embedding model on the Models page first.';
      }

      updateExperimentSubmitState();
    } catch (error) {
      generationList.innerHTML = '<div class="empty-state"><p class="error-text">Could not load models: ' + escapeHtml(error.message) + '</p></div>';
      if (experimentModel) experimentModel.disabled = true;
      if (embeddingModel) embeddingModel.disabled = true;
      updateExperimentSubmitState();
    }
  }

  document.getElementById('refresh-models').addEventListener('click', loadConfiguredModels);
  document.querySelector('[data-tab="models"]')?.addEventListener('click', loadConfiguredModels);
  [datasetSelect, experimentModel, embeddingModel].forEach(select => select?.addEventListener('change', updateExperimentSubmitState));
  // novalidate is used by the form, so enforce the same requirement in code as well.
  experimentForm?.addEventListener('submit', event => {
    if (!datasetSelect?.value || !experimentModel?.value || !embeddingModel?.value ||
        experimentModel.disabled || embeddingModel.disabled) {
      event.preventDefault();
      event.stopImmediatePropagation();
      updateExperimentSubmitState();
      setStatus('Select a dataset, generation model, and embedding model before creating an experiment.', 'error');
    }
  }, true);
  experimentForm?.addEventListener('reset', () => {
    if (experimentSubmit) experimentSubmit.disabled = true;
  });

  updateProviderFields();
  loadConfiguredModels();
})();
