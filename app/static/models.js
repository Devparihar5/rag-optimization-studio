/* Provider setup, model discovery, and experiment model selector. */
'use strict';

(() => {
  const form = document.getElementById('model-provider-form');
  if (!form) return;

  const provider = document.getElementById('model-provider');
  const apiKey = document.getElementById('model-api-key');
  const apiKeyWrap = document.getElementById('model-api-key-wrap');
  const baseUrl = document.getElementById('model-base-url');
  const baseUrlWrap = document.getElementById('model-base-url-wrap');
  const discoverButton = document.getElementById('discover-models');
  const choiceWrap = document.getElementById('model-choice-wrap');
  const modelChoice = document.getElementById('discovered-model');
  const saveButton = document.getElementById('save-model');
  const status = document.getElementById('model-discovery-status');
  const modelsList = document.getElementById('configured-models-list');
  const experimentModel = document.querySelector('#experiment-form [name="generation_model"]');
  const experimentSubmit = document.querySelector('#experiment-form [type="submit"]');
  const experimentHelp = document.getElementById('generation-model-help');

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

  function currentProviderPayload() {
    const value = { provider: provider.value };
    if (provider.value === 'ollama') {
      value.base_url = baseUrl.value.trim() || 'http://localhost:11434';
    } else {
      value.api_key = apiKey.value.trim();
    }
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
  apiKey.addEventListener('input', invalidateDiscovery);
  baseUrl.addEventListener('input', invalidateDiscovery);

  discoverButton.addEventListener('click', async () => {
    invalidateDiscovery();
    discoverButton.disabled = true;
    discoverButton.textContent = 'Checking provider…';
    setStatus('Checking credentials and loading available models…');
    try {
      const payload = currentProviderPayload();
      const result = await request('/models/discover', 'POST', payload);
      discovered = result.models || [];
      discoveredFor = JSON.stringify(payload);
      if (!discovered.length) {
        setStatus('Connection succeeded, but no generation models were found for this account.', 'error');
        return;
      }
      modelChoice.replaceChildren(new Option('Choose a model…', ''));
      discovered.forEach(model => {
        modelChoice.add(new Option(model.name === model.id ? model.id : model.name + ' (' + model.id + ')', model.id));
      });
      choiceWrap.classList.remove('hidden');
      setStatus('Verified. Found ' + discovered.length + ' available model(s).', 'success');
    } catch (error) {
      setStatus(error.message, 'error');
    } finally {
      discoverButton.disabled = false;
      discoverButton.textContent = 'Verify connection & load models';
    }
  });

  modelChoice.addEventListener('change', () => {
    saveButton.disabled = !modelChoice.value || !discovered.some(item => item.id === modelChoice.value);
  });

  form.addEventListener('submit', async event => {
    event.preventDefault();
    const payload = currentProviderPayload();
    if (!modelChoice.value || discoveredFor !== JSON.stringify(payload)) {
      setStatus('Verify the provider again and select one of its available models.', 'error');
      return;
    }
    saveButton.disabled = true;
    saveButton.textContent = 'Saving…';
    try {
      const result = await request('/models', 'POST', { ...payload, model_id: modelChoice.value });
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

  async function loadConfiguredModels() {
    try {
      const models = await request('/models');
      if (!models.length) {
        modelsList.innerHTML = '<div class="empty-state"><div class="empty-icon">🤖</div><p>No models configured yet. Add an online or local model above.</p></div>';
      } else {
        modelsList.innerHTML = '<div class="configured-models">' + models.map(model => {
          const providerName = { openai: 'OpenAI', gemini: 'Google Gemini', ollama: 'Ollama · Local' }[model.provider] || model.provider;
          const credential = model.provider === 'ollama' ? 'Local endpoint' : 'API key saved securely';
          return '<article class="configured-model">' +
            '<div class="configured-model-icon">🤖</div>' +
            '<div class="configured-model-info"><strong>' + escapeHtml(model.name) + '</strong>' +
            '<span>' + escapeHtml(providerName) + ' · ' + escapeHtml(model.model_id) + '</span>' +
            '<small>' + escapeHtml(credential) + '</small></div>' +
            '<button type="button" class="btn btn-danger" data-remove-model="' + escapeHtml(model.id) + '">Remove</button>' +
            '</article>';
        }).join('') + '</div>';
        modelsList.querySelectorAll('[data-remove-model]').forEach(button => {
          button.addEventListener('click', async () => {
            if (!confirm('Remove this configured model? Existing experiments will keep their saved configuration.')) return;
            button.disabled = true;
            try {
              await request('/models/' + encodeURIComponent(button.dataset.removeModel), 'DELETE');
              await loadConfiguredModels();
            } catch (error) {
              alert(error.message);
              button.disabled = false;
            }
          });
        });
      }

      if (experimentModel) {
        const previous = experimentModel.value;
        experimentModel.replaceChildren();
        if (!models.length) {
          experimentModel.add(new Option('Configure a model first…', ''));
          experimentModel.disabled = true;
          if (experimentSubmit) experimentSubmit.disabled = true;
          if (experimentHelp) experimentHelp.textContent = 'Add a model on the Models page before creating an experiment.';
        } else {
          experimentModel.add(new Option('Choose a configured model…', ''));
          models.forEach(model => {
            const providerName = { openai: 'OpenAI', gemini: 'Gemini', ollama: 'Ollama' }[model.provider] || model.provider;
            experimentModel.add(new Option(providerName + ' · ' + model.name, model.id));
          });
          experimentModel.value = models.some(model => model.id === previous) ? previous : '';
          experimentModel.disabled = false;
          if (experimentSubmit) experimentSubmit.disabled = !experimentModel.value;
          if (experimentHelp) experimentHelp.textContent = 'Choose a model configured on the Models page. Provider credentials are stored on the server.';
        }
      }
    } catch (error) {
      modelsList.innerHTML = '<div class="empty-state"><p class="error-text">Could not load configured models: ' + escapeHtml(error.message) + '</p></div>';
      if (experimentModel) {
        experimentModel.disabled = true;
        experimentModel.replaceChildren(new Option('Could not load models', ''));
      }
      if (experimentSubmit) experimentSubmit.disabled = true;
    }
  }

  document.getElementById('refresh-models').addEventListener('click', loadConfiguredModels);
  document.querySelector('[data-tab="models"]')?.addEventListener('click', loadConfiguredModels);
  experimentModel?.addEventListener('change', () => {
    if (experimentSubmit) experimentSubmit.disabled = !experimentModel.value;
  });

  updateProviderFields();
  loadConfiguredModels();
})();
