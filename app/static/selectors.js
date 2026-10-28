/* Replace manual UUID inputs with API-backed selectors. */
'use strict';

(() => {
  const experimentForm = document.getElementById('experiment-form');
  const benchmarkForm = document.getElementById('benchmark-form');
  if (!experimentForm || !benchmarkForm) return;

  function replaceWithSelect(form, name, placeholder, required = false) {
    const input = form.querySelector(`[name="${name}"]`);
    if (!input || input.tagName === 'SELECT') return input;
    const select = document.createElement('select');
    select.className = input.className;
    select.name = name;
    select.required = required || input.required;
    select.setAttribute('aria-label', placeholder);
    input.replaceWith(select);
    const label = select.closest('label');
    if (label) {
      const helper = document.createElement('small');
      helper.className = 'selector-help';
      helper.textContent = '';
      helper.dataset.selectorHelp = name;
      label.appendChild(helper);
    }
    return select;
  }

  const experimentDataset = replaceWithSelect(
    experimentForm, 'dataset_id', 'Select a dataset', true
  );
  const experimentBenchmark = replaceWithSelect(
    experimentForm, 'benchmark_id', 'No benchmark (optional)'
  );
  const benchmarkDataset = replaceWithSelect(
    benchmarkForm, 'dataset_id', 'Select a dataset', true
  );

  if (!experimentDataset || !experimentBenchmark || !benchmarkDataset) return;

  function addOption(select, value, label, disabled = false) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    option.disabled = disabled;
    select.appendChild(option);
  }

  function optionLabel(item) {
    const name = item.name || 'Unnamed';
    const id = String(item.id || '');
    // Names are not unique in the API, so show a stable ID prefix as well.
    return id ? `${name} (${id.slice(0, 8)})` : name;
  }

  async function getJson(path) {
    const response = await fetch(`/api${path}`);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || `Request failed (${response.status})`);
    }
    return payload;
  }

  function setHelp(select, message) {
    const help = select.closest('label')?.querySelector('[data-selector-help]');
    if (help) help.textContent = message;
  }

  function fillDatasetSelect(select, datasets, selectedId) {
    select.replaceChildren();
    addOption(select, '', datasets.length ? 'Choose a dataset…' : 'No datasets available', true);
    datasets.forEach(dataset => {
      addOption(select, dataset.id, optionLabel(dataset));
    });
    select.disabled = datasets.length === 0;
    select.value = datasets.some(dataset => dataset.id === selectedId) ? selectedId : '';
    setHelp(select, datasets.length ? '' : 'Create a dataset first, then reopen this form.');
  }

  function fillBenchmarkSelect(select, benchmarks, datasetId, selectedId) {
    const matching = benchmarks.filter(item => item.dataset_id === datasetId);
    select.replaceChildren();
    addOption(
      select,
      '',
      matching.length ? 'No benchmark (optional)' : 'No benchmark for this dataset'
    );
    matching.forEach(benchmark => addOption(select, benchmark.id, optionLabel(benchmark)));
    select.disabled = false;
    select.value = matching.some(item => item.id === selectedId) ? selectedId : '';
    setHelp(select, datasetId && !matching.length
      ? 'You can save without a benchmark, or create one for this dataset.'
      : '');
  }

  async function refreshSelectors() {
    const previousExperimentDataset = experimentDataset.value;
    const previousExperimentBenchmark = experimentBenchmark.value;
    const previousBenchmarkDataset = benchmarkDataset.value;
    experimentDataset.disabled = true;
    benchmarkDataset.disabled = true;

    try {
      const [datasets, benchmarks] = await Promise.all([
        getJson('/datasets'),
        getJson('/benchmarks'),
      ]);
      fillDatasetSelect(experimentDataset, datasets, previousExperimentDataset);
      fillDatasetSelect(benchmarkDataset, datasets, previousBenchmarkDataset);
      fillBenchmarkSelect(
        experimentBenchmark, benchmarks, experimentDataset.value, previousExperimentBenchmark
      );
      const experimentSubmit = experimentForm.querySelector('[type="submit"]');
      const benchmarkSubmit = benchmarkForm.querySelector('[type="submit"]');
      if (experimentSubmit) experimentSubmit.disabled = datasets.length === 0;
      if (benchmarkSubmit) benchmarkSubmit.disabled = datasets.length === 0;
    } catch (error) {
      [experimentDataset, benchmarkDataset].forEach(select => {
        select.replaceChildren();
        addOption(select, '', 'Could not load datasets', true);
        select.disabled = true;
        setHelp(select, `Refresh the page or try again. ${error.message}`);
      });
      experimentBenchmark.replaceChildren();
      addOption(experimentBenchmark, '', 'Could not load benchmarks', true);
      experimentBenchmark.disabled = true;
      [experimentForm, benchmarkForm].forEach(form => {
        const submit = form.querySelector('[type="submit"]');
        if (submit) submit.disabled = true;
      });
    }
  }

  async function refreshBenchmarksForDataset() {
    try {
      const benchmarks = await getJson('/benchmarks');
      fillBenchmarkSelect(experimentBenchmark, benchmarks, experimentDataset.value, '');
    } catch (error) {
      experimentBenchmark.replaceChildren();
      addOption(experimentBenchmark, '', 'Could not load benchmarks', true);
      setHelp(experimentBenchmark, error.message);
    }
  }

  experimentDataset.addEventListener('change', refreshBenchmarksForDataset);
  document.getElementById('btn-new-experiment')?.addEventListener('click', refreshSelectors);
  document.getElementById('btn-new-benchmark')?.addEventListener('click', refreshSelectors);

  // Refresh choices when datasets or benchmarks are created/deleted in their lists.
  ['datasets-list', 'benchmarks-list'].forEach(id => {
    const target = document.getElementById(id);
    if (!target || !window.MutationObserver) return;
    new MutationObserver(() => refreshSelectors()).observe(target, { childList: true, subtree: true });
  });

  refreshSelectors();
})();
