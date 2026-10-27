<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>RAG Optimization Studio</title>
    <link rel="stylesheet" href="/static/styles.css" />
  </head>
  <body>
    <div class="app-shell">
      <div class="topbar">
        <div>
          <div class="badge">Single-server MVP</div>
        </div>
      </div>

      <section class="hero">
        <h1>RAG Optimization Studio</h1>
        <p>
          A benchmarking platform for comparing chunking, retrieval, reranking, and answer
          generation strategies in a fair, reproducible way.
        </p>
      </section>

      <section id="overview" class="grid">
        <div id="metric-cards"></div>
      </section>

      <div class="layout">
        <section class="card">
          <h2>Experiment matrix</h2>
          <p>Start with a controlled set of retrieval experiments before expanding into ablations.</p>
          <ul id="focus-areas"></ul>
          <div id="experiment-table" style="margin-top: 18px;"></div>
        </section>

        <section class="card">
          <h2>Create experiment</h2>
          <form id="experiment-form">
            <label>
              Experiment name
              <input name="name" placeholder="Hybrid rerank benchmark" required />
            </label>

            <label>
              Dataset name
              <input name="dataset_name" value="technical-docs" />
            </label>

            <label>
              Chunking strategy
              <select name="chunking_strategy">
                <option value="fixed-size">Fixed-size</option>
                <option value="sentence-aware">Sentence-aware</option>
                <option value="recursive">Recursive</option>
              </select>
            </label>

            <label>
              Chunk size
              <input name="chunk_size" type="number" value="500" min="100" max="2000" />
            </label>

            <label>
              Overlap
              <input name="overlap" type="number" value="50" min="0" max="500" />
            </label>

            <label>
              Retrieval method
              <select name="retrieval_method">
                <option value="dense">Dense</option>
                <option value="bm25">BM25</option>
                <option value="hybrid_rrf" selected>Hybrid RRF</option>
              </select>
            </label>

            <label>
              Top-k
              <input name="top_k" type="number" value="5" min="1" max="50" />
            </label>

            <label>
              Embedding model
              <input name="embedding_model" value="sentence-transformers/all-MiniLM-L6-v2" />
            </label>

            <label>
              Generation model
              <input name="generation_model" value="gpt-4o-mini" />
            </label>

            <label>
              <input name="reranking" type="checkbox" checked /> Enable reranking
            </label>

            <label>
              <input name="query_rewrite" type="checkbox" /> Query rewriting
            </label>

            <label>
              Notes
              <textarea name="notes" placeholder="Example: compare hybrid retrieval against dense baseline with reranking enabled."></textarea>
            </label>

            <button type="submit">Save experiment</button>
            <div id="status-message" class="empty"></div>
          </form>
        </section>
      </div>
    </div>

    <script src="/static/app.js"></script>
  </body>
</html>
