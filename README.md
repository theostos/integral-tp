# integral-tp

Hands-on Rocq workshop material for certified numerical analysis.

## Files

- `source.v`: compact reference development.
- `workshop_api/`: small Python API over `rocq-ml-server` for adding Rocq elements, opening lemmas, running tactics, local retrieval hooks, and optional LLM calls.
- `integral_workshop.ipynb`: 90-minute notebook decomposing `source.v` step by step.
- `scripts/build_retrieval_cache.py`: offline builder for the FAISS docstring retrieval cache.
- `pyproject.toml`: package metadata and Colab extras used by the notebook setup.
- `requirements-colab.txt`: one-line remote install spec for Colab.

## Colab Run

The intended workshop notebook runs in Google Colab. In this setup, "local"
retrieval means local to the Colab runtime: the precomputed FAISS cache is
downloaded into `/content` and queried there.

Open `integral_workshop.ipynb` in Colab from this repository. The setup cells:

- install `integral-tp[colab]` from GitHub with `%pip`;
- connect to a remote `rocq-ml-server` endpoint;
- download the FAISS retrieval cache into `/content`;
- then all Rocq interaction goes through that remote endpoint, while
  retrieval queries run inside the Colab Python process.

The install spec used by default is:

```bash
integral-tp[colab] @ git+https://github.com/theostos/integral-tp.git
```

The notebook uses this Colab cell:

```python
%pip install -q integral-tp[colab]@git+https://github.com/theostos/integral-tp.git
```

The `colab` extra depends on `rocq-ml-toolbox[client]` from the public
`llm4rocq/rocq-ml-toolbox` repository, plus public `pytanque` and retrieval
packages. The Rocq server stack still comes from the Docker image.

For a branch, fork, or local development variant, edit the `%pip install` line
in the notebook.

Run the Rocq server once on a remote machine with Docker, then point all Colab
notebooks at that host:

```bash
docker run -d -p 5000:5000 \
  theostos/coq-coquelicot:8.20-3.4.4 \
  rocq-ml-server --host 0.0.0.0 --port 5000 \
    --num-pet-server 16 --workers 16 --timeout 600
```

In the notebook, configure:

```python
ROCQ_SERVER_HOST = "rocq-workshop.example.org"
ROCQ_SERVER_PORT = 5000
```

For a larger group, run several server instances on different ports or
machines and assign participants to endpoints.

## Retrieval Cache

Build the retrieval cache once from the Hugging Face dataset
`theostos/pile-of-rocq`, config `coq-coquelicot-toc_nodes`:

```bash
pip install datasets faiss-cpu "sentence-transformers>=2.7.0" "transformers>=4.51.0"
python scripts/build_retrieval_cache.py \
  --hf-dataset theostos/pile-of-rocq \
  --hf-config coq-coquelicot-toc_nodes \
  --library Stdlib \
  --library Coquelicot \
  --output-dir build/retrieval_cache \
  --zip build/retrieval_cache.zip \
  --batch-size 4 \
  --force
```

Those Hugging Face dataset/config/library values are the script defaults, so
the explicit form above can be shortened to just the output paths. The builder
classifies `Corelib/...` and `Bignums/...` rows as `Stdlib`, and `Coquelicot/...`
rows as `Coquelicot`.

The default embedding model is `Qwen/Qwen3-Embedding-4B`. You can override it
with `--model`, but the query-side Colab runtime must use the same model as the
one recorded in `manifest.json`. For Qwen3 embedding caches, the builder records
`query_prompt_name = "query"` in the manifest, and the local retriever uses that
prompt when embedding user queries.

The builder writes both `embeddings.npy` and `index.faiss`. The `.npy` file is a
portable float32 matrix in the same order as `metadata.jsonl`, so you can rebuild
another FAISS index strategy later without recomputing embeddings. The default
index is exact `Flat` / inner product over normalized vectors, which is a good
fit for this cache size. Use `--index-factory` only when you want to experiment
with approximate FAISS indexes after the workshop path is stable.

For a rented GPU run, use `--resume` if the embedding process is interrupted.
The script records progress in `embedding_progress.json` and refuses to resume
if the model, entry order, or cache parameters do not match.

Upload `build/retrieval_cache.zip` to a public host. In Colab, point the
notebook to the direct zip URL before the retrieval cell:

```python
DOCSTRING_CACHE_URL = "https://example.org/retrieval_cache.zip"
```

The notebook downloads the zip, loads `index.faiss` + `metadata.jsonl`, and
only computes embeddings for user queries. If the cache is already unpacked in
the runtime, set `DOCSTRING_CACHE_DIR` instead. Retrieval is local-only: there
is no remote semantic service and no lexical fallback.

Minimal retrieval use in the notebook:

```python
from workshop_api import (
    RetrievalClient,
    format_retrieval_hits,
    prepare_colab_retrieval_cache,
)

cache_path = prepare_colab_retrieval_cache(DOCSTRING_CACHE_URL)
retriever = RetrievalClient.from_env(cache_dir=cache_path)
hits = retriever.search(
    "RInt fundamental theorem derivative continuous interval integral",
    library="Coquelicot",
    kind="start_theorem_proof",
    k=5,
)
print(format_retrieval_hits(hits))

hits = retriever.search(
    "field_simplify ring exponential positivity exp_plus denominator nonzero",
    library="Stdlib",
    kind=["ltac", "start_theorem_proof"],
    k=5,
)
print(format_retrieval_hits(hits))
```
