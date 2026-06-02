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

Open `integral_workshop.ipynb` in Colab from this repository. The setup shell
cells:

- installs `integral-tp[colab]` from GitHub using this repo's `pyproject.toml`;
- pulls `theostos/coq-coquelicot:8.20-3.4.4`;
- starts `rocq-ml-server` from that Docker image with host networking;
- then all Rocq interaction goes through `http://127.0.0.1:5000`.

The install spec used by default is:

```bash
integral-tp[colab] @ git+https://github.com/theostos/integral-tp.git
```

When running it manually in Colab, quote the PEP 508 requirement:

```python
%pip install -q "integral-tp[colab] @ git+https://github.com/theostos/integral-tp.git"
```

The `colab` extra depends on `rocq-ml-toolbox[client]` from the public
`llm4rocq/rocq-ml-toolbox` repository, plus public `pytanque` and retrieval
packages. The Rocq server stack still comes from the Docker image.

For a branch, fork, or local development variant, set
`INTEGRAL_TP_INSTALL_SPEC` before the import/setup cell. Set
`ROCQ_DOCKER_IMAGE` to override the Docker image.

The Docker shell cell used by the notebook is the detached equivalent of the
interactive terminal command:

```bash
docker run -it --network host theostos/coq-coquelicot:8.20-3.4.4 rocq-ml-server
```

In notebook form:

```bash
docker run -d --network host \
  --name integral-tp-rocq-server \
  theostos/coq-coquelicot:8.20-3.4.4 \
  rocq-ml-server --host 0.0.0.0 --port 5000 \
    --num-pet-server 2 --workers 4 --timeout 600
```

If you run outside Colab, either use the same Docker command or start
`rocq-ml-server` locally on port 5000.

## Retrieval Cache

Build the retrieval cache once on a machine that has access to the generated
Rocq/Coq docstring JSON files:

```bash
pip install faiss-cpu sentence-transformers
python scripts/build_retrieval_cache.py \
  --input /path/to/stdlib-and-coquelicot-docstrings \
  --output-dir build/retrieval_cache \
  --zip build/retrieval_cache.zip
```

Upload `build/retrieval_cache.zip` to a public host. In Colab, point the
notebook to the direct zip URL before the retrieval cell:

```python
%env DOCSTRING_CACHE_URL=https://example.org/retrieval_cache.zip
```

The notebook downloads the zip, loads `index.faiss` + `metadata.jsonl`, and
only computes embeddings for user queries. If the cache is already unpacked in
the runtime, set `DOCSTRING_CACHE_DIR` instead.
