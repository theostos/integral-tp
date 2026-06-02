from __future__ import annotations

import json
import os
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests


FALLBACK_DOC_HINTS: list[dict[str, str]] = [
    {
        "name": "is_RInt_derive",
        "library": "Coquelicot",
        "statement": "If F has derivative f on an interval and f is continuous, then RInt f a b = F b - F a.",
        "docstring": "Fundamental theorem of calculus for Coquelicot integrals.",
    },
    {
        "name": "is_RInt_unique",
        "library": "Coquelicot",
        "statement": "Turn an is_RInt witness into equality with RInt.",
        "docstring": "Uniqueness principle for the Coquelicot RInt operator.",
    },
    {
        "name": "is_derive_plus",
        "library": "Coquelicot",
        "statement": "Derivative rule for a sum of two functions.",
        "docstring": "Use when proving that the derivative of F + G is f + g.",
    },
    {
        "name": "ex_derive_continuous",
        "library": "Coquelicot",
        "statement": "Differentiability implies continuity.",
        "docstring": "Useful bridge from ex_derive goals to continuous goals.",
    },
    {
        "name": "is_derive_unique",
        "library": "Coquelicot",
        "statement": "The derivative computed by Derive is equal to a known is_derive value.",
        "docstring": "Often used after auto_derive introduces Derive terms.",
    },
    {
        "name": "auto_derive",
        "library": "Coquelicot",
        "statement": "Tactic for symbolic differentiation of real expressions.",
        "docstring": "Produces side conditions such as non-zero denominators.",
    },
    {
        "name": "field_simplify",
        "library": "Stdlib",
        "statement": "Simplify rational field expressions under non-zero side conditions.",
        "docstring": "Useful after unfolding definitions built from divisions.",
    },
    {
        "name": "ring",
        "library": "Stdlib",
        "statement": "Solve polynomial/ring equalities.",
        "docstring": "Useful for algebraic cleanup after rewriting.",
    },
    {
        "name": "exp_plus",
        "library": "Stdlib",
        "statement": "exp (x + y) = exp x * exp y.",
        "docstring": "Needed to relate exp (2*u) and exp u squared.",
    },
    {
        "name": "exp_pos",
        "library": "Stdlib",
        "statement": "0 < exp x.",
        "docstring": "Positivity fact for exponential denominators.",
    },
]


def _default_cache_parent() -> Path:
    if _running_in_colab():
        return Path("/content/.cache/integral-tp")
    return Path(os.getenv("WORKSHOP_CACHE_DIR", "~/.cache/integral-tp")).expanduser()


def _running_in_colab() -> bool:
    return "google.colab" in sys.modules or bool(os.getenv("COLAB_RELEASE_TAG"))


def _resolve_cache_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "manifest.json").exists():
        return path
    children = [child for child in path.iterdir() if child.is_dir()] if path.exists() else []
    matches = [child for child in children if (child / "manifest.json").exists()]
    if len(matches) == 1:
        return matches[0].resolve()
    return path


def download_retrieval_cache(
    cache_url: str,
    *,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Download and unpack a retrieval cache zip.

    `cache_url` can be a direct HTTP(S) zip URL, a local zip path, or a local
    directory. The returned directory contains `manifest.json`, `metadata.jsonl`,
    and `index.faiss`.
    """

    if not cache_url:
        raise ValueError("cache_url is empty.")

    raw = cache_url.strip()
    if raw.startswith("file://"):
        raw = raw[len("file://") :]

    candidate = Path(raw).expanduser()
    if candidate.exists() and candidate.is_dir():
        return _resolve_cache_dir(candidate)

    target = Path(cache_dir).expanduser() if cache_dir is not None else _default_cache_parent() / "retrieval"
    if target.exists() and force:
        shutil.rmtree(target)
    resolved = _resolve_cache_dir(target) if target.exists() else target
    if (resolved / "manifest.json").exists() and not force:
        return resolved

    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / "retrieval_cache.zip"

    if candidate.exists() and candidate.is_file():
        shutil.copyfile(candidate, zip_path)
    else:
        with requests.get(raw, stream=True, timeout=120) as response:
            response.raise_for_status()
            with zip_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)

    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(target)
    except zipfile.BadZipFile as exc:
        raise ValueError(
            f"Downloaded file is not a valid zip: {zip_path}. "
            "Use a direct download URL for retrieval_cache.zip; public preview pages "
            "from cloud drives usually download HTML instead of the archive."
        ) from exc
    zip_path.unlink(missing_ok=True)
    resolved = _resolve_cache_dir(target)
    if not (resolved / "manifest.json").exists():
        raise FileNotFoundError(
            f"Downloaded cache did not contain manifest.json at {target}. "
            "The zip should contain manifest.json, metadata.jsonl, and index.faiss."
        )
    return resolved


def prepare_colab_retrieval_cache(
    cache_url: str | None = None,
    *,
    cache_dir: str | Path = "/content/rocq-doc-cache",
    force: bool = False,
) -> Path:
    """Prepare the retrieval cache inside the Colab runtime.

    In the workshop notebook, the FAISS index is meant to be local to Colab:
    the cache zip is downloaded once into `/content`, unpacked, and queried
    without recomputing library embeddings.
    """

    explicit_dir = os.getenv("DOCSTRING_CACHE_DIR")
    if explicit_dir and not cache_url:
        return _resolve_cache_dir(Path(explicit_dir))

    url = cache_url or os.getenv("DOCSTRING_CACHE_URL")
    if not url:
        raise ValueError(
            "No retrieval cache URL configured. Set DOCSTRING_CACHE_URL to the "
            "public zip URL, or set DOCSTRING_CACHE_DIR to an already-unpacked cache."
        )
    cache_path = download_retrieval_cache(url, cache_dir=cache_dir, force=force)
    os.environ["DOCSTRING_CACHE_DIR"] = str(cache_path)
    return cache_path


@dataclass
class LocalFaissRetriever:
    cache_dir: str | Path
    model_name: str | None = None
    top_k_multiplier: int = 4

    def __post_init__(self) -> None:
        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - dependency error path.
            raise RuntimeError(
                "Local retrieval requires `faiss-cpu`, `numpy`, and `sentence-transformers`. "
                "In Colab, run: `pip install -q faiss-cpu sentence-transformers`."
            ) from exc

        self._faiss = faiss
        self._np = np
        self.cache_dir = _resolve_cache_dir(Path(self.cache_dir))
        manifest_path = self.cache_dir / "manifest.json"
        metadata_path = self.cache_dir / "metadata.jsonl"
        index_path = self.cache_dir / "index.faiss"
        if not manifest_path.exists() or not metadata_path.exists() or not index_path.exists():
            raise FileNotFoundError(
                f"Invalid retrieval cache at {self.cache_dir}; expected manifest.json, "
                "metadata.jsonl, and index.faiss."
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.model_name = self.model_name or self.manifest.get("model_name")
        if not self.model_name:
            raise ValueError("No embedding model name found in manifest or constructor.")
        self.metadata = [
            json.loads(line)
            for line in metadata_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.index = faiss.read_index(str(index_path))
        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"Index/metadata mismatch: index has {self.index.ntotal}, "
                f"metadata has {len(self.metadata)}."
            )
        self.model = SentenceTransformer(self.model_name)

    def search(self, query: str, *, k: int = 8) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []
        vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=bool(self.manifest.get("normalize_embeddings", True)),
            show_progress_bar=False,
        ).astype("float32")
        scores, ids = self.index.search(vector, max(int(k) * self.top_k_multiplier, int(k)))
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for score, idx in zip(scores[0], ids[0], strict=False):
            if idx < 0:
                continue
            item = dict(self.metadata[int(idx)])
            key = str(item.get("uid") or (item.get("source"), item.get("name"), item.get("content")))
            if key in seen:
                continue
            seen.add(key)
            item["score"] = float(score)
            out.append(item)
            if len(out) >= k:
                break
        return out


@dataclass
class RetrievalClient:
    """Docstring retrieval hook.

    Resolution order:
    1. local FAISS cache (`cache_dir` or downloadable `cache_url`);
    2. remote semantic search service (`base_url`);
    3. tiny lexical fallback so the notebook remains runnable.
    """

    base_url: str | None = None
    env: str = "stdlib-coquelicot"
    route: str = "/search"
    api_key: str | None = None
    timeout: float = 30.0
    cache_dir: str | Path | None = None
    cache_url: str | None = None
    model_name: str | None = None
    fallback_entries: list[dict[str, str]] = field(default_factory=lambda: list(FALLBACK_DOC_HINTS))
    _local: LocalFaissRetriever | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_env(
        cls,
        *,
        cache_url: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> "RetrievalClient":
        return cls(
            cache_url=cache_url or os.getenv("DOCSTRING_CACHE_URL") or None,
            cache_dir=cache_dir or os.getenv("DOCSTRING_CACHE_DIR") or None,
            model_name=os.getenv("DOCSTRING_EMBEDDING_MODEL") or None,
            base_url=os.getenv("DOCSTRING_SEARCH_BASE_URL") or None,
            route=os.getenv("DOCSTRING_SEARCH_ROUTE", "/search"),
            api_key=os.getenv("DOCSTRING_SEARCH_API_KEY") or None,
            env=os.getenv("DOCSTRING_SEARCH_ENV", "stdlib-coquelicot"),
        )

    def _local_retriever(self) -> LocalFaissRetriever | None:
        if self._local is not None:
            return self._local
        cache_dir = self.cache_dir
        if cache_dir is None and self.cache_url:
            cache_dir = download_retrieval_cache(self.cache_url)
        if cache_dir is None:
            return None
        self._local = LocalFaissRetriever(cache_dir=cache_dir, model_name=self.model_name)
        return self._local

    def search(self, query: str, *, k: int = 8) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []

        local = self._local_retriever()
        if local is not None:
            return local.search(query, k=k)

        if self.base_url:
            url = self.base_url.rstrip("/") + "/" + self.route.lstrip("/")
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            response = requests.post(
                url,
                json={"query": query, "env": self.env, "k": int(k)},
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results", payload) if isinstance(payload, dict) else payload
            if not isinstance(results, list):
                raise ValueError("Retrieval response must be a list or contain a `results` list.")
            return [item for item in results if isinstance(item, dict)][:k]

        tokens = {tok for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_']*", query.lower()) if len(tok) > 2}
        scored: list[tuple[int, dict[str, str]]] = []
        for entry in self.fallback_entries:
            haystack = " ".join(str(value).lower() for value in entry.values())
            score = sum(1 for tok in tokens if tok in haystack)
            if score:
                scored.append((score, entry))
        scored.sort(key=lambda item: (-item[0], item[1]["name"]))
        return [dict(entry, score=score) for score, entry in scored[:k]]
