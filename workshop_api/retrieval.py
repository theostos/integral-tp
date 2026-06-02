from __future__ import annotations

import json
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import requests


DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"


def _normalize_library_name(name: str) -> str:
    key = name.strip().lower()
    aliases = {
        "coq": "Stdlib",
        "coq-stdlib": "Stdlib",
        "corelib": "Stdlib",
        "bignums": "Stdlib",
        "stdlib": "Stdlib",
        "std": "Stdlib",
        "coquelicot": "Coquelicot",
        "mathcomp": "MathComp",
        "math-comp": "MathComp",
    }
    return aliases.get(key, name.strip())


def _normalize_libraries(value: str | Sequence[str] | None) -> set[str] | None:
    if value is None:
        return None
    raw_values = [value] if isinstance(value, str) else list(value)
    out: set[str] = set()
    for item in raw_values:
        for raw in str(item).split(","):
            raw = raw.strip()
            if raw:
                out.add(_normalize_library_name(raw))
    return out or None


def _normalize_kinds(value: str | Sequence[str] | None) -> set[str] | None:
    if value is None:
        return None
    raw_values = [value] if isinstance(value, str) else list(value)
    out: set[str] = set()
    for item in raw_values:
        for raw in str(item).split(","):
            raw = raw.strip().lower()
            if raw:
                out.add(raw)
    return out or None


def _library_matches(item: dict[str, Any], libraries: set[str] | None) -> bool:
    if libraries is None:
        return True
    library = item.get("library")
    if not isinstance(library, str) or not library.strip():
        return False
    return _normalize_library_name(library) in libraries


def _kind_matches(item: dict[str, Any], kinds: set[str] | None) -> bool:
    if kinds is None:
        return True
    kind = item.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        return False
    return kind.strip().lower() in kinds


def _shorten(text: object, *, limit: int) -> str:
    raw = " ".join(str(text or "").split())
    if len(raw) <= limit:
        return raw
    return raw[: max(limit - 3, 0)].rstrip() + "..."


def format_retrieval_hits(
    hits: list[dict[str, Any]],
    *,
    statement_chars: int = 500,
    docstring_chars: int = 260,
) -> str:
    """Render retrieval hits as workshop-friendly plain text."""

    if not hits:
        return "No retrieval hits."

    lines: list[str] = []
    for index, hit in enumerate(hits, start=1):
        name = hit.get("name") or hit.get("uid") or "<unnamed>"
        kind = hit.get("kind")
        library = hit.get("library")
        source = hit.get("source")
        score = hit.get("score")
        heading_parts = [f"{index}. {name}"]
        if kind:
            heading_parts.append(f"({kind})")
        if score is not None:
            try:
                heading_parts.append(f"[score {float(score):.3f}]")
            except (TypeError, ValueError):
                heading_parts.append(f"[score {score}]")
        lines.append(" ".join(heading_parts))
        if library:
            lines.append(f"   library: {_shorten(library, limit=80)}")
        if source:
            lines.append(f"   source: {_shorten(source, limit=140)}")

        statement = hit.get("statement") or hit.get("content")
        if statement:
            lines.append(f"   statement: {_shorten(statement, limit=statement_chars)}")

        docstring = hit.get("docstring")
        if docstring:
            lines.append(f"   docstring: {_shorten(docstring, limit=docstring_chars)}")
    return "\n".join(lines)


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
                "In Colab, install `integral-tp[colab]`, or run: "
                "`pip install -q faiss-cpu 'sentence-transformers>=2.7.0' 'transformers>=4.51.0'`."
            ) from exc

        self._faiss = faiss
        self._np = np
        self.cache_dir = _resolve_cache_dir(Path(self.cache_dir))
        manifest_path = self.cache_dir / "manifest.json"
        metadata_path = self.cache_dir / "metadata.jsonl"
        if not manifest_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Invalid retrieval cache at {self.cache_dir}; expected manifest.json, "
                "metadata.jsonl, and the manifest index file."
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index_path = self.cache_dir / str(self.manifest.get("index_file", "index.faiss"))
        if not index_path.exists():
            raise FileNotFoundError(f"Retrieval cache manifest points to missing index file: {index_path}")
        manifest_model = self.manifest.get("model_name")
        if self.model_name and manifest_model and self.model_name != manifest_model:
            raise ValueError(
                f"Embedding model mismatch: cache manifest uses {manifest_model!r}, "
                f"but query model was set to {self.model_name!r}."
            )
        self.model_name = self.model_name or manifest_model or DEFAULT_EMBEDDING_MODEL
        self.query_prompt_name = self.manifest.get("query_prompt_name")
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

    def search(
        self,
        query: str,
        *,
        k: int = 8,
        library: str | Sequence[str] | None = None,
        kind: str | Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        query = query.strip()
        k = int(k)
        if not query or k <= 0:
            return []
        libraries = _normalize_libraries(library)
        kinds = _normalize_kinds(kind)
        encode_kwargs: dict[str, Any] = {}
        if isinstance(self.query_prompt_name, str) and self.query_prompt_name.strip():
            encode_kwargs["prompt_name"] = self.query_prompt_name
        vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=bool(self.manifest.get("normalize_embeddings", True)),
            show_progress_bar=False,
            **encode_kwargs,
        ).astype("float32")
        search_k = self.index.ntotal if libraries is not None or kinds is not None else max(k * self.top_k_multiplier, k)
        scores, ids = self.index.search(vector, min(int(search_k), int(self.index.ntotal)))
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for score, idx in zip(scores[0], ids[0], strict=False):
            if idx < 0:
                continue
            item = dict(self.metadata[int(idx)])
            if not _library_matches(item, libraries):
                continue
            if not _kind_matches(item, kinds):
                continue
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
    """Local FAISS docstring retriever.

    The client never calls a remote semantic service and has no lexical
    fallback. Configure either `cache_dir` or `cache_url`; otherwise the first
    search raises a clear setup error. Pass `library="Stdlib"` or
    `library="Coquelicot"` to restrict matches to one source library. Pass
    `kind="definition"` or `kind=["definition", "start_theorem_proof"]` to
    restrict matches by Rocq element kind.
    """

    cache_dir: str | Path | None = None
    cache_url: str | None = None
    model_name: str | None = None
    libraries: str | Sequence[str] | None = None
    kinds: str | Sequence[str] | None = None
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
            libraries=os.getenv("DOCSTRING_LIBRARIES") or None,
            kinds=os.getenv("DOCSTRING_KINDS") or None,
        )

    def _local_retriever(self) -> LocalFaissRetriever:
        if self._local is not None:
            return self._local
        cache_dir = self.cache_dir
        if cache_dir is None and self.cache_url:
            cache_dir = download_retrieval_cache(self.cache_url)
        if cache_dir is None:
            raise ValueError(
                "Local retrieval cache is not configured. Set DOCSTRING_CACHE_URL "
                "to a direct retrieval_cache.zip URL, set DOCSTRING_CACHE_DIR to an "
                "unpacked cache directory, or pass cache_url/cache_dir explicitly."
            )
        self._local = LocalFaissRetriever(cache_dir=cache_dir, model_name=self.model_name)
        return self._local

    def search(
        self,
        query: str,
        *,
        k: int = 8,
        library: str | Sequence[str] | None = None,
        kind: str | Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []

        active_library = library if library is not None else self.libraries
        active_kind = kind if kind is not None else self.kinds
        return self._local_retriever().search(query, k=k, library=active_library, kind=active_kind)
