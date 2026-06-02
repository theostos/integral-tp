#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_files(inputs: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for path in inputs:
        path = path.expanduser().resolve()
        if path.is_dir():
            out.extend(sorted(path.rglob("*.toc.json")))
            out.extend(sorted(path.rglob("*.docstrings.json")))
            out.extend(sorted(path.rglob("*.jsonl")))
        elif path.is_file():
            out.append(path)
        else:
            raise FileNotFoundError(path)
    dedup: list[Path] = []
    seen: set[Path] = set()
    for path in out:
        if path not in seen:
            dedup.append(path)
            seen.add(path)
    return dedup


def _load_jsonish(path: Path) -> Any:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(path.read_text(encoding="utf-8"))


def _entry_from_node(node: dict[str, Any], *, source: str) -> dict[str, Any] | None:
    data = node.get("data", {})
    if not isinstance(data, dict):
        data = {}
    docstring = node.get("docstring")
    content = data.get("content") or node.get("content") or node.get("statement")
    name = node.get("name") or data.get("name")
    if not any(isinstance(value, str) and value.strip() for value in (docstring, content, name)):
        return None
    return {
        "uid": data.get("uid") or node.get("uid"),
        "name": name,
        "kind": node.get("kind"),
        "range": node.get("range"),
        "docstring": docstring,
        "content": content,
        "source": source,
    }


def _walk_toc_nodes(nodes: list[Any], *, source: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in nodes:
        if not isinstance(raw, dict):
            continue
        entry = _entry_from_node(raw, source=source)
        if entry is not None:
            out.append(entry)
        members = raw.get("members")
        if isinstance(members, list):
            out.extend(_walk_toc_nodes(members, source=source))
    return out


def _extract_entries(payload: Any, *, source: str) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        docstrings = payload.get("docstrings")
        if isinstance(docstrings, list):
            return [
                dict(item, source=item.get("source") or source)
                for item in docstrings
                if isinstance(item, dict)
            ]
        entries = payload.get("entries")
        if isinstance(entries, list):
            return [
                dict(item, source=item.get("source") or source)
                for item in entries
                if isinstance(item, dict)
            ]
        nodes = payload.get("nodes")
        if isinstance(nodes, list):
            return _walk_toc_nodes(nodes, source=source)
        return []

    if isinstance(payload, list):
        return _walk_toc_nodes(payload, source=source)

    return []


def _text_for_embedding(entry: dict[str, Any]) -> str:
    parts = [
        str(entry.get("name") or ""),
        str(entry.get("kind") or ""),
        str(entry.get("content") or ""),
        str(entry.get("docstring") or ""),
        str(entry.get("source") or ""),
    ]
    return "\n".join(part.strip() for part in parts if part and part.strip())


def _stable_uid(entry: dict[str, Any]) -> str:
    raw = entry.get("uid")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    text = json.dumps(
        {
            "source": entry.get("source"),
            "name": entry.get("name"),
            "kind": entry.get("kind"),
            "content": entry.get("content"),
            "docstring": entry.get("docstring"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def collect_entries(inputs: list[Path], *, docstrings_only: bool = False) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in _json_files(inputs):
        source = path.as_posix()
        try:
            payload = _load_jsonish(path)
        except Exception as exc:
            print(f"skip unreadable JSON {path}: {exc}")
            continue
        entries.extend(_extract_entries(payload, source=source))

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        docstring = entry.get("docstring")
        content = entry.get("content")
        if docstrings_only and not (isinstance(docstring, str) and docstring.strip()):
            continue
        if not any(isinstance(value, str) and value.strip() for value in (docstring, content, entry.get("name"))):
            continue
        uid = _stable_uid(entry)
        if uid in seen:
            continue
        seen.add(uid)
        normalized = dict(entry)
        normalized["uid"] = uid
        normalized["text"] = _text_for_embedding(normalized)
        out.append(normalized)
    return out


def build_cache(
    *,
    entries: list[dict[str, Any]],
    output_dir: Path,
    model_name: str,
    batch_size: int,
    source_roots: list[str],
) -> None:
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError(
            "Cache building requires `faiss-cpu`, `numpy`, and `sentence-transformers`.\n"
            "Install with: pip install faiss-cpu sentence-transformers"
        ) from exc

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    model = SentenceTransformer(model_name)
    texts = [entry["text"] for entry in entries]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    if embeddings.ndim != 2:
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(output_dir / "index.faiss"))

    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for entry in entries:
            stored = {key: value for key, value in entry.items() if key != "text"}
            handle.write(json.dumps(stored, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "version": 1,
        "generated_at": _utc_now(),
        "model_name": model_name,
        "entry_count": len(entries),
        "dimension": dim,
        "metric": "cosine",
        "normalize_embeddings": True,
        "source_roots": source_roots,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def zip_cache(output_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(output_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local FAISS docstring retrieval cache for the Rocq workshop."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Directory or JSON/JSONL file containing *.toc.json / docstring entries. Repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        default="build/retrieval_cache",
        help="Directory where manifest.json, metadata.jsonl and index.faiss are written.",
    )
    parser.add_argument(
        "--zip",
        default="build/retrieval_cache.zip",
        help="Zip archive to publish/download in Colab.",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model used for both index and query embeddings.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="Debug limit; 0 means no limit.")
    parser.add_argument(
        "--docstrings-only",
        action="store_true",
        help="Drop entries that have content but no docstring.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(item) for item in args.input]
    entries = collect_entries(input_paths, docstrings_only=bool(args.docstrings_only))
    if args.limit:
        entries = entries[: args.limit]
    if not entries:
        raise SystemExit("No entries found. Check the --input paths and JSON format.")
    output_dir = Path(args.output_dir).expanduser().resolve()
    zip_path = Path(args.zip).expanduser().resolve()
    build_cache(
        entries=entries,
        output_dir=output_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        source_roots=[str(path.expanduser().resolve()) for path in input_paths],
    )
    zip_cache(output_dir, zip_path)
    print(f"entries: {len(entries)}")
    print(f"cache:   {output_dir}")
    print(f"zip:     {zip_path}")


if __name__ == "__main__":
    main()
