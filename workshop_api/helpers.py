from __future__ import annotations

from typing import Any

from .llm import LLMUsage


def expected_hit(
    retriever: Any,
    name: str,
    *,
    library: str | None = None,
    kind: str | None = None,
    query: str | None = None,
    k: int = 12,
) -> dict[str, Any]:
    """Return a retrieval hit by exact name, preferring cache metadata."""

    def same(value: object, expected: str | None) -> bool:
        return expected is None or str(value).lower() == str(expected).lower()

    local_retriever = retriever._local_retriever()
    for item in local_retriever.metadata:
        if (
            item.get("name") == name
            and same(item.get("library"), library)
            and same(item.get("kind"), kind)
        ):
            hit = dict(item)
            hit.setdefault("score", 1.0)
            return hit

    hits = retriever.search(query or name, library=library, kind=kind, k=k)
    for hit in hits:
        if hit.get("name") == name:
            return hit
    names = [hit.get("name") for hit in hits]
    raise ValueError(f"Could not find retrieval hit {name!r}. Closest hits: {names}")


def set_hits(target: list[dict[str, Any]], *hits: Any) -> list[dict[str, Any]]:
    """Replace a selected-hit list and print the loaded names."""

    target.clear()
    for hit in hits:
        if hasattr(hit, "as_retrieval_hit"):
            hit = hit.as_retrieval_hit()
        target.append(hit)
    print([hit.get("name") for hit in target])
    return target


def show_usage(label: str, results: Any, *, show_cache: bool = False) -> LLMUsage:
    """Print total LLM usage for one strategy or proof."""

    if not isinstance(results, (list, tuple)):
        results = [results]
    usages = [
        result.usage
        for result in results
        if getattr(result, "usage", None) is not None
    ]
    usage = LLMUsage.aggregate(usages)
    print(f"{label}: {len(results)} request(s), {usage.summary(show_cache=show_cache)}")
    return usage
