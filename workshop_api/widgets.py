from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Any, Sequence

from .retrieval import format_retrieval_hits


def _hit_key(hit: dict[str, Any]) -> str:
    return str(hit.get("uid") or (hit.get("source"), hit.get("name"), hit.get("content")))


def _shorten(value: object, *, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def _highlight_rocq(text: object) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    try:
        from pygments import highlight
        from pygments.formatters import HtmlFormatter
        from pygments.lexers import get_lexer_by_name

        formatter = HtmlFormatter(nowrap=False)
        return highlight(raw, get_lexer_by_name("coq"), formatter)
    except Exception:
        return f"<pre>{html.escape(raw)}</pre>"


def _hit_html(hit: dict[str, Any]) -> str:
    name = html.escape(str(hit.get("name") or hit.get("uid") or "<unnamed>"))
    kind = html.escape(str(hit.get("kind") or "?"))
    library = html.escape(str(hit.get("library") or "?"))
    source = html.escape(_shorten(hit.get("source"), limit=120))
    score = hit.get("score")
    score_text = ""
    if score is not None:
        try:
            score_text = f" score={float(score):.3f}"
        except (TypeError, ValueError):
            score_text = f" score={html.escape(str(score))}"

    statement = hit.get("statement") or hit.get("content")
    docstring = hit.get("docstring")
    doc_html = ""
    if docstring:
        doc_html = (
            "<div class='rocq-docstring'>"
            f"{html.escape(_shorten(docstring, limit=420))}"
            "</div>"
        )

    return (
        "<div class='rocq-hit'>"
        f"<div class='rocq-hit-head'><b>{name}</b> "
        f"<span>{kind}</span> <span>{library}</span><span>{score_text}</span></div>"
        f"<div class='rocq-source'>{source}</div>"
        f"{_highlight_rocq(statement)}"
        f"{doc_html}"
        "</div>"
    )


def _style_html() -> str:
    try:
        from pygments.formatters import HtmlFormatter

        pygments_css = HtmlFormatter().get_style_defs(".rocq-hit .highlight")
    except Exception:
        pygments_css = ""
    return f"""
<style>
.rocq-retrieval {{
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
.rocq-hit {{
  border: 1px solid #d0d7de;
  border-radius: 6px;
  padding: 8px 10px;
  margin: 0 0 8px 0;
  background: #ffffff;
}}
.rocq-hit-head {{
  display: flex;
  gap: 8px;
  align-items: baseline;
  flex-wrap: wrap;
  margin-bottom: 3px;
}}
.rocq-hit-head span,
.rocq-source {{
  color: #57606a;
  font-size: 12px;
}}
.rocq-source {{
  margin-bottom: 6px;
}}
.rocq-docstring {{
  color: #24292f;
  background: #f6f8fa;
  border-radius: 4px;
  padding: 6px 8px;
  margin-top: 6px;
  font-size: 12px;
}}
.rocq-hit pre,
.rocq-hit .highlight {{
  margin: 0;
  overflow-x: auto;
  background: #f6f8fa;
  border-radius: 4px;
  padding: 8px;
  font-size: 12px;
  line-height: 1.4;
}}
{pygments_css}
</style>
"""


@dataclass
class RetrievalExplorer:
    """Small ipywidgets UI for searching and curating retrieval hits."""

    retriever: Any
    selected_hits: list[dict[str, Any]] = field(default_factory=list)
    default_query: str = ""
    default_library: str | None = None
    default_kind: str = ""
    default_k: int = 8
    _ui: Any = field(default=None, init=False, repr=False)
    _results_box: Any = field(default=None, init=False, repr=False)
    _selected_box: Any = field(default=None, init=False, repr=False)
    _context_box: Any = field(default=None, init=False, repr=False)
    _selected_label: Any = field(default=None, init=False, repr=False)

    @property
    def hits(self) -> list[dict[str, Any]]:
        return self.selected_hits

    def context(self, *, statement_chars: int = 1000, docstring_chars: int = 500) -> str:
        return format_retrieval_hits(
            self.selected_hits,
            statement_chars=statement_chars,
            docstring_chars=docstring_chars,
        )

    def clear(self) -> None:
        self.selected_hits.clear()
        self._render_selected()

    def add(self, hit: dict[str, Any]) -> None:
        key = _hit_key(hit)
        if not any(_hit_key(old) == key for old in self.selected_hits):
            self.selected_hits.append(hit)
        self._render_selected()

    def remove(self, key: str) -> None:
        self.selected_hits[:] = [hit for hit in self.selected_hits if _hit_key(hit) != key]
        self._render_selected()

    def render(self) -> Any:
        if self._ui is not None:
            return self._ui

        try:
            import ipywidgets as widgets
        except Exception as exc:
            raise RuntimeError(
                "RetrievalExplorer requires ipywidgets. In Colab, install "
                "`integral-tp[colab]` and rerun the runtime."
            ) from exc

        try:
            from google.colab import output

            output.enable_custom_widget_manager()
        except Exception:
            pass

        query = widgets.Textarea(
            value=self.default_query,
            placeholder="Search Stdlib / Coquelicot...",
            layout=widgets.Layout(width="100%", height="76px"),
        )
        library = widgets.Dropdown(
            options=[
                ("All libraries", None),
                ("Stdlib", "Stdlib"),
                ("Coquelicot", "Coquelicot"),
            ],
            value=self.default_library,
            description="Library",
            layout=widgets.Layout(width="230px"),
        )
        kind = widgets.Text(
            value=self.default_kind,
            placeholder="definition,start_theorem_proof,ltac",
            description="Kind",
            layout=widgets.Layout(width="360px"),
        )
        k = widgets.IntSlider(
            value=int(self.default_k),
            min=1,
            max=20,
            step=1,
            description="Hits",
            continuous_update=False,
            layout=widgets.Layout(width="280px"),
        )
        search_button = widgets.Button(description="Search", icon="search")
        clear_button = widgets.Button(description="Clear selection", icon="trash")
        self._selected_label = widgets.HTML()
        self._results_box = widgets.VBox()
        self._selected_box = widgets.VBox()
        self._context_box = widgets.Textarea(
            description="Context",
            layout=widgets.Layout(width="100%", height="180px"),
        )

        def run_search(_: object = None) -> None:
            hits = self.retriever.search(
                query.value,
                library=library.value,
                kind=kind.value.strip() or None,
                k=k.value,
            )
            self._render_results(hits)

        search_button.on_click(run_search)
        clear_button.on_click(lambda _: self.clear())

        self._ui = widgets.VBox(
            [
                widgets.HTML(_style_html()),
                query,
                widgets.HBox([library, kind, k, search_button]),
                widgets.HTML("<b>Search results</b>"),
                self._results_box,
                widgets.HBox([widgets.HTML("<b>Selected context</b>"), clear_button]),
                self._selected_label,
                self._selected_box,
                self._context_box,
            ]
        )
        self._render_selected()
        return self._ui

    def display(self) -> None:
        try:
            from IPython.display import display
        except Exception as exc:
            raise RuntimeError("RetrievalExplorer.display() requires IPython.") from exc
        display(self.render())

    def _render_results(self, hits: Sequence[dict[str, Any]]) -> None:
        if self._results_box is None:
            return
        import ipywidgets as widgets

        rows: list[Any] = []
        for hit in hits:
            button = widgets.Button(
                description="Add",
                icon="plus",
                layout=widgets.Layout(width="80px"),
            )
            def add_current(_: object, current: dict[str, Any] = hit) -> None:
                self.add(current)

            button.on_click(add_current)
            rows.append(
                widgets.HBox(
                    [button, widgets.HTML(_hit_html(dict(hit)))],
                    layout=widgets.Layout(align_items="flex-start"),
                )
            )
        self._results_box.children = tuple(rows) if rows else (widgets.HTML("<em>No hits.</em>"),)

    def _render_selected(self) -> None:
        if self._selected_box is None or self._context_box is None or self._selected_label is None:
            return
        import ipywidgets as widgets

        self._selected_label.value = f"{len(self.selected_hits)} selected item(s)"
        self._context_box.value = self.context()
        rows: list[Any] = []
        for hit in self.selected_hits:
            key = _hit_key(hit)
            button = widgets.Button(
                description="Remove",
                icon="minus",
                layout=widgets.Layout(width="100px"),
            )

            def remove_current(_: object, current_key: str = key) -> None:
                self.remove(current_key)

            button.on_click(remove_current)
            rows.append(
                widgets.HBox(
                    [button, widgets.HTML(_hit_html(dict(hit)))],
                    layout=widgets.Layout(align_items="flex-start"),
                )
            )
        self._selected_box.children = tuple(rows) if rows else (widgets.HTML("<em>No selected items.</em>"),)
