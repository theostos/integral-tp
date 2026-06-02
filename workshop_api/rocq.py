from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from rocq_ml_toolbox.inference.client import PytanqueExtended
except Exception as exc:  # pragma: no cover - exercised in notebooks.
    PytanqueExtended = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


LEMMA_HEADER_RE = re.compile(
    r"^\s*(Lemma|Theorem|Fact|Proposition|Corollary)\s+([A-Za-z_][A-Za-z0-9_']*)\b",
    re.MULTILINE,
)


def _clean_command(command: str) -> str:
    out = textwrap.dedent(command).strip()
    if not out:
        raise ValueError("Empty Rocq command.")
    return out


def _goal_text(goal: Any) -> str:
    return str(getattr(goal, "pp", None) or getattr(goal, "ty", None) or goal)


def _feedback(state: Any) -> list[str]:
    raw = getattr(state, "feedback", None)
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            out.append(str(item[1]))
        elif isinstance(item, dict):
            out.append(str(item.get("message", item)))
        else:
            out.append(str(item))
    return out


@dataclass
class StateNode:
    index: int
    parent_index: int | None
    command: str | None
    state: Any


@dataclass
class LemmaSession:
    name: str
    header: str
    nodes: list[StateNode]
    latest_index: int = 0
    completed: bool = False
    qed_state: Any | None = None

    def node(self, state_index: int) -> StateNode:
        if state_index < 0 or state_index >= len(self.nodes):
            raise ValueError(
                f"Unknown state_index={state_index}; "
                f"available={[node.index for node in self.nodes]}"
            )
        return self.nodes[state_index]

    def active_commands(self, state_index: int | None = None) -> list[str]:
        idx = self.latest_index if state_index is None else state_index
        commands: list[str] = []
        while idx is not None and idx > 0:
            node = self.node(idx)
            if node.command:
                commands.append(node.command)
            idx = node.parent_index
        commands.reverse()
        return commands

    def source(self, *, state_index: int | None = None, close: bool | None = None) -> str:
        if close is None:
            close = self.completed
        commands = self.active_commands(state_index)
        proof = "\n".join(commands).strip()
        if proof:
            proof = textwrap.indent(proof, "  ")
        else:
            proof = "  (* no proof steps yet *)"
        end = "Qed." if close else "Abort."
        return f"{self.header}\nProof.\n{proof}\n{end}\n"


class RocqWorkshop:
    """Small teaching API over rocq-ml-server/Petanque.

    The object keeps one global Rocq state for imports/definitions/completed
    lemmas. Each open lemma has its own state history. Calling `run_tac` from an
    older state creates a branch and moves that lemma's active head to the new
    state.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5000,
        *,
        timeout: float = 180.0,
        connect: bool = True,
    ):
        if PytanqueExtended is None:
            raise RuntimeError(
                "Could not import rocq_ml_toolbox.inference.client.PytanqueExtended. "
                "Install rocq-ml-toolbox or add its `src` directory to sys.path."
            ) from _IMPORT_ERROR
        self.client = PytanqueExtended(host, port)
        if connect:
            self.client.connect()
        self.timeout = timeout
        self.host = host
        self.port = port
        self.root_path = Path(self.client.tmp_file(content="")).resolve()
        self.root_state = self.client.get_root_state(str(self.root_path), timeout=timeout)
        self.global_state = self.root_state
        self.elements: list[str] = []
        self.lemmas: dict[str, LemmaSession] = {}
        self.completed_order: list[str] = []
        self.timeline: list[tuple[str, str]] = []

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def add_element(self, command: str, *, refresh_open_lemmas: bool = True) -> dict[str, Any]:
        command = _clean_command(command)
        new_state = self.client.run(self.global_state, command, timeout=self.timeout)
        self.global_state = new_state
        self.elements.append(command)
        self.timeline.append(("element", command))
        refreshed = self.refresh_open_lemmas() if refresh_open_lemmas else []
        return {
            "ok": True,
            "kind": "element",
            "command": command,
            "feedback": _feedback(new_state),
            "refreshed_open_lemmas": refreshed,
        }

    def add_import(self, libname: str, source: str) -> dict[str, Any]:
        return self.add_element(f"From {libname} Require Import {source}.")

    def open_scope(self, scope: str) -> dict[str, Any]:
        return self.add_element(f"Open Scope {scope}.")

    def add_definition(self, command: str) -> dict[str, Any]:
        if not re.search(r"^\s*(Definition|Fixpoint|CoFixpoint|Let)\b", command):
            raise ValueError("add_definition expects a Definition/Fixpoint/CoFixpoint/Let command.")
        return self.add_element(command)

    def add_lemma(self, header: str, *, name: str | None = None) -> dict[str, Any]:
        header = _clean_command(header)
        if re.search(r"\b(Proof|Qed|Admitted|Defined|Abort)\b", header):
            raise ValueError("Pass only the lemma/theorem statement, without Proof/Qed.")
        match = LEMMA_HEADER_RE.search(header)
        if not match and name is None:
            raise ValueError("Could not infer lemma name. Pass name=... explicitly.")
        inferred_name = match.group(2) if match else name
        assert inferred_name is not None
        if inferred_name in self.lemmas:
            raise ValueError(f"Lemma `{inferred_name}` already exists in this session.")

        proof_state = self.client.run(self.global_state, header, timeout=self.timeout)
        lemma = LemmaSession(
            name=inferred_name,
            header=header,
            nodes=[StateNode(index=0, parent_index=None, command=None, state=proof_state)],
        )
        self.lemmas[inferred_name] = lemma
        return {
            "ok": True,
            "lemma": inferred_name,
            "state_index": 0,
            "goals": self.goals(inferred_name)["goals"],
        }

    def list_lemmas(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for name, lemma in self.lemmas.items():
            out.append(
                {
                    "name": name,
                    "completed": lemma.completed,
                    "latest_state_index": lemma.latest_index,
                    "state_count": len(lemma.nodes),
                    "active_step_count": len(lemma.active_commands()),
                }
            )
        return out

    def goals(self, lemma_name: str, *, state_index: int | None = None) -> dict[str, Any]:
        lemma = self.lemmas[lemma_name]
        idx = lemma.latest_index if state_index is None else state_index
        state = lemma.node(idx).state
        goals = [_goal_text(goal) for goal in self.client.goals(state, timeout=self.timeout)]
        return {
            "lemma": lemma_name,
            "state_index": idx,
            "goals": goals,
            "goal_count": len(goals),
            "proof_finished": bool(getattr(state, "proof_finished", False)) or len(goals) == 0,
        }

    def run_tac(
        self,
        lemma_name: str,
        tactic: str,
        *,
        state_index: int | None = None,
    ) -> dict[str, Any]:
        tactic = _clean_command(tactic)
        lemma = self.lemmas[lemma_name]
        idx = lemma.latest_index if state_index is None else state_index
        parent = lemma.node(idx)
        try:
            new_state = self.client.run(parent.state, tactic, timeout=self.timeout)
        except Exception as exc:
            return {
                "ok": False,
                "lemma": lemma_name,
                "source_state_index": idx,
                "tactic": tactic,
                "error": str(exc),
                "feedback": _feedback(parent.state),
            }
        new_idx = len(lemma.nodes)
        lemma.nodes.append(
            StateNode(index=new_idx, parent_index=idx, command=tactic, state=new_state)
        )
        lemma.latest_index = new_idx
        goal_info = self.goals(lemma_name, state_index=new_idx)
        return {
            "ok": True,
            "lemma": lemma_name,
            "source_state_index": idx,
            "new_state_index": new_idx,
            "tactic": tactic,
            "goals": goal_info["goals"],
            "goal_count": goal_info["goal_count"],
            "proof_finished": goal_info["proof_finished"],
            "feedback": _feedback(new_state),
        }

    def run_script(self, lemma_name: str, steps: Iterable[str]) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for step in steps:
            out = self.run_tac(lemma_name, step)
            outputs.append(out)
            if not out.get("ok", False):
                break
        return outputs

    def complete_lemma(self, lemma_name: str, *, refresh_open_lemmas: bool = True) -> dict[str, Any]:
        lemma = self.lemmas[lemma_name]
        if lemma.completed:
            return {"ok": True, "lemma": lemma_name, "already_completed": True}
        latest = lemma.node(lemma.latest_index)
        try:
            qed_state = self.client.run(latest.state, "Qed.", timeout=self.timeout)
        except Exception as exc:
            return {
                "ok": False,
                "lemma": lemma_name,
                "error": str(exc),
                "goals": self.goals(lemma_name)["goals"],
            }
        lemma.completed = True
        lemma.qed_state = qed_state
        self.global_state = qed_state
        self.completed_order.append(lemma_name)
        self.timeline.append(("lemma", lemma_name))
        refreshed = self.refresh_open_lemmas(except_names={lemma_name}) if refresh_open_lemmas else []
        return {
            "ok": True,
            "lemma": lemma_name,
            "feedback": _feedback(qed_state),
            "refreshed_open_lemmas": refreshed,
        }

    def restart_lemma(self, lemma_name: str) -> dict[str, Any]:
        lemma = self.lemmas[lemma_name]
        if lemma.completed:
            raise ValueError(f"Lemma `{lemma_name}` is already completed.")
        commands = lemma.active_commands()
        proof_state = self.client.run(self.global_state, lemma.header, timeout=self.timeout)
        new_nodes = [StateNode(index=0, parent_index=None, command=None, state=proof_state)]
        latest_index = 0
        for command in commands:
            new_state = self.client.run(new_nodes[latest_index].state, command, timeout=self.timeout)
            latest_index = len(new_nodes)
            new_nodes.append(
                StateNode(
                    index=latest_index,
                    parent_index=latest_index - 1,
                    command=command,
                    state=new_state,
                )
            )
        lemma.nodes = new_nodes
        lemma.latest_index = latest_index
        return {
            "ok": True,
            "lemma": lemma_name,
            "replayed_step_count": len(commands),
            "latest_state_index": lemma.latest_index,
            "goals": self.goals(lemma_name)["goals"],
        }

    def refresh_open_lemmas(self, *, except_names: set[str] | None = None) -> list[dict[str, Any]]:
        except_names = except_names or set()
        refreshed: list[dict[str, Any]] = []
        for name, lemma in list(self.lemmas.items()):
            if lemma.completed or name in except_names:
                continue
            try:
                refreshed.append(self.restart_lemma(name))
            except Exception as exc:
                refreshed.append({"ok": False, "lemma": name, "error": str(exc)})
        return refreshed

    def proof_script(self, lemma_name: str) -> str:
        return "\n".join(self.lemmas[lemma_name].active_commands())

    def lemma_source(self, lemma_name: str, *, close: bool | None = None) -> str:
        return self.lemmas[lemma_name].source(close=close)

    def source(self, *, include_open: bool = True) -> str:
        blocks: list[str] = []
        for kind, value in self.timeline:
            if kind == "element":
                blocks.append(value)
            elif kind == "lemma":
                blocks.append(self.lemmas[value].source(close=True))
        if include_open:
            for _name, lemma in self.lemmas.items():
                if not lemma.completed:
                    blocks.append(lemma.source(close=False))
        return "\n\n".join(block.rstrip() for block in blocks if block.strip()) + "\n"
