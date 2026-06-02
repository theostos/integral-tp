from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class LLMClient:
    """Minimal OpenAI-compatible chat-completions client."""

    model: str
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    timeout: float = 60.0

    @classmethod
    def from_env(cls, *, model: str | None = None) -> "LLMClient":
        return cls(
            model=model or os.getenv("LLM_MODEL", "gpt-4.1-mini"),
            api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def chat(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        if not self.api_key:
            raise RuntimeError(
                "No LLM API key configured. Set LLM_API_KEY or OPENAI_API_KEY, "
                "and optionally LLM_BASE_URL / LLM_MODEL."
            )
        url = self.base_url.rstrip("/") + "/chat/completions"
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload["choices"][0]["message"]["content"])


def proof_prompt(
    *,
    lemma_header: str,
    goals: list[str],
    retrieval_hits: list[dict[str, Any]],
    extra_context: str = "",
) -> str:
    hits = "\n".join(
        f"- {hit.get('name')}: {hit.get('statement') or hit.get('docstring') or hit}"
        for hit in retrieval_hits
    )
    goal_text = "\n\n".join(goals) if goals else "(no focused goals)"
    return textwrap.dedent(
        f"""
        We are proving this Rocq statement:

        {lemma_header}

        Current goals:

        {goal_text}

        Retrieved library facts/tactics:

        {hits or "(none)"}

        Extra context:

        {extra_context or "(none)"}

        Return a short Rocq proof script. Use only tactic commands, no `Proof.` and no `Qed.`.
        Prefer robust scripts over clever one-liners.
        """
    ).strip()
