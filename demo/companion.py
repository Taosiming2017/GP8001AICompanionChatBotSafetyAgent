"""The companion: the LLM that actually talks to the user (Qwen in LM Studio by default).

The system prompt is demo/prompts/companion_system.md plus one scenario card from
demo/prompts/scenarios.yaml. Both are sent with every request, so nothing has to be typed into
LM Studio. The safety agent's directives for the turn are appended under "Instructions for this
turn". The reply is checked by the output guard in the orchestrator, not here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from safety.llm_providers import LLMProvider

PROMPTS = Path(__file__).parent / "prompts"


def load_scenarios(path: Path = PROMPTS / "scenarios.yaml") -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class Companion:
    def __init__(self, provider: LLMProvider, scenarios: Optional[dict] = None,
                 system_prompt: Optional[str] = None, temperature: float = 0.7,
                 max_tokens: int = 160, history_turns: int = 12):
        self.provider = provider
        self.scenarios = scenarios or load_scenarios()
        self.system_prompt = system_prompt or (PROMPTS / "companion_system.md").read_text(encoding="utf-8")
        self.temperature, self.max_tokens, self.history_turns = temperature, max_tokens, history_turns

    @property
    def name(self) -> str:
        return f"{self.provider.name}/{self.provider.model}"

    def system(self, scenario: str, directives: list[str], paused: bool, hint: Optional[str] = None) -> str:
        card = self.scenarios.get(scenario) or self.scenarios["free_chat"]
        parts = [self.system_prompt, f"## Scenario card\nScenario id: {scenario}\n{card['role']}"]
        extra = list(directives)
        if paused and not any("paused" in d.lower() for d in extra):
            extra.insert(0, "The role-play is paused. Speak as Companion, not as the character.")
        if hint:
            extra.append(hint)
        if extra:
            parts.append("## Instructions for this turn\n" + "\n".join(f"- {d}" for d in extra))
        return "\n\n".join(parts)

    def reply(self, scenario: str, history: list[dict], directives: list[str] = (),
              paused: bool = False, hint: Optional[str] = None) -> str:
        """history: [{"role": "user"|"companion", "text": ...}, ...] oldest first, user message last."""
        msgs = [{"role": "system", "content": self.system(scenario, list(directives), paused, hint)}]
        for h in history[-self.history_turns:]:
            msgs.append({"role": "assistant" if h["role"] == "companion" else "user", "content": h["text"]})
        text = self.provider.chat(msgs, temperature=self.temperature, max_tokens=self.max_tokens)
        return " ".join(text.split()) if text else ""
