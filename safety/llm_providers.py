"""Pluggable LLM back-ends. Every provider exposes two methods:

    complete_json(system, user, temperature=0.0, max_tokens=400, schema=None) -> dict   # safety check
    chat(messages, temperature=0.7, max_tokens=200) -> str                              # companion replies

`schema` is an optional JSON schema for the answer. LM Studio uses it to force valid
JSON (grammar-constrained decoding); the cloud providers ignore it. `messages` is the
usual OpenAI list of {"role": system|user|assistant, "content": ...}.

Keys come from the environment: OPENAI_API_KEY or GEMINI_API_KEY (or GOOGLE_API_KEY).
LM Studio needs no key and no extra package: start its local server (default http://localhost:1234/v1).
Model names change often; override them in config.yaml (llm.model) or with --model.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Optional, Protocol

DEFAULT_MODELS = {
    "openai": "gpt-5-mini",        # check https://platform.openai.com/docs/models
    "gemini": "gemini-3.6-flash",  # check https://ai.google.dev/gemini-api/docs/models
}


class LLMProvider(Protocol):
    name: str
    model: str

    def complete_json(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 400,
                      schema: Optional[dict] = None) -> dict: ...

    def chat(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 200) -> str: ...


_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove a reasoning block that some local models (Qwen3) put before the answer."""
    text = _THINK.sub("", text or "")
    if "<think>" in text:            # unterminated block: the model ran out of tokens while thinking
        text = text.split("<think>", 1)[0]
    return text.strip()


def _parse_json(text: str) -> dict:
    text = strip_thinking(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)  # tolerate ```json fences
        if not m:
            raise
        return json.loads(m.group(0))


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: Optional[str] = None, timeout_s: float = 20.0):
        from openai import OpenAI

        self.model = model or DEFAULT_MODELS["openai"]
        self.client = OpenAI(timeout=timeout_s, max_retries=1)

    def complete_json(self, system, user, temperature=0.0, max_tokens=400, schema=None):
        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            max_completion_tokens=max_tokens,
        )
        try:
            resp = self.client.chat.completions.create(temperature=temperature, **kwargs)
        except Exception as e:  # some reasoning models only accept the default temperature
            if "temperature" not in str(e):
                raise
            resp = self.client.chat.completions.create(**kwargs)
        return _parse_json(resp.choices[0].message.content)

    def chat(self, messages, temperature=0.7, max_tokens=200):
        kwargs = dict(model=self.model, messages=messages, max_completion_tokens=max_tokens)
        try:
            resp = self.client.chat.completions.create(temperature=temperature, **kwargs)
        except Exception as e:
            if "temperature" not in str(e):
                raise
            resp = self.client.chat.completions.create(**kwargs)
        return strip_thinking(resp.choices[0].message.content)


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: Optional[str] = None, timeout_s: float = 20.0):
        from google import genai
        from google.genai import types

        self.model = model or DEFAULT_MODELS["gemini"]
        self.types = types
        self.client = genai.Client(http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))

    def complete_json(self, system, user, temperature=0.0, max_tokens=400, schema=None):
        cfg = self.types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        resp = self.client.models.generate_content(model=self.model, contents=user, config=cfg)
        return _parse_json(resp.text)

    def chat(self, messages, temperature=0.7, max_tokens=200):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
        contents = [self.types.Content(role="model" if m["role"] == "assistant" else "user",
                                       parts=[self.types.Part.from_text(text=m["content"])])
                    for m in messages if m["role"] != "system"]
        cfg = self.types.GenerateContentConfig(system_instruction=system, temperature=temperature,
                                               max_output_tokens=max_tokens)
        resp = self.client.models.generate_content(model=self.model, contents=contents, config=cfg)
        return strip_thinking(resp.text)


class LMStudioProvider:
    """A model served locally by LM Studio through its OpenAI-compatible server.

    In LM Studio: load a model, then Developer tab -> Start Server. The model id is the
    one shown there (or set LMSTUDIO_MODEL); if none is given, the first loaded model is used.
    Qwen3-style "thinking" is switched off per request (chat_template_kwargs) and any
    <think> block that still comes back is stripped.

    Talks plain HTTP with the standard library (no openai package needed): fewer things to
    install, no proxy surprises on localhost, and nothing for Windows Smart App Control to block.
    """

    name = "lmstudio"

    def __init__(self, model: Optional[str] = None, timeout_s: float = 60.0, base_url: Optional[str] = None,
                 disable_thinking: bool = True):
        self.base_url = (base_url or os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")).rstrip("/")
        self.timeout_s = timeout_s
        self._open = urllib.request.build_opener(urllib.request.ProxyHandler({})).open  # never via a proxy
        self.model = model or os.environ.get("LMSTUDIO_MODEL") or self._first_loaded_model()
        self.extra = {"chat_template_kwargs": {"enable_thinking": False}} if disable_thinking else {}

    def _request(self, path: str, body: Optional[dict] = None) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data,
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer lm-studio"})
        with self._open(req, timeout=self.timeout_s) as r:
            return json.load(r)

    def list_models(self) -> list[str]:
        return [m["id"] for m in self._request("/models").get("data", [])]

    def _first_loaded_model(self) -> str:
        models = self.list_models()
        if not models:
            raise RuntimeError("LM Studio server is running but no model is loaded")
        return models[0]

    def _create(self, **body) -> str:
        """POST /chat/completions and return the assistant text."""
        try:
            resp = self._request("/chat/completions", {**body, **self.extra})
        except urllib.error.HTTPError as e:  # a server that rejects chat_template_kwargs: retry plain, remember that
            if not self.extra or e.code not in (400, 422):
                raise
            self.extra = {}
            resp = self._request("/chat/completions", body)
        return resp["choices"][0]["message"]["content"] or ""

    def complete_json(self, system, user, temperature=0.0, max_tokens=400, schema=None):
        fmt = {"type": "json_schema",
               "json_schema": {"name": "answer", "strict": True, "schema": schema or {"type": "object"}}}
        text = self._create(model=self.model,
                            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                            response_format=fmt, temperature=temperature, max_tokens=max_tokens)
        return _parse_json(text)

    def chat(self, messages, temperature=0.7, max_tokens=200):
        return strip_thinking(self._create(model=self.model, messages=messages, temperature=temperature,
                                           max_tokens=max_tokens))


class MockProvider:
    """Offline stand-in for tests and for running the pipeline without an API key.

    behaviour:
      "keyword" - crude keyword guess (NOT a real safety check)
      "fail"    - raises, to test the fail-safe path
      int 0..3  - always returns that level
    """

    name = "mock"

    def __init__(self, behaviour="keyword", model: str = "mock"):
        self.behaviour = behaviour
        self.model = model
        self.calls = 0

    def complete_json(self, system, user, temperature=0.0, max_tokens=400, schema=None):
        self.calls += 1
        if self.behaviour == "fail":
            raise TimeoutError("mock provider failure")
        if isinstance(self.behaviour, int):
            return {"level": self.behaviour, "category": None, "confidence": 0.9, "rationale": "mock"}
        m = re.search(r"<user_message>(.*?)</user_message>", user, re.DOTALL)
        msg = (m.group(1) if m else user).lower()
        if re.search(r"\b(die|dead|suicid\w*|kill myself|end it|hurt myself|burden|disappear\w*|without me|goodbye)\b", msg):
            return {"level": 3, "category": None, "confidence": 0.6, "rationale": "mock keyword"}
        if re.search(r"\b(can'?t do this|too much|too loud|stop|overwhelm\w*|leave me alone)\b", msg):
            return {"level": 2, "category": None, "confidence": 0.6, "rationale": "mock keyword"}
        if re.search(r"\b(nervous|anxious|scared|worried|stressed|um+|uh+|is this ok)\b", msg):
            return {"level": 1, "category": None, "confidence": 0.6, "rationale": "mock keyword"}
        return {"level": 0, "category": None, "confidence": 0.6, "rationale": "mock keyword"}

    # Canned companion lines per scenario (the system prompt names the scenario as "Scenario id: ...").
    CHAT_LINES = {
        "hawker_centre": ["Okay. One chicken rice. Do you want chilli with it?",
                          "Sure. Do you want it here or take away?",
                          "That is 4 dollars 50. Cash or PayNow?", "Here you go. Enjoy your meal."],
        "new_colleague": ["Nice to meet you. I am Wei Ling from the design team. Which team are you in?",
                          "Oh nice. How is your first week going so far?",
                          "Same here. Do you want to get lunch together on Friday?", "Great. See you then."],
        "job_interview": ["Thank you for coming. Can you tell me a bit about yourself?",
                          "Thank you. Why are you interested in this role?",
                          "I see. Can you tell me about a time you worked in a team?", "Thank you. Do you have any questions for me?"],
        "free_chat": ["Thanks for telling me. What would you like to talk about today?",
                      "I see. Can you tell me a bit more about that?",
                      "That sounds like something you know a lot about. What do you like most about it?",
                      "Thank you for sharing that with me."],
    }

    def chat(self, messages, temperature=0.7, max_tokens=200):
        self.calls += 1
        system = " ".join(m["content"] for m in messages if m["role"] == "system")
        m = re.search(r"Scenario id: (\w+)", system)
        lines = self.CHAT_LINES.get(m.group(1) if m else "free_chat", self.CHAT_LINES["free_chat"])
        n_said = sum(1 for x in messages if x["role"] == "assistant")
        if "role-play is paused" in system.lower():
            return "Okay. I am here. Take your time."
        return lines[min(n_said, len(lines) - 1)]


def get_provider(name: str, model: Optional[str] = None, timeout_s: float = 20.0,
                 base_url: Optional[str] = None, disable_thinking: bool = True) -> LLMProvider:
    name = (name or "mock").lower()
    if name == "openai":
        return OpenAIProvider(model, timeout_s)
    if name == "gemini":
        return GeminiProvider(model, timeout_s)
    if name == "lmstudio":
        return LMStudioProvider(model, max(timeout_s, 60.0), base_url, disable_thinking)
    if name == "mock":
        return MockProvider()
    if name == "mock-fail":
        return MockProvider("fail")
    raise ValueError(f"unknown LLM provider {name!r} (use openai | gemini | lmstudio | mock)")
