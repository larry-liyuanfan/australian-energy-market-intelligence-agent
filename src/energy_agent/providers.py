from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.request import Request, urlopen

from .tools import ToolRegistry


class ProviderUnavailable(RuntimeError):
    pass


Transport = Callable[[Request, float], bytes]


def _urlopen_transport(request: Request, timeout: float) -> bytes:
    with urlopen(request, timeout=timeout) as response:  # nosec B310 - operator-configured HTTPS endpoint
        return bytes(response.read())


@dataclass(frozen=True)
class ModelStudioPlanner:
    """OpenAI-compatible Model Studio planner constrained to registered typed tools."""

    base_url: str
    api_key: str
    model: str = "qwen-plus"
    timeout_seconds: float = 20.0
    transport: Transport = _urlopen_transport
    name: str = "model_studio_openai_compatible"

    @classmethod
    def from_environment(cls) -> ModelStudioPlanner | None:
        base_url = os.getenv("MODEL_STUDIO_BASE_URL", "").strip()
        api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not base_url or not api_key:
            return None
        if not base_url.startswith("https://"):
            raise ProviderUnavailable("MODEL_STUDIO_BASE_URL must use HTTPS")
        return cls(base_url, api_key, os.getenv("MODEL_STUDIO_MODEL", "qwen-plus"))

    def plan(
        self, question: str, registry: ToolRegistry, max_tool_calls: int
    ) -> list[tuple[str, dict[str, object]]]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Select only registered typed tools. Never emit SQL, Elasticsearch DSL, or optimisation code.",
                },
                {"role": "user", "content": question},
            ],
            "tools": [{"type": "function", "function": spec} for spec in registry.specs()],
            "tool_choice": "auto",
            "temperature": 0,
            "stream": False,
        }
        request = Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        decoded: dict[str, Any] = json.loads(self.transport(request, self.timeout_seconds))
        try:
            calls = decoded["choices"][0]["message"].get("tool_calls", [])
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderUnavailable("provider response did not contain a valid message") from exc
        output: list[tuple[str, dict[str, object]]] = []
        for call in calls[:max_tool_calls]:
            function = call.get("function", {})
            name = str(function.get("name", ""))
            arguments = json.loads(function.get("arguments", "{}"))
            if not isinstance(arguments, dict):
                raise TypeError("tool arguments must be a JSON object")
            validated = registry.validate(name, arguments)
            output.append((name, validated.model_dump(mode="json")))
        return output
