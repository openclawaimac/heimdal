"""Cloud teacher providers (OpenAI, Anthropic).

These providers are intentionally minimal: they import their SDK lazily so a
Heimdal install without ``openai`` / ``anthropic`` keeps working, and they
read API keys only from env vars (never from manifest, never logged). They
raise an actionable error when the SDK is missing or the key isn't set --
Mirror Mode then records the call as ``status="skipped"`` rather than
crashing the run.

CI never exercises these providers; the stub teacher is the test path.
"""

from __future__ import annotations

import os

from heimdal.mirror.provider import TeacherInput, TeacherProvider, TeacherResult


class CloudProviderUnavailable(RuntimeError):
    """Raised when a cloud provider's SDK or credential is missing."""


def _require_env(var: str) -> str:
    value = os.environ.get(var)
    if not value:
        raise CloudProviderUnavailable(
            f"{var} is not set. Set the env var or use --teacher stub."
        )
    return value


class OpenAITeacher(TeacherProvider):
    name = "openai"

    def __init__(self, model: str):
        self.model = model

    def generate(self, input_: TeacherInput) -> TeacherResult:
        api_key = _require_env("OPENAI_API_KEY")
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise CloudProviderUnavailable(
                "openai SDK is not installed. `pip install openai`, or use "
                "--teacher stub."
            ) from exc
        client = OpenAI(api_key=api_key)
        # Build a minimal prompt -- the input has already been redacted by
        # the runner before reaching the provider.
        prompt = (
            "You are an external teacher reviewing a local LLM agent's "
            "answer. Provide a better answer that follows the task's "
            "constraints. Do not invent facts.\n\n"
            f"TASK: {input_.task.get('objective','')}\n"
            f"CONSTRAINTS: {input_.constraints}\n"
            f"LOCAL ANSWER:\n{input_.local_output}\n"
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        return TeacherResult(
            provider=self.name,
            model=self.model,
            status="pass",
            output=text,
            usage={
                "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "estimated_cost": None,
            },
            metadata={},
        )


class AnthropicTeacher(TeacherProvider):
    name = "anthropic"

    def __init__(self, model: str):
        self.model = model

    def generate(self, input_: TeacherInput) -> TeacherResult:
        api_key = _require_env("ANTHROPIC_API_KEY")
        try:
            import anthropic  # type: ignore
        except ImportError as exc:
            raise CloudProviderUnavailable(
                "anthropic SDK is not installed. `pip install anthropic`, "
                "or use --teacher stub."
            ) from exc
        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            "You are reviewing a local LLM agent's answer. Produce a better "
            "answer that follows the task constraints. Do not invent facts.\n\n"
            f"TASK: {input_.task.get('objective','')}\n"
            f"CONSTRAINTS: {input_.constraints}\n"
            f"LOCAL ANSWER:\n{input_.local_output}\n"
        )
        message = client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )
        usage = getattr(message, "usage", None)
        return TeacherResult(
            provider=self.name,
            model=self.model,
            status="pass",
            output=text,
            usage={
                "input_tokens": getattr(usage, "input_tokens", 0) or 0,
                "output_tokens": getattr(usage, "output_tokens", 0) or 0,
                "estimated_cost": None,
            },
            metadata={},
        )
