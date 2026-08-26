"""
agent_loop.py
-------------
The autonomous tool-calling loop: given a user question and a DataFrame,
the agent repeatedly asks the LLM what to do, executes any tool calls it
requests against the live data, feeds the results back, and keeps going
until the model produces a final natural-language answer (or a max-step
budget is hit, to guarantee termination).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

from agent.llm_client import DEFAULT_MODEL, chat
from agent.tools import TOOL_SCHEMAS, build_tool_implementations

SYSTEM_PROMPT = """You are an autonomous data analytics agent. You have tools to \
inspect a transactions dataset spanning multiple sales channels/sources. \
Use the tools to gather real numbers before answering -- never invent figures. \
Call as many tools as needed across multiple turns to fully answer the question. \
When you have enough information, give a concise, decision-useful answer \
that references the concrete numbers you found. Flag anomalies proactively \
if they're relevant to the question."""


@dataclass
class AgentStep:
    role: str  # "tool_call" | "tool_result" | "final"
    content: str
    tool_name: str | None = None


@dataclass
class AgentRun:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)


def run_agent(
    question: str,
    df: pd.DataFrame,
    model: str = DEFAULT_MODEL,
    max_steps: int = 6,
) -> AgentRun:
    """Runs the tool-calling loop until the model answers or max_steps is hit."""
    tool_impls = build_tool_implementations(df)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    run = AgentRun(answer="")

    for _ in range(max_steps):
        message = chat(messages=messages, tools=TOOL_SCHEMAS, model=model)
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            final_text = message.get("content", "").strip()
            run.answer = final_text or "I wasn't able to produce an answer."
            run.steps.append(AgentStep(role="final", content=run.answer))
            return run

        messages.append(message)

        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name")
            raw_args = fn.get("arguments", {})
            args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")

            run.steps.append(AgentStep(
                role="tool_call",
                tool_name=name,
                content=json.dumps(args),
            ))

            impl = tool_impls.get(name)
            if impl is None:
                result = json.dumps({"error": f"unknown tool {name}"})
            else:
                try:
                    result = impl(**args)
                except Exception as e:  # noqa: BLE001 - surface tool errors to the model
                    result = json.dumps({"error": str(e)})

            run.steps.append(AgentStep(role="tool_result", tool_name=name, content=result))
            messages.append({
                "role": "tool",
                "content": result,
            })

    run.answer = "Reached the step limit before the model finished reasoning. Try a narrower question."
    run.steps.append(AgentStep(role="final", content=run.answer))
    return run
