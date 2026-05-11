"""Plan/execute loop with LangGraph, sequential stages of parallel steps.

The planner produces a "plan" shaped as a list of stages. Each stage is a
list of steps that can run concurrently. Stages run sequentially - later
stages may depend on results from earlier ones.

This handles both extremes with the same code:
  - Fully sequential goal -> N stages of 1 step each
  - Fully parallel goal   -> 1 stage of N steps
  - Mixed                 -> some stages of 1, some of many

Nodes:
  make_plan    -> LLM produces the stages
  execute_plan_step  -> ReAct sub-agent does ONE step (many run in parallel via Send)
  pop_stage    -> remove the just-finished stage from the plan
  final_answer -> summarize the whole run for the user
"""

from __future__ import annotations

import asyncio
import os
import urllib.request
from operator import add
from pathlib import Path
from typing import Annotated, List, Tuple, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Send
from pydantic import BaseModel, Field

load_dotenv()


@tool
def fetch_url(url: str) -> str:
    """Fetch the given URL and return the response body as text (truncated to 4000 chars)."""
    req = urllib.request.Request(url, headers={"User-Agent": "plan-execute-demo/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return body[:4000]


@tool
def write_file(path: str, content: str) -> str:
    """Write the given text content to a file. Creates parent dirs if needed."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {p}"


TOOLS = [fetch_url, write_file]


class RunState(TypedDict):
    goal: str
    plan: List[List[str]]
    past_steps: Annotated[List[Tuple[str, str]], add]
    response: str


class StepState(TypedDict):
    goal: str
    step: str
    past_steps: List[Tuple[str, str]]


class Stage(BaseModel):
    """A group of steps that are independent of each other and run in parallel."""

    steps: List[str] = Field(
        description="Independent steps. None of these may depend on the output of any other step in the same stage."
    )


class Plan(BaseModel):
    """An ordered list of stages. Stages run sequentially; steps inside a stage run in parallel."""

    stages: List[Stage] = Field(
        description=(
            "Sequential stages. Steps inside one stage run in parallel, "
            "so they must be independent. Steps in later stages may use "
            "results from earlier stages. Use one step per stage if the goal "
            "is purely sequential."
        )
    )


def build_app():
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", max_tokens=2048)
    planner = llm.with_structured_output(Plan)
    executor = create_react_agent(llm, TOOLS)

    async def make_plan(state: RunState) -> dict:
        prompt = (
            "Break this goal into 1-5 stages with 1-5 steps each.\n"
            "Group steps into the same stage only when they are truly independent.\n"
            "Otherwise put them in separate stages so later ones can use earlier results.\n\n"
            f"Goal: {state['goal']}"
        )
        plan: Plan = await planner.ainvoke(prompt)
        return {"plan": [s.steps for s in plan.stages]}

    async def execute_plan_step(state: StepState) -> dict:
        context = "\n".join(f"- {s}: {r[:200]}" for s, r in state["past_steps"]) or "(none)"
        prompt = (
            f"Overall goal: {state['goal']}\n"
            f"Results from previous stages:\n{context}\n\n"
            f"Do this single step using the available tools: {state['step']}\n"
            "Report only what you did and what you observed."
        )
        result = await executor.ainvoke({"messages": [("user", prompt)]})
        output = result["messages"][-1].content
        return {"past_steps": [(state["step"], output)]}

    async def pop_stage(state: RunState) -> dict:
        return {"plan": state["plan"][1:]}

    async def final_answer(state: RunState) -> dict:
        history = "\n".join(f"- {s}\n  -> {r}" for s, r in state["past_steps"])
        msg = await llm.ainvoke(
            f"Goal: {state['goal']}\n\nWhat was done:\n{history}\n\n"
            "Write a short final answer for the user."
        )
        return {"response": msg.content}

    def dispatch(state: RunState):
        if not state["plan"]:
            return [Send("final_answer", state)]

        current_stage = state["plan"][0]
        sends = []
        for step in current_stage:
            step_state: StepState = {
                "goal": state["goal"],
                "step": step,
                "past_steps": state["past_steps"],
            }
            sends.append(Send("execute_plan_step", step_state))
        return sends

    graph = StateGraph(RunState)
    graph.add_node("make_plan", make_plan)
    graph.add_node("execute_plan_step", execute_plan_step)
    graph.add_node("pop_stage", pop_stage)
    graph.add_node("final_answer", final_answer)
    graph.set_entry_point("make_plan")
    graph.add_conditional_edges("make_plan", dispatch, ["execute_plan_step", "final_answer"])
    graph.add_edge("execute_plan_step", "pop_stage")
    graph.add_conditional_edges("pop_stage", dispatch, ["execute_plan_step", "final_answer"])
    graph.add_edge("final_answer", END)
    return graph.compile()


async def run(goal: str) -> str:
    app = build_app()
    initial: RunState = {"goal": goal, "plan": [], "past_steps": [], "response": ""}
    final = await app.ainvoke(initial, config={"recursion_limit": 50})
    return final["response"]


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY before running.")

    goal = (
        "Fetch the page titles of https://example.com, https://example.org, "
        "and https://example.net, then write all three titles into one file "
        "at /tmp/all_titles.txt, one per line."
    )

    answer = asyncio.run(run(goal))
    print("\n=== FINAL ANSWER ===")
    print(answer)
