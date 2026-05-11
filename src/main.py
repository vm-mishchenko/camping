"""Plan/execute loop with LangGraph, sequential stages of parallel steps.

The planner produces a "plan" shaped as a list of stages. Each stage is a
list of steps that can run concurrently. Stages run sequentially - later
stages may depend on results from earlier ones.

This handles both extremes with the same code:
  - Fully sequential goal -> N stages of 1 step each
  - Fully parallel goal   -> 1 stage of N steps
  - Mixed                 -> some stages of 1, some of many

Nodes:
  make_plan          -> LLM produces the stages
  execute_plan_step  -> ReAct sub-agent does ONE step (many run in parallel via Send)
  pop_stage          -> remove the just-finished stage from the plan
  final_answer       -> summarize the whole run for the user

State persists in a SQLite checkpointer (data/checkpoints.db). Each run is
identified by a thread_id. Kill the process at any point and restart with
the same thread_id to resume from the last completed node.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import urllib.request
import uuid
from operator import add
from pathlib import Path
from random import random
from typing import Annotated, List, Tuple, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Send, RetryPolicy
from pydantic import BaseModel, Field

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DB = REPO_ROOT / "data" / "checkpoints.db"


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

def build_graph():
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", max_tokens=2048)
    planner = llm.with_structured_output(Plan)
    executor = create_agent(llm, TOOLS)

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
        if random() > 0.5:
            raise Exception("test exception")
        
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
    graph.add_node(
        "execute_plan_step", 
        execute_plan_step,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_interval=1.0,
            backoff_factor=2.0,
        ),
    )
    graph.add_node("pop_stage", pop_stage)
    graph.add_node("final_answer", final_answer)
    graph.set_entry_point("make_plan")
    graph.add_conditional_edges("make_plan", dispatch, ["execute_plan_step", "final_answer"])
    graph.add_edge("execute_plan_step", "pop_stage")
    graph.add_conditional_edges("pop_stage", dispatch, ["execute_plan_step", "final_answer"])
    graph.add_edge("final_answer", END)
    return graph


async def run_new(goal: str, thread_id: str) -> str:
    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver:
        app = build_graph().compile(checkpointer=saver)
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
        initial: RunState = {"goal": goal, "plan": [], "past_steps": [], "response": ""}
        final = await app.ainvoke(initial, config=config)
        return final.get("response", "")


async def run_resume(thread_id: str) -> str:
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver:
        app = build_graph().compile(checkpointer=saver)
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
        snapshot = await app.aget_state(config)
        if not snapshot or not snapshot.values:
            raise SystemExit(f"No state found for thread_id={thread_id}")
        final = await app.ainvoke(None, config=config)
        return final.get("response", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan/execute agent with resumable runs.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--goal", help="Start a new run with this goal.")
    group.add_argument("--resume", metavar="THREAD_ID", help="Resume an existing run.")
    parser.add_argument("--thread-id", help="Override the auto-generated thread_id (new runs only).")
    return parser.parse_args()


async def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY before running.")

    args = parse_args()

    if args.resume:
        print(f"Resuming thread_id={args.resume}")
        answer = await run_resume(args.resume)
    else:
        goal = args.goal or (
            "Fetch the page titles of https://example.com, https://example.org, "
            "and https://example.net, then write all three titles into one file "
            "at /tmp/all_titles.txt, one per line."
        )
        thread_id = args.thread_id or str(uuid.uuid4())
        print(f"Starting new run, thread_id={thread_id}")
        print(f"To resume: python -m src.main --resume {thread_id}")
        print(f"Goal: {goal}\n")
        answer = await run_new(goal, thread_id)

    print("\n=== FINAL ANSWER ===")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
