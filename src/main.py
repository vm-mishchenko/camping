from __future__ import annotations

import asyncio
import os
import urllib.request
from pathlib import Path
from typing import Annotated, List, Tuple, TypedDict
from operator import add

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

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


class PlanState(TypedDict):
    goal: str
    plan: List[str]
    past_steps: Annotated[List[Tuple[str, str]], add]
    response: str


def build_app():
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", max_tokens=2048)
    executor = create_react_agent(llm, TOOLS)

    async def plan_node(state: PlanState) -> dict:
        prompt = (
            "Break this goal into 2-5 short, concrete steps a tool-using agent can do.\n"
            "Return one step per line, no numbering, no extra commentary.\n\n"
            f"Goal: {state['goal']}"
        )
        msg = await llm.ainvoke(prompt)
        steps = [line.strip(" -*0123456789.") for line in msg.content.splitlines() if line.strip()]
        return {"plan": steps}

    async def execute_node(state: PlanState) -> dict:
        step = state["plan"][0]
        context = "\n".join(f"- {s}: {r[:200]}" for s, r in state["past_steps"]) or "(none)"
        prompt = (
            f"Overall goal: {state['goal']}\n"
            f"Previous steps and results:\n{context}\n\n"
            f"Now do this step using the available tools: {step}\n"
            "Report only what you did and what you observed."
        )
        result = await executor.ainvoke({"messages": [("user", prompt)]})
        output = result["messages"][-1].content
        return {"past_steps": [(step, output)], "plan": state["plan"][1:]}

    async def replan_node(state: PlanState) -> dict:
        if state["plan"]:
            return {}
        history = "\n".join(f"- {s}\n  -> {r}" for s, r in state["past_steps"])
        msg = await llm.ainvoke(
            f"Goal: {state['goal']}\n\nWhat was done:\n{history}\n\n"
            "Write a short final answer for the user."
        )
        return {"response": msg.content}

    def route(state: PlanState) -> str:
        return END if state.get("response") else "execute"

    graph = StateGraph(PlanState)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("replan", replan_node)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "replan")
    graph.add_conditional_edges("replan", route, {"execute": "execute", END: END})
    return graph.compile()


async def run(goal: str) -> str:
    app = build_app()
    initial: PlanState = {"goal": goal, "plan": [], "past_steps": [], "response": ""}
    final = await app.ainvoke(initial, config={"recursion_limit": 50})
    return final["response"]


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY before running.")

    goal = (
        "Fetch https://example.com, extract the page title, "
        "and save just the title text to /tmp/example_title.txt"
    )
    answer = asyncio.run(run(goal))
    print("\n=== FINAL ANSWER ===")
    print(answer)
