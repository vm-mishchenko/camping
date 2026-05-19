"""Plan/execute loop with LangGraph, sequential stages of parallel steps.

The planner produces a "plan" shaped as a list of stages. Each stage is a
list of steps that can run concurrently. Stages run sequentially - later
stages may depend on results from earlier ones.

This handles both extremes with the same code:
  - Fully sequential goal -> N stages of 1 step each
  - Fully parallel goal   -> 1 stage of N steps
  - Mixed                 -> some stages of 1, some of many

Nodes:
  classify           -> LLM decides: NEWTASK (re-plan) or FOLLOWUP (answer directly)
  make_plan          -> LLM produces the stages
  execute_plan_step  -> ReAct sub-agent does ONE step (many run in parallel via Send)
  pop_stage          -> remove the just-finished stage from the plan
  final_answer       -> summarize the whole run for the user

Each user message triggers a full graph run from START to END. The checkpointer
(SQLite) persists the conversation history across turns via the add_messages
reducer; scratch fields like plan and completed_steps are reset at the start of
each run by the classify node.

State persists in a SQLite checkpointer (data/checkpoints.db). Each run is
identified by a thread_id. Kill the process at any point and restart with
the same thread_id to resume from the last completed node.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import urllib.request
import uuid
from pathlib import Path
from typing import Annotated, List, Literal, Tuple, TypedDict

VERBOSE = 15
logging.addLevelName(VERBOSE, "VERBOSE")


class _VerboseLogger(logging.Logger):
    def verbose(self, msg: str, *args, **kwargs) -> None:
        if self.isEnabledFor(VERBOSE):
            self._log(VERBOSE, msg, args, **kwargs)


logging.setLoggerClass(_VerboseLogger)
log: _VerboseLogger = logging.getLogger(__name__)  # type: ignore[assignment]

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, Send, RetryPolicy
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


def _steps_reducer(left: list, right: list) -> list:
    """Like operator.add, but an empty right-hand side resets the list (for classify reset)."""
    if not right:
        return []
    return (left or []) + right


# Shared state passed between graph nodes. `stage_index` advances through `plan.stages`.
class RunState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]  # full conversation history across turns
    goal: str  # current run goal, extracted from the latest human message
    plan: Plan  # plan contains stages; each stage is a list of steps executed in parallel
    stage_index: int  # the next stage in the plan to execute
    completed_steps: Annotated[List[Tuple[str, str]], _steps_reducer]
    response: str  # final agent response for the current turn


# Per-step state for the worker subgraph; carries the overall goal plus prior results for context.
class StepState(TypedDict):
    goal: str  # original agent goal
    step: str  # specific step goal
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
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=2048)
    planner = llm.with_structured_output(Plan)
    executor = create_agent(llm, TOOLS)

    async def classify(
        state: RunState,
    ) -> Command[Literal["make_plan", "final_answer"]]:
        """Classify the latest user message as NEWTASK or FOLLOWUP and route accordingly.

        Also resets per-run scratch (completed_steps, goal) so previous run data
        does not leak into the current run.
        """
        user_msg = state["messages"][-1].content
        log.verbose("classify: message=%r", user_msg)

        # First-ever message is always a new task.
        if len(state["messages"]) == 1:
            log.verbose("classify: first message -> NEWTASK")
            return Command(
                update={"completed_steps": [], "goal": "", "plan": Plan(stages=[]), "stage_index": 0},
                goto="make_plan",
            )

        decision = await llm.ainvoke(
            list(state["messages"][:-1])
            + [
                HumanMessage(
                    content=(
                        "The user sent a new message. Reply with exactly one word:\n"
                        "  FOLLOWUP  — if it is a question or comment about the previous result\n"
                        "  NEWTASK   — if it is a new independent task to execute\n\n"
                        f"Message: {user_msg}"
                    )
                )
            ]
        )
        verdict = decision.content.strip().upper()
        log.verbose("classify: verdict=%r", verdict)
        goto: Literal["make_plan", "final_answer"] = (
            "make_plan" if "NEWTASK" in verdict else "final_answer"
        )
        return Command(
            update={"completed_steps": [], "goal": "", "plan": Plan(stages=[]), "stage_index": 0},
            goto=goto,
        )

    async def make_plan(state: RunState) -> dict:
        goal = state["messages"][-1].content
        prompt = (
            "Break this goal into 1-5 stages with 1-5 steps each.\n"
            "Group steps into the same stage only when they are truly independent.\n"
            "Otherwise put them in separate stages so later ones can use earlier results.\n\n"
            f"Goal: {goal}"
        )
        plan: Plan = await planner.ainvoke(prompt)
        log.verbose(
            "make_plan: produced %d stage(s): %s",
            len(plan.stages),
            [[s for s in stage.steps] for stage in plan.stages],
        )
        return {"goal": goal, "plan": plan, "stage_index": 0, "completed_steps": []}

    async def execute_plan_step(step_state: StepState) -> dict:
        # node policy.max_attempts defines how many times langgraph retries the node
        # if random() > 0.9:
        #     raise Exception("execute_plan_step exception")

        context = "\n".join(f"- {s}: {r[:200]}" for s, r in step_state["past_steps"]) or "(none)"
        prompt = (
            f"Overall goal: {step_state['goal']}\n"
            f"Results from previous stages:\n{context}\n\n"
            f"Do this single step using the available tools: {step_state['step']}\n"
            "Report only what you did and what you observed."
        )
        result = await executor.ainvoke({"messages": [("user", prompt)]})
        output = result["messages"][-1].content
        return {"completed_steps": [(step_state["step"], output)]}

    async def pop_stage(state: RunState) -> dict:
        return {"stage_index": state["stage_index"] + 1}

    async def final_answer(state: RunState) -> dict:
        history = "\n".join(f"- {s}\n  -> {r}" for s, r in state["completed_steps"])
        # For follow-up turns history may be empty; answer from conversation context instead.
        if history:
            prompt = (
                f"Goal: {state['goal']}\n\nWhat was done:\n{history}\n\n"
                "Write a short final answer for the user."
            )
        else:
            prompt = "Answer the user's latest message based on the conversation so far."
        msg = await llm.ainvoke(state["messages"] + [HumanMessage(content=prompt)])
        log.verbose("final_answer: %r", msg.content[:120])
        return {"response": msg.content, "messages": [AIMessage(content=msg.content)]}

    def dispatch(state: RunState) -> list[Send]:
        plan = state["plan"]
        idx = state["stage_index"]
        if idx >= len(plan.stages):
            return [Send("final_answer", state)]

        current_stage = plan.stages[idx]
        sends = []
        for step in current_stage.steps:
            step_state: StepState = {
                "goal": state["goal"],
                "step": step,
                "past_steps": state["completed_steps"],
            }
            sends.append(Send("execute_plan_step", step_state))
        return sends

    graph = StateGraph(RunState)
    graph.add_node("classify", classify)
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
    graph.set_entry_point("classify")
    graph.add_conditional_edges("make_plan", dispatch, ["execute_plan_step", "final_answer"])
    graph.add_edge("execute_plan_step", "pop_stage")
    graph.add_conditional_edges("pop_stage", dispatch, ["execute_plan_step", "final_answer"])
    graph.add_edge("final_answer", END)
    return graph


async def run_chat(thread_id: str) -> None:
    """Interactive REPL. Each user message triggers a full graph run from START to END."""
    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver:
        app = build_graph().compile(checkpointer=saver)
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

        snapshot = await app.aget_state(config)
        if snapshot and snapshot.values:
            print(f"Resuming session (thread_id={thread_id}). Type your follow-up.\n")

        print("Type your message and press Enter. Ctrl-C or Ctrl-D to quit.\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
            if not user_input:
                continue

            final = await app.ainvoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )
            answer = final.get("response", "")
            print(f"\nAgent: {answer}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan/execute chatbot agent.")
    parser.add_argument(
        "--thread-id",
        help="Session ID. Omit to start a new session; provide to resume an existing one.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args()


async def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY before running.")

    args = parse_args()

    logging.basicConfig(
        level=VERBOSE if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    thread_id = args.thread_id or str(uuid.uuid4())
    print(f"Session thread_id={thread_id}")
    print(f"To resume later: python -m src.main --thread-id {thread_id}\n")

    await run_chat(thread_id)


if __name__ == "__main__":
    asyncio.run(main())
