"""Plan/execute agent with a thin app layer on top of LangGraph.

Architecture:
  App layer  — owns conversation history and thread state in memory,
               classifies each user message, decides what to run
               (graph, direct LLM call, custom logic), records every run.
  Graph layer — pure plan/execute state machine; START to END on every
               invocation; receives goal + context, returns response.

Graph nodes:
  make_plan          -> LLM produces sequential stages of parallel steps
  execute_plan_step  -> ReAct sub-agent executes ONE step (fan-out via Send)
  pop_stage          -> advance to the next stage
  final_answer       -> summarise the completed run for the user

LangGraph checkpoints (SQLite) are kept for observability and mid-run
durability; they are NOT the source of truth for conversation history.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Dict, List, Tuple, TypedDict

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
from langgraph.types import Send, RetryPolicy
from pydantic import BaseModel, Field

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DB = REPO_ROOT / "data" / "checkpoints.db"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# App-layer data structures
# ---------------------------------------------------------------------------

@dataclass
class GraphExecution:
    """Record of one graph invocation within a single app turn."""
    graph_thread_id: str
    response: str


@dataclass
class UserMessage:
    """A user turn in the thread history."""
    content: str


@dataclass
class AppMessage:
    """An app turn in the thread history."""
    classification: str                                    # "NEWTASK" | "FOLLOWUP"
    response: str
    started_at: datetime
    elapsed_ms: int
    graph_executions: List[GraphExecution] = field(default_factory=list)


HistoryEntry = UserMessage | AppMessage


class ThreadState:
    """In-memory conversation state for one thread.

    history is a flat alternating list: [UserMessage, AppMessage, UserMessage, AppMessage, ...]
    """

    def __init__(self) -> None:
        self.history: List[HistoryEntry] = []

    def to_messages(self, max_turns: int = 10) -> List[BaseMessage]:
        """Flatten recent history into a LangChain message list for LLM context."""
        # Take the last max_turns pairs (each pair = 2 entries).
        recent = self.history[-(max_turns * 2):]
        msgs: List[BaseMessage] = []
        for entry in recent:
            if isinstance(entry, UserMessage):
                msgs.append(HumanMessage(content=entry.content))
            else:
                msgs.append(AIMessage(content=entry.response))
        return msgs


# ---------------------------------------------------------------------------
# Graph types
# ---------------------------------------------------------------------------

def _steps_reducer(left: list, right: list) -> list:
    """Append new steps, or reset to empty when right is []."""
    if not right:
        return []
    return (left or []) + right


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


class RunState(TypedDict):
    """State owned by the graph for the duration of a single run."""
    messages: List[BaseMessage]                            # context built by app; not accumulated
    goal: str
    plan: Plan
    stage_index: int
    completed_steps: Annotated[List[Tuple[str, str]], _steps_reducer]
    response: str


class StepState(TypedDict):
    """Per-step state for the parallel worker fan-out."""
    goal: str
    step: str
    past_steps: List[Tuple[str, str]]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(llm: ChatAnthropic) -> StateGraph:
    planner = llm.with_structured_output(Plan)
    executor = create_agent(llm, TOOLS)

    async def make_plan(state: RunState) -> dict:
        goal = state["goal"]
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
        return {"plan": plan, "stage_index": 0, "completed_steps": []}

    async def execute_plan_step(step_state: StepState) -> dict:
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
        if history:
            prompt = (
                f"Goal: {state['goal']}\n\nWhat was done:\n{history}\n\n"
                "Write a short final answer for the user."
            )
        else:
            prompt = "Answer the user's latest message based on the conversation so far."
        msg = await llm.ainvoke(state["messages"] + [HumanMessage(content=prompt)])
        log.verbose("final_answer: %r", msg.content[:120])
        return {"response": msg.content}

    def dispatch(state: RunState) -> list[Send]:
        plan = state["plan"]
        idx = state["stage_index"]
        if idx >= len(plan.stages):
            return [Send("final_answer", state)]
        current_stage = plan.stages[idx]
        return [
            Send("execute_plan_step", StepState(
                goal=state["goal"],
                step=step,
                past_steps=state["completed_steps"],
            ))
            for step in current_stage.steps
        ]

    graph = StateGraph(RunState)
    graph.add_node("make_plan", make_plan)
    graph.add_node(
        "execute_plan_step",
        execute_plan_step,
        retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0, backoff_factor=2.0),
    )
    graph.add_node("pop_stage", pop_stage)
    graph.add_node("final_answer", final_answer)
    graph.set_entry_point("make_plan")
    graph.add_conditional_edges("make_plan", dispatch, ["execute_plan_step", "final_answer"])
    graph.add_edge("execute_plan_step", "pop_stage")
    graph.add_conditional_edges("pop_stage", dispatch, ["execute_plan_step", "final_answer"])
    graph.add_edge("final_answer", END)
    return graph


# ---------------------------------------------------------------------------
# App layer
# ---------------------------------------------------------------------------

async def _classify(user_input: str, thread: ThreadState, llm: ChatAnthropic) -> str:
    """Return 'NEWTASK' or 'FOLLOWUP' for the given user message."""
    if not thread.history:
        return "NEWTASK"

    # Build context from the last 3 pairs (6 entries) of history.
    recent_pairs: List[str] = []
    entries = thread.history[-6:]
    for i in range(0, len(entries) - 1, 2):
        if isinstance(entries[i], UserMessage) and isinstance(entries[i + 1], AppMessage):
            recent_pairs.append(f"User: {entries[i].content}\nAssistant: {entries[i + 1].response}")
    recent = "\n".join(recent_pairs)
    decision = await llm.ainvoke([
        HumanMessage(content=(
            f"Conversation so far:\n{recent}\n\n"
            f"New message: {user_input}\n\n"
            "Reply with exactly one word:\n"
            "  FOLLOWUP  — question or comment about the previous result\n"
            "  NEWTASK   — new independent task to execute"
        ))
    ])
    verdict = decision.content.strip().upper()
    return "NEWTASK" if "NEWTASK" in verdict else "FOLLOWUP"


async def run_chat(thread_id: str) -> None:
    """REPL loop. The app classifies each message and dispatches accordingly."""
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=2048)
    threads: Dict[str, ThreadState] = {}
    thread = threads.setdefault(thread_id, ThreadState())

    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver:
        graph_app = build_graph(llm).compile(checkpointer=saver)

        print("Type your message and press Enter. Ctrl-C or Ctrl-D to quit.\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
            if not user_input:
                continue

            started_at = datetime.now()
            classification = await _classify(user_input, thread, llm)
            log.verbose("app: classification=%r for %r", classification, user_input[:60])

            graph_executions: List[GraphExecution] = []

            if classification == "NEWTASK":
                graph_thread_id = str(uuid.uuid4())
                context_messages = thread.to_messages()
                result = await graph_app.ainvoke(
                    {
                        "goal": user_input,
                        "messages": context_messages + [HumanMessage(content=user_input)],
                        "plan": Plan(stages=[]),
                        "stage_index": 0,
                        "completed_steps": [],
                        "response": "",
                    },
                    config={
                        "configurable": {"thread_id": graph_thread_id},
                        "recursion_limit": 50,
                    },
                )
                response = result["response"]
                graph_executions.append(GraphExecution(
                    graph_thread_id=graph_thread_id,
                    response=response,
                ))
            else:
                # FOLLOWUP: answer directly from conversation context, no graph needed.
                context_messages = thread.to_messages() + [HumanMessage(content=user_input)]
                msg = await llm.ainvoke(
                    context_messages
                    + [HumanMessage(content="Answer the user's latest message based on the conversation above.")]
                )
                response = msg.content

            elapsed_ms = int((datetime.now() - started_at).total_seconds() * 1000)
            thread.history.append(UserMessage(content=user_input))
            thread.history.append(AppMessage(
                classification=classification,
                response=response,
                started_at=started_at,
                elapsed_ms=elapsed_ms,
                graph_executions=graph_executions,
            ))
            print(f"\nAgent: {response}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan/execute chatbot agent.")
    parser.add_argument(
        "--thread-id",
        help="Session ID. Omit to start a new session; provide to resume an existing one.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args()


async def main() -> None:
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
