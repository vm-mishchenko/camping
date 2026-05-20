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
from enum import Enum
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Optional, Tuple, TypedDict

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
import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
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
    """Fetch the given URL and return status, headers, metrics, and body as text (body truncated to 4000 chars)."""
    import time
    req = urllib.request.Request(url, headers={"User-Agent": "plan-execute-demo/1.0"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=15) as resp:
        status = resp.status
        headers = dict(resp.headers)
        body = resp.read().decode("utf-8", errors="replace")
    latency_ms = round((time.perf_counter() - t0) * 1000)
    metrics = f"latency_ms={latency_ms} body_bytes={len(body.encode())}"
    return f"Status: {status}\nMetrics: {metrics}\nHeaders: {headers}\n\nBody:\n{body[:4000]}"


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

class Classification(str, Enum):
    NEWTASK = "NEWTASK"
    FOLLOWUP = "FOLLOWUP"


class ClassificationResult(BaseModel):
    classification: Classification = Field(
        description=(
            f"{Classification.NEWTASK} — new independent task to execute; "
            f"{Classification.FOLLOWUP} — question or comment about the previous result"
        )
    )


@dataclass
class GraphExecution:
    """Record of one graph invocation within a single app turn."""
    graph_thread_id: str
    response: str


@dataclass
class UserMessage:
    """A user turn in the thread history."""
    content: str


TaskStatus = Literal["running", "completed", "stopped", "failed"]


@dataclass
class TaskAppMessage:
    """An app turn that ran the graph (Classification.NEWTASK)."""
    goal: str
    response: str
    started_at: datetime
    elapsed_ms: int
    status: TaskStatus = "running"
    graph_executions: List[GraphExecution] = field(default_factory=list)


@dataclass
class FollowupAppMessage:
    """An app turn answered directly from context (Classification.FOLLOWUP)."""
    response: str
    started_at: datetime
    elapsed_ms: int


AppMessage = TaskAppMessage | FollowupAppMessage
HistoryEntry = UserMessage | AppMessage


@dataclass
class CurrentRun:
    """The single in-flight run on the thread, whether running or paused.

    The full TaskAppMessage lives in thread.history; this struct is just a
    pointer (graph_thread_id) plus phase metadata (stopped_at). All fields are
    scalars so ThreadState stays fully serializable.

    Phase is encoded by stopped_at:
        is_running: stopped_at is None. The graph is actively executing.
        is_paused:  stopped_at is set. The graph was halted by /stop and is
                    resumable via /continue.
    """
    graph_thread_id: str
    stopped_at: Optional[datetime] = None

    @property
    def is_running(self) -> bool:
        return self.stopped_at is None

    @property
    def is_paused(self) -> bool:
        return self.stopped_at is not None


class ThreadState:
    """In-memory conversation state for one thread.

    history is a flat alternating list: [UserMessage, AppMessage, UserMessage, AppMessage, ...]
    current_run holds a pointer to the one in-flight run (running or paused), or
    None when the thread is idle. All fields are intended to be serializable so
    a thread can later be saved and resumed from disk.
    """

    def __init__(self) -> None:
        self.history: List[HistoryEntry] = []
        self.current_run: Optional[CurrentRun] = None

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

    def find_task_message(self, graph_thread_id: str) -> TaskAppMessage:
        """Look up the TaskAppMessage owning a given LangGraph thread id.

        O(N) over history; v1 history is small enough that this is fine.
        """
        for entry in reversed(self.history):
            if isinstance(entry, TaskAppMessage):
                if any(ge.graph_thread_id == graph_thread_id for ge in entry.graph_executions):
                    return entry
        raise LookupError(f"no TaskAppMessage owns graph_thread_id={graph_thread_id!r}")

    def current_task_message(self) -> Optional[TaskAppMessage]:
        """The TaskAppMessage of the current run (running or paused), if any."""
        if self.current_run is None:
            return None
        return self.find_task_message(self.current_run.graph_thread_id)


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

class Reporter:
    """Progress sink for one user turn. Knows nothing about the domain.

    Can be swapped for a WebSocket emitter, a logging sink, or a no-op
    without changing any call sites.
    """

    _DIM = "\033[2m"
    _RESET = "\033[0m"

    def print(self, message: str) -> None:
        print(f"{self._DIM}  {message}{self._RESET}")


async def _run_graph(
    graph_app,
    inputs: Optional[dict],
    config: dict,
    reporter: Reporter,
) -> str:
    """Stream a graph run and forward progress messages to reporter. Returns the final response.

    Pass ``inputs=None`` to resume from the latest LangGraph checkpoint for the
    ``thread_id`` in ``config``.
    """
    response = ""
    async for chunk in graph_app.astream(inputs, config=config, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "make_plan":
                plan = update.get("plan")
                if plan and plan.stages:
                    total = sum(len(s.steps) for s in plan.stages)
                    reporter.print(f"Plan: {len(plan.stages)} stage(s), {total} step(s)")
                    for i, stage in enumerate(plan.stages):
                        if len(stage.steps) == 1:
                            reporter.print(f"  Stage {i + 1}: {stage.steps[0]}")
                        else:
                            reporter.print(f"  Stage {i + 1} (parallel):")
                            for step in stage.steps:
                                reporter.print(f"    - {step}")
            elif node == "execute_plan_step":
                for step, _ in update.get("completed_steps", []):
                    reporter.print(f"  done: {step[:100]}")
            elif node == "final_answer":
                reporter.print("Composing answer...")
                response = update.get("response", "")
    return response


async def _classify(user_input: str, thread: ThreadState, llm: ChatAnthropic) -> Classification:
    """Classify the user message as NEWTASK or FOLLOWUP."""
    if not thread.history:
        return Classification.NEWTASK

    # Build context from the last 3 pairs (6 entries) of history.
    recent_pairs: List[str] = []
    entries = thread.history[-6:]
    for i in range(0, len(entries) - 1, 2):
        if isinstance(entries[i], UserMessage) and isinstance(entries[i + 1], AppMessage):
            recent_pairs.append(f"User: {entries[i].content}\nAssistant: {entries[i + 1].response}")
    recent = "\n".join(recent_pairs)
    classifier = llm.with_structured_output(ClassificationResult)
    result: ClassificationResult = await classifier.ainvoke([
        HumanMessage(content=(
            f"Conversation so far:\n{recent}\n\n"
            f"Classify this new message: {user_input}"
        ))
    ])
    return result.classification


STOP_CMD = "/stop"
CONTINUE_CMD = "/continue"
_EOF_TOKEN = "\x00EOF"  # private sentinel; cannot collide with user input


async def run_chat(thread_id: str) -> None:
    """REPL with a concurrent input loop and a background graph task.

    The input loop reads stdin via ``asyncio.to_thread`` and pushes lines onto a
    queue. The dispatch loop owns lifecycle: it spawns the graph as a background
    task on NEWTASK, accepts ``/stop`` while the graph is running, and resumes a
    paused graph on ``/continue`` via the LangGraph checkpointer.
    """
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=2048)
    threads: Dict[str, ThreadState] = {}
    thread = threads.setdefault(thread_id, ThreadState())

    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    # Allow-list our own state types so the checkpointer doesn't warn
    # about deserializing "unregistered" __main__.Plan / __main__.Stage.
    # Other types still fall back to LangGraph's default safe set.
    serde = JsonPlusSerializer(allowed_msgpack_modules=[Plan, Stage])
    async with aiosqlite.connect(str(CHECKPOINT_DB)) as conn:
        saver = AsyncSqliteSaver(conn, serde=serde)
        graph_app = build_graph(llm).compile(checkpointer=saver)

        # Queue of user input lines: input_loop reads each line typed and 
        # pushes it here for the dispatch loop to consume.
        input_queue: "asyncio.Queue[str]" = asyncio.Queue()
        reporter = Reporter()
        # Cancellation handle for the in-flight asyncio coroutine. The dispatch
        # loop reads `thread.current_run` (state) to decide whether a graph is
        # running; `graph_task` exists solely so we can call `.cancel()` on the
        # coroutine when /stop arrives. Not on ThreadState because asyncio.Task
        # is not serializable.
        graph_task: Optional[asyncio.Task] = None

        def show_prompt() -> None:
            """Print the 'You: ' prompt at the current cursor position.

            Called after each turn ends so the prompt sits below the agent's
            output instead of being buried by it. Not called before/during a
            running graph: the user can still type while a graph is running
            (e.g. /stop), but they type without a visible prompt until the
            slice ends.
            """
            print("You: ", end="", flush=True)

        async def input_loop() -> None:
            while True:
                try:
                    # No prompt arg: show_prompt() handles prompt printing on
                    # the dispatch side so it appears at the right time.
                    line = await asyncio.to_thread(input)
                except (EOFError, KeyboardInterrupt):
                    await input_queue.put(_EOF_TOKEN)
                    return
                await input_queue.put(line.strip())

        async def run_graph_task(
            task_msg: TaskAppMessage,
            inputs: Optional[dict],
            graph_thread_id: str,
        ) -> None:
            slice_started = datetime.now()
            try:
                response = await _run_graph(
                    graph_app,
                    inputs=inputs,
                    config={
                        "configurable": {"thread_id": graph_thread_id},
                        "recursion_limit": 50,
                    },
                    reporter=reporter,
                )
                task_msg.graph_executions[-1].response = response
                task_msg.response = response
                task_msg.status = "completed"
                task_msg.elapsed_ms = int((datetime.now() - slice_started).total_seconds() * 1000)
                thread.current_run = None
                reporter.print(f"({task_msg.elapsed_ms}ms)")
                print(f"\nAgent: {response}\n")
                show_prompt()
            except asyncio.CancelledError:
                # /stop path: flip status; stop_running_graph stamps stopped_at
                # on thread.current_run and the dispatch loop re-shows the
                # prompt after stop_running_graph returns.
                task_msg.status = "stopped"
                task_msg.elapsed_ms = int((datetime.now() - slice_started).total_seconds() * 1000)
            except Exception as e:
                task_msg.status = "failed"
                task_msg.elapsed_ms = int((datetime.now() - slice_started).total_seconds() * 1000)
                thread.current_run = None
                log.exception("graph failed")
                print(f"\nAgent: [graph failed: {e}]\n")
                show_prompt()

        def start_new_task(user_input: str) -> None:
            nonlocal graph_task
            graph_thread_id = str(uuid.uuid4())
            started_at = datetime.now()
            context_messages = thread.to_messages()
            task_msg = TaskAppMessage(
                goal=user_input,
                response="",
                started_at=started_at,
                elapsed_ms=0,
                status="running",
                graph_executions=[GraphExecution(
                    graph_thread_id=graph_thread_id,
                    response="",
                )],
            )
            thread.history.append(UserMessage(content=user_input))
            thread.history.append(task_msg)
            thread.current_run = CurrentRun(graph_thread_id=graph_thread_id)
            reporter.print("Starting a new task...")
            graph_task = asyncio.create_task(run_graph_task(
                task_msg,
                inputs={
                    "goal": user_input,
                    "messages": context_messages + [HumanMessage(content=user_input)],
                    "plan": Plan(stages=[]),
                    "stage_index": 0,
                    "completed_steps": [],
                    "response": "",
                },
                graph_thread_id=graph_thread_id,
            ))

        def resume_paused() -> None:
            nonlocal graph_task
            current = thread.current_run
            assert current is not None and current.is_paused
            try:
                task_msg = thread.find_task_message(current.graph_thread_id)
            except LookupError as e:
                log.error("resume: %s", e)
                reporter.print("paused task is missing from history; cannot resume.")
                thread.current_run = None
                return
            task_msg.graph_executions.append(GraphExecution(
                graph_thread_id=current.graph_thread_id,
                response="",
            ))
            task_msg.status = "running"
            task_msg.started_at = datetime.now()
            current.stopped_at = None  # back to running phase
            reporter.print(f"resuming: {task_msg.goal!r}")
            graph_task = asyncio.create_task(run_graph_task(
                task_msg,
                inputs=None,
                graph_thread_id=current.graph_thread_id,
            ))

        async def stop_running_graph() -> None:
            nonlocal graph_task
            if graph_task is None or graph_task.done():
                return
            current_task = graph_task
            task_msg = thread.current_task_message()
            current_task.cancel()
            try:
                await current_task
            except asyncio.CancelledError:
                pass
            # If the cancel landed cleanly the run_graph_task handler set
            # status="stopped". If it raced with a natural completion/failure,
            # run_graph_task already cleared thread.current_run for us.
            if (
                thread.current_run is not None
                and task_msg is not None
                and task_msg.status == "stopped"
            ):
                thread.current_run.stopped_at = datetime.now()
                reporter.print(
                    f"paused after {task_msg.elapsed_ms}ms. type /continue to resume."
                )
            graph_task = None

        # FOLLOWUP path: answer the user's message directly from the LLM using
        # prior conversation as context. No graph is involved, so this runs
        # synchronously inside the dispatch loop.
        async def answer_followup(user_input: str) -> None:
            started_at = datetime.now()
            context_messages = thread.to_messages()
            thread.history.append(UserMessage(content=user_input))
            reporter.print("Following up on previous answer...")
            reporter.print("Answering from context...")
            msg = await llm.ainvoke(
                context_messages
                + [
                    HumanMessage(content=user_input),
                    HumanMessage(content="Answer the user's latest message based on the conversation above."),
                ]
            )
            response = msg.content
            elapsed_ms = int((datetime.now() - started_at).total_seconds() * 1000)
            thread.history.append(FollowupAppMessage(
                response=response,
                started_at=started_at,
                elapsed_ms=elapsed_ms,
            ))
            reporter.print(f"({elapsed_ms}ms)")
            print(f"\nAgent: {response}\n")

        # Handle one user line. Branches on slash commands and the current run's
        # phase (running / paused / idle). Single entry point for the dispatch loop.
        async def dispatch(line: str) -> None:
            is_running = thread.current_run is not None and thread.current_run.is_running
            is_paused = thread.current_run is not None and thread.current_run.is_paused

            if line == STOP_CMD:
                if is_running:
                    await stop_running_graph()
                else:
                    reporter.print("nothing to stop.")
                return

            if line == CONTINUE_CMD:
                if is_running:
                    reporter.print("graph is running. type /stop to halt it.")
                elif is_paused:
                    resume_paused()
                else:
                    reporter.print("nothing to continue.")
                return

            # Regular message. While a graph is running, ignore everything but /stop.
            if is_running:
                reporter.print("graph is running. type /stop to halt it.")
                return

            classification = await _classify(line, thread, llm)
            if classification == Classification.NEWTASK:
                if is_paused:
                    abandoned = thread.current_task_message()
                    goal = abandoned.goal if abandoned is not None else "?"
                    reporter.print(f"abandoned paused task: {goal!r}")
                    thread.current_run = None
                start_new_task(line)
            else:
                await answer_followup(line)

        print(
            "Type your message and press Enter. "
            "Use /stop and /continue to control a running task. "
            "Ctrl-C or Ctrl-D to quit.\n"
        )
        show_prompt()

        input_task = asyncio.create_task(input_loop())
        try:
            while True:
                # Block until the user types something (or the input_loop signals EOF).
                line = await input_queue.get()
                if line == _EOF_TOKEN:
                    if graph_task is not None and not graph_task.done():
                        graph_task.cancel()
                        try:
                            await graph_task
                        except asyncio.CancelledError:
                            pass
                    print("\nGoodbye.")
                    return
                if line:
                    await dispatch(line)
                # Re-arm the prompt unless a graph is now running; in that case
                # run_graph_task will print the prompt itself when its slice
                # ends (success/failure paths) or this same check will fire on
                # the next iteration after stop_running_graph stamps paused.
                if thread.current_run is None or thread.current_run.is_paused:
                    show_prompt()
        finally:
            input_task.cancel()
            try:
                await input_task
            except asyncio.CancelledError:
                pass


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
