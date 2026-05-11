# Camping

Plan/execute agent loop built with LangGraph. The agent breaks a goal into steps, runs each step with a ReAct sub-agent, then writes a final answer.

## Requirements

- Python 3.10 or newer
- An Anthropic API key from https://console.anthropic.com/

Check your Python:

```
python3 --version
```

## Install

Clone and enter the repo, then:

```
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

## Configure the API key

```
cp .env.example .env
```

Open `.env` and set:

```
ANTHROPIC_API_KEY=sk-ant-...
```

The script auto-loads `.env` via `python-dotenv`, so no `export` needed.

## Run

Run as a module from the repo root:

```
.venv/bin/python -m src.main
```

Expected output: a printed final answer, plus a new file `/tmp/example_title.txt` containing the page title from `https://example.com`.

## How it works

The graph has three nodes that loop until the plan is empty:

- `plan`: LLM turns the goal into a short list of steps
- `execute`: a ReAct sub-agent does the next step using the tools
- `replan`: stops when no steps remain and writes the final answer

Two tools are wired in:

- `fetch_url(url)`: HTTP GET, returns body text
- `write_file(path, content)`: writes text to a file

## Project layout

```
/camping
 /src
  /main.py
 /requirements.txt
 /.env.example
 /README.md
```

## Customize

Change the `goal` string at the bottom of `src/main.py` to try other tasks. Add new tools by writing a function decorated with `@tool` and appending it to the `TOOLS` list. Swap in MCP tools later by replacing `TOOLS` with the output of `MultiServerMCPClient.get_tools()`.

## Troubleshooting

`Set ANTHROPIC_API_KEY before running.`
The `.env` file is missing or the key is empty. Re-check the value in `.env`.

`ModuleNotFoundError: No module named 'langgraph'`
You ran the system Python instead of the venv. Use `.venv/bin/python` explicitly, or activate with `source .venv/bin/activate`.

`urllib.error.URLError`
Network is blocked or the target site is unreachable. Try a different URL in the `goal`.
