PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: install run resume dump dump-all list clean clean-data help

help:
	@echo "make install              Create venv and install deps"
	@echo "make run                  Start a new run with the default goal"
	@echo "make run GOAL='...'       Start a new run with a custom goal"
	@echo "make resume THREAD=...    Resume an existing thread"
	@echo "make list                 List all checkpointed thread_ids"
	@echo "make dump THREAD=...      Dump latest state for a thread as JSON"
	@echo "make dump-all THREAD=...  Dump full checkpoint history for a thread"
	@echo "make clean-data           Delete all checkpoints"
	@echo "make clean                Delete venv and __pycache__"

install:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
ifdef GOAL
	$(PY) -m src.main --goal "$(GOAL)"
else
	$(PY) -m src.main
endif

resume:
ifndef THREAD
	@echo "Usage: make resume THREAD=<thread_id>"; exit 1
endif
	$(PY) -m src.main --resume $(THREAD)

list:
	$(PY) scripts/dump.py

dump:
ifndef THREAD
	@echo "Usage: make dump THREAD=<thread_id>"; exit 1
endif
	$(PY) scripts/dump.py $(THREAD)

dump-all:
ifndef THREAD
	@echo "Usage: make dump-all THREAD=<thread_id>"; exit 1
endif
	$(PY) scripts/dump.py $(THREAD) --all

clean-data:
	rm -rf data/

clean:
	rm -rf .venv
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
