# The stack: `make up train serve demo`, then `make down`.
# The offline measurements need no containers: `make experiments`.

COMPOSE := docker compose
RUN     := $(COMPOSE) run --rm tools
# The stack's dataset: seed 7, prices up 50% from day 78 so the drift alert has
# something to find while the demo replays into it.
DATASET := --seed 7 --drift-day 78 --drift-inflation 1.5

.PHONY: up train serve demo down test lint typecheck check experiments e2e

up:  ## build, start the infrastructure, generate data, load the online store
	$(COMPOSE) --profile serve --profile jobs build
	$(COMPOSE) up -d --wait kafka redis postgres mlflow prometheus
	$(RUN) python -m rtml.serve.bootstrap $(DATASET)

train:  ## train v1 (champion) and v2 (challenger) and register them in MLflow
	$(RUN) python -m rtml.train

serve:  ## the API, the stream processor and the drift monitor
	$(COMPOSE) --profile serve up -d --wait api processor monitor

demo:  ## replay, measure latency and parity, gate, promote, drift alert, refuse
	$(RUN) python -m rtml.demo

down:  ## stop everything and delete the volumes
	$(COMPOSE) --profile serve --profile jobs down -v --remove-orphans

experiments:  ## every offline measurement in MEASUREMENTS.md
	uv run python -m rtml.experiments.offline all
	uv run python -m rtml.experiments.drift_eval

test:
	uv run pytest

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

typecheck:
	uv run mypy

check: lint typecheck test
