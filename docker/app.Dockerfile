# One image for the API, the stream processor, the monitor and the batch jobs.
FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 git \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev
ENV RTML_ROOT=/app RTML_DATA=/app/data/stack RTML_RESULTS=/app/results PYTHONUNBUFFERED=1
