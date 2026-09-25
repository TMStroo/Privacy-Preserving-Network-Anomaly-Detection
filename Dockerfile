# Reproducible research environment for DriftGuard.
#
# The image carries the code and its pinned dependencies, not the datasets.
# The research datasets are far too large to bake in, so they are mounted or
# fetched explicitly with `python -m driftguard data fetch`.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# git is used by the experiment tracker to record the commit a run came from.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip && python -m pip install -e .

COPY configs ./configs
COPY tests ./tests
COPY fixtures ./fixtures
COPY scripts ./scripts

# Mount a dataset cache here:  docker run -v /path/to/raw:/data/raw ...
ENV DRIFTGUARD_RAW_DIR=/data/raw
VOLUME ["/data/raw", "/workspace/results"]

CMD ["python", "-m", "driftguard", "--help"]
