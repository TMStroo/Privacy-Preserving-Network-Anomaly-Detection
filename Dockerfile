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

# pytest is a runtime dependency of the verification story, not a test-only
# extra: the documented `docker compose run --rm tests` command runs it from
# this image, so it has to be present or that command fails immediately.
RUN python -m pip install pytest

COPY configs ./configs
COPY tests ./tests
COPY fixtures ./fixtures
COPY scripts ./scripts
# tools/ holds the run verifier and the results indexer. The verifier is
# imported by tests/test_verify_run.py and run by the CI smoke job, so an
# image without it fails its own test suite.
COPY tools ./tools

# Mount a dataset cache here:  docker run -v /path/to/raw:/data/raw ...
ENV DRIFTGUARD_RAW_DIR=/data/raw
VOLUME ["/data/raw", "/workspace/results"]


# The experiment tracker records the commit a run came from. The image
# deliberately does NOT copy .git, because that would put the remote URL and
# anything else in the repository metadata into a distributable artifact. Pass
# the commit explicitly instead:
#   docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t driftguard .
ARG GIT_COMMIT=unknown
ENV DRIFTGUARD_GIT_COMMIT=${GIT_COMMIT}

CMD ["python", "-m", "driftguard", "--help"]
