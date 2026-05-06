FROM python:3.12-slim

RUN addgroup --system app && adduser --system --ingroup app app

WORKDIR /app
# Lock first so dep changes are the only thing that bust the install
# layers; pyproject.toml is for pytest config + entrypoint and only
# matters at runtime, so it ships alongside the source below.
COPY requirements.lock ./

# Install build tools for implicit (ALS) + OpenMP runtime, then cleanup
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ libgomp1 && \
    pip install --no-cache-dir -r requirements.lock && \
    apt-get purge -y gcc g++ && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Optional: vectorised torch backend (--backend=torch). Pytest tags
# along so the suite under ``tests/src_torch/`` is runnable inside the
# image. Default index is the CPU wheel (~190 MB) so the benchmark
# image stays self-contained. For CUDA, override with
# ``--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128``
# (or whichever flavour matches the host driver). Set
# ``--build-arg INSTALL_TORCH=0`` to skip both and keep the image lean.
ARG INSTALL_TORCH=1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN if [ "$INSTALL_TORCH" = "1" ]; then \
      pip install --no-cache-dir --index-url "$TORCH_INDEX_URL" "torch>=2.2,<3" && \
      pip install --no-cache-dir "pytest>=8.0,<9" ; \
    fi

# Model storage directory with restricted permissions (not /tmp)
RUN mkdir -p /app/data/models && chown app:app /app/data/models

COPY pyproject.toml ./
COPY src/ ./src/
COPY src_torch/ ./src_torch/
COPY tests/ ./tests/
COPY scheduler.sh ./scheduler.sh

USER app
ENTRYPOINT ["python", "-m", "src.cli"]
