FROM python:3.12-slim

RUN addgroup --system app && adduser --system --ingroup app app

WORKDIR /app
COPY pyproject.toml requirements.lock ./

# Install build tools for implicit (ALS) + OpenMP runtime, then cleanup
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ libgomp1 && \
    pip install --no-cache-dir -r requirements.lock && \
    apt-get purge -y gcc g++ && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Model storage directory with restricted permissions (not /tmp)
RUN mkdir -p /app/data/models && chown app:app /app/data/models

COPY src/ ./src/
COPY scheduler.sh ./scheduler.sh

USER app
ENTRYPOINT ["python", "-m", "src.cli"]
