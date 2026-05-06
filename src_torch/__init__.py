"""PyTorch backend for the recommendation pipeline.

Vectorised mirror of ``src/``: the hybrid scorer, evaluation kernel,
Thompson Sampling bandit and offline warmup all run as batched tensor
operations on the configured device (CPU or CUDA). Metric values
agree with ``src/`` on the same dataset within tolerance.

Selected at runtime via ``--backend=torch`` on the CLI.
"""
