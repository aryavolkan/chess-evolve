# syntax=docker/dockerfile:1
# Reproducible training image for the Rust CPU backend: the Python harness plus
# the three PyO3 crates built from source. No Godot, no GPU.
#
#   docker build -t chess-evolve .
#   docker run --rm chess-evolve                                # 3-generation smoke run (W&B offline)
#   docker run --rm chess-evolve scripts/bench_throughput.py    # games/s and moves/s on this machine
#   docker run --rm -e WANDB_MODE=online -e WANDB_API_KEY=... \
#       chess-evolve train_wandb.py --config configs/steady_progress_config.json
#
# Python dependencies come from requirements.txt; the crates build with
# --locked against their committed Cargo.lock files, so the image is built
# from pinned inputs end to end.

FROM python:3.11-slim AS builder
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl build-essential \
 && rm -rf /var/lib/apt/lists/* \
 && curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable \
 && pip install --no-cache-dir "maturin>=1.0,<2.0"
WORKDIR /src
COPY rust/chess-cpu rust/chess-cpu
COPY rust/evolve-ga rust/evolve-ga
COPY rust/neat-ga rust/neat-ga
RUN for crate in chess-cpu evolve-ga neat-ga; do \
      maturin build --release --locked --manifest-path rust/$crate/Cargo.toml --out /wheels || exit 1; \
    done

FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=builder /wheels /tmp/wheels
RUN pip install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels
COPY train_wandb.py conftest.py pyproject.toml ./
COPY python/ python/
COPY configs/ configs/
COPY overnight-agent/ overnight-agent/
COPY scripts/ scripts/
# Offline by default so the image runs without a W&B account; pass
# -e WANDB_MODE=online -e WANDB_API_KEY=... to log to wandb.ai instead.
ENV WANDB_MODE=offline \
    PYTHONUNBUFFERED=1
ENTRYPOINT ["python"]
CMD ["train_wandb.py", "--config", "configs/smoke_config.json"]
