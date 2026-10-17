# Headless render image for the Kubernetes fan-out (deploy/k8s/): the
# `alebrijes` CLI plus everything it needs baked in, so a render pod never
# touches the network. Build from the repo root:
#
#   docker build -f deploy/docker/Dockerfile -t alebrijes:dev .
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf

WORKDIR /app

# CPU-only torch first, before anything that depends on it -- a plain
# `pip install` resolves a ~1.8GB CUDA build instead (PHASES.md, Phase 4).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Model weights + every built-in voice, fetched once at build time. Only
# depends on pocket-tts and the voice list, so this layer stays cached
# across source changes. With no HF login, pocket-tts falls back to the
# ungated kyutai/pocket-tts-without-voice-cloining weights (see README).
RUN pip install pocket-tts soundfile
COPY src/alebrijes/config.py /tmp/config.py
RUN python -c "import runpy; from pocket_tts import TTSModel; \
voices = runpy.run_path('/tmp/config.py')['BUILTIN_VOICES']; \
model = TTSModel.load_model(); \
[model.get_state_for_audio_prompt(v) for v in voices]; \
print('prefetched', voices)"

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . && python -m spacy download en_core_web_sm

# Pods run fully offline, one core each: pocket-tts synthesis is
# single-threaded, so parallelism comes from more pods, not more threads.
ENV HF_HUB_OFFLINE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

ENTRYPOINT ["alebrijes"]
