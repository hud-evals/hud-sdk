FROM govindhud/novnc-ubuntu:latest

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY requirements.txt /hud/requirements.txt
COPY controller/ /hud/controller/
RUN cd /hud && uv venv --python 3.12 && uv pip install --requirements requirements.txt

HEALTHCHECK --interval=5s --timeout=3s --retries=1 \
    CMD wget -q --spider http://localhost:8080 || exit 1
