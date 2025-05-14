FROM govindhud/novnc-ubuntu:latest

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /hud
COPY requirements.txt ./requirements.txt
COPY controller ./controller
RUN uv venv && uv pip install --requirements requirements.txt
EXPOSE 8483

HEALTHCHECK --interval=5s --timeout=3s --retries=1 \
    CMD wget -q --spider http://localhost:8080 || exit 1
