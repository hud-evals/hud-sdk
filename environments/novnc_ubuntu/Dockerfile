FROM govindhud/novnc-ubuntu:latest

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /controller
COPY requirements.txt *.py ./
RUN uv venv --system-site-packages && uv pip sync requirements.txt

EXPOSE 25565
HEALTHCHECK --interval=5s --timeout=3s --retries=1 \
    CMD wget -q --spider http://localhost:8080 || exit 1

# note: when you make changes to the controller, hud-sdk will automatically reinstall the python package corresponding to the controller before every step.
# Any changes to the Dockerfile though will only be reflected on the next environment creation
