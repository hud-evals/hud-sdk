#!/bin/bash

cd /hud/
# Not using uv run here cause that obscures true PID
.venv/bin/python -m controller.server &> /hud/controller.log &
echo $! > /hud/controller.pid
echo $$ > /hud/controller.sh.pid
tail -f /hud/controller.log
