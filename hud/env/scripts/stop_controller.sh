#!/bin/bash

PID=$(cat /hud/controller.pid)
if [ -z "$PID" ]; then
    echo "No PID file found. Is the controller running?"
    exit 1
fi
kill -9 $PID

SH_PID=$(cat /hud/controller.sh.pid)
if [ -z "$SH_PID" ]; then
    echo "No PID file found. Is the controller.sh running?"
    exit 1
fi
kill -9 $SH_PID


