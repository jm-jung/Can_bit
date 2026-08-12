#!/bin/sh
# Generic keep-awake guard: run a command under caffeinate -i -s.
# Lifecycle is bound to the wrapped command (no orphan assertion).
set -eu

if [ "$#" -eq 0 ]; then
    echo "No command supplied to caffeinate guard" >&2
    exit 64
fi

if [ ! -x /usr/bin/caffeinate ]; then
    echo "/usr/bin/caffeinate is unavailable" >&2
    exit 69
fi

exec /usr/bin/caffeinate -i -s "$@"
