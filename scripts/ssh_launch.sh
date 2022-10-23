#! /bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"

COMMAND=$1

shift

for host in "$@"; do
    echo "Launching on host: $host"
    ssh $host 'tmux new -d -s launch '"$COMMAND" &
done

wait
