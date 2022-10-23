#! /bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"

for host in "$@"; do
    rsync -avP --exclude=logs --exclude=__pycache__ --exclude=.git --exclude=local --exclude='**/cache/' --exclude='**/*.hdf5' --exclude=".*" $PROJECT_DIR $host:~/ &
done

wait
