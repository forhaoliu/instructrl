#! /bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"

for host in "$@"; do
    scp $SCRIPT_DIR/tpu_vm_setup.sh $host:~/
    ssh $host '~/tpu_vm_setup.sh' &
done

wait
