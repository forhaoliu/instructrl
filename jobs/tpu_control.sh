#! /bin/bash

trap "exit;" SIGINT SIGTERM

export SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
echo $SCRIPT_DIR
echo $PROJECT_DIR

cd ${PROJECT_DIR}

# read -r -d '' TPU_IPS << EOM
# EOM
TPU_IPS=$(gcloud alpha compute tpus tpu-vm describe $1 --format="value[delimiter=' '](networkEndpoints.accessConfig.externalIp)")
echo $TPU_IPS

OPT=$2
echo $OPT

SCRIPT=${3:-NA}
echo $SCRIPT

COMMAND='~/instructrl/jobs/'${SCRIPT}
echo $COMMAND

if [ "$OPT" = "setup" ]; then
    ./scripts/ssh_vm_setup.sh $TPU_IPS
elif [ "$OPT" = "copy" ]; then
    ./scripts/ssh_copy.sh $TPU_IPS
elif [ "$OPT" = "launch" ]; then
    ./scripts/ssh_launch.sh $COMMAND $TPU_IPS
elif [ "$OPT" = "stop" ]; then
    ./scripts/ssh_stop.sh $TPU_IPS
elif [ "$OPT" = "check" ]; then
    for host in $TPU_IPS; do
        echo "Checking host: $host"
        ssh $host 'tmux capture-pane -pt launch'
    done
elif [ "$OPT" = "all" ]; then
    ./scripts/ssh_copy.sh $TPU_IPS
    ./scripts/ssh_copy.sh $TPU_IPS
    ./scripts/ssh_vm_setup.sh $TPU_IPS
    ./scripts/ssh_vm_setup.sh $TPU_IPS
    ./scripts/ssh_launch.sh $COMMAND $TPU_IPS
else
    echo 'Invalid option!'
fi
