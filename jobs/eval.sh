#! /bin/bash

export SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd $PROJECT_DIR

export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR/instructrl/models"
echo $PYTHONPATH
export WANDB_API_KEY=''

export bucket_name='instruct-rl'

export experiment_name='instructrl'

ONLINE=False
DATASET="reach_target"
MODEL_TYPE="vit_base"
TRANSFER_TYPE="m3ae_vit_b16"
INSTRUCTION="moving to one of the spheres"
CKPT="gs://instruct-rl/instructrl/experiment_output/hao/Hao-instructrl-3/85c64241b762445fa079d65cf0c7b4c3/model.pkl"
NOTE="Local rollout. pt: $TRANSFER_TYPE inst: $INSTRUCTION policy: $MODEL_TYPE dataset: $DATASET"

python3 -m instructrl.local_run \
    --load_checkpoint "$CKPT" \
    --dataset_name="$DATASET" \
    --model.model_type="$MODEL_TYPE" \
    --model.transfer_type="$TRANSFER_TYPE" \
    --window_size=1 \
    --instruct="$INSTRUCTION" \
    --log_all_worker=False \
    --data.path="$PROJECT_DIR/data/variation" \
    --logging.online="$ONLINE" \
    --logging.prefix='' \
    --logging.project="$experiment_name" \
    --logging.gcs_output_dir="gs://$bucket_name/instructrl/experiment_output/hao/$experiment_name" \
    --logging.output_dir="$HOME/experiment_output/$experiment_name" \
    --logging.random_delay=0.0 \
    --logging.notes="$NOTE"

read


