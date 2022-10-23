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

ONLINE=True
DATASET="reach_target"
MODEL_TYPE="vit_base"
TRANSFER_TYPE="m3ae_vit_b16"
BATCH_SIZE=2048
INSTRUCTION="moving to one of the spheres"
NOTE="pt: $TRANSFER_TYPE inst: $INSTRUCTION batch size: $BATCH_SIZE policy: $MODEL_TYPE dataset: $DATASET"

python3 -m instructrl.instructrl_main \
    --is_tpu=True \
    --dataset_name="$DATASET" \
    --model.model_type="$MODEL_TYPE" \
    --model.transfer_type="$TRANSFER_TYPE" \
    --window_size=4 \
    --val_every_epochs=1 \
    --test_every_epochs=1 \
    --instruct="$INSTRUCTION" \
    --batch_size="$BATCH_SIZE" \
    --weight_decay=0.0 \
    --lr=3e-4 \
    --auto_scale_lr=False \
    --lr_schedule=cos \
    --warmup_epochs=5 \
    --momentum=0.9 \
    --clip_gradient=10.0 \
    --epochs=200 \
    --dataloader_n_workers=16 \
    --dataloader_shuffle=False \
    --log_all_worker=False \
    --logging.online="$ONLINE" \
    --logging.prefix='' \
    --logging.project="$experiment_name" \
    --logging.gcs_output_dir="gs://$bucket_name/instructrl/experiment_output/hao/$experiment_name" \
    --logging.output_dir="$HOME/experiment_output/$experiment_name" \
    --logging.random_delay=0.0 \
    --logging.notes="$NOTE"

read
