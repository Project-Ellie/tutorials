export PYTHONPATH=${PYTHONPATH}:${PWD}
python -m train.task \
  --eval_steps="10"  \
  --throttle_secs="30"  \
  --eval_batch_size="1024"  \
  --eval_data_pattern="gs://going-tfx/samples/eval_data/atl_june_tfr*"  \
  --learning_rate="0.001"  \
  --distribute="False"  \
  --prefetch_buffer_size="10000"  \
  --log_step_count_steps="200"  \
  --hypothesis="linear"  \
  --model_dir="gs://going-tfx/samples/model"  \
  --train_batch_size="256"  \
  --max_train_steps="5000"  \
  --metadata_dir="gs://going-tfx/samples/metadata"  \
  --train_data_pattern="gs://going-tfx/samples/train_data/atl_june_tfr*"  \
  --save_checkpoints_steps="2000"  \
  --base_dir="gs://going-tfx/samples"  \
