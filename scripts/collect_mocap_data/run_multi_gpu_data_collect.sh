#!/bin/bash
logs_folder="$(pwd)/logs"
mkdir -p "$logs_folder"

for gpu_id in {0..7}; do
    for instance_id in {1..4}; do
        session_name="gpu${gpu_id}_instance${instance_id}"
        tmux new-session -d -s "$session_name" "source $HOME/mambaforge/bin/activate sabr && CUDA_VISIBLE_DEVICES=$gpu_id python collect_data.py --instance $instance_id > $logs_folder/output_${session_name}.log 2>&1"
    done
done