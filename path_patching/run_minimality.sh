#!/bin/bash
source ~/.bashrc
cd ../
conda activate anima
python "/data/nikhil_prakash/anima-2.0/path_patching/minimality.py" --datafile="/data/nikhil_prakash/anima-2.0/box_datasets/no_instructions/alternative/Random/7/train.jsonl" --circuit_root_path="/data/nikhil_prakash/anima-2.0/path_patching/Sat_Dec__9_15:16:21_2023/path_patching_results/26" --model_name="llama" --n_value_fetcher=66 --n_pos_trans=15 --n_pos_detect=30 --n_struct_read=5 --num_samples=100 --batch_size=100 --results_path="/data/nikhil_prakash/anima-2.0/path_patching/results/minimality_results"
