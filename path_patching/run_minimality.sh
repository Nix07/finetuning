#!/bin/bash
source ~/.bashrc
cd ../
conda activate anima
python "/data/nikhil_prakash/anima-2.0/path_patching/minimality.py" --datafile="/data/nikhil_prakash/anima-2.0/box_datasets/no_instructions/alternative/Random/7/train.jsonl" --circuit_root_path="/data/nikhil_prakash/anima-2.0/path_patching/goat_circuits/1404" --model_name="goat" --n_value_fetcher=75 --n_pos_trans=10 --n_pos_detect=35 --n_struct_read=5 --num_samples=100 --results_path="/data/nikhil_prakash/anima-2.0/path_patching/results/minimality"
