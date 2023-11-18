import os
import time
import numpy as np

# Submit slurm jobs for many tasks

job_path = str(time.ctime()).replace(" ", "_")
print(job_path)
os.makedirs(job_path, exist_ok=True)

d_name_to_cmd = {}
model_name = "goat"

## creating the jobs
for percentage in np.arange(0.05, 0.55, 0.05):
    results_path = (
        "/data/nikhil_prakash/anima-2.0/path_patching/minimality_res/minimality"
    )
    datafile = "/data/nikhil_prakash/anima-2.0/box_datasets/no_instructions/alternative/Random/7/train.jsonl"
    circuit_root_path = (
        "/data/nikhil_prakash/anima-2.0/path_patching/goat_circuits/1404"
    )

    cmd = f"python /data/nikhil_prakash/anima-2.0/path_patching/minimality.py --datafile='{datafile}' --circuit_root_path='{circuit_root_path}' --model_name='{model_name}' --n_value_fetcher=75 --n_pos_trans=10 --n_pos_detect=35 --n_struct_read=5 --num_samples=100 --results_path='{results_path}' --percentage={percentage}"

    d_name_to_cmd[percentage] = cmd


for key in d_name_to_cmd:
    with open("template.sh", "r") as f:
        bash_template = f.readlines()
        bash_template.append(d_name_to_cmd[key])

    with open(f"{job_path}/percentage_{key}.sh", "w") as f:
        f.writelines(bash_template)


## running the jobs
for job in os.listdir(job_path):
    job_script = f"{job_path}/{job}"
    cmd = f"export MKL_SERVICE_FORCE_INTEL=1; sbatch --gpus=1 --time=24:00:00 {job_script}"
    print("submitting job: ", job)
    print(cmd)
    os.system(cmd)
    print("\n\n")

print("------------------------------------------------------------------")
print(f"submitted {len(os.listdir(job_path))} jobs!")
