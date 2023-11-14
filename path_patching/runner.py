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
for _ in range(10):
    current_seed = np.random.randint(1000000)
    results_path = os.path.join("results", f"{current_seed}")
    datafile = "/data/nikhil_prakash/anima-2.0/box_datasets/no_instructions/alternative/Random/7/train.jsonl"

    cmd = f"python /data/nikhil_prakash/anima-2.0/path_patching/path_patching.py --datafile='{datafile}' --model_name='{model_name}' --output_path='{results_path}' --seed={current_seed}"

    d_name_to_cmd[current_seed] = cmd


for key in d_name_to_cmd:
    with open("template.sh", "r") as f:
        bash_template = f.readlines()
        bash_template.append(d_name_to_cmd[key])

    with open(f"{job_path}/{key}.sh", "w") as f:
        f.writelines(bash_template)


## running the jobs
for job in os.listdir(job_path):
    job_script = f"{job_path}/{job}"
    cmd = f"export MKL_SERVICE_FORCE_INTEL=1; sbatch --gpus=1 --time=48:00:00 {job_script}"
    print("submitting job: ", job)
    print(cmd)
    os.system(cmd)
    print("\n\n")

print("------------------------------------------------------------------")
print(f"submitted {len(os.listdir(job_path))} jobs!")
