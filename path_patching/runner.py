import os
import time
import numpy as np

# Submit slurm jobs for many tasks

root_path = "/data/nikhil_prakash/anima-2.0/path_patching/"
job_path = root_path + str(time.ctime()).replace(" ", "_")
os.makedirs(job_path, exist_ok=True)
results_directory = os.path.join(job_path, "path_patching_results")
os.makedirs(results_directory, exist_ok=True)

d_name_to_cmd = {}
model_name = "llama"

## creating the jobs
i = 0
while len(d_name_to_cmd.keys()) < 20:
    if i == 0:
        current_seed = 38
    elif i == 1:
        current_seed = 83
    else:
        current_seed = np.random.randint(100)
    if current_seed in d_name_to_cmd:
        continue
    results_path = os.path.join(results_directory, f"{str(current_seed)}/")
    os.makedirs(results_path, exist_ok=True)
    datafile = "/data/nikhil_prakash/anima-2.0/box_datasets/no_instructions/alternative/Random/7/train.jsonl"

    cmd = f"python /data/nikhil_prakash/anima-2.0/path_patching/path_patching.py --datafile='{datafile}' --model_name='{model_name}' --output_path='{results_path}' --seed={current_seed} --batch_size=100 --num_samples=300"

    i += 1
    d_name_to_cmd[current_seed] = cmd


for key in d_name_to_cmd:
    with open("template.sh", "r") as f:
        bash_template = f.readlines()
        bash_template.append(d_name_to_cmd[key])

    with open(f"{job_path}/seed_{key}.sh", "w") as f:
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
