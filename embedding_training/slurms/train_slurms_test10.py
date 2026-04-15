import os, sys
import numpy as np

nv_lst = np.array([[0.25, 0.5]])#, [0.25, 0.375]]) #np.array([[1, 2], [0.5, 1], [0.5, 1.5]])
nv_norm_lst = np.array([[0.025, 0.05]])#, [0.025, 0.0375]])#np.array([[0.1, 0.2], [0.05, 0.1], [0.05, 0.15]])
lambdas_lst = np.array([0.1, 0.5])#np.array([0, 1, 3, 6, 9])
lm_lr_lst = np.array([1e-6, 7e-5])#([1e-6,8e-6]) #,2e-5])
lm_regularization_lst = np.array([1e-4,1e-7])
run_num = 10.8

#------------hyperparameters for training-----------------------

epochs = 75
batch_size = 1024
train_num = 2100000 
val_num = 1200000
test_num = 200000

load_initial_model = False #True
if load_initial_model:
    initial_model = "/n/home07/tshelley2002/Uncertainty_Project/cl4ad_new/runs409/checkpoints/_valfold_4_vae_ep_5.pth"
if load_initial_model:
    freeze_initial_model = False  # or True

freeze_model = False
model_base_lr = 0.00002
loss_temp = 0.1   

zero_init = True
quadratic_model = False
load_initial_lm = False

if load_initial_lm:
    initial_lm = ""
if load_initial_lm:
    freeze_initial_lm = False

freeze_lm = False

lambda_1 = 1

#-----------Directories and Paths--------------------
base_dir = "/n/home07/tshelley2002/Uncertainty_Project/cl4ad_new/"
storage_dir = "/n/holystore01/LABS/iaifi_lab/Users/tshelley2002/Uncertainty_Project/"
pyscript_path = os.path.join(base_dir, "train_with_signal_lxplus_test1.py")
slurm_path = os.path.join(base_dir, f"slurm_files/Run{run_num}_slurm/")
log_dir = os.path.join(base_dir, f"logs/Run{run_num}/")
output_path = os.path.join(storage_dir, f"Run{run_num}/")

os.makedirs(slurm_path, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

#--------------------------------------------------
for i in range(len(nv_lst)):
    
    for lm_lr in lm_lr_lst:
        
        for lm_regularization in lm_regularization_lst:
            
            nv = nv_lst[i]
            nv_norm = nv_norm_lst[i]
        
            job_tag = f"R{run_num}_nv{nv[0]}_{nv[1]}_lr{lm_lr}_r{lm_regularization}"
            script_name = f"{job_tag}.sh"
            script_file = os.path.join(slurm_path, script_name)
        
            with open(script_file, "w") as script_sbatch:
                script_sbatch.write("#!/bin/bash\n")
                script_sbatch.write("#SBATCH --account=iaifi_lab\n")
                script_sbatch.write("#SBATCH --ntasks=1\n")
                script_sbatch.write("#SBATCH --cpus-per-task=2\n")
                script_sbatch.write("#SBATCH -t 48:00:00\n")
                script_sbatch.write("#SBATCH -p gpu\n")
                script_sbatch.write("#SBATCH --mem=48G\n")
                script_sbatch.write("#SBATCH --gres=gpu:1\n")
        
                script_sbatch.write(f"#SBATCH -J {job_tag}\n")
                script_sbatch.write(f"#SBATCH -o {log_dir}/%j.out\n")
                script_sbatch.write(f"#SBATCH -e {log_dir}/%j.err\n")
        
                script_sbatch.write("\n")
                script_sbatch.write("set -euo pipefail\n")
                script_sbatch.write("source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh\n")
                script_sbatch.write("conda activate cl4ad_new\n\n")
        
                # fixed broken quote
                script_sbatch.write(f"echo '\n Run {run_num}, nv_norm = {nv_norm.tolist()}, paramodel lr = {lm_lr}, lm regularization = {lm_regularization}'\n\n")
        
                for l in lambdas_lst:
                    script_sbatch.write(f"echo 'lambda = {l}'\n")
        
                    para_model_name = "quadratic" if quadratic_model else "linear"
                    full_output_path = os.path.join(output_path,f"nv{nv[0]}_{nv[1]}_l{l}_lr{lm_lr}_r{lm_regularization}/")
                    os.makedirs(full_output_path, exist_ok=True)
                    
                    # Build command in pieces so conditional args are easy and don't break parsing
                    cmd_parts = [
                        "srun --export=ALL --cpu-bind=none python -u",
                        f"{pyscript_path}",
                        f"/n/holystore01/LABS/iaifi_lab/Users/tshelley2002/Uncertainty_Project/hdf5_files-v2/ADC_Delphes_scaled_-{nv_norm[1]}.hdf5",
                        f"/n/holystore01/LABS/iaifi_lab/Users/tshelley2002/Uncertainty_Project/hdf5_files-v2/ADC_Delphes_scaled_-{nv_norm[0]}.hdf5",
                        "/n/holystore01/LABS/iaifi_lab/Users/tshelley2002/Uncertainty_Project/hdf5_files-v2/ADC_Delphes_scaled_0.0.hdf5",
                        f"/n/holystore01/LABS/iaifi_lab/Users/tshelley2002/Uncertainty_Project/hdf5_files-v2/ADC_Delphes_scaled_{nv_norm[0]}.hdf5",
                        f"/n/holystore01/LABS/iaifi_lab/Users/tshelley2002/Uncertainty_Project/hdf5_files-v2/ADC_Delphes_scaled_{nv_norm[1]}.hdf5",
                        f"--nvs -{nv[1]} -{nv[0]} {nv[0]} {nv[1]}",
                        f"--train-num {train_num}",
                        f"--val-num {val_num}",
                        f"--test-num {test_num}",
                        f"--batch-size {batch_size}",
                        f"--epochs {epochs}",
                        "--type Delphes",
                        f"--base-lr {model_base_lr}",
                        f"--loss-temp {loss_temp}",
                        f"--linear-model-lr {lm_lr}",
                        f"--linear-model-regularization {lm_regularization}",
                        f"--lambda {l}",
                        f"--lambda1 {lambda_1}",
                        "--supervision supervised",
                        f"--output-path {full_output_path}",
                        "--model-name simCLR",
                        f"--para-model-name {para_model_name}",
                        "--train",
                    ]
                    # conditional args
                    if load_initial_model:
                        cmd_parts.append("--load-initial-model")
                        cmd_parts.append(f"--initial-model {initial_model}")
                        
                    if load_initial_model and ('freeze_initial_model' in locals()) and freeze_initial_model:
                        cmd_parts.append("--freeze-initial-model")
                    if freeze_model:
                        cmd_parts.append("--freeze-model")
        
                    if quadratic_model:
                        cmd_parts.append("--quadratic-model")

                    if zero_init:
                        cmd_parts.append("--zero-init")
        
                    if load_initial_lm:
                        cmd_parts.append("--load-initial-linear-model")
                        cmd_parts.append(f"--initial-linear-model {initial_lm}")
                        
                    if load_initial_lm and ('freeze_initial_lm' in locals()) and freeze_initial_lm:
                        cmd_parts.append("--freeze-initial-linear-model")
                        
                    if freeze_lm:
                        cmd_parts.append("--freeze-linear-model")
        
                    full_cmd = " \\\n    ".join(cmd_parts) + "\n"
                    script_sbatch.write(full_cmd)
                    script_sbatch.write("\n")
        
            os.system(f"chmod a+x {script_file}")
            os.system(f"sbatch {script_file}")