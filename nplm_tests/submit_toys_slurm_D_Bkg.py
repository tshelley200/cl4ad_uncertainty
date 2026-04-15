#!/usr/bin/env python3
"""
submit_nplm_jobs.py

Purpose
-------
This script builds the NPLM config JSON, writes it to disk, and then either:
  1) runs the target python script locally, or
  2) creates and submits Slurm sbatch scripts.

This is a cleaned-up version of the original wrapper, but with the common
hardcoded paths exposed as command-line arguments so you do not need to edit
the file every time.

Typical usage
-------------
python submit_nplm_jobs.py \
    --pyscript run_toys.py \
    --test-num 7 \
    --output-dir /path/to/output \
    --embeddings-file /path/to/val_embeddings_fold0_trimmed.npz \
    --shape-model /path/to/cubic_12a_final.pth \
    --toys 100 \
    --firstseed 0

Notes
-----
- Adjust the defaults in the "environment defaults" section if your base
  directories or conda environment change.
- I left a lot of comments on purpose so future-you can edit this quickly.
"""

import os
import sys
import json
import argparse
import datetime
import numpy as np

# ---------------------------------------------------------------------
# Import your project utilities exactly like in your current script.
# Change these if your project moves.
# ---------------------------------------------------------------------
UTILS_PATH = "/n/home07/tshelley2002/Uncertainty_Project/cl4ad_new/NPLM_testing/code_updated/utils/"
sys.path.insert(1, UTILS_PATH)

import NUconfig as nu
import UTILSutils as util

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build config JSON and submit/run NPLM toy jobs."
    )

    parser.add_argument("--test-num", type=int, required=True, help="Test number used for labeling outputs/logs.")
    parser.add_argument("--shape-nuisance-val", type = float, required = True)
    parser.add_argument("--embeddings-file", type=str, required=True)
    parser.add_argument("--shape-model", type=str, action="append", required=True, default=[])
    
    parser.add_argument("-p", "--pyscript",type=str, default="NPLM_SimCLR_smallSample.py")
    parser.add_argument("--output-dir", type=str, default = "/n/holystore01/LABS/iaifi_lab/Users/tshelley2002/Uncertainty_Project/NPLM_tests")
    parser.add_argument("--nv-values", type=float, nargs="+", default=[-0.5, -0.25, 0.25, 0.5],  help="Values used to compute nv_std")
    
    parser.add_argument("--unscaled-data-name", type=str, default="gx_0")
    parser.add_argument("--scaled-data-name", type=str, default="gx_pos0.25")
    
    parser.add_argument("--weight-clipping",type=float, default=1.97, help="novelty_finder_weight_clipping")
    parser.add_argument("--n-ref", type=int, default=52000, help="N_Ref")
    parser.add_argument("--n-bkg", type=int, default=10000, help="N_Bkg")
    
    parser.add_argument("--novelty-arch", type=int,nargs="+", default=[4, 4, 4, 1], help="Novelty finder architecture, e.g. --novelty-arch 4 4 4 1") 

    parser.add_argument("--epochs-tau", type=int, default=50000)
    parser.add_argument("--patience-tau", type=int, default=10)
    parser.add_argument("--epochs-delta", type=int, default=1000)
    parser.add_argument("--patience-delta", type=int, default=10)


    # ==============================================================
    # Runtime controls
    # ==============================================================

    parser.add_argument(
        "-l", "--local",
        type=int,
        default=0,
        help="1 = run locally, 0 = submit Slurm jobs")

    parser.add_argument(
        "-t", "--toys",
        type=int,
        default=100,
        help="Number of toys")

    parser.add_argument(
        "-s", "--firstseed",
        type=int,
        default=-1,
        help="Starting seed. If < 0, seeds are generated from current time.")

    # ==============================================================

    
    parser.add_argument("--is-tail-excess", action="store_true", help="Set tail excess mode")

    parser.add_argument(
        "--correction",
        type=str,
        default="SHAPE",
        choices=["SHAPE", "NORM", ""],
        help='Correction mode: "SHAPE", "NORM", or ""'
    )

    parser.add_argument(
        "--shape-nuisance-id",
        type=str,
        action="append",
        default=["scale"],
        help=(
            "Shape nuisance ID. Can be repeated, e.g. "
            "--shape-nuisance-id scale --shape-nuisance-id something_else"
        )
    )

    parser.add_argument("--make-plots", type=int, default=1, help="1=True, 0=False")
    parser.add_argument("--verbose", type=int, default=1, help="1=True, 0=False")

    # ==============================================================
    # Slurm / environment settings
    # ==============================================================

    parser.add_argument(
        "--base-dir",
        type=str,
        default="/n/home07/tshelley2002/Uncertainty_Project/cl4ad_new/NPLM_testing/code_updated/",
        help="Base project directory"
    )

    parser.add_argument(
        "--log-base-dir",
        type=str,
        default="/n/home07/tshelley2002/Uncertainty_Project/cl4ad_new/NPLM_testing/code_updated/logs",
        help="Base directory for Slurm log files"
    )

    parser.add_argument(
        "--slurm-label-base",
        type=str,
        default="slurms",
        help="Base directory for generated sbatch scripts"
    )

    parser.add_argument("--account", type=str, default="iaifi_lab")
    parser.add_argument("--partition", type=str, default="shared")
    parser.add_argument("--time", type=str, default="3:00:00")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", type=str, default="12G")
    parser.add_argument("--conda-env", type=str, default="cl4ad_new")
    parser.add_argument(
        "--conda-sh",
        type=str,
        default="/n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh"
    )

    return parser.parse_args()


def validate_args(args):
    # Check path-like inputs early so failures happen before submission.
    if not os.path.isfile(args.embeddings_file):
        raise FileNotFoundError(f"embeddings file not found:\n{args.embeddings_file}")


    for model in args.shape_model:
        if not os.path.isfile(model):
            raise FileNotFoundError(f"shape model not found:\n{model}")

    if not args.pyscript.endswith(".py"):
        print(f"Warning: pyscript does not end with .py: {args.pyscript}")

    if args.correction == "SHAPE" and len(args.shape_model) == 0:
        raise ValueError('correction is SHAPE, but no --shape-model was provided.')


def build_config(args):
    """
    Rebuild the config_json structure from your original wrapper, but now
    using command-line arguments instead of hardcoded values.
    """
    shape_nuisances_id_list = args.shape_nuisance_id

    config_json = {
        "make_plots": bool(args.make_plots),
        "verbose": bool(args.verbose),
        "N_Ref": args.n_ref,
        "N_Bkg": args.n_bkg,
        "is_tail_excess": bool(args.is_tail_excess),
        "output_directory": f"{args.output_dir}/Test{args.test_num}/D/",
        "shape_nuisances_id": shape_nuisances_id_list,
        "shape_nuisances_data": [args.shape_nuisance_val],#[nu.nuisances_data[k] for k in shape_nuisances_id_list],
        "shape_nuisances_reference": [nu.nuisances_reference[k] for k in shape_nuisances_id_list],
        "shape_nuisances_sigma": [nu.nuisances_sigma[k] for k in shape_nuisances_id_list],
        "shape_models": args.shape_model,
        "norm_nuisances_data": nu.nuisances_data["norm"],
        "norm_nuisances_reference": nu.nuisances_reference["norm"],
        "norm_nuisances_sigma": nu.nuisances_sigma["norm"],
        "epochs_tau": args.epochs_tau,
        "patience_tau": args.patience_tau,
        "epochs_delta": args.epochs_delta,
        "patience_delta": args.patience_delta,
        "novelty_finder_architecture": args.novelty_arch,
        "novelty_finder_weight_clipping": args.weight_clipping,
        "correction": args.correction,
        "nv_std": float(np.std(np.array(args.nv_values))),
        "unscaled_data_name": args.unscaled_data_name,
        "scaled_data_name": args.scaled_data_name,
        "embeddings_filename": args.embeddings_file,
        "test_num": args.test_num,
       
    }

    return config_json


def validate_config(config_json):
    """
    Keep the same checks you had in the original wrapper.
    """
    if config_json["correction"] not in ["SHAPE", "NORM", ""]:
        raise ValueError('Error: "correction" must be one of ["SHAPE", "NORM", ""]')

    if len(config_json["shape_nuisances_sigma"]) != len(config_json["shape_models"]):
        raise ValueError('Error: length of "shape_nuisances_sigma" and "shape_models" must match.')

    if len(config_json["shape_nuisances_sigma"]) != len(config_json["shape_nuisances_data"]):
        raise ValueError('Error: length of "shape_nuisances_sigma" and "shape_nuisances_data" must match.')

    if len(config_json["shape_nuisances_sigma"]) != len(config_json["shape_nuisances_reference"]):
        raise ValueError('Error: length of "shape_nuisances_sigma" and "shape_nuisances_reference" must match.')

    if config_json["correction"] == "SHAPE" and not len(config_json["shape_models"]):
        raise ValueError('Error: correction is SHAPE but no "shape_models" were provided.')


def build_output_id(config_json):
    """
    Reproduce your folder naming logic so output structure stays familiar.
    """
    ID = (
        "Nref" + str(config_json["N_Ref"]) +
        "_Nbkg" + str(config_json["N_Bkg"]) 
    )

    correction_details = config_json["correction"]

    if config_json["correction"] == "SHAPE":
        correction_details += str(len(config_json["shape_models"])) + "_"
        for i in range(len(config_json["shape_nuisances_id"])):
            key = config_json["shape_nuisances_id"][i]
            if config_json["shape_nuisances_data"][i] != 0:
                correction_details += f"nu{key}{config_json['shape_nuisances_data'][i]}_"

    if config_json["correction"] in ["NORM", "SHAPE"]:
        if config_json["correction"] == "NORM":
            correction_details += "_"
        correction_details += "nuN" + str(config_json["norm_nuisances_data"]) + "_"

    ID += "/" + correction_details
    ID += (
        "_epochsTau" + str(config_json["epochs_tau"]) +
        "_epochsDelta" + str(config_json["epochs_delta"])
    )
    ID += (
        "_arc" +
        str(config_json["novelty_finder_architecture"]).replace(", ", "_").replace("[", "").replace("]", "") +
        "_wclip" + str(config_json["novelty_finder_weight_clipping"])
    )

    return ID


def generate_seed(firstseed, i):
    if firstseed >= 0:
        return firstseed + i
    now = datetime.datetime.now()
    return now.microsecond + now.second + now.minute


def submit_or_run(args, config_json, json_path):
    ntoys = args.toys
    pyscript = args.pyscript
    firstseed = args.firstseed

    if args.local:
        # Local mode: run jobs directly from this machine/session.
        for i in range(ntoys):
            seed = generate_seed(firstseed, i)
            cmd = f"python {os.getcwd()}/{pyscript} -j {json_path} -s {seed}"
            print(f"Running locally: {cmd}")
            os.system(cmd)

    else:
        # Slurm mode: create sbatch files and submit them.
        test_num = int(config_json["test_num"])
        wc = float(config_json["novelty_finder_weight_clipping"])
        N_Ref = int(config_json["N_Ref"])
        N_Bkg = int(config_json["N_Bkg"])

        label = os.path.join(
            args.slurm_label_base,
            f"Test{test_num}",
            f"D_WC{wc}_NRef{N_Ref}_NBkg{N_Bkg}_Bkg"
        )
        log_dir = os.path.join(
            args.log_base_dir,
            f"Test{test_num}",
            f"D_WC{wc}_NRef{N_Ref}_NBkg{N_Bkg}_Bkg"
        )
        job_tag = f"T{test_num}D{wc}_Bkg"

        os.makedirs(label, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        script_path = os.path.join(args.base_dir, pyscript)

        for i in range(ntoys):
            seed = generate_seed(firstseed, i)

            script_name = f"{job_tag}_submit_{seed}.sh"
            script_file = os.path.join(label, script_name)

            with open(script_file, "w") as script_sbatch:
                script_sbatch.write("#!/bin/bash\n")
                script_sbatch.write(f"#SBATCH --account={args.account}\n")
                script_sbatch.write("#SBATCH --ntasks=1\n")
                script_sbatch.write(f"#SBATCH --cpus-per-task={args.cpus_per_task}\n")
                script_sbatch.write(f"#SBATCH -t {args.time}\n")
                script_sbatch.write(f"#SBATCH -p {args.partition}\n")
                script_sbatch.write(f"#SBATCH --mem={args.mem}\n")
                script_sbatch.write(f"#SBATCH -J {job_tag}_submit\n")
                script_sbatch.write(f"#SBATCH -o {log_dir}/%j.out\n")
                script_sbatch.write(f"#SBATCH -e {log_dir}/%j.err\n")
                script_sbatch.write("\n")
                script_sbatch.write("set -euo pipefail\n")
                script_sbatch.write(f"echo 'Test{test_num}, WC{wc}, D'\n")
                script_sbatch.write(f"source {args.conda_sh}\n")
                script_sbatch.write(f"conda activate {args.conda_env}\n\n")
                script_sbatch.write(
                    f"srun --export=ALL --cpu-bind=none python -u {script_path} -j {json_path} -s {seed}\n"
                )

            os.system(f"chmod a+x {script_file}")
            print(f"Submitting: {script_file}")
            os.system(f"sbatch {script_file}")


def main():
    args = parse_args()
    validate_args(args)

    config_json = build_config(args)
    validate_config(config_json)

    ID = build_output_id(config_json)
    config_json["output_directory"] = os.path.join(
        args.output_dir,
        f"Test{args.test_num}",
        "D",
        ID)

    os.makedirs(config_json["output_directory"], exist_ok=True)
    config_json["pyscript"] = args.pyscript

    json_path = util.create_config_file(config_json, config_json["output_directory"])

    print("Config written to:")
    print(f"  {json_path}")
    print("Final output directory:")
    print(f"  {config_json['output_directory']}")

    submit_or_run(args, config_json, json_path)

if __name__ == "__main__":
    main()