import os
import glob
import json
from typing import Any, List, Tuple, Optional

import numpy as np


def _as_list(x: Any) -> List[float]:
    if x is None:
        return []
    if isinstance(x, list):
        return [float(v) for v in x]
    return [float(x)]


def _compute_mlp_params(layer_sizes: List[int], bias: bool = True) -> int:
    """
    Fully-connected MLP params for sizes [in, h1, ..., out]:
      sum_i (n_i*n_{i+1} + (n_{i+1} if bias else 0))
    """
    if len(layer_sizes) < 2:
        return 0
    total = 0
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        total += n_in * n_out
        if bias:
            total += n_out
    return int(total)


def collect_results_with_nparams_from_config(
    folder: str,
    results_pattern: str = "*results.json",
    config_name: str = "config.json",
    *,
    config_arch_key: str = "novelty_finder_architecture",
    bias: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[List[float]], List[List[float]], np.ndarray, List[str]]:
    """
    Reads:
      - all results JSONs matching `results_pattern`
      - ONE config JSON (config_name) in the same folder
    Extracts per-results file:
      - tau (scalar)
      - delta (scalar)
      - tau_loss_history (list)
      - delta_loss_history (list)

    And computes `n_params` ONCE from config.json's architecture, then returns
    it as an array of length N (same value repeated for each results file).

    Returns:
      taus, deltas, tau_loss_histories, delta_loss_histories, n_params, files
    """
    folder = os.path.expanduser(folder)

    # ---- read config and compute nparams ----
    config_path = os.path.join(folder, config_name)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find config file: {config_path}")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    arch = cfg.get(config_arch_key, None)
    if not (isinstance(arch, list) and all(isinstance(x, (int, float)) for x in arch)):
        raise KeyError(
            f"Config key '{config_arch_key}' missing or not a numeric list in {config_path}."
        )

    arch_int = [int(x) for x in arch]
    nparams_one = _compute_mlp_params(arch_int, bias=bias)

    # ---- read results files ----
    files = sorted(glob.glob(os.path.join(folder, results_pattern)))

    taus: List[float] = []
    deltas: List[float] = []
    tau_loss_histories: List[List[float]] = []
    delta_loss_histories: List[List[float]] = []

    for fp in files:
        with open(fp, "r") as f:
            data = json.load(f)

        # common top-level keys (as in your screenshot)
        tau_val = data.get("tau", data.get("final_tau", data.get("tau_fit", None)))
        delta_val = data.get("delta", data.get("final_delta", data.get("delta_fit", None)))

        taus.append(float(tau_val) if tau_val is not None else np.nan)
        deltas.append(float(delta_val) if delta_val is not None else np.nan)

        tau_loss_histories.append(_as_list(data.get("tau_loss_history")))
        delta_loss_histories.append(_as_list(data.get("delta_loss_history")))

    taus_arr = np.asarray(taus, dtype=float)
    deltas_arr = np.asarray(deltas, dtype=float)
    nparams_arr = np.full(len(files), nparams_one, dtype=int)

    return taus_arr, deltas_arr, tau_loss_histories, delta_loss_histories, nparams_arr, files


# ---------------- example ----------------
# folder = "/n/holystore01/LABS/iaifi_lab/Users/tshelley2002/Uncertainty_Project/NPLM_tests/Test2/Dj/Nref200000_Nbkg39000_Nsig0/SHAPE1_nuscale-0.1_nuN0__epochsTau10000_epochsDelta1000_arc4_4_4_1_wclip1.8"
# taus, deltas, tau_hist, delta_hist, nparams, files = collect_results_with_nparams_from_config(folder)
# print("nparams:", nparams[0], "N files:", len(files))