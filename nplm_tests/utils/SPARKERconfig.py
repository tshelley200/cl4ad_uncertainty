sparker_config = {
    "number_centroids": 10,
    "width_init": 5,
    "width_fin":0.05,
    "t_ini": 0,
    "decay_epochs": 0.9,
    #"coeffs_init": 0, set it by default and not tunable
    "coeffs_clip": 1,
    "coeffs_reg": "L2", #either "L1" or "L2" or ""   
    "coeffs_reg_lambda":  1,
    "train_coeffs": True,
    "train_width": False,
    "train_centroids": True,
}
