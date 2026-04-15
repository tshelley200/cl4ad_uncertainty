import sys, os, time, datetime, argparse, json, copy
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
#import matplotlib.pyplot as plt
#import matplotlib.font_manager as font_manager
from scipy.stats import norm, expon, chi2, uniform, chisquare

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(1, '/n/home07/tshelley2002/Uncertainty_Project/cl4ad_new/NPLM_testing/code_updated/utils/')
import GOFutils as gof
import PLOTutils as plot
import TAYLORutils as taylor
import UTILSutils as util

parser   = argparse.ArgumentParser()
parser.add_argument('-j', '--jsonfile', type=str, help="json file", required=True)
parser.add_argument('-s', '--seed', type=int, help="toy seed", required=False, default=None)
args     = parser.parse_args()

# random seed 
seed = args.seed
if seed==None:
    seed=util.generate_random_seed_from_time()
np.random.seed(seed)
print('Random seed:'+str(seed))

# train on GPU?                                                                                 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# setup parameters                                                                               
with open(args.jsonfile, 'r') as jsonfile:
    config_json = json.load(jsonfile)

# statistics    
N_ref      = config_json['N_Ref']
N_Bkg      = config_json['N_Bkg']

# variables to define the reference sample weight: N_D/N_R
N_R        = N_ref
N_D        = N_Bkg

# systematics:  
correction = config_json['correction'] #'', 'NORM', 'SHAPE'
shape_nuisances_id_list = config_json['shape_nuisances_id'] # list
Scale   = config_json['shape_nuisances_data'] # list
Norm    = config_json['norm_nuisances_data']
sigma_s = config_json['shape_nuisances_sigma'] # list
sigma_n = config_json['norm_nuisances_sigma'] #[0.08]

print("sigma_s:",sigma_s," sima_n", sigma_n)
# generate the auxiliary values of the nuisance parameters
NU0_S     = [torch.tensor([np.random.normal(loc=Scale_i, scale=sigma_s_i, size=1)[0]]) for Scale_i, sigma_s_i in zip(Scale, sigma_s)]
NU0_N = torch.tensor([np.random.normal(loc=Norm, scale=sigma_n, size=1)[0]])
print('Auxiliary measurements of nuisances parameters gave:')
print('normalization NU0_N:', NU0_N)
print('scal NU0_S:', NU0_S)

# initialization 
train_norm_syst_finder=False
train_shape_syst_finder=False 
Shape_syst_finder_list=None
Norm_syst_finder=None

# training time                                                 
total_epochs_tau   = config_json["epochs_tau"]
patience_tau       = config_json["patience_tau"]
total_epochs_delta = config_json["epochs_delta"]
patience_delta     = config_json["patience_delta"]

# define output path
OUTPUT_PATH    = config_json["output_directory"]
OUTPUT_FILE_ID = '/seed'+str(seed)
make_plots = config_json["make_plots"]
verbose =  config_json["verbose"]
###################################################
###################################################

# Load the models with the data

# the number of events to generate fluctuates poissonianly:
N_Bkg_Pois  = np.random.poisson(lam=N_Bkg*np.exp(Norm), size=1)[0]
"""if N_Sig:
    N_Sig_Pois = np.random.poisson(lam=N_Sig*np.exp(Norm), size=1)[0]"""

# the featureData are generated with a distortion in shape and normalization 
# determined by the "Scale" and "Norm" parameters respectively:
"""
scale = 1
if 'scale' in shape_nuisances_id_list:
    scale = np.exp(Scale)
featureData = np.random.exponential(scale=scale, size=(N_Bkg_Pois, 1))
if N_Sig:
    featureSig  = np.random.normal(loc=Sig_loc, scale=Sig_std, size=(N_Sig_Pois,1))*scale
    featureData = np.concatenate((featureData, featureSig), axis=0)

# the featureRef are generated according to the central value (e.g. no distortion): 
featureRef  = np.random.exponential(scale=1, size=(N_ref, 1))
feature     = np.concatenate((featureData, featureRef), axis=0)
"""
###############################################################################################

filename = config_json["embeddings_filename"]
data_x = np.load(filename); 
N_Bkg_Pois  = np.random.poisson(lam=N_Bkg*np.exp(Norm), size=1)[0]

name_unscaled_data = config_json["unscaled_data_name"]
unscaled_train0_X = data_x[name_unscaled_data] 
unscaled_rng = np.random.default_rng(seed)
unscaled_idx = unscaled_rng.choice(unscaled_train0_X.shape[0], size=N_ref, replace=False)  
unscaled_train0_x = unscaled_train0_X[unscaled_idx, :]

name_scaled_data = config_json["scaled_data_name"]
scaled_train0_X = data_x[name_scaled_data] 
scaled_rng = np.random.default_rng(seed)
scaled_idx = scaled_rng.choice(scaled_train0_X.shape[0], size=N_Bkg_Pois, replace=False)   # random unique rows    
scaled_train0_x = scaled_train0_X[scaled_idx, :]

featureData_lst_x = [scaled_train0_x];

targetRef_x  = np.zeros_like(unscaled_train0_x[:,0])
feature_x    = np.empty((0,4))
target_x    = np.empty((0))
nuisance_x   = np.empty((0))

nv = [Scale[0]/0.1] #[Scale[0]/sigma_s[0]]
print(nv)
nv_std =config_json["nv_std"]

for i in range(0,len(nv)):
    
    featureData_x = featureData_lst_x[i]
    targetData_x  = np.ones_like(featureData_x[:,0])

    feature_x = np.append(feature_x, unscaled_train0_x,axis=0)
    feature_x = np.append(feature_x, featureData_x, axis=0)
    
    target_x  = np.append(target_x, targetRef_x,  axis=0)
    target_x  = np.append(target_x, targetData_x, axis=0)
    
    nuisance_x = np.append(nuisance_x, np.ones_like(unscaled_train0_x[:,0])*nv[i]/nv_std,axis=0)
    nuisance_x = np.append(nuisance_x, np.ones_like(featureData_x[:,0])*nv[i]/nv_std,axis=0)
    
mean_0_x = np.mean(unscaled_train0_x, axis=0, keepdims=True)
std_0_x  = np.std(unscaled_train0_x, axis=0, keepdims=True)
feature_x = (feature_x - (mean_0_x)) / (std_0_x)
target_x = np.stack([target_x, nuisance_x], axis=1)

#pmodel = taylor.CubicModel(4).to(device)#taylor.QuadraticModel(4).to(device)
pmodel = taylor.LinearModel(4).to(device)
state_dict = torch.load(config_json["shape_models"][0], map_location=device) 
pmodel.load_state_dict(state_dict)
pmodel.eval()

X   = torch.as_tensor(feature_x, dtype=torch.float32, device=device)   # shape [N, inputsize] as a tensor
true = torch.as_tensor(target_x,  dtype=torch.float32, device=device)   # cols [y, w, nu] as a tensor
polyn_degree_ = 1 #2

with torch.no_grad():
    fx_x = pmodel(X,true)
    shape_coeffs_x = pmodel.get_coeffs(X) # [N, polyn_degree] 

featureRef = X[0:N_ref,]
featureData = X[N_ref:N_ref+N_Bkg_Pois,]
feature   = torch.concatenate((featureData, featureRef), dim=0)

coeffs_Ref = shape_coeffs_x[0:N_ref,:]
coeffs_Data = shape_coeffs_x[N_ref:N_ref+N_Bkg_Pois,:]
shape_coeffs = torch.concatenate((coeffs_Data,coeffs_Ref),dim=0)

###########################################################################################

# keep shape [N,1] like your numpy [:,0:1]
targetData = torch.ones_like(featureData[:, 0:1])        # D label = 1
targetRef  = torch.zeros_like(featureRef[:, 0:1])        # R label = 0

weightsData = torch.ones_like(featureData[:, 0:1])
weightsRef  = torch.ones_like(featureRef[:, 0:1]) * (N_D / N_R)  # scalar scale

target      = torch.concatenate((targetData, targetRef), dim=0)
weights     = torch.concatenate((weightsData, weightsRef), dim=0)

target      = torch.concatenate((target, weights), dim=1)

batch_size  = feature.shape[0]
inputsize   = feature.shape[1]
input_dim = feature.shape[1]

print('feature shape', feature.shape)

############################################################################################
if correction!='': #e.g. "NORM" or "SHAPE"
    # ACTIVATE NORM corrections ###################################
    # if we choose to study the shape effects we also keep the normalization ones 
    # (SHAPE is like an upgrade of NORM)
    train_norm_syst_finder=True
    ## initialize parameters
    NUR_N     = torch.tensor([config_json['norm_nuisances_reference']])
    NU_N      = torch.tensor([0. ]) #initialization
    SIGMA_N   = torch.tensor([sigma_n]) 
    input_dim = feature.shape[1]
    input_shape = (None, input_dim)
    Norm_syst_finder= gof.norm_syst_finder(input_shape, 
                                           nu=NUR_N, 
                                           nu_central=NU0_N, 
                                           nu_ref=NUR_N, 
                                           nu_sigma=SIGMA_N, 
                                           trainable=train_norm_syst_finder)

if correction=="SHAPE":

    # ACTIVATE SHAPE corrections ######################################
    train_shape_syst_finder=True
    Shape_syst_finder_list = []
    Shape_coeffs_list = []
    Shape_coeffs_list.append(shape_coeffs)
    
    for i, k in enumerate(shape_nuisances_id_list): #note: len(shape_nuisances_id_list)=1
        ##initialize parameters
        #filename = config_json["shape_models"][i]
        #hidden_layers_, nu_std, polyn_degree_, epoch_ = taylor.parse_model_filename(filename)
        
        NUR_S     = torch.tensor([config_json['shape_nuisances_reference'][i]])
        NU_S_STD  = torch.tensor([nv_std])
        SIGMA_S   = torch.tensor([sigma_s[i]])
        input_dim = feature.shape[1]
        input_shape = (None, input_dim)
        
        """pmodel = taylor.PolynomialModel(input_dim, 
                                        polyn_degree=polyn_degree_, 
                                        hidden_layers=hidden_layers_ )
        pmodel.load_state_dict(torch.load(filename))
        pmodel.to(device)
        
        with torch.no_grad():
            coeffs_i = pmodel.get_coeffs(torch.from_numpy(feature).to(device).float()) # [N, polyn_degree]
            Shape_coeffs_list.append(coeffs_i)"""
        
        Shape_syst_finder_list.append( 
            gof.shape_syst_finder(input_shape, 
                              nu=NUR_S, #initialization
                              nu_central=NU0_S[i], 
                              nu_ref=NUR_S, 
                              nu_sigma=SIGMA_S,
                              nu_std=NU_S_STD*0.1,
                              #nu_std=NU_S_STD*SIGMA_S,
                              polyn_degrees=polyn_degree_,
                              trainable=train_shape_syst_finder))

# Create data loader

def iter_batches_no_shuffle(xb_all, sb_all, yb_all, batch_size):
    """
    xb_all: [N, d]
    sb_all: [N, ...]  (same first dim N)
    yb_all: [N, k]
    Works on CPU or GPU tensors.
    """
    N = xb_all.shape[0]
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        yield xb_all[start:end], sb_all[start:end], yb_all[start:end]

xb_all = feature.to(device=device, dtype=torch.float32)
sb_all = Shape_coeffs_list[0].to(device=device, dtype=torch.float32)
yb_all = target.to(device=device, dtype=torch.float32)

"""train_ds = TensorDataset(xb_all, sb_all, yb_all)
train_loader = DataLoader(train_ds, batch_size=batch_size_, shuffle=False, num_workers=0)"""

#################################################################################
##################### TAU MODEL############################
# initialize the novelty finder

train_novelty_finder=True
Novelty_finder = gof.novelty_finder(input_dim,
                                    architecture=config_json['novelty_finder_architecture'],
                                    activation="sigmoid",
                                    weight_clipping=config_json["novelty_finder_weight_clipping"],
                                    trainable=True,
                                   )


TAU = gof.NP_GOF_sys(input_shape, 
           novelty_finder=Novelty_finder, 
           shape_syst_finder_list=copy.deepcopy(Shape_syst_finder_list), 
           norm_syst_finder=copy.deepcopy(Norm_syst_finder),
           train_novelty_finder=train_novelty_finder, 
           train_norm_syst_finder=train_norm_syst_finder,
           train_shape_syst_finder=train_shape_syst_finder, 
           device=device)

# Move internal modules/objects you passed into TAU to device
util.move_modules_to_device(TAU, device)

batch_size_ = 1000
# optimizer
learning_rate_ = 1.2e-3
optimizer = torch.optim.Adam(TAU.parameters(), lr=learning_rate_)

tau_epoch_losses = []
# Train
for epoch in range(total_epochs_tau):
    running_loss = 0
    optimizer.zero_grad()
    for xb, sb, yb in iter_batches_no_shuffle(xb_all, sb_all, yb_all, batch_size_):
        #xb, sb, yb = xb.to(device), sb.to(device), yb.to(device)
        pred = TAU([xb, sb])
        loss = TAU.loss_evidence(pred, yb)
        loss.backward()
        running_loss += loss.detach().item()
    aux = TAU.loss_auxiliary()
    aux.backward()
    running_loss += aux.detach().item()
    optimizer.step()
    TAU.novelty_finder.clip_weights()
    tau_epoch_losses.append(running_loss)
    if verbose and not epoch%patience_tau:
        TAU.eval()
        print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")
        print('nu_shape', TAU.shape_syst_finder_list[0].nu.detach().item(),
              'nu_norm', TAU.norm_syst_finder.nu.detach().item(),
             )
        TAU.train()

print("TAU model finished training")

#####################################################
#### DELTA model
# reset novelty finder:
train_novelty_finder=False
Novelty_finder=None

# DELTA MODEL
DELTA = gof.NP_GOF_sys(input_shape, 
           novelty_finder=Novelty_finder, 
           shape_syst_finder_list=copy.deepcopy(Shape_syst_finder_list), 
           norm_syst_finder=copy.deepcopy(Norm_syst_finder),
           train_novelty_finder=train_novelty_finder, 
           train_norm_syst_finder=train_norm_syst_finder,
           train_shape_syst_finder=train_shape_syst_finder, 
           device=device)

# reset the trainable nuisance parameters to their initial values:
#DELTA.set_initialization_nu_shapes(NU0_S)
#DELTA.set_initialization_nu_norm(NU0_N)

# Move internal modules/objects you passed into DELTA to device                 
util.move_modules_to_device(DELTA, device)

batch_size_ = 1000
learning_rate_ = 1.00e-3
# optimizer
optimizer = torch.optim.Adam(DELTA.parameters(), lr=learning_rate_)

delta_epoch_losses = []

print("Start training DELTA model")

# Train
for epoch in range(total_epochs_delta):
    running_loss = 0
    optimizer.zero_grad()
    for xb, sb, yb in iter_batches_no_shuffle(xb_all, sb_all, yb_all, batch_size_):
        #xb, sb, yb = xb.to(device), sb.to(device), yb.to(device)
        pred = DELTA([xb, sb])
        loss = DELTA.loss_evidence(pred, yb)
        loss.backward()
        running_loss += loss.detach().item() 
    aux = DELTA.loss_auxiliary()
    aux.backward()
    running_loss+=aux.detach().item()
    optimizer.step()
    delta_epoch_losses.append(running_loss)
    if verbose and not epoch%patience_delta:
        DELTA.eval()
        print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")
        print('nu_shape', DELTA.shape_syst_finder_list[0].nu.detach().item(),
              'nu_norm', DELTA.norm_syst_finder.nu.detach().item(),
             )
        DELTA.train()
        
print("DELTA model finished training")
#####################################################
# EVAL
pred_delta = []
pred_tau = []
targets = []
weights = []
data = []

TAU   = TAU.to(device).eval()
DELTA = DELTA.to(device).eval()

with torch.no_grad():
    running_loss_tau, running_loss_delta = 0, 0
    for xb, sb, yb in iter_batches_no_shuffle(xb_all, sb_all, yb_all, batch_size_):
        #xb, sb, yb = xb.float().to(device), sb.to(device), yb.to(device)
        pred_delta_tmp = DELTA([xb, sb])
        pred_tau_tmp = TAU([xb, sb])
        pred_delta.append(pred_delta_tmp.detach().cpu().numpy())
        pred_tau.append(pred_tau_tmp.detach().cpu().numpy()) 
        targets.append(yb[:, 0].cpu().numpy())
        weights.append(yb[:, 1].cpu().numpy())
        data.append(xb.cpu().numpy())
        running_loss_tau += TAU.loss_evidence(pred_tau_tmp, yb).detach().item()
        running_loss_delta += DELTA.loss_evidence(pred_delta_tmp, yb).detach().item()  
    running_loss_tau+= TAU.loss_auxiliary().detach().item()
    running_loss_delta+= DELTA.loss_auxiliary().detach().item()
    
pred_delta = np.concatenate(pred_delta, axis=0)
pred_tau = np.concatenate(pred_tau, axis=0)
targets = np.concatenate(targets, axis=0)
weights = np.concatenate(weights, axis=0)
data = np.concatenate(data, axis=0)

output_tau_ref = pred_tau[targets==0]
output_delta_ref = pred_delta[targets==0]

weight_DATA = weights[targets==1]
weight_REF = weights[targets==0]

DATA = data[targets==1]
REF = data[targets==0]

tau_OBS = -2*running_loss_tau
delta_OBS = -2*running_loss_delta
print("tau:", tau_OBS, "delta:", delta_OBS, "tau-delta:", tau_OBS - delta_OBS)

#####################################################
# collect results           
config_results = {}
## loss histories
config_results['tau_loss_history'] = tau_epoch_losses
config_results['delta_loss_history'] = delta_epoch_losses
## TAU and DELTA values
config_results['tau'] = float(tau_OBS)
config_results['delta'] = float(delta_OBS)
## nus:
config_results['shape_nuisances_id_list']=shape_nuisances_id_list
### nu fit by TAU
shape_nuisances_list_tau_fit = TAU.get_nu_shape_list()

for i,k in enumerate(shape_nuisances_id_list):
    config_results[k+'_tau_fit'] = shape_nuisances_list_tau_fit[i].detach().cpu().item()
norm_nuisance_tau_fit = TAU.get_nu_norm().detach().cpu().item()
config_results['norm_tau_fit'] = norm_nuisance_tau_fit
### nu fit by DELTA
shape_nuisances_list_delta_fit = DELTA.get_nu_shape_list()
for i,k in enumerate(shape_nuisances_id_list):
    config_results[k+'_delta_fit'] = shape_nuisances_list_delta_fit[i].detach().cpu().item()
norm_nuisance_delta_fit = DELTA.get_nu_norm().detach().cpu().item()
config_results['norm_delta_fit'] = norm_nuisance_delta_fit

# save json
with open(OUTPUT_PATH+OUTPUT_FILE_ID+'results.json', 'w') as outfile:
    json.dump(util.json_safe(config_results), outfile, indent=4)

