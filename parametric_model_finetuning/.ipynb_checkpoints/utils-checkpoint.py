import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import random
from scipy import stats
from scipy.stats import chi2_contingency

def clip_weights(module):
    if isinstance(module, nn.Linear):
        module.weight.data.clamp_(-1.0, 1.0)

class LinearModel(nn.Module):
    
    def __init__(self, input_dim):
        
        super().__init__()
        
        self.network1 = nn.Sequential(
        nn.Linear(input_dim,512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256,128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1))
        
        
    def forward(self, x: torch.Tensor,true:torch.Tensor) -> torch.Tensor:
        a1 = self.network1(x)[:,0]
        nu = true[:,1]
        f1 = a1*nu
        return f1
        
    def get_coeffs(self, x: torch.Tensor):
        a1 = self.network1(x)[:, 0]   # [N]
        return torch.stack((a1), dim=1)   # [N, 2]

        
class QuadraticModel(nn.Module):
    
    def __init__(self, input_dim):
        
        super().__init__()
        
        self.network1 = nn.Sequential(
        nn.Linear(input_dim,512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256,128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1))
        
        self.network2 = nn.Sequential(
        nn.Linear(input_dim,256),
        nn.ReLU(),
        nn.Linear(256,128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1))
        
    def forward(self, x: torch.Tensor,true:torch.Tensor) -> torch.Tensor:
        a1 = self.network1(x)[:,0]
        a2 = self.network2(x)[:,0]
        nu = true[:,1]
        f1 = a1*nu + a2*(nu**2)
        return f1
        
    def get_coeffs(self, x: torch.Tensor):
        a1 = self.network1(x)[:, 0]   # [N]
        a2 = self.network2(x)[:, 0]   # [N]
        return torch.stack((a1, a2), dim=1)   # [N, 2]


class CubicModel(nn.Module):
    
    def __init__(self, input_dim):
        
        super().__init__()
        
        self.network1 = nn.Sequential(
        nn.Linear(input_dim,512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256,128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1))
        
        self.network2 = nn.Sequential(
        nn.Linear(input_dim,256),
        nn.ReLU(),
        nn.Linear(256,128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1))

        #to-do
        self.network3 = nn.Sequential(
        nn.Linear(input_dim,128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1))
        
    def forward(self, x: torch.Tensor,true:torch.Tensor) -> torch.Tensor:
        a1 = self.network1(x)[:,0]
        a2 = self.network2(x)[:,0]
        a3 = self.network3(x)[:,0]
        
        nu = true[:,1]
        
        f1 = a1*nu + a2*(nu**2) + a3*(nu**3)
        
        return f1
          
    def get_coeffs(self, x: torch.Tensor):
        a1 = self.network1(x)[:, 0]   # [N]
        a2 = self.network2(x)[:, 0]   # [N]
        a3 = self.network3(x)[:, 0]
        
        return torch.stack((a1, a2, a3), dim=1)   # [N, 3]   


class ExpoLoss(nn.Module):

    def __init__(self):
        super().__init__()  # initializes base nn.Module

    def forward(self, true: torch.Tensor, fx: torch.Tensor) -> torch.Tensor:
        
        y  = true[:, 0]
        w = true [:,2]
        c = torch.sigmoid(-fx)
        loss = torch.mean((y * w * c**2) + ((1 - y) * w * (1 - c)**2))
        #loss = torch.mean((y * c**2) + ((1 - y) * (1 - c)**2))
        return loss


def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

    
def load_data(nv, filename, N_Bkg_Pois= 450000, N_ref= 450000, sigma_s = 0.1, alt_name = False, nv_std = np.array([-1., -0.5, 0.5, 1.])):
    
    #nv   = np.array([-1.5, -0.5, 0.5, 1.5])
    #nv_std = np.std(nv)
    
    N_diff = N_Bkg_Pois - N_ref
    
    neg3_start = N_ref
    neg3_end = N_ref+N_Bkg_Pois
    neg1_start = (2*N_ref)+N_Bkg_Pois
    neg1_end = 2*(N_ref+N_Bkg_Pois)
    pos1_start = (3*N_ref)+(2*N_Bkg_Pois)
    pos1_end = (3*N_ref)+(3*N_Bkg_Pois)
    pos3_start = (4*N_ref)+(3*N_Bkg_Pois)
    pos3_end = (4*N_ref)+(4*N_Bkg_Pois)
    
    #Load the data
    data = np.load(filename);
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if alt_name:
        unscaled_train0 = data['gx_0'][0:N_ref,:]
        scaled_1_train0 = data[f'gx_pos{nv[-2]}'][0:N_Bkg_Pois,:]
        scaled_3_train0 = data[f'gx_pos{nv[-1]}'][0:N_Bkg_Pois,:]
        scaled_neg1_train0 = data[f'gx_neg{nv[-2]}'][0:N_Bkg_Pois,:]
        scaled_neg3_train0 = data[f'gx_neg{nv[-1]}'][0:N_Bkg_Pois,:]
    else:
        unscaled_train0 = data['arr_2'][0:N_ref,:]
        scaled_1_train0 = data['arr_3'][0:N_Bkg_Pois,:]
        scaled_3_train0 = data['arr_4'][0:N_Bkg_Pois,:]
        scaled_neg1_train0 = data['arr_1'][0:N_Bkg_Pois,:]
        scaled_neg3_train0 = data['arr_0'][0:N_Bkg_Pois,:]
    
    featureData_lst = [scaled_neg3_train0,scaled_neg1_train0,scaled_1_train0,scaled_3_train0];
    
    targetRef  = np.zeros_like(unscaled_train0[:,0])
    feature    = np.empty((0,4))
    target     = np.empty((0))
    nuisance   = np.empty((0))
    weights    = np.empty((0))
    
    for i in range(0,len(nv)):
        
        featureData = featureData_lst[i]
        targetData  = np.ones_like(featureData[:,0])
    
        feature = np.append(feature, unscaled_train0,axis=0)
        feature = np.append(feature, featureData, axis=0)
        
        target  = np.append(target, targetRef,  axis=0)
        target  = np.append(target, targetData, axis=0)
        
        nuisance = np.append(nuisance, np.ones_like(unscaled_train0[:,0])*nv[i]/nv_std,axis=0)
        nuisance = np.append(nuisance, np.ones_like(featureData[:,0])*nv[i]/nv_std,axis=0)

        weights =  np.append(weights,np.ones_like(unscaled_train0[:,0])* N_Bkg_Pois*1./N_ref)
        weights =  np.append(weights,np.ones_like(featureData[:,0]))
        
    mean_0 = np.mean(unscaled_train0, axis=0, keepdims=True)
    std_0  = np.std(unscaled_train0, axis=0, keepdims=True)
    feature = (feature - (mean_0)) / (std_0)
    target = np.stack([target, nuisance, weights], axis=1)

    return feature,target

#to-do
def prepare_training(feature,target,load_model, load_model_path,total_epochs, patience,batch_size,model_name="quadratic"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #X   = torch.as_tensor(feature, dtype=torch.float32, device=device)   # shape [N, inputsize] as a tensor
    #true = torch.as_tensor(target,  dtype=torch.float32, device=device)   # cols [y, w, nu] as a tensor
    X = torch.as_tensor(feature, dtype=torch.float32)
    true = torch.as_tensor(target, dtype=torch.float32)
    
    ds   = TensorDataset(X, true) # paired up datasets
    dl   = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    num_blocks = int(total_epochs // patience)

    if model_name == "linear":
        model = LinearModel(4).to(device);
    elif model_name == "quadratic":
        model = QuadraticModel(4).to(device);
    elif model_name == "cubic":
        model = CubicModel(4).to(device);
        
    if load_model:
        model.load_state_dict(torch.load(load_model_path,map_location=torch.device(device)))
        
    """if optimizer_name = "Adam":
        optimizer = torch.optim.Adam([{"params": [p for p in model.parameters() if p.requires_grad],"lr": lr, "weight_decay":regularization}])
    
    elif optimizer_name = "AdamW":
        optimizer = torch.optim.AdamW([{"params": [p for p in model.parameters() if p.requires_grad],"lr": lr, "weight_decay":regularization}])"""
    
    return ds, dl, X, true, model, num_blocks, patience

def get_fitted_data_fx(N_ref, N_Bkg_Pois,fx,feature_np,targets):  
    
    N_diff = N_Bkg_Pois - N_ref
    
    neg3_start = N_ref
    neg3_end = N_ref+N_Bkg_Pois
    neg1_start = (2*N_ref)+N_Bkg_Pois
    neg1_end = 2*(N_ref+N_Bkg_Pois)
    pos1_start = (3*N_ref)+(2*N_Bkg_Pois)
    pos1_end = (3*N_ref)+(3*N_Bkg_Pois)
    pos3_start = (4*N_ref)+(3*N_Bkg_Pois)
    pos3_end = (4*N_ref)+(4*N_Bkg_Pois)

    ref = feature_np[0:N_ref]
    data_negnv2 = feature_np[neg3_start:neg3_end]
    data_negnv1 = feature_np[neg1_start:neg1_end]
    data_nv1 = feature_np[pos1_start:pos1_end]
    data_nv2 = feature_np [pos3_start:pos3_end]

    fx_negnv2 = fx[0:neg3_start]
    fx_negnv1 = fx[neg3_end:neg1_start]
    fx_nv1 = fx[neg1_end:pos1_start]
    fx_nv2 = fx[pos1_end:pos3_start]
    
    weights_ref = targets[0:N_ref,2]
    weights_negnv2 = targets[neg3_start:neg3_end,2]
    weights_negnv1 = targets[neg1_start:neg1_end,2]
    weights_nv1 = targets[pos1_start:pos1_end,2]
    weights_nv2 = targets[pos3_start:pos3_end,2]
    
    data_list = [data_negnv1,data_nv1,data_negnv2,data_nv2]
    fx_list = [fx_negnv1,fx_nv1,fx_negnv2,fx_nv2]
    weights_list =[weights_negnv1,weights_nv1,weights_negnv2,weights_nv2]

    return data_list,fx_list,ref,weights_list,weights_ref
    
def plot_stephist(feature_np, delta_np, targets,run_num, run_series, nvs,  N_Bkg_Pois, N_ref):

    fig, axs = plt.subplots(2, 4, figsize=(40,12), sharey=True)
    axs = axs.flatten()

    [data_negnv1,data_nv1,data_negnv2,data_nv2], [fx_negnv1,fx_nv1,fx_negnv2,fx_nv2], ref, [weights_negnv1,weights_nv1,weights_negnv2,weights_nv2],weights_ref = get_fitted_data_fx(N_ref, N_Bkg_Pois, delta_np, feature_np, targets)
    
    for i in range(0, int(len(axs) / 2)):

        #--------------------------Unweighted/Ground Truth-------------------------------------
        h = axs[i].hist(ref[:, i], bins=45, alpha=0.3, edgecolor='black',weights = weights_ref,
            linewidth=0.0, label='Original', color='grey', density=False)

        counts_orig = h[0]
        bin_edges   = h[1]

        axs[i].hist(data_nv2[:, i], bins=bin_edges, histtype='step',linewidth=2.1,weights = weights_nv2,
            color='mediumturquoise', density=False, label=f'Scaled, nv = {nvs[-1]}')


        axs[i].hist(data_nv1[:, i], bins=bin_edges, histtype='step', linewidth=2.1,weights = weights_nv1,
            color='dodgerblue', density=False, label=f'Scaled, nv = {nvs[-2]}')


        axs[i].hist(data_negnv1[:, i], bins=bin_edges, histtype='step', linewidth=2.1,weights = weights_negnv1,
            color='blue', density=False, label=f'Scaled, nv = {nvs[1]}')
        
        axs[i].hist(data_negnv2[:, i], bins=bin_edges, histtype='step', linewidth=2.1,weights = weights_negnv2,
            color='midnightblue', density=False,label=f'Scaled, nv = {nvs[0]}')

        axs[i].set_title(f'Feature {i} (Ground Truth)', fontsize=20)
        axs[i].legend(fontsize=10)

        #------------------------Weighted/Parametric Fitting-----------------------------------
        axs[i+4].hist(ref[:,i], bins=bin_edges, alpha=0.3, edgecolor='black', weights = weights_ref,
            linewidth=0.0, label='Original', color='grey', density=False)

        # nv = 0.1 * 3
        axs[i+4].hist(ref[:, i],bins=bin_edges, histtype='step', weights=np.exp(fx_nv2)*weights_ref,
            linewidth=2.1, color='mediumturquoise', density=False, label=f'Scaled, nv = {nvs[-1]}')

        # nv = 0.1 * 1
        axs[i+4].hist(ref[:, i],bins=bin_edges, histtype='step', weights=np.exp(fx_nv1)*weights_ref,
            linewidth=2.1, color='dodgerblue',density=False,label=f'Scaled, nv = {nvs[-2]}')

        # nv = 0.1 * -1
        axs[i+4].hist(ref[:, i], bins=bin_edges, histtype='step', weights=np.exp(fx_negnv1)*weights_ref,
            linewidth=2.1, color='blue', density=False, label=f'Scaled, nv = {nvs[1]}')

        # nv = 0.1 * -3
        axs[i+4].hist(ref[:, i], bins=bin_edges, histtype='step', weights=np.exp(fx_negnv2)*weights_ref,
            linewidth=2.1,color='midnightblue', density=False, label=f'Scaled, nv = {nvs[0]}')

        axs[i+4].set_title(f'Feature {i} (Parametric Fit)')
        axs[i+4].legend(fontsize=10)

    plt.tight_layout()
    plt.suptitle(f"Run {run_num}, {run_series}", y=1.03, fontsize=36)
    plt.show()


    
def plot_reco_og(N_ref, N_Bkg_Pois, fx, feature_np, targets, nv_list):

    data_list,fx_list, ref,weights_list,weights_ref = get_fitted_data_fx(N_ref, N_Bkg_Pois, fx, feature_np, targets)
    #[data_negnv1,data_nv1,data_negnv2,data_nv2], [fx_negnv1,fx_nv1,fx_negnv2,fx_nv2],ref = get_fitted_data_fx(N_ref, N_Bkg_Pois, fx, feature_np)
    for j,data in enumerate(data_list):
        
        fig, axs = plt.subplots(1, 4, figsize=(30,6), sharey=True)
        data_list,fx_list, ref,weights_list,weights_ref = get_fitted_data_fx(N_ref, N_Bkg_Pois, fx, feature_np, targets)
        
        for i in range(0, 4):
                h = axs[i].hist(
                    ref[:,i],
                    bins=45,
                    weights = weights_ref,
                    alpha=0.3,
                    edgecolor='black',
                    linewidth=0.0,
                    label='Original embedding,nv = 0',
                    color='grey',
                    density=False
                )
        
                # h[0] = counts, h[1] = bin edges
                counts_orig = h[0]
                bin_edges   = h[1]
                
                axs[i].hist(
                    data[:,i],
                    bins=bin_edges,
                    weights = weights_list[i],
                    histtype='step',
                    linewidth=1.8,
                    color='dodgerblue',
                    density=False,
                    label=f'nv = {nv_list[j]} Embedding'
                )
                
                axs[i].hist(
                    ref[:,i],
                    bins=bin_edges,
                    weights = np.exp(fx_list[j])*weights_ref,
                    histtype='step',
                    linewidth=1.8,
                    color='red',
                    density=False,
                    label=f'nv = {nv_list[j]} Fitted Model'
                )
            
                axs[i].set_title(f'Feature_{i}', fontsize=20)
                axs[i].legend(fontsize=10)
    
        plt.tight_layout()
        plt.show()
    
    #Difference

def plot_log_sbs(feature_np, fx, nv_vals, 
                 N_ref, N_Bkg_Pois, targets,
                 x_target_list = [-1.2, -0.9,-0.6,-0.3, 0, 0.3, 0.6, 0.9, 1.2]):
    
    #nv_vals = [-0.15,-0.05, 0.0, 0.05, 0.15]
    [data_negnv1,data_nv1,data_negnv2,data_nv2], [fx_negnv1,fx_nv1,fx_negnv2,fx_nv2],ref,[weights_negnv1,weights_nv1,weights_negnv2,weights_nv2], weights_ref = get_fitted_data_fx(N_ref, N_Bkg_Pois, fx, feature_np, targets)
    colors = ["blue","black","red","green","orange","purple","grey","brown","pink","coralred"]
       
    _, global_bins = np.histogram(ref[:,0],bins=40)
        
    global_bin_centers = (global_bins[:-1] + global_bins[1:]) / 2 #find the bin center for each bin
        
    fig, axs = plt.subplots(1, 4, figsize=(24, 6),sharey=False)
    axs = axs.flatten()
            
    for i in range(0,4):
        
            counts_orig, _ = np.histogram(feature_np[0:N_ref,i], bins=global_bins, weights =  weights_ref,density=False)
        
            counts_negnv2_unw, _ = np.histogram(data_negnv2[:,i], bins=global_bins, weights = weights_negnv2, density=False)
            counts_negnv1_unw, _ = np.histogram(data_negnv1[:,i], bins=global_bins, weights = weights_negnv1,density=False)
            counts_orig_unw, _ = np.histogram(ref[:,i], bins=global_bins, weights = weights_ref, density=False)
            counts_nv1_unw, _ = np.histogram(data_nv1[:,i], bins=global_bins, weights = weights_nv1, density=False)
            counts_nv2_unw, _ = np.histogram(data_nv2[:,i], bins=global_bins, weights = weights_nv2, density=False)
        
            counts_negnv2_w, _ = np.histogram(ref[:,i], bins=global_bins, weights =  weights_ref*np.exp(fx_negnv2),density=False)
            counts_negnv1_w, _ = np.histogram(ref[:,i], bins=global_bins, weights =  weights_ref*np.exp(fx_negnv1),density=False)
            counts_orig_w, _ = np.histogram(ref[:,i], bins=global_bins, weights = weights_ref, density=False)
            counts_nv1_w, _ = np.histogram(ref[:,i], bins=global_bins, weights =  weights_ref*np.exp(fx_nv1),density=False)
            counts_nv2_w, _ = np.histogram(ref[:,i], bins=global_bins, weights =  weights_ref*np.exp(fx_nv2),density=False)

            for j,x_target in enumerate(x_target_list):
            
                shared_bin_idx = np.argmin(np.abs(global_bin_centers - x_target)) #find the index of the bin closest to x_target
                shared_bin_center = global_bin_centers[shared_bin_idx] #store the x-value
            
                y_counts_unw = np.array([
                            np.log(counts_negnv2_unw[shared_bin_idx]/counts_orig[shared_bin_idx]),
                            np.log(counts_negnv1_unw[shared_bin_idx]/counts_orig[shared_bin_idx]),
                            np.log(counts_orig_unw[shared_bin_idx]/counts_orig[shared_bin_idx]),
                            np.log(counts_nv1_unw[shared_bin_idx]/counts_orig[shared_bin_idx]),
                            np.log(counts_nv2_unw[shared_bin_idx]/counts_orig[shared_bin_idx])])
                
                y_counts_w = np.array([
                            np.log(counts_negnv2_w[shared_bin_idx]/counts_orig_w[shared_bin_idx]),
                            np.log(counts_negnv1_w[shared_bin_idx]/counts_orig_w[shared_bin_idx]),
                            np.log(counts_orig_w[shared_bin_idx]/counts_orig_w[shared_bin_idx]),
                            np.log(counts_nv1_w[shared_bin_idx]/counts_orig_w[shared_bin_idx]),
                            np.log(counts_nv2_w[shared_bin_idx]/counts_orig_w[shared_bin_idx])])

                axs[i].errorbar(nv_vals, y_counts_unw, fmt='o-', color=colors[j],
                                    capsize=4, elinewidth=1.1, markeredgewidth=1, markersize=3, label=f'x = {x_target},Embeddings')
                axs[i].errorbar(nv_vals, y_counts_w, fmt='*-.', color=colors[j],
                                    capsize=4, elinewidth=0.9, markeredgewidth=1, markersize=5, label=f'x = {x_target},Parametric')
            
            axs[i].set_title(f'Log Ratios Feature {i}')
            axs[i].set_xlabel('ν')
            axs[i].set_ylabel('log(normalized counts)')
            axs[i].grid(True, linestyle=':', alpha=0.5)
            axs[i].legend(fontsize = 9)
    
    plt.suptitle(f'Embeddings Log Ratios', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

#to-do: function that randomizes data it loads everytime the function is called, and uses weights to do plots, returns the bins
def twosamples_tests(feature_np, fx, nv_only_val, N_ref, N_Bkg_Pois, targets,bin_num=45):
    """
    For each feature i in {0,1,2,3}:
      - KS test between (data_*[:,i]) and (ref[:,i] reweighted by exp(fx_*))
      - Chi2 contingency test between unweighted vs weighted hist counts
    """

    # returns: [data_negnv1,data_nv1,data_negnv2,data_nv2], [fx_negnv1,fx_nv1,fx_negnv2,fx_nv2], ref
    (data_negnv1, data_nv1, data_negnv2, data_nv2), (fx_negnv1, fx_nv1, fx_negnv2, fx_nv2), ref,[weights_negnv1,weights_nv1,weights_negnv2,weights_nv2], weights_ref = get_fitted_data_fx(N_ref, N_Bkg_Pois, fx, feature_np, targets)

    for i in range(4):
        print(f"\nFeature {i}:")

        _, bin_edges = np.histogram(ref[:, i], bins=bin_num, weights = weights_ref ,density=False)

        # Unweighted hist counts from the *data* samples
        counts_negnv2_unw, _ = np.histogram(data_negnv2[:, i], bins=bin_edges, weights = weights_negnv2 ,density=False)
        counts_negnv1_unw, _ = np.histogram(data_negnv1[:, i], bins=bin_edges, weights = weights_negnv1,density=False)
        counts_nv1_unw,  _   = np.histogram(data_nv1[:, i],    bins=bin_edges, weights = weights_nv1, density=False)
        counts_nv2_unw,  _   = np.histogram(data_nv2[:, i],    bins=bin_edges,  weights = weights_nv2,density=False)

        # Weighted hist counts by reweighting *ref* with exp(fx_*)
        # IMPORTANT: fx_* must be shape (N_ref, 4) matching ref.
        counts_negnv2_w, _ = np.histogram(ref[:, i], bins=bin_edges, weights=weights_ref*np.exp(fx_negnv2), density=False)
        counts_negnv1_w, _ = np.histogram(ref[:, i], bins=bin_edges, weights=weights_ref*np.exp(fx_negnv1), density=False)
        counts_nv1_w,  _   = np.histogram(ref[:, i], bins=bin_edges, weights=weights_ref*np.exp(fx_nv1),    density=False)
        counts_nv2_w,  _   = np.histogram(ref[:, i], bins=bin_edges, weights=weights_ref*np.exp(fx_nv2),    density=False)

        # Each tables should be shape (2, nbins) for chi2_contingency
        counts_list = [
            np.vstack([counts_negnv2_unw, counts_negnv2_w]),
            np.vstack([counts_negnv1_unw, counts_negnv1_w]),
            np.vstack([counts_nv1_unw,    counts_nv1_w]),
            np.vstack([counts_nv2_unw,    counts_nv2_w]),
        ]

        ks_table = [([counts_negnv2_unw, counts_negnv2_w]),
            ([counts_negnv1_unw, counts_negnv1_w]),
            ([counts_nv1_unw,    counts_nv1_w]),
            ([counts_nv2_unw,    counts_nv2_w]),
        ]

        # -------------------- print out test stats --------------------
        for j, tables in enumerate(counts_list):
            x, y = ks_table[j]

            ks_stat, ks_p = stats.ks_2samp(ks_table[j][0],ks_table[j][1])
            chi2_stat, chi2_p, dof, expected = chi2_contingency(tables)

            print(f" NV = {nv_only_val[j]}:")
            print(f"   KS:  stat={ks_stat:.6g}, p={ks_p:.6g}")
            print(f"   Chi2: stat={chi2_stat:.6g}, p={chi2_p:.6g}, dof={dof}")
            print(f"   |Δcounts| sum = {np.sum(np.abs(tables[0] - tables[1])):.6g},|counts| sum = {np.sum(tables[0]):.6g} and {np.sum(tables[1]):.6g}")
            print(f"   |Δcounts| = {np.abs(tables[0] - tables[1])}")
            print(f"   |Δcounts|/|counts| = {np.abs(tables[0] - tables[1])/tables[1]}")