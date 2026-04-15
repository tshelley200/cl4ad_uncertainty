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
    
    neg1_start = N_ref
    neg1_end = N_ref+N_Bkg_Pois
    pos1_start = (2*N_ref)+N_Bkg_Pois
    pos1_end = 2*(N_ref+N_Bkg_Pois)
    
    #Load the data
    data = np.load(filename);
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if alt_name:
        unscaled_train0 = data['gx_0'][0:N_ref,:]
        scaled_1_train0 = data[f'gx_pos{nv[1]}'][0:N_Bkg_Pois,:]
        #scaled_3_train0 = data[f'gx_pos{nv[-1]}'][0:N_Bkg_Pois,:]
        scaled_neg1_train0 = data[f'gx_neg{nv[1]}'][0:N_Bkg_Pois,:]
        #scaled_neg3_train0 = data[f'gx_neg{nv[-1]}'][0:N_Bkg_Pois,:]
    else:
        unscaled_train0 = data['arr_2'][0:N_ref,:]
        scaled_1_train0 = data['arr_3'][0:N_Bkg_Pois,:]
        #scaled_3_train0 = data['arr_4'][0:N_Bkg_Pois,:]
        scaled_neg1_train0 = data['arr_1'][0:N_Bkg_Pois,:]
        #scaled_neg3_train0 = data['arr_0'][0:N_Bkg_Pois,:]
    
    featureData_lst = [scaled_neg1_train0,scaled_1_train0];
    
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
    
    neg1_start = N_ref
    neg1_end = N_ref+N_Bkg_Pois
    pos1_start = (2*N_ref)+N_Bkg_Pois
    pos1_end = 2*(N_ref+N_Bkg_Pois)

    ref = feature_np[0:N_ref]

    data_negnv1 = feature_np[neg1_start:neg1_end]
    data_nv1 = feature_np[pos1_start:pos1_end]
   
    fx_negnv1 = fx[0:neg1_start]
    fx_nv1 = fx[neg1_end:pos1_start]
    
    weights_ref = targets[0:N_ref,2]
    weights_negnv1 = targets[neg1_start:neg1_end,2]
    weights_nv1 = targets[pos1_start:pos1_end,2]
    
    data_list = [data_negnv1,data_nv1]
    fx_list = [fx_negnv1,fx_nv1]
    weights_list =[weights_negnv1,weights_nv1]

    return data_list,fx_list,ref,weights_list,weights_ref



def plot_stephist(feature_np, delta_np, targets, run_num, run_series, nvs, N_Bkg_Pois, N_ref):

    fig, axs = plt.subplots(
        4, 4,
        figsize=(40, 18),
        sharex='col',
        gridspec_kw={'height_ratios': [3.2, 3.2, 1.2, 1.2], 'hspace': 0.22}
    )

    [data_negnv1, data_nv1], [fx_negnv1, fx_nv1], ref, [weights_negnv1, weights_nv1], weights_ref = get_fitted_data_fx(
        N_ref, N_Bkg_Pois, delta_np, feature_np, targets
    )
    
    for i in range(4):

        ax_emb = axs[0, i]
        ax_fit = axs[1, i]
        ax_ratio_emb = axs[2, i]
        ax_ratio_fit = axs[3, i]

        # -------------------------- Embedding histogram --------------------------
        h = ax_emb.hist(
            ref[:, i],
            bins=45,
            alpha=0.3,
            edgecolor='black',
            weights=weights_ref,
            linewidth=0.0,
            label='Original',
            color='grey',
            density=False
        )

        counts_orig = h[0]
        bin_edges = h[1]
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        counts_nv1, _, _ = ax_emb.hist(
            data_nv1[:, i],
            bins=bin_edges,
            histtype='step',
            linewidth=2.1,
            weights=weights_nv1,
            color='dodgerblue',
            density=False,
            label=f'Scaled, nv = {nvs[1]}'
        )

        counts_negnv1, _, _ = ax_emb.hist(
            data_negnv1[:, i],
            bins=bin_edges,
            histtype='step',
            linewidth=2.1,
            weights=weights_negnv1,
            color='navy',
            density=False,
            label=f'Scaled, nv = {nvs[0]}'
        )

        ax_emb.set_title(f'Feature {i} (Embedding)', fontsize=20)
        ax_emb.legend(fontsize=10)

        # -------------------------- Fitted histogram --------------------------
        ax_fit.hist(
            ref[:, i],
            bins=bin_edges,
            alpha=0.3,
            edgecolor='black',
            weights=weights_ref,
            linewidth=0.0,
            label='Original',
            color='grey',
            density=False
        )

        counts_fit_nv1, _, _ = ax_fit.hist(
            ref[:, i],
            bins=bin_edges,
            histtype='step',
            weights=np.exp(fx_nv1) * weights_ref,
            linewidth=2.1,
            color='dodgerblue',
            density=False,
            label=f'Scaled, nv = {nvs[1]}'
        )

        counts_fit_negnv1, _, _ = ax_fit.hist(
            ref[:, i],
            bins=bin_edges,
            histtype='step',
            weights=np.exp(fx_negnv1) * weights_ref,
            linewidth=2.1,
            color='navy',
            density=False,
            label=f'Scaled, nv = {nvs[0]}'
        )

        ax_fit.set_title(f'Feature {i} (Parametric Fit)', fontsize=20)
        ax_fit.legend(fontsize=10)

        # -------------------------- Embedding log-ratio --------------------------
        ratio_nv1 = np.divide(
            counts_nv1,
            counts_orig,
            out=np.full_like(counts_nv1, np.nan, dtype=float),
            where=(counts_orig > 0) & (counts_nv1 > 0)
        )
        ratio_negnv1 = np.divide(
            counts_negnv1,
            counts_orig,
            out=np.full_like(counts_negnv1, np.nan, dtype=float),
            where=(counts_orig > 0) & (counts_negnv1 > 0)
        )

        log_ratio_nv1 = np.log(ratio_nv1)
        log_ratio_negnv1 = np.log(ratio_negnv1)

        ax_ratio_emb.axhline(0.0, linestyle='--', linewidth=1.2, color='black')
        ax_ratio_emb.plot(
            bin_centers, log_ratio_nv1,
            marker='o', linestyle='-',
            linewidth=1.5, markersize=3.5,
            color='dodgerblue'
        )
        ax_ratio_emb.plot(
            bin_centers, log_ratio_negnv1,
            marker='o', linestyle='-',
            linewidth=1.5, markersize=3.5,
            color='navy'
        )
        ax_ratio_emb.set_ylabel('log(Emb / Orig)', fontsize=11)
        ax_ratio_emb.tick_params(axis='both', labelsize=10)

        # -------------------------- Fit log-ratio --------------------------
        ratio_fit_nv1 = np.divide(
            counts_fit_nv1,
            counts_orig,
            out=np.full_like(counts_fit_nv1, np.nan, dtype=float),
            where=(counts_orig > 0) & (counts_fit_nv1 > 0)
        )
        ratio_fit_negnv1 = np.divide(
            counts_fit_negnv1,
            counts_orig,
            out=np.full_like(counts_fit_negnv1, np.nan, dtype=float),
            where=(counts_orig > 0) & (counts_fit_negnv1 > 0)
        )

        log_ratio_fit_nv1 = np.log(ratio_fit_nv1)
        log_ratio_fit_negnv1 = np.log(ratio_fit_negnv1)

        ax_ratio_fit.axhline(0.0, linestyle='--', linewidth=1.2, color='black')
        ax_ratio_fit.plot(
            bin_centers, log_ratio_fit_nv1,
            marker='o', linestyle='-',
            linewidth=1.5, markersize=3.5,
            color='dodgerblue'
        )
        ax_ratio_fit.plot(
            bin_centers, log_ratio_fit_negnv1,
            marker='o', linestyle='-',
            linewidth=1.5, markersize=3.5,
            color='navy'
        )
        ax_ratio_fit.set_ylabel('log(Fit / Orig)', fontsize=11)
        ax_ratio_fit.tick_params(axis='both', labelsize=10)
        ax_ratio_fit.set_xlabel(f'Feature {i}', fontsize=12)

        # -------------------------- Shared ylim only within same feature --------------------------
        all_ratio_vals = np.concatenate([
            log_ratio_nv1,
            log_ratio_negnv1,
            log_ratio_fit_nv1,
            log_ratio_fit_negnv1
        ])

        finite_vals = all_ratio_vals[np.isfinite(all_ratio_vals)]

        if len(finite_vals) > 0:
            ymin = finite_vals.min()
            ymax = finite_vals.max()

            if ymin == ymax:
                pad = 0.1 if ymin == 0 else 0.1 * abs(ymin)
            else:
                pad = 0.08 * (ymax - ymin)

            ax_ratio_emb.set_ylim(ymin - pad, ymax + pad)
            ax_ratio_fit.set_ylim(ymin - pad, ymax + pad)

    plt.tight_layout()
    plt.suptitle(f"Run {run_num}, {run_series}", y=1.01, fontsize=36)
    plt.show()




    
def plot_reco_og(N_ref, N_Bkg_Pois, fx, feature_np, targets, nv_list):

    data_list,fx_list, ref,weights_list,weights_ref = get_fitted_data_fx(N_ref, N_Bkg_Pois, fx, feature_np, targets)
    
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
                    weights = weights_list[j],
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



#to-do: function that randomizes data it loads everytime the function is called, and uses weights to do plots, returns the bins
def twosamples_tests(feature_np, fx, nv_only_val, N_ref, N_Bkg_Pois, targets,bin_num=45):
    """
    For each feature i in {0,1,2,3}:
      - KS test between (data_*[:,i]) and (ref[:,i] reweighted by exp(fx_*))
      - Chi2 contingency test between unweighted vs weighted hist counts
    """

  
    (data_negnv1, data_nv1), (fx_negnv1, fx_nv1), ref,[weights_negnv1,weights_nv1], weights_ref = get_fitted_data_fx(N_ref, N_Bkg_Pois, fx, feature_np, targets)

    for i in range(4):
        print(f"\nFeature {i}:")

        _, bin_edges = np.histogram(ref[:, i], bins=bin_num, weights = weights_ref ,density=False)

        # Unweighted hist counts from the *data* samples
        
        counts_negnv1_unw, _ = np.histogram(data_negnv1[:, i], bins=bin_edges, weights = weights_negnv1,density=False)
        counts_nv1_unw,  _   = np.histogram(data_nv1[:, i],    bins=bin_edges, weights = weights_nv1, density=False)
        

        # Weighted hist counts by reweighting *ref* with exp(fx_*)
        # IMPORTANT: fx_* must be shape (N_ref, 4) matching ref.
       
        counts_negnv1_w, _ = np.histogram(ref[:, i], bins=bin_edges, weights=weights_ref*np.exp(fx_negnv1), density=False)
        counts_nv1_w,  _   = np.histogram(ref[:, i], bins=bin_edges, weights=weights_ref*np.exp(fx_nv1),    density=False)
        

        # Each tables should be shape (2, nbins) for chi2_contingency
        counts_list = [
            np.vstack([counts_negnv1_unw, counts_negnv1_w]),
            np.vstack([counts_nv1_unw,    counts_nv1_w]),
        ]

        ks_table = [
            ([counts_negnv1_unw, counts_negnv1_w]),
            ([counts_nv1_unw,    counts_nv1_w]),
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
            print(f"   |Δcounts|/|counts| = {np.abs(tables[0] - tables[1])*100/tables[1]}%")
            print(f"    Average |Δcounts|/|counts| = {np.mean(np.abs(tables[0] - tables[1])/tables[1])}")
            



def plot_log_sbs(feature_np_list, N_Bkg_Pois,N_ref,nv_vals, x_target_list = [-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.1,1.2,]):
    
    #nv_vals = [-0.1,-0.05, 0.0, 0.05, 0.1]
    N_diff = N_Bkg_Pois - N_ref

    neg1_start = N_ref
    neg1_end = N_ref+N_Bkg_Pois
    pos1_start = (2*N_ref)+N_Bkg_Pois
    pos1_end = 2*(N_ref+N_Bkg_Pois)
    
    colors = ["blue","black","red","green","orange","purple","grey","brown","lightcoral","forestgreen","darkgrey","khaki","steelblue"]

    for k,feature_np in enumerate(feature_np_list):
        
        _, global_bins = np.histogram(feature_np[0:N_ref,0],bins=40)
        
        global_bin_centers = (global_bins[:-1] + global_bins[1:]) / 2 #find the bin center for each bin
        
        fig, axs = plt.subplots(1, 4, figsize=(28, 6),sharey=True)
        axs = axs.flatten()
            
        for i in range(0,4):
                
            counts_orig, _ = np.histogram(feature_np[0:N_ref,i], bins=global_bins, density=False)
        
            counts_neg01_unw, _ = np.histogram(feature_np[neg1_start:neg1_end,i], bins=global_bins, density=False)
            counts_orig_unw, _ = np.histogram(feature_np[0:N_ref,i], bins=global_bins, density=False)
            counts_01_unw, _ = np.histogram(feature_np[pos1_start:pos1_end,i], bins=global_bins, density=False)
            
    
            for j,x_target in enumerate(x_target_list):
            
                shared_bin_idx = np.argmin(np.abs(global_bin_centers - x_target)) #find the index of the bin closest to x_target
                shared_bin_center = global_bin_centers[shared_bin_idx] #store the x-value
            
                y_counts_unw = np.array([
                           
                            np.log(counts_neg01_unw[shared_bin_idx]/counts_orig[shared_bin_idx]),
                            np.log(counts_orig_unw[shared_bin_idx]/counts_orig[shared_bin_idx]),
                            np.log(counts_01_unw[shared_bin_idx]/counts_orig[shared_bin_idx]),])
                            
                axs[i].errorbar(nv_vals, y_counts_unw, fmt='o-', color=colors[j],
                                    capsize=4, elinewidth=1.2, markeredgewidth=1, markersize=3, label=f'x = {x_target}')
            
            axs[i].set_title(f'Feature {i}',fontsize=18.5)
            axs[i].set_xlabel('ν',fontsize = 17)
            axs[i].set_ylabel(r'$\log(R_{\nu}/R_{0})$',fontsize = 17)
            axs[i].grid(True, linestyle=':', alpha=0.5)
            axs[i].legend(fontsize = 8)
            axs[i].set_xticks(nv_vals)
            axs[i].tick_params(axis='y', labelleft=True)
            #axs[i].set_ylim(top=0.120,bottom=-0.12)
    
        plt.suptitle(f'Embeddings Log Ratios', fontsize=14, y=1.02)
        plt.tight_layout()
        #print(filename_list[k])
        plt.show()