import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
import os

def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def LoadData(filepath,N_Bkg_Pois,N_ref,nv):

    nv_std = np.std(nv)

    #Load the data
    #data = np.load('/n/home07/tshelley2002/Uncertainty_Project/cl4ad_new/Run6/test14/embedding.npz');
    N_diff = N_Bkg_Pois - N_ref

    neg2_start = N_ref
    neg2_end = N_ref+N_Bkg_Pois
    neg1_start = (2*N_ref)+N_Bkg_Pois
    neg1_end = 2*(N_ref+N_Bkg_Pois)
    pos1_start = (3*N_ref)+(2*N_Bkg_Pois)
    pos1_end = (3*N_ref)+(3*N_Bkg_Pois)
    pos2_start = (4*N_ref)+(3*N_Bkg_Pois)
    pos2_end = (4*N_ref)+(4*N_Bkg_Pois)
    
    data = np.load(filepath)
    #print(data)
    unscaled_train0 = data['arr_2'][0:N_ref,:]
    #print(unscaled_train0.shape)

    scaled_1_train0 = data['arr_3'][0:N_Bkg_Pois,:]
    #print(scaled_1_train0.shape)

    scaled_2_train0 = data['arr_4'][0:N_Bkg_Pois,:]
    #print(scaled_3_train0.shape)

    scaled_neg1_train0 = data['arr_1'][0:N_Bkg_Pois,:]
    scaled_neg2_train0 = data['arr_0'][0:N_Bkg_Pois,:]
    
    featureData_lst = [scaled_neg2_train0,scaled_neg1_train0,scaled_1_train0,scaled_2_train0];

    targetRef  = np.zeros_like(unscaled_train0[:,0])
    feature    = np.empty((0,4))
    target     = np.empty((0))
    nuisance   = np.empty((0))

    for i in range(0,len(nv)):

        featureData = featureData_lst[i]
        targetData  = np.ones_like(featureData[:,0])

        feature = np.append(feature, unscaled_train0,axis=0)
        feature = np.append(feature, featureData, axis=0)

        target  = np.append(target, targetRef,  axis=0)
        target  = np.append(target, targetData, axis=0)

        nuisance = np.append(nuisance, np.ones_like(unscaled_train0[:,0])*nv[i]/nv_std,axis=0)
        nuisance = np.append(nuisance, np.ones_like(featureData[:,0])*nv[i]/nv_std,axis=0)

    mean_0 = np.mean(unscaled_train0, axis=0, keepdims=True)
    std_0  = np.std(unscaled_train0, axis=0, keepdims=True)
    #print(mean_0,std_0)
    feature = (feature - (mean_0)) / (std_0)
    target = np.stack([target, nuisance], axis=1)
    #print(feature.shape)
    #print(target)
    #plot_stephist(feature,target)
    return to_np(feature),filepath


def plot_stephist_with_ratio_old(f, t,N_Bkg_Pois,N_ref,nv):
    feature_np = to_np(f)
    target_np  = to_np(t)
    N_diff = N_Bkg_Pois - N_ref

    neg2_start = N_ref
    neg2_end = N_ref+N_Bkg_Pois
    neg1_start = (2*N_ref)+N_Bkg_Pois
    neg1_end = 2*(N_ref+N_Bkg_Pois)
    pos1_start = (3*N_ref)+(2*N_Bkg_Pois)
    pos1_end = (3*N_ref)+(3*N_Bkg_Pois)
    pos2_start = (4*N_ref)+(3*N_Bkg_Pois)
    pos2_end = (4*N_ref)+(4*N_Bkg_Pois)
    fig, axs = plt.subplots(
        2, 4,
        figsize=(46, 9),
        sharex='col',
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08}
    )

    colors = {
    "pos2": "deepskyblue",
    "pos1": "royalblue",
    "neg1": "blueviolet",
    "neg2": "navy",
    }

    labels = {
        "pos2": rf'Embedding w/ $\nu$ = {0.1*nv[-1]:.2f}',
        "pos1": rf'Embedding w/ $\nu$ ={0.1*nv[-2]:.2f}',
        "neg1": rf'Embedding w/ $\nu$ ={0.1*nv[1]:.2f}',
        "neg2": rf'Embedding w/ $\nu$ = {0.1*nv[0]:.2f}',
    }

    for i in range(4):
        ax  = axs[0, i]
        rax = axs[1, i]

        # ---------- top histogram panel ----------
        counts_ref, bins, _ = ax.hist(
            feature_np[0:N_ref, i],
            bins=40,
            alpha=0.3,
            edgecolor='black',
            linewidth=0.0,
            label=r'Original Embedding; Reference ($R_0$)',
            color='grey',
            density=False
        )

        samples = {
            "pos2": feature_np[pos2_start:pos2_end, i],
            "pos1": feature_np[pos1_start:pos1_end, i],
            "neg1": feature_np[neg1_start:neg1_end, i],
            "neg2": feature_np[neg2_start:neg2_end, i],}

        counts_dict = {}
        for key, arr in samples.items():
            counts, _ = np.histogram(arr, bins=bins)
            counts_dict[key] = counts

            ax.hist(
                arr,
                bins=bins,
                histtype='step',
                linewidth=2.2,
                color=colors[key],
                density=False,
                label=labels[key]
            )

        ax.set_title(f'Neural Embeddings Dimension {i}',fontsize=19)
        ax.set_ylabel('Counts',fontsize=18)
        ax.tick_params(axis='y', labelleft=True)
        ax.legend(fontsize=14)

        # ---------- bottom ratio panel ----------
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        ref_band = np.full_like(counts_ref, np.nan, dtype=float)
        mask_ref = counts_ref > 0
        ref_band[mask_ref] = 1.0 / np.sqrt(counts_ref[mask_ref])

        rax.axhline(1.0, color='black', linewidth=1.2)
        rax.fill_between(
            bin_centers,
            1.0 - ref_band,
            1.0 + ref_band,
            step='mid',
            color='navy',
            alpha=0.18,
            linewidth=0
        )

        for key in ["pos2", "pos1", "neg1", "neg2"]:
            counts_nv = counts_dict[key].astype(float)
            counts_0  = counts_ref.astype(float)

            ratio = np.full_like(counts_0, np.nan, dtype=float)
            mask = (counts_0 > 0) & (counts_nv > 0)
            ratio[mask] = counts_0[mask] / counts_nv[mask]

            rax.plot(
                bin_centers[mask],
                ratio[mask],
                'o-',
                color=colors[key],
                markersize=3,
                linewidth=1.4
            )

        rax.set_xlabel('x',fontsize=18)
        rax.set_ylabel(r'$R_0/R_{\nu}$',fontsize=18)
        rax.tick_params(axis='y', labelleft=True)
        rax.grid(True, linestyle=':', alpha=0.4)
        #rax.set_ylim(0.4, 1.6)

    plt.tight_layout()
    plt.show()

def plot_stephist_with_ratio_v1(f, t, N_Bkg_Pois, N_ref, nv):
    feature_np = to_np(f)
    target_np  = to_np(t)
    N_diff = N_Bkg_Pois - N_ref

    neg2_start = N_ref
    neg2_end   = N_ref + N_Bkg_Pois
    neg1_start = (2 * N_ref) + N_Bkg_Pois
    neg1_end   = 2 * (N_ref + N_Bkg_Pois)
    pos1_start = (3 * N_ref) + (2 * N_Bkg_Pois)
    pos1_end   = (3 * N_ref) + (3 * N_Bkg_Pois)
    pos2_start = (4 * N_ref) + (3 * N_Bkg_Pois)
    pos2_end   = (4 * N_ref) + (4 * N_Bkg_Pois)

    fig, axs = plt.subplots(
        3, 4,
        figsize=(46, 13),
        sharex='col',
        gridspec_kw={'height_ratios': [3, 1, 1], 'hspace': 0.08}
    )

    colors = {
        "pos2": "deepskyblue",
        "pos1": "royalblue",
        "neg1": "blueviolet",
        "neg2": "navy",
    }

    labels = {
        "pos2": rf'Embedding w/ $\nu$ = {0.1 * nv[-1]:.2f}',
        "pos1": rf'Embedding w/ $\nu$ = {0.1 * nv[-2]:.2f}',
        "neg1": rf'Embedding w/ $\nu$ = {0.1 * nv[1]:.2f}',
        "neg2": rf'Embedding w/ $\nu$ = {0.1 * nv[0]:.2f}',
    }

    for i in range(4):
        ax   = axs[0, i]
        rax  = axs[1, i]
        lrax = axs[2, i]

        # ---------- top histogram panel ----------
        counts_ref, bins, _ = ax.hist(
            feature_np[0:N_ref, i],
            bins=40,
            alpha=0.3,
            edgecolor='black',
            linewidth=0.0,
            label=r'Original Embedding; Reference ($R_0$)',
            color='grey',
            density=False
        )

        samples = {
            "pos2": feature_np[pos2_start:pos2_end, i],
            "pos1": feature_np[pos1_start:pos1_end, i],
            "neg1": feature_np[neg1_start:neg1_end, i],
            "neg2": feature_np[neg2_start:neg2_end, i],
        }

        counts_dict = {}
        for key, arr in samples.items():
            counts, _ = np.histogram(arr, bins=bins)
            counts_dict[key] = counts

            ax.hist(
                arr,
                bins=bins,
                histtype='step',
                linewidth=2.2,
                color=colors[key],
                density=False,
                label=labels[key]
            )

        ax.set_title(f'Neural Embeddings Dimension {i}', fontsize=19)
        ax.set_ylabel('Counts', fontsize=18)
        ax.tick_params(axis='y', labelleft=True)
        ax.legend(fontsize=14)

        # ---------- middle ratio panel: R0 / Rnu ----------
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        ref_band = np.full_like(counts_ref, np.nan, dtype=float)
        mask_ref = counts_ref > 0
        ref_band[mask_ref] = 1.0 / np.sqrt(counts_ref[mask_ref])

        rax.axhline(1.0, color='black', linewidth=1.2)
        rax.fill_between(
            bin_centers,
            1.0 - ref_band,
            1.0 + ref_band,
            step='mid',
            color='navy',
            alpha=0.18,
            linewidth=0
        )

        for key in ["pos2", "pos1", "neg1", "neg2"]:
            counts_nv = counts_dict[key].astype(float)
            counts_0  = counts_ref.astype(float)

            ratio = np.full_like(counts_0, np.nan, dtype=float)
            mask = (counts_0 > 0) & (counts_nv > 0)
            ratio[mask] = counts_0[mask] / counts_nv[mask]

            rax.plot(
                bin_centers[mask],
                ratio[mask],
                'o-',
                color=colors[key],
                markersize=3,
                linewidth=1.4
            )

        rax.set_ylabel(r'$R_0 / R_{\nu}$', fontsize=18)
        rax.tick_params(axis='y', labelleft=True)
        rax.grid(True, linestyle=':', alpha=0.4)

        # ---------- bottom log-ratio panel: log(Rnu / R0) ----------
        lrax.axhline(0.0, color='black', linewidth=1.2)

        for key in ["pos2", "pos1", "neg1", "neg2"]:
            counts_nv = counts_dict[key].astype(float)
            counts_0  = counts_ref.astype(float)

            log_ratio = np.full_like(counts_0, np.nan, dtype=float)
            mask = (counts_0 > 0) & (counts_nv > 0)
            log_ratio[mask] = np.log(counts_nv[mask] / counts_0[mask])

            lrax.plot(
                bin_centers[mask],
                log_ratio[mask],
                'o-',
                color=colors[key],
                markersize=3,
                linewidth=1.4
            )

        lrax.set_xlabel('x', fontsize=18)
        lrax.set_ylabel(r'$\log(R_{\nu}/R_0)$', fontsize=18)
        lrax.tick_params(axis='y', labelleft=True)
        lrax.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.show()

def plot_stephist_with_ratio(f, t, N_Bkg_Pois, N_ref, nv):
    feature_np = to_np(f)
    target_np  = to_np(t)

    neg2_start = N_ref
    neg2_end   = N_ref + N_Bkg_Pois
    neg1_start = (2 * N_ref) + N_Bkg_Pois
    neg1_end   = 2 * (N_ref + N_Bkg_Pois)
    pos1_start = (3 * N_ref) + (2 * N_Bkg_Pois)
    pos1_end   = (3 * N_ref) + (3 * N_Bkg_Pois)
    pos2_start = (4 * N_ref) + (3 * N_Bkg_Pois)
    pos2_end   = (4 * N_ref) + (4 * N_Bkg_Pois)

    fig, axs = plt.subplots(
        3, 4,
        figsize=(46, 13),
        sharex='col',
        gridspec_kw={'height_ratios': [3, 1, 1], 'hspace': 0.08}
    )

    colors = {
        "pos2": "deepskyblue",
        "pos1": "royalblue",
        "neg1": "blueviolet",
        "neg2": "navy",
    }

    labels = {
        "pos2": rf'Embedding w/ $\nu$ = {0.1 * nv[-1]:.2f}',
        "pos1": rf'Embedding w/ $\nu$ = {0.1 * nv[-2]:.2f}',
        "neg1": rf'Embedding w/ $\nu$ = {0.1 * nv[1]:.2f}',
        "neg2": rf'Embedding w/ $\nu$ = {0.1 * nv[0]:.2f}',
    }

    for i in range(4):
        ax   = axs[0, i]
        rax  = axs[1, i]
        lrax = axs[2, i]

        # ---------- top histogram panel ----------
        counts_ref, bins, _ = ax.hist(
            feature_np[0:N_ref, i],
            bins=40,
            alpha=0.3,
            edgecolor='black',
            linewidth=0.0,
            label=r'Original Embedding; Reference ($R_0$)',
            color='grey',
            density=False
        )

        samples = {
            "pos2": feature_np[pos2_start:pos2_end, i],
            "pos1": feature_np[pos1_start:pos1_end, i],
            "neg1": feature_np[neg1_start:neg1_end, i],
            "neg2": feature_np[neg2_start:neg2_end, i],
        }

        counts_dict = {}
        for key, arr in samples.items():
            counts, _ = np.histogram(arr, bins=bins)
            counts_dict[key] = counts

            ax.hist(
                arr,
                bins=bins,
                histtype='step',
                linewidth=2.2,
                color=colors[key],
                density=False,
                label=labels[key]
            )

        ax.set_title(f'Neural Embeddings Dimension {i}', fontsize=19)
        ax.set_ylabel('Counts', fontsize=18)
        ax.tick_params(axis='y', labelleft=True)
        ax.legend(fontsize=14)

        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # ---------- middle ratio panel: R0 / Rnu ----------
        ref_band = np.full_like(counts_ref, np.nan, dtype=float)
        mask_ref = counts_ref > 0
        ref_band[mask_ref] = 1.0 / np.sqrt(counts_ref[mask_ref])

        rax.axhline(1.0, color='black', linewidth=1.2)
        rax.fill_between(
            bin_centers,
            1.0 - ref_band,
            1.0 + ref_band,
            step='mid',
            color='navy',
            alpha=0.18,
            linewidth=0
        )

        for key in ["pos2", "pos1", "neg1", "neg2"]:
            counts_nv = counts_dict[key].astype(float)
            counts_0  = counts_ref.astype(float)

            ratio = np.full_like(counts_0, np.nan, dtype=float)
            mask = (counts_0 > 0) & (counts_nv > 0)
            ratio[mask] = counts_0[mask] / counts_nv[mask]

            rax.plot(
                bin_centers[mask],
                ratio[mask],
                'o-',
                color=colors[key],
                markersize=3,
                linewidth=1.4
            )

        rax.set_ylabel(r'$R_0 / R_{\nu}$', fontsize=18)
        rax.tick_params(axis='y', labelleft=True)
        rax.grid(True, linestyle=':', alpha=0.4)

        # ---------- bottom log-ratio panel: log(Rnu / R0) with errors ----------
        lrax.axhline(0.0, color='black', linewidth=1.2)

        for key in ["pos2", "pos1", "neg1", "neg2"]:
            counts_nv = counts_dict[key].astype(float)
            counts_0  = counts_ref.astype(float)

            log_ratio = np.full_like(counts_0, np.nan, dtype=float)
            log_err   = np.full_like(counts_0, np.nan, dtype=float)

            mask = (counts_0 > 0) & (counts_nv > 0)

            log_ratio[mask] = np.log(counts_nv[mask] / counts_0[mask])
            log_err[mask]   = np.sqrt(1.0 / counts_nv[mask] + 1.0 / counts_0[mask])

            lrax.errorbar(
                bin_centers[mask],
                log_ratio[mask],
                yerr=log_err[mask],
                fmt='o-',
                color=colors[key],
                markersize=3,
                linewidth=1.2,
                elinewidth=1.0,
                capsize=2
            )

        lrax.set_xlabel('x', fontsize=18)
        lrax.set_ylabel(r'$\log(R_{\nu}/R_0)$', fontsize=18)
        lrax.tick_params(axis='y', labelleft=True)
        lrax.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.show()

def plot_log_sbs(feature_np_list, filename_list, N_Bkg_Pois,N_ref,nv_vals, x_target_list = [-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.1,1.2,]):
    
    #nv_vals = [-0.1,-0.05, 0.0, 0.05, 0.1]
    N_diff = N_Bkg_Pois - N_ref

    neg2_start = N_ref
    neg2_end = N_ref+N_Bkg_Pois
    neg1_start = (2*N_ref)+N_Bkg_Pois
    neg1_end = 2*(N_ref+N_Bkg_Pois)
    pos1_start = (3*N_ref)+(2*N_Bkg_Pois)
    pos1_end = (3*N_ref)+(3*N_Bkg_Pois)
    pos2_start = (4*N_ref)+(3*N_Bkg_Pois)
    pos2_end = (4*N_ref)+(4*N_Bkg_Pois)
    colors = ["blue","black","red","green","orange","purple","pink","grey","brown","lightcoral","forestgreen","darkgrey","khaki","steelblue"]

    for k,feature_np in enumerate(feature_np_list):
        
        _, global_bins = np.histogram(feature_np[0:N_ref,0],bins=40)
        
        global_bin_centers = (global_bins[:-1] + global_bins[1:]) / 2 #find the bin center for each bin
        
        fig, axs = plt.subplots(1, 4, figsize=(28, 6),sharey=True)
        axs = axs.flatten()
            
        for i in range(0,4):
                
            counts_orig, _ = np.histogram(feature_np[0:N_ref,i], bins=global_bins, density=False)
        
            counts_neg03_unw, _ = np.histogram(feature_np[neg2_start:neg2_end,i], bins=global_bins, density=False)
            counts_neg01_unw, _ = np.histogram(feature_np[neg1_start:neg1_end,i], bins=global_bins, density=False)
            counts_orig_unw, _ = np.histogram(feature_np[0:N_ref,i], bins=global_bins, density=False)
            counts_01_unw, _ = np.histogram(feature_np[pos1_start:pos1_end,i], bins=global_bins, density=False)
            counts_03_unw, _ = np.histogram(feature_np[pos2_start:pos2_end,i], bins=global_bins, density=False)
    
            for j,x_target in enumerate(x_target_list):
            
                shared_bin_idx = np.argmin(np.abs(global_bin_centers - x_target)) #find the index of the bin closest to x_target
                shared_bin_center = global_bin_centers[shared_bin_idx] #store the x-value
            
                y_counts_unw = np.array([
                            np.log(counts_neg03_unw[shared_bin_idx]/counts_orig[shared_bin_idx]),
                            np.log(counts_neg01_unw[shared_bin_idx]/counts_orig[shared_bin_idx]),
                            np.log(counts_orig_unw[shared_bin_idx]/counts_orig[shared_bin_idx]),
                            np.log(counts_01_unw[shared_bin_idx]/counts_orig[shared_bin_idx]),
                            np.log(counts_03_unw[shared_bin_idx]/counts_orig[shared_bin_idx])])
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

