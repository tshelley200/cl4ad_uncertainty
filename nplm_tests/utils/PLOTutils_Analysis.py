import glob, json, h5py, math, time, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from scipy.stats import norm, expon, chi2, uniform, chisquare

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp


def plot_1distribution(t, df, xmin=0, xmax=300, nbins=10, label='', save=False, save_path='', file_name=''):
    '''
    Plot the histogram of a test statistics sample (t) and the target chi2 distribution. 
    The median and the error on the median are calculated in order to calculate the median Z-score and its error.
    
    t:  (numpy array shape (None,))
    df: (int) chi2 degrees of freedom
    '''
    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    fig  = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor('white')
    # plot distribution histogram
    bins      = np.linspace(xmin, xmax, nbins+1)
    Z_obs     = norm.ppf(chi2.cdf(np.median(t), df))
    t_obs_err = 1.2533*np.std(t)*1./np.sqrt(t.shape[0])
    Z_obs_p   = norm.ppf(chi2.cdf(np.median(t)+t_obs_err, df))
    Z_obs_m   = norm.ppf(chi2.cdf(np.median(t)-t_obs_err, df))
    label  = 'sample %s\nsize: %i \nmedian: %s, std: %s\n'%(label, t.shape[0], str(np.around(np.median(t), 2)),str(np.around(np.std(t), 2)))
    label += 'Z = %s (+%s/-%s)'%(str(np.around(Z_obs, 2)), str(np.around(Z_obs_p-Z_obs, 2)), str(np.around(Z_obs-Z_obs_m, 2)))
    binswidth = (xmax-xmin)*1./nbins
    h = plt.hist(t, weights=np.ones_like(t)*1./(t.shape[0]*binswidth), color='lightblue', ec='#2c7fb8',
                 bins=bins, label=label)
    err = np.sqrt(h[0]/(t.shape[0]*binswidth))
    x   = 0.5*(bins[1:]+bins[:-1])
    plt.errorbar(x, h[0], yerr = err, color='#2c7fb8', marker='o', ls='')
    # plot reference chi2
    x  = np.linspace(chi2.ppf(0.0001, df), chi2.ppf(0.9999, df), 100)
    plt.plot(x, chi2.pdf(x, df),'midnightblue', lw=5, alpha=0.8, label=r'$\chi^2$('+str(df)+')')
    font = font_manager.FontProperties(family='serif', size=14) 
    plt.legend(prop=font)
    plt.xlabel('t', fontsize=18, fontname="serif")
    plt.ylabel('Probability', fontsize=18, fontname="serif")
    plt.yticks(fontsize=16, fontname="serif")
    plt.xticks(fontsize=16, fontname="serif")
    if save:
        if save_path=='': print('argument save_path is not defined. The figure will not be saved.')
        else:
            if file_name=='': file_name = '1distribution'
            else: file_name += '_1distribution'
            plt.savefig(save_path+file_name+'.pdf')
    plt.show()
    plt.close(fig)
    return

def plot_2distribution(t1, t2, df, xmin=0, xmax=300, nbins=10, label1='1', label2='2', save=False, save_path='', file_name=''):
    '''
    Plot the histogram of two test statistics samples (t1 and t2) and the target chi2 distribution.
    The median and the error on the median are calculated and thus the median Z-score and its error.
    
    t1:  (numpy array shape (None,))
    t2:  (numpy array shape (None,))
    df: (int) chi2 degrees of freedom
    '''
    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    fig  = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor('white')
    # plot distribution histogram
    bins      = np.linspace(xmin, xmax, nbins+1)
    binswidth = (xmax-xmin)*1./nbins
    # t1
    Z_obs     = norm.ppf(chi2.cdf(np.median(t1), df))
    t_obs_err = 1.2533*np.std(t1)*1./np.sqrt(t1.shape[0])
    Z_obs_p   = norm.ppf(chi2.cdf(np.median(t1)+t_obs_err, df))
    Z_obs_m   = norm.ppf(chi2.cdf(np.median(t1)-t_obs_err, df))
    label  = 'sample %s\nsize: %i\nmedian: %s\nstd: %s\n'%(label1, t1.shape[0], str(np.around(np.median(t1), 2)),str(np.around(np.std(t1), 2)))
    label += 'Z = %s (+%s/-%s)'%(str(np.around(Z_obs, 2)), str(np.around(Z_obs_p-Z_obs, 2)), str(np.around(Z_obs-Z_obs_m, 2)))
    h = plt.hist(t1, weights=np.ones_like(t1)*1./(t1.shape[0]*binswidth), color='lightblue', ec='#2c7fb8',
                 bins=bins, label=label)
    err = np.sqrt(h[0]/(t1.shape[0]*binswidth))
    x   = 0.5*(bins[1:]+bins[:-1])
    plt.errorbar(x, h[0], yerr = err, color='#2c7fb8', marker='o', ls='')
    # t2
    Z_obs     = norm.ppf(chi2.cdf(np.median(t2), df))
    t_obs_err = 1.2533*np.std(t2)*1./np.sqrt(t2.shape[0])
    Z_obs_p   = norm.ppf(chi2.cdf(np.median(t2)+t_obs_err, df))
    Z_obs_m   = norm.ppf(chi2.cdf(np.median(t2)-t_obs_err, df))
    label  = 'sample %s\nsize: %i\nmedian: %s\nstd: %s\n'%(label2, t2.shape[0], str(np.around(np.median(t2), 2)),str(np.around(np.std(t2), 2)))
    label += 'Z = %s (+%s/-%s)'%(str(np.around(Z_obs, 2)), str(np.around(Z_obs_p-Z_obs, 2)), str(np.around(Z_obs-Z_obs_m, 2)))
    h = plt.hist(t2, weights=np.ones_like(t2)*1./(t2.shape[0]*binswidth), color='#8dd3c7', ec='seagreen',
                 bins=bins, label=label)
    err = np.sqrt(h[0]/(t2.shape[0]*binswidth))
    x   = 0.5*(bins[1:]+bins[:-1])
    plt.errorbar(x, h[0], yerr = err, color='seagreen', marker='o', ls='')
    # plot reference chi2
    x  = np.linspace(chi2.ppf(0.0001, df), chi2.ppf(0.9999, df), 100)
    plt.plot(x, chi2.pdf(x, df),'midnightblue', lw=5, alpha=0.8, label=r'$\chi^2$('+str(df)+')')
    font = font_manager.FontProperties(family='serif', size=14) #weight='bold', style='normal', )
    plt.legend(ncol=1, loc='upper right', prop=font)
    plt.xlabel('t', fontsize=14, fontname="serif")
    plt.ylabel('Probability', fontsize=14, fontname="serif")
    plt.ylim(0., np.max(chi2.pdf(x, df))*1.3)
    plt.yticks(fontsize=16, fontname="serif")
    plt.xticks(fontsize=16, fontname="serif")
    if save:
        if save_path=='': print('argument save_path is not defined. The figure will not be saved.')
        else:
            if file_name=='': file_name = '2distribution'
            else: file_name += '_2distribution'
            plt.savefig(save_path+file_name+'.pdf')
    plt.show()
    plt.close()
    return

def compute_df(input_size, hidden_layers, output_size=1):
    """
    Compute degrees of freedom of a neural net (number of trainable params)

    input_size :    (int) size of the input layer
    hidden_layers : (list) list specifiying size of hidden layers
    latentsize :    (int) number of hidden units for each layer

    df : (int) degrees of freedom
    """
    nn_arch = [input_size] + hidden_layers + [output_size]
    df = sum(map(lambda x, y : x*(y+1), nn_arch[1:], nn_arch[:-1]))
    return df

def Plot_Percentiles(tvalues_check, patience=1, checkpoints=[], ylabel='t', ymax=300, ymin=0, save=False, file_name='', save_path=''):
    '''
    The function creates the plot of the evolution in the epochs of the [2.5%, 25%, 50%, 75%, 97.5%] quantiles of the toy sample distribution.
    
    patience: (int) interval between two check points (epochs).
    tvalues_check: (numpy array, shape (N_toys, N_check_points)) array of t=-2*loss, 
    '''
    colors = ['seagreen','mediumseagreen','lightseagreen','#2c7fb8','midnightblue']
    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    epochs_check = []
    mask         = []
    nr_check_points = tvalues_check.shape[1]
    for i in range(nr_check_points):
        epoch_check = patience*(i+1)
        epochs_check.append(epoch_check)
        if len(checkpoints): mask.append(np.any(np.array(checkpoints)==epoch_check))
        else: mask.append(True)
    mask = np.array(mask)
    epochs_check = np.array(epochs_check)
    fig = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor('white')
    quantiles   = [2.5, 25, 50, 75, 97.5]
    percentiles = np.array([])
    plt.xlabel('Epoch', fontsize=16, fontname="serif")
    plt.ylabel(ylabel, fontsize=16, fontname="serif")
    plt.ylim(ymin, ymax)
    for i in range(tvalues_check.shape[1]):
        percentiles_i = np.percentile(tvalues_check[:, i], quantiles)
        percentiles_i = np.expand_dims(percentiles_i, axis=1)
        if not i: percentiles = percentiles_i.T
        else: percentiles = np.concatenate((percentiles, percentiles_i.T))
    legend = []
    print(percentiles.shape)
    for j in range(percentiles.shape[1]):
        y = percentiles[:, j]
        plt.plot(epochs_check[mask], y[mask], marker='.', color=colors[j])
        legend.append(str(quantiles[j])+' % quantile')
    plt.legend(legend, fontsize=16)
    plt.yticks(fontsize=16, fontname="serif")
    plt.xticks(fontsize=16, fontname="serif")
    plt.grid()
    if save:
        if save_path=='': print('argument save_path is not defined. The figure will not be saved.')
        else:
            if file_name=='': file_name = 'PlotPercentiles'
            else: file_name += '_PlotPercentiles'
            fig.savefig(save_path+file_name+'.pdf')
    plt.show()
    plt.close(fig)
    return



def Plot_Percentiles_ref(tvalues_check, df, patience=1, wc=None, ymax=300, ymin=0, save=False, save_path='', file_name=''):
    '''
    The funcion creates the plot of the evolution in the epochs of the [2.5%, 25%, 50%, 75%, 97.5%] quantiles of the toy sample distribution.
    The percentile lines for the target chi2 distribution are shown as a reference.
    
    patience:      (int) interval between two check points (epochs).
    tvalues_check: (numpy array shape (N_toys, N_check_points)) array of t=-2*loss
    df:            (int) chi2 degrees of freedom
    '''
    colors = ['seagreen', 'mediumseagreen', 'lightseagreen', '#2c7fb8', 'midnightblue']
    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    epochs_check = []
    nr_check_points = tvalues_check.shape[1]
    for i in range(nr_check_points):
        epoch_check = patience*(i+1)
        epochs_check.append(epoch_check)
    fig = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor('white')
    quantiles   = [2.5, 25, 50, 75, 97.5]
    percentiles = np.array([])
    plt.xlabel('Training Epochs', fontsize=16, fontname="serif")
    plt.ylabel('t', fontsize=16, fontname="serif")
    plt.ylim(ymin, ymax)
    if wc != None:
        plt.title('Weight Clipping = '+wc, fontsize=16,  fontname="serif")
    for i in range(tvalues_check.shape[1]):
        percentiles_i = np.percentile(tvalues_check[:, i], quantiles)
        percentiles_i = np.expand_dims(percentiles_i, axis=1)
        if not i: percentiles = percentiles_i.T
        else: percentiles = np.concatenate((percentiles, percentiles_i.T))
    legend = []
    for j in range(percentiles.shape[1]):
        plt.plot(epochs_check, percentiles[:, j], marker='.', linewidth=3, color=colors[j])
        legend.append(str(quantiles[j])+' % quantile')
    for j in range(percentiles.shape[1]):
        plt.plot(epochs_check, chi2.ppf(quantiles[j]/100., df=df, loc=0, scale=1)*np.ones_like(epochs_check),
                color=colors[j], ls='--', linewidth=1)
        if j==0: legend.append("Target "+r"$\chi^2(df=$"+str(df)+")")
    font = font_manager.FontProperties(family='serif', size=16)         
    plt.legend(legend, prop=font)
    plt.yticks(fontsize=16, fontname="serif")
    plt.xticks(fontsize=16, fontname="serif")
    if save:
        if save_path=='': print('argument save_path is not defined. The figure will not be saved.')
        else:
            if file_name=='': file_name = 'PlotPercentiles'
            else: file_name += '_PlotPercentiles'
            fig.savefig(save_path+file_name+'.pdf')
    plt.show()
    plt.close(fig)
    return


def plot_2samples_overlay(
    t1,
    t2,
    xmin=None,
    xmax=None,
    nbins=12,
    label1="sample 1",
    label2="sample 2",
    *,
    density=True,
    show_errors=True,
    alpha=0.45,
    color1="lightblue",
    color2="lightgreen",
    save=False,
    save_path="",
    file_name="2samples_overlay",
):
    """
    Overlay two 1D samples as histograms (no chi2 curve), and show the KS statistic in the legend.
    """
    # --- sanitize inputs ---
    t1 = np.asarray(t1, dtype=float).reshape(-1)
    t2 = np.asarray(t2, dtype=float).reshape(-1)
    t1 = t1[np.isfinite(t1)]
    t2 = t2[np.isfinite(t2)]
    if t1.size == 0 or t2.size == 0:
        raise ValueError("plot_2samples_overlay: one of the samples has no finite values.")

    # --- range ---
    if xmin is None:
        xmin = float(min(t1.min(), t2.min()))
    if xmax is None:
        xmax = float(max(t1.max(), t2.max()))
    if xmax <= xmin:
        raise ValueError(f"plot_2samples_overlay: bad range xmin={xmin}, xmax={xmax}")

    bins = np.linspace(xmin, xmax, nbins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_widths = np.diff(bins)

    # --- KS test ---
    ks_stat, ks_pvalue = ks_2samp(t1, t2, alternative="two-sided", mode="auto")

    # --- histogram data ---
    h1, _ = np.histogram(t1, bins=bins)
    h2, _ = np.histogram(t2, bins=bins)

    if density:
        y1 = h1 / (t1.size * bin_widths)
        y2 = h2 / (t2.size * bin_widths)
        ylabel = "Probability density"
    else:
        y1 = h1
        y2 = h2
        ylabel = "Counts"

    # --- plot ---
    plt.rcParams["font.family"] = "serif"
    plt.style.use("classic")
    fig = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor("white")
    ax = plt.gca()

    ax.bar(
        bin_centers, y1, width=bin_widths, align="center",
        alpha=alpha, edgecolor="k", color=color1,
        label=f"{label1}\nN={t1.size}, median={np.median(t1):.2f}, std={np.std(t1):.2f}"
    )
    ax.bar(
        bin_centers, y2, width=bin_widths, align="center",
        alpha=alpha, edgecolor="k", color=color2,
        label=(f"{label2}\nN={t2.size}, median={np.median(t2):.2f}, std={np.std(t2):.2f}\n"
               f"KS D={ks_stat:.3f}, p={ks_pvalue:.3g}")
    )

    if show_errors:
        e1 = np.sqrt(h1)
        e2 = np.sqrt(h2)
        if density:
            e1 = e1 / (t1.size * bin_widths)
            e2 = e2 / (t2.size * bin_widths)
        ax.errorbar(bin_centers, y1, yerr=e1, fmt="o", ms=4, capsize=2, linestyle="none", color="k")
        ax.errorbar(bin_centers, y2, yerr=e2, fmt="o", ms=4, capsize=2, linestyle="none", color="k")

    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("t", fontsize=18, fontname="serif")
    ax.set_ylabel(ylabel, fontsize=18, fontname="serif")
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(prop={"family": "serif", "size": 12})

    if save:
        if save_path == "":
            print("save=True but save_path is empty; not saving.")
        else:
            if not save_path.endswith("/"):
                save_path += "/"
            fig.savefig(save_path + file_name + ".pdf", bbox_inches="tight")

    plt.show()
    plt.close(fig)
    return fig, ax, ks_stat, ks_pvalue
