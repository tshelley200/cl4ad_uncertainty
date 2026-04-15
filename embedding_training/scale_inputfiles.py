import numpy as np
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser


def print_datashapes(input_files):
    
    """Print data shapes"""
    print("-------------------------------------------------------------------")
    print("-------------------------------------------------------------------")
    print(f"{input_files} contained:", input_files.files,"\n")
    for key in input_files.files:
        print(f"    {key}'s shape:", input_files[key].shape,"\n")


def plot_jetpT(npzname, filename, save_filename, saved_path, bins=40, hist_color="lightgrey"):
    
    """Plot the histogram of a chosen file"""
    
    pT_jetdata = npzname[filename][:, 9:, 0, 0]
    pT_jetdata = pT_jetdata.flatten()

    plt.hist(pT_jetdata, bins=bins, range=(15, 215), alpha=0.5, edgecolor='black',
             linewidth=1.1, color=hist_color)
    plt.xlabel('pT (GeV)')
    plt.ylabel('Entries/Bins')
    plt.title("Jet pT Histogram")

    os.makedirs(saved_path, exist_ok=True)  # ensure output directory exists
    plt.savefig(os.path.join(saved_path, save_filename), dpi=150)
    plt.close()


def create_scaled_npzfile(data, nv, outputpath, modifiedfile_name):
    
    """Multiply jet pT by exp(nv) and save to new .npz"""
    cut_pt = 23

    modified_data = {}
    
    total_cut = 0
    total_total = 0

    print("------------------------------------------------------------------")
    print(f" Scaling factor nv = {nv}: \n")
    for key in data.files:
            
            arr = data[key].copy()
           
            
            # Only apply to x_* datasets since y_* datasets are not used
            if True:
            #if key.startswith("x_"):
                print(f"\n    For {key}: ")
                arr[:, 9:, 0, 0] *= np.exp(nv)
                print(f'        Before cut (exluding 0s): average jet pT = {np.mean(arr[:, 9:, 0, 0][arr[:, 9:, 0, 0] > 0]):.3f} GeV')
                
                # Count how many pT values < 23
                mask = (arr[:, 9:, 0, 0] < cut_pt) & (arr[:, 9:, 0, 0] > 0)
                n_cut = np.sum(mask)
                n_total = mask.size
                efficiency = 1 - n_cut / n_total
        
                # Apply the cut
                arr[:, 9:, 0, 0][mask] = 0
                print(f'        After cut (exluding 0s): average jet pT = {np.mean(arr[:, 9:, 0, 0][arr[:, 9:, 0, 0] > 0]):.3f} GeV')
                
                print(f"        Efficiency after cut: {efficiency:.4f} ({n_cut} cut out of {n_total})")
                # Track totals for overall x_* efficiency
                total_cut += n_cut
                total_total += n_total

            modified_data[key] = arr

    overall_efficiency = 1 - total_cut / total_total
    print(f"\n  Overall x_* efficiency (all files): {overall_efficiency:.4f} ({total_cut} cut out of {total_total}) \n")
    
    os.makedirs(outputpath, exist_ok=True)
    np.savez(os.path.join(outputpath, modifiedfile_name), **modified_data)
    
 
    
def main(args):
    
    ## Step 0: Define variables from arguments
    inputfile = args.inputfile
    outputfile = args.outputfile
    nv_values = args.scale_factors
    output_path = args.output_path

    # Load input data
    input_bgfiles = np.load(inputfile, mmap_mode="r")

    ## Step 1: Optionally print and plot input
    if args.printing_input:
        print_datashapes(input_bgfiles)

    if args.plotting_input:
        unscaled_plot_path = os.path.join(output_path, "Unscaled_jetpT_histograms")
        for key in input_bgfiles.files:
            plot_jetpT(input_bgfiles, key,
                       "Unscaled_jetpT_" + key.split('.')[0] + ".png",
                       saved_path=unscaled_plot_path)

    ## Step 2: Scale and save new npz files
          
    color_lst = ["pink","skyblue","honeydew","bisque"]
    color_index = 0
    for nv in nv_values:
       
        scaled_dir = os.path.join(output_path, "scaled_datasets")
        scaled_filename = f"nv_{nv}-{outputfile}"
        create_scaled_npzfile(input_bgfiles, nv, scaled_dir, scaled_filename)

        full_path_output = os.path.join(scaled_dir, scaled_filename)
        output_npz = np.load(full_path_output, mmap_mode="r")

        if args.printing_output:
            print_datashapes(output_npz)

        if args.plotting_output:
            scaled_plot_path = os.path.join(output_path, "Scaled_jetpT_histograms")
            
            for key in output_npz.files:
                
                if key.startswith("x_"):
                    plot_jetpT(output_npz, key,
                               f"nv_{nv}_jetpT_" + key.split('.')[0] + ".png",
                               saved_path=scaled_plot_path, hist_color = color_lst[color_index % len(color_lst)])
                    color_index +=1

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('inputfile', type=str,
                        help='Path to input .npz file')
    parser.add_argument('scale_factors', nargs='+', type=float,
                        help='List of nv scale factors, e.g. 0.1 0.2')
    parser.add_argument('outputfile', type=str,
                        help='Name for output .npz file(s)')
    parser.add_argument('output_path', type=str,
                        help='Directory to write output files and plots')

    parser.add_argument('--printing_input', action='store_true',
                        help='Print shapes of input arrays')
    parser.add_argument('--plotting_input', action='store_true',
                        help='Plot jet pT histograms from input file')
    parser.add_argument('--printing_output', action='store_true',
                        help='Print shapes of scaled arrays')
    parser.add_argument('--plotting_output', action='store_true',
                        help='Plot jet pT histograms from scaled file(s)')

    args = parser.parse_args()
    
    main(args)
