import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import os
import time, datetime
import random

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
#from torchsummary import summary

import losses
from models import CVAE, SimpleDense, DeepSets, SimpleDense_small, SimpleDense_JetClass, CVAE_JetClass, SimpleDense_ADC
from transformer import TransformerEncoder
import augmentations
import math
import h5py
from tqdm import tqdm
from sklearn.utils import shuffle
 


def main(args):
    
    '''
    Infastructure for training CVAE (background specific and with anomalies)
    '''
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    #Seed
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    
    
    feat_dim = 57
        
    if args.type == 'JetClass' or args.type == 'JetClass_Transformer':
        feat_dim = 512
        
    #Initialize transform (empty list: None)
    transform = augmentations.Transform(["naive_masking"], feat_dim)
    
    datasets_lst = args.datasets
    
    leptoquark, ato4l, hChToTauNu, hToTauTau = [None] * len(datasets_lst),[None] * len(datasets_lst),[None] * len(datasets_lst),[None] * len(datasets_lst)
    leptoquark_labels,ato4l_labels,hChToTauNu_labels,hToTauTau_labels = [None] * len(datasets_lst),[None] * len(datasets_lst),[None] * len(datasets_lst),[None] * len(datasets_lst)
    leptoquark_data_loader,ato4l_data_loader,hChToTauNu_data_loader,hToTauTau_data_loader = [None] * len(datasets_lst),[None] * len(datasets_lst),[None] * len(datasets_lst),[None] * len(datasets_lst)
    
    #n_train = args.train_num
    #n_val   = args.val_num   
    #n_test  = args.test_num
        
    for i in range(0,len(datasets_lst)):
            
            with h5py.File(f'{datasets_lst[i]}', 'r') as f:
                
                leptoquark[i] = np.array(f['leptoquark'][...])
                leptoquark_labels[i] = np.array(f['leptoquark_labels'][...])
                ato4l[i] = np.array(f['ato4l'][...])
                ato4l_labels[i] = np.array(f['ato4l_labels'][...])
                hChToTauNu[i] = np.array(f['hChToTauNu'][...])
                hChToTauNu_labels[i] = np.array(f['hChToTauNu_labels'][...])
                hToTauTau[i] = np.array(f['hToTauTau'][...])
                hToTauTau_labels[i] = np.array(f['hToTauTau_labels'][...])
                
            leptoquark_data_loader[i] = DataLoader(
                TorchCLDataset(leptoquark[i].reshape(-1,feat_dim), leptoquark_labels[i].reshape(-1), device),
                batch_size=args.batch_size,
                shuffle=False,drop_last=True)
        
            ato4l_data_loader[i] = DataLoader(
                    TorchCLDataset(ato4l[i].reshape(-1,feat_dim), ato4l_labels[i].reshape(-1), device),
                    batch_size=args.batch_size,
                    shuffle=False,drop_last=True)
        
            hChToTauNu_data_loader[i] = DataLoader(
                    TorchCLDataset(hChToTauNu[i].reshape(-1,feat_dim), hChToTauNu_labels[i].reshape(-1), device),
                    batch_size=args.batch_size,
                    shuffle=False,drop_last=True)

            hToTauTau_data_loader[i] = DataLoader(
                    TorchCLDataset(hToTauTau[i].reshape(-1,feat_dim), hToTauTau_labels[i].reshape(-1), device),
                    batch_size=args.batch_size,
                    shuffle=False,drop_last=True)

    dataset_names = args.dataset_names #default  = ["neg3","neg1","0","pos1","pos3"]
    signal_data_loader_masterlist = [leptoquark_data_loader,ato4l_data_loader,hChToTauNu_data_loader,hToTauTau_data_loader]
    embedding_name_list = ["leptoquark.npz","ato4l.npz","hChToTauNu.npz","hToTauTau.npz"]
    
    if args.type == 'JetClass':
        model = SimpleDense_JetClass(args.latent_dim).to(device)
        
    elif args.type == 'JetClass_Transformer':
        transformer_args_jetclass = dict(
        input_dim=4, 
        model_dim=128, 
        output_dim=64,
        embed_dim=6,   #Only change embed_dim without describing new transformer architecture
        n_heads=8, 
        dim_feedforward=256, 
        n_layers=4,
        hidden_dim_dino_head=256,
        bottleneck_dim_dino_head=64,
        pos_encoding = True,
        use_mask = True,
        mode='cls',
        )
        model = TransformerEncoder(**transformer_args_jetclass).to(device)
        
    elif args.type == 'Delphes':
        transformer_args_standard = dict(
        input_dim=3, 
        model_dim=64, 
        output_dim=64, 
        embed_dim=4,
        n_heads=8, 
        dim_feedforward=256, 
        n_layers=4,
        hidden_dim_dino_head=256,
        bottleneck_dim_dino_head=64,
        pos_encoding = True,
        use_mask = False,
        mode='cls',
        dropout=0.025,
        )
        
        model = TransformerEncoder(**transformer_args_standard).to(device)
    else:
        assert False
        

    #Loading initial models
    if args.load_initial_model:
        
        initial_model = args.initial_model
        model.load_state_dict(torch.load(initial_model, map_location=device))
    
        print(f"Loaded inital model:{initial_model}")
    else:
        print("No initial models loaded.\n")
    

    def val_one_epoch(tb_writer,data_loader,embeddings_name):
        
        with torch.no_grad():
                
                num_batches_nominal = len(data_loader[0])  # all loaders same length
                processed_batches = 0
        
                num_datasets = len(datasets_lst)
                all_g_per_dataset = [[] for _ in range(num_datasets)]
        
                unscaled_idx = 2 
                
                for batches_idx, batches in enumerate(
                    tqdm(zip(*data_loader), total=num_batches_nominal,
                    disable=True),start=1
                ):
                    # batches[i] = (val_i, labels_i)
                    vals   = [b[0] for b in batches]
                    labels = [b[1] for b in batches]
    
                    # keep lockstep: skip if any partial
                    if any(v.shape[0] != args.batch_size for v in vals):
                        continue
                                   
                    # per-dataset forward
                    for i in range(len(datasets_lst)):
                        v = vals[i].to(device)
                        if args.supervision == 'selfsupervised':
                            if args.type == 'JetClass_Transformer':
                                v_aug1 = transform(vals[i]).reshape(-1,128,4).to(device)
                                v_aug2 = transform(vals[i]).reshape(-1,128,4).to(device)
                                v_in   = v.reshape(-1,128,4)
                                g  = model.representation(v_aug1, v_in)
                            elif args.type == 'JetClass':
                                v_aug1 = transform(vals[i]).to(device)
                                v_aug2 = transform(vals[i]).to(device)
                                g  = model.representation(v_aug1)
                            elif args.type == 'Delphes':
                                v_aug1 = transform(vals[i]).reshape(-1,19,3).to(device)
                                v_aug2 = transform(vals[i]).reshape(-1,19,3).to(device)
                                v_in   = v.reshape(-1,19,3)
                                g  = model.representation(v_aug1, v_in)
    
                        else:  # supervised
                            if args.type == 'JetClass_Transformer':
                                v_in = v.reshape(-1,128,4)
                                g = model.representation(v_in, v_in)
                                
                            elif args.type == 'JetClass':
                                g = model.representation(v)
                                
                            elif args.type == 'Delphes':
                                v_in = v.reshape(-1,19,3)
                                g = model.representation(v_in, v_in)
    
                        all_g_per_dataset[i].append(g)
            
                all_g_cat = [torch.cat(all_g_per_dataset[i], dim=0) for i in range(num_datasets)]
                gx_np = [g.detach().cpu().numpy() for g in all_g_cat]
                named = {f"gx_{name}": arr for name, arr in zip(dataset_names, gx_np)}
                os.makedirs(args.output_path, exist_ok=True)
                np.savez(os.path.join(args.output_path, embeddings_name), **named)

    summary_dir = os.path.join(args.output_path,"results")
    os.makedirs(summary_dir, exist_ok=True)
    writer = SummaryWriter(summary_dir, comment="Similarity with LR=1e-3", flush_secs=5)

    model.train(False)

    for i in range(0,len(signal_data_loader_masterlist)):
        data_loader = signal_data_loader_masterlist[i]
        embed_name = embedding_name_list[i]
        val_one_epoch(writer, data_loader, embed_name)  

class TorchCLDataset(Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, features, labels, device):
        'Initialization'
        self.device = device
        self.mean = np.mean(features)
        self.std = np.std(features)
        #print(f"Mean: {self.mean} and std: {self.std}")
        self.features = torch.from_numpy(features).to(dtype=torch.float32)
        self.labels = torch.from_numpy(labels).to(dtype=torch.float32)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.features)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = self.features[index]
        y = self.labels[index]

        return X, y



if __name__ == '__main__':
    # Parses terminal command
    parser = ArgumentParser()

    # If not using full data, name of smaller dataset to pull from
    parser.add_argument('datasets', nargs='+', type=str)

    parser.add_argument('--output-path',type=str,default = "/n/home07/tshelley2002/Uncertainty_Project/cl4ad_new/test_run/")
    
    parser.add_argument('--load-initial-model', action='store_true')
    parser.add_argument('--initial-model',type=str, default='/n/home07/tshelley2002/Uncertainty_Project/cl4ad_new/runs409/vae4.pth')

    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--model-name', type=str, default='test409') #"models/runs409/vae4.pth"
    parser.add_argument('--latent-dim', type=int, default=4)
    
    parser.add_argument('--type', choices=('Delphes', 'JetClass', 'JetClass_Transformer'))
    parser.add_argument('--supervision', choices=('selfsupervised', 'supervised'))
    parser.add_argument('--dataset-names', nargs="+", default=["neg3","neg1","0","pos1","pos3"], help="List of dataset names")

    args = parser.parse_args()
    main(args)
    
    