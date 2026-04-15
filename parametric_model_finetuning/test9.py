import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import random
import os
from argparse import ArgumentParser
from utils import *

def main(args):

    nv = np.array(args.nv)
    nbg, nref = args.nbg, args.nref
    models_path = args.models_path
    input_file_name = args.input_file_name
    input_model_name = args.input_model_name
    output_model_name = args.output_model_name
    
    epochs = args.epochs
    batch_size = args.batch_size
    run_num = args.run_num
    nv_only_val = (np.array(args.nv))/10
    learning_rate = args.learning_rate
    regularization = args.regularization

    model_name = "cubic" if args.cubic else "quadratic"
    alt_name = True if args.alt_name else False
    not_save = True if args.not_save_model else False
    load_model = True if args.load_model else False
    input_file_path = os.path.join(models_path, input_file_name)
    para_model_path = os.path.join(models_path, "final_models", input_model_name)

    optimizer_name = args.optimizer
    
    feature,target = load_data(nv=nv,
                               filename = input_file_path,
                               N_Bkg_Pois = nbg,
                               N_ref = nref,
                               alt_name = alt_name)
    
    ds, dl, X, true, model, num_blocks, patience = prepare_training(feature = feature,target = target, 
                     load_model=load_model, 
                     load_model_path = para_model_path,
                     total_epochs = epochs,
                     patience = 5,
                     batch_size = batch_size,
                     model_name=model_name)
    
    device = next(model.parameters()).device  
    expo_loss = ExpoLoss().to(device)
    history,epoch_ix  = [],[]    # last loss of each block
    
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW([{"params": [p for p in model.parameters() if p.requires_grad],
                 "lr": learning_rate, "weight_decay":regularization} ])
        
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam([{"params": [p for p in model.parameters() if p.requires_grad],
                 "lr": learning_rate, "weight_decay":regularization} ])
    
    model.train()
    
    for i in range(num_blocks):
        last_epoch_loss = None
        
        for e in range(patience):
    
            running_loss, nseen = 0.0, 0
            
            for xb, tb in dl:
                
                xb, tb = xb.to(device), tb.to(device)
                optimizer.zero_grad() 
                pred = model(xb,tb)
                loss = expo_loss(tb,pred) 
                loss.backward()
                optimizer.step()
                model.apply(clip_weights) 
                running_loss += loss.item() * xb.size(0) # xb.size = batch size
                nseen += xb.size(0)  #total num of samples processed so far
                
            last_epoch_loss = running_loss / max(nseen, 1) 
            
            print(last_epoch_loss)
    
        k = patience * (i + 1)
        history.append(last_epoch_loss)
        epoch_ix.append(k)
        print(f"epoch: {k}, loss: {last_epoch_loss:.8f}")
        
    model.eval()
    with torch.no_grad():
        fx_chunks = []
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size].to(device)
            tb = true[i:i+batch_size].to(device)
            with torch.no_grad():
                fxb = model(xb, tb)
            fx_chunks.append(fxb.cpu())
        fx = torch.cat(fx_chunks, dim=0)
        #fx = model(X,true)
        
        
    fx_np    = to_np(fx) 
    twosamples_tests(feature_np = feature, 
                     fx = fx_np , 
                     nv_only_val = nv_only_val, 
                     N_ref = nref, 
                     N_Bkg_Pois = nbg, 
                     targets = target,
                     bin_num=45)
    
    if not not_save:
        output_model_path =os.path.join(models_path, "final_models", output_model_name)
        #os.makedirs(output_model_path, exist_ok=True)
        torch.save(model.state_dict(),output_model_path)
        

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--nv', nargs='+', type=float, default=[-1.5, -0.5, 0.5, 1.5])
    parser.add_argument('--nbg', type=int, default=1000000)
    parser.add_argument('--nref', type=int, default=1000000)
    parser.add_argument('--cubic', action='store_true')
    parser.add_argument('--alt_name', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--models_path',type=str)
    parser.add_argument('--input_file_name', type=str)
    parser.add_argument('--input_model_name', type=str)
    parser.add_argument('--output_model_name', type=str)
    parser.add_argument('--optimizer', type=str,default = "AdamW")
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=80000)
    parser.add_argument('--run_num', type=int, default=9)
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--regularization', type=float, default=0.05e-6)
    parser.add_argument('--not_save_model', action='store_true')

    args = parser.parse_args()
    
    main(args)

    
    

    
    
