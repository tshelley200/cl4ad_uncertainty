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
 


def main(args, train_idx, val_idx):
    '''
    Infastructure for training CVAE (background specific and with anomalies)
    '''
    if args.linear_model_simple:
        from models import LinearModelSimple as LinearModel
    elif args.quadratic_model:
        from models import QuadraticModel as LinearModel
    else:
        from models import LinearModel
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}',flush=True)

    #Seed
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    #Import dataset with four SM background classes (ADC_delphes_original_divisions.hdf5)
    print(f"Using train folds: {train_idx}",flush=True)
    print(f"and using val fold: {val_idx}",flush=True)
    
    
    feat_dim = 57
        
    if args.type == 'JetClass' or args.type == 'JetClass_Transformer':
        feat_dim = 512
        
    #Initialize transform (empty list: None)
    transform = augmentations.Transform(["naive_masking"], feat_dim)
    
    datasets_lst = args.datasets
    nv_lst = args.nvs
    lambda_ = args.lambda_
    lambda1_=args.lambda1_
    
    train_fold,labels_train_fold,val_fold = [None] * len(datasets_lst),[None] * len(datasets_lst),[None] * len(datasets_lst)
    labels_val_fold,x_test,labels_test = [None] * len(datasets_lst),[None] * len(datasets_lst),[None] * len(datasets_lst)
    charac_trainset =[None] * len(datasets_lst)
    train_data_loader,test_data_loader,val_data_loader = [None] * len(datasets_lst),[None] * len(datasets_lst),[None] * len(datasets_lst)
    
    n_train = args.train_num
    n_val   = args.val_num   
    n_test  = args.test_num   
    
    for i in range(0,len(datasets_lst)):
        
        with h5py.File(f'{datasets_lst[i]}', 'r') as f:
            
            #TO-DO: check hdf5 file structures
            train_fold[i] = np.concatenate([np.array(f[f'x_train_fold_{idx}'][...]) for idx in train_idx], axis=0)[:n_train]
            labels_train_fold[i] = np.concatenate([np.array(f[f'labels_train_fold_{idx}'][...]) for idx in train_idx], axis=0)[:n_train]
            
            val_fold[i] = np.array(f[f'x_train_fold_{val_idx}'][...])[:n_val]
            labels_val_fold[i] = np.array(f[f'labels_train_fold_{val_idx}'][...])[:n_val]
            
            x_test[i] = np.array(f['x_test'][...])[:n_test]
            labels_test[i] = np.array(f['labels_test'][...])[:n_test]
            
            #Shuffle the train dataset
        train_fold[i], labels_train_fold[i] = shuffle(train_fold[i], labels_train_fold[i], random_state=0)


        #For corruption augm. get min, max, mean, std values of all the features in the training dataset
        charac_trainset[i] = dict(
            feat_low = np.min(train_fold[i].reshape(-1,feat_dim), axis=0), 
            feat_high = np.max(train_fold[i].reshape(-1,feat_dim), axis=0),
            feat_mean = np.mean(train_fold[i].reshape(-1,feat_dim), axis=0), 
            feat_std = np.std(train_fold[i].reshape(-1, feat_dim), axis=0),
        )
        
    
        train_data_loader[i] = DataLoader(
            TorchCLDataset(train_fold[i].reshape(-1,feat_dim), labels_train_fold[i].reshape(-1), device),
            batch_size=args.batch_size,
            shuffle=True,drop_last=True)

        test_data_loader[i] = DataLoader(
            TorchCLDataset(x_test[i].reshape(-1,feat_dim), labels_test[i].reshape(-1), device),
            batch_size=args.batch_size,
            shuffle=False,drop_last=True)

        val_data_loader[i] = DataLoader(
            TorchCLDataset(val_fold[i].reshape(-1,feat_dim), labels_val_fold[i].reshape(-1), device),
            batch_size=args.batch_size,
            shuffle=False,drop_last=True)

    #Defining modles
    #Try to load models
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
        
    para_model = LinearModel(4).to(device)
    if args.zero_init:
        para_model.zero_init()                  
    
    #Freezing models if chooses to
    if args.freeze_linear_model:
        for param in para_model.parameters(): 
            param.requires_grad = False
            
    if args.freeze_model:
        for param in model.parameters(): 
            param.requires_grad = False
            
    #Loading initial models
    if args.load_initial_model:
        
        initial_model = args.initial_model
        model.load_state_dict(torch.load(initial_model, map_location=device))
        
        #Freeze or train the initial loaded models, depending on the input arguments
        for param in model.parameters():
            if args.freeze_initial_model:
                #if freezes model, set requries_grad to False
                param.requires_grad = False
            else:
                param.requires_grad = True
        print(f"Loaded inital model:{initial_model}",flush=True)
        print(f"Freezes initial model = {args.freeze_initial_model}",flush=True)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters (Kyles): {num_params}",flush=True)
        print("\n")
        
    else:
        print("No initial models loaded.\n",flush=True)
    
    if args.load_initial_linear_model:
        
        initial_linear_model = args.initial_linear_model
        para_model.load_state_dict(torch.load(initial_linear_model, map_location=device))
        
        for param in para_model.parameters():
            if args.freeze_initial_linear_model:
                #if freezes model, set requries_grad to False
                param.requires_grad = False
            #else:
                #param.requires_grad = True
                
        print(f"Loaded inital linear model:{initial_linear_model}",flush=True)
        print(f"Freezes initial linear model = {args.freeze_initial_linear_model}",flush=True)
        para_num_params = sum(p.numel() for p in para_model.parameters() if p.requires_grad)
        print(f"Total trainable parameters (parametric): {para_num_params}",flush=True)
        print("\n")
    else:
        print("No initial linear models loaded.\n",flush=True)    
        
        
    #Setup SimCLR loss: defaults to self-supervised version if no labels are given
    criterion = losses.SimCLRloss_nolabels_fast(temperature=args.loss_temp, base_temperature=args.loss_temp, contrast_mode='one')
    linear_criterion = losses.ExpoLoss_Linear().to(device)
    
    # Build optimizer with two models; only applies function of adjusting lr to Kyle's model
    
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad],
         "lr": 0, "weight_decay": args.model_regularization, "apply_schedule": True}, 
        
        {"params": [p for p in para_model.parameters() if p.requires_grad],
         "lr": args.linear_model_lr, "weight_decay":args.linear_model_regularization, "apply_schedule": False} ])
    

    # ====== torch-native prepare (keeps graph) ======
    def torch_prepare_data_from_gx(gx_list, unscaled_idx, nv_lst, device):
        # gx_list: list of tensors shaped [B, 4] (your latent dim)
        dtype = gx_list[0].dtype
        B = gx_list[0].shape[0]

        # nv as torch tensor for a differentiable std (no grads, but fine on-device)
        nv = torch.tensor(nv_lst, device=device, dtype=dtype)
        nv_std = nv.std(unbiased=False) + 1e-12

        unscaled_gx = gx_list[unscaled_idx]               # <-- NO detach
        
        # Build pairs (unscaled, scaled_i) for every i (including the unscaled one)
        feature_chunks = []
        y_chunks = []
        nu_chunks = []
        
        scaled_list = [gx_list[i] for i in range(0,len(gx_list)) if i != unscaled_idx]
        
        for i, g_i in enumerate(scaled_list):
            # append unscaled then scaled_i (to mimic your original layout)
            feature_chunks.append(unscaled_gx)
            feature_chunks.append(g_i)

            # targets: 0 for unscaled, 1 for scaled_i
            y_chunks.append(torch.zeros(B, device=device, dtype=dtype))
            y_chunks.append(torch.ones(B,  device=device, dtype=dtype))

            # nuisance labels for each row (normalized)
            nu_val = (nv[i] / nv_std).expand(B)
            nu_chunks.append(nu_val)         # for the unscaled block
            nu_chunks.append(nu_val)         # for the scaled block
        
        

        feature = torch.cat(feature_chunks, dim=0)        # [2*len(nv_lst)*B, 4]
        y       = torch.cat(y_chunks,      dim=0)         # [2*len(nv_lst)*B]
        nu      = torch.cat(nu_chunks,     dim=0)         # [2*len(nv_lst)*B]
        
        # Normalize by stats of the unscaled representation (keep graph!)
        mean0 = unscaled_gx.mean(dim=0, keepdim=True)
        std0  = unscaled_gx.std(dim=0, unbiased=False, keepdim=True) + 1e-6
        
        
        #mean0 = mean0.detach()
        #std0  = std0.detach()
        feature = (feature - mean0) / std0

        target = torch.stack([y, nu], dim=1)              # shape [N, 2]
        
        
        return feature, target

    
    def train_one_epoch(epoch_index, tb_writer):
        
        model.train()
        para_model.train()

        num_batches_nominal = len(train_data_loader[0])  # nominal length (all loaders equal)
        processed_batches = 0                            # actual processed (after drops)

        # epoch accumulators
        epoch_total_sum = 0.0
        epoch_expo_sum  = 0.0
        epoch_sim_sum   = 0.0

        # window accumulators for periodic logging
        #window_total = window_expo = window_sim = 0.0
        #window_count = 0
        LOG_EVERY = 500  # batches
        
        if args.epochs == epoch_index:
            num_datasets = len(datasets_lst)
            all_g_per_dataset = [[] for _ in range(num_datasets)]
            
        unscaled_idx = int((len(datasets_lst)-1)/2) # should be 2
        para_num_params = sum(p.numel() for p in para_model.parameters() if p.requires_grad)
        print(f"Total trainable parameters (parametric): {para_num_params}",flush=True)
        # iterate all datasets in lockstep
        for idx, batches in enumerate(
            tqdm(zip(*train_data_loader), total=num_batches_nominal, desc=f"\nEpoch {epoch_index}",
            disable=True),start=1
        ):
            # batches[i] = (val_i, labels_i)
            vals   = [b[0] for b in batches]
            labels = [b[1] for b in batches]

            # drop partials if loaders weren't built with drop_last=True
            if any(v.shape[0] != args.batch_size for v in vals):
                continue
            
            if args.average_sim:
                sim_losses = []
            
            gx_list = []
            
            for i in range(len(datasets_lst)):

                v = vals[i].to(device)
                    
                if args.supervision == 'selfsupervised':
                    
                    if args.type == 'JetClass_Transformer':
                        v_aug1 = transform(vals[i]).reshape(-1,128,4).to(device)
                        v_aug2 = transform(vals[i]).reshape(-1,128,4).to(device)
                        v_in   = v.reshape(-1,128,4)
                        g  = model.representation(v_aug1, v_in)
                        if (args.average_sim) or (i == unscaled_idx):
                            z1 = model(v_aug1, v_in)
                            z2 = model(v_aug2, v_in)
                    elif args.type == 'JetClass':
                        v_aug1 = transform(vals[i]).to(device)
                        v_aug2 = transform(vals[i]).to(device)
                        g  = model.representation(v_aug1)
                        
                        if (args.average_sim) or (i == unscaled_idx):
                            z1 = model(v_aug1)
                            z2 = model(v_aug2)
                        
                    elif args.type == 'Delphes':
                        v_aug1 = transform(vals[i]).reshape(-1,19,3).to(device)
                        v_aug2 = transform(vals[i]).reshape(-1,19,3).to(device)
                        v_in   = v.reshape(-1,19,3)
                        g  = model.representation(v_aug1, v_in)
                        if (args.average_sim) or (i == unscaled_idx):
                            z1 = model(v_aug1, v_in)
                            z2 = model(v_aug2, v_in)
                            
                    if args.average_sim:
                        feat = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)
                        sim_losses.append(criterion(feat))
                        
                    elif i == unscaled_idx:
                        feat = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)
                        sim_loss = criterion(feat)

                else:  # supervised
                    
                    if args.type == 'JetClass_Transformer':
                        v_in = v.reshape(-1,128,4)
                        g = model.representation(v_in, v_in)
                        if (args.average_sim) or (i == unscaled_idx):
                            z = model(v_in, v_in)
                        
                    elif args.type == 'JetClass':
                        g = model.representation(v)
                        if (args.average_sim) or (i == unscaled_idx):
                            z = model(v)
                            
                    elif args.type == 'Delphes':
                        
                        v_in = v.reshape(-1,19,3)
                        g = model.representation(v_in, v_in)
                        if (args.average_sim) or (i == unscaled_idx):
                            z = model(v_in, v_in)
                    
                    if args.average_sim:
                        sim_losses.append(criterion(z.unsqueeze(1), labels[i].reshape(-1).to(device)))
                        
                    elif i == unscaled_idx:
                        sim_loss = criterion(z.unsqueeze(1), labels[i].reshape(-1).to(device))

                gx_list.append(g)
                if epoch_index == args.epochs:
                        all_g_per_dataset[i].append(g)
            X, true = torch_prepare_data_from_gx(gx_list, unscaled_idx, nv_lst, device)

            fx = para_model(X, true)
            expo_loss = linear_criterion(true, fx)
            
            if args.average_sim:
                sim_loss  = torch.stack(sim_losses).mean()

            total_loss = (lambda_ * expo_loss) + (lambda1_ * sim_loss)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # values for logging
            total_val = float(total_loss.item())
            expo_val  = float(expo_loss.item())
            sim_val   = float(sim_loss.item())

            # epoch sums
            epoch_total_sum += total_val
            epoch_expo_sum  += expo_val
            epoch_sim_sum   += sim_val
            processed_batches += 1

        if epoch_index == args.epochs:
            all_g_cat = [torch.cat(all_g_per_dataset[i], dim=0) for i in range(num_datasets)]
            gx_np = [g.detach().cpu().numpy() for g in all_g_cat]
            np.savez(os.path.join(args.output_path,args.output_filename), *gx_np)
            
        # epoch averages (use processed_batches for correctness)
        if processed_batches > 0:
            epoch_total_avg = epoch_total_sum / processed_batches
            epoch_expo_avg  = epoch_expo_sum  / processed_batches
            epoch_sim_avg   = epoch_sim_sum   / processed_batches
        else:
            epoch_total_avg = epoch_expo_avg = epoch_sim_avg = float('nan')

        print(f"[Training Epoch {epoch_index}] total={epoch_total_avg:.6f} | expo={epoch_expo_avg:.6f} | sim={epoch_sim_avg:.6f}\n",flush=True)
        """tb_writer.add_scalar('Loss/train_total_epoch', epoch_total_avg, epoch_index)
        tb_writer.add_scalar('Loss/train_expo_epoch',  epoch_expo_avg,  epoch_index)
        tb_writer.add_scalar('Loss/train_sim_epoch',   epoch_sim_avg,   epoch_index)"""

        return epoch_total_avg, epoch_expo_avg, epoch_sim_avg

    
    
    def val_one_epoch(epoch_index, tb_writer):

        num_batches_nominal = len(val_data_loader[0])  # all loaders same length
        processed_batches = 0

        # epoch accumulators
        epoch_total_sum = 0.0
        epoch_expo_sum  = 0.0
        epoch_sim_sum   = 0.0

        # window accumulators for periodic logging
        #window_total = window_expo = window_sim = 0.0
        #window_count = 0
        
        LOG_EVERY_VAL = 50  # batches
        unscaled_idx = int((len(datasets_lst)-1)/2) # should be 2
        
        with torch.no_grad():
            
            for idx, batches in enumerate(
                tqdm(zip(*val_data_loader), total=num_batches_nominal, desc=f"[VAL] Epoch {epoch_index}",
                disable=True),start=1
            ):
                # batches[i] = (val_i, labels_i)
                vals   = [b[0] for b in batches]
                labels = [b[1] for b in batches]

                # keep lockstep: skip if any partial
                if any(v.shape[0] != args.batch_size for v in vals):
                    continue
                gx_list = []
                if args.average_sim:
                    sim_losses = []
                
                # per-dataset forward
                for i in range(len(datasets_lst)):
                    v = vals[i].to(device)
                    if args.supervision == 'selfsupervised':
                        if args.type == 'JetClass_Transformer':
                            v_aug1 = transform(vals[i]).reshape(-1,128,4).to(device)
                            v_aug2 = transform(vals[i]).reshape(-1,128,4).to(device)
                            v_in   = v.reshape(-1,128,4)
                            g  = model.representation(v_aug1, v_in)
                            if (args.average_sim) or (i == unscaled_idx):
                                z1 = model(v_aug1, v_in)
                                z2 = model(v_aug2, v_in)
                            
                        elif args.type == 'JetClass':
                            v_aug1 = transform(vals[i]).to(device)
                            v_aug2 = transform(vals[i]).to(device)
                            g  = model.representation(v_aug1)
                            if (args.average_sim) or (i == unscaled_idx):
                                z1 = model(v_aug1)
                                z2 = model(v_aug2)
                            
                        elif args.type == 'Delphes':
                            v_aug1 = transform(vals[i]).reshape(-1,19,3).to(device)
                            v_aug2 = transform(vals[i]).reshape(-1,19,3).to(device)
                            v_in   = v.reshape(-1,19,3)
                            g  = model.representation(v_aug1, v_in)
                            if (args.average_sim) or (i == unscaled_idx):
                                z1 = model(v_aug1, v_in)
                                z2 = model(v_aug2, v_in)
            
                        if args.average_sim:
                            feat = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)
                            sim_losses.append(criterion(feat))
                        
                        elif i == unscaled_idx:
                            feat = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)
                            sim_loss = criterion(feat)
                        

                    else:  # supervised
                        if args.type == 'JetClass_Transformer':
                            v_in = v.reshape(-1,128,4)
                            g = model.representation(v_in, v_in)
                            if (args.average_sim) or (i == unscaled_idx):
                                z = model(v_in, v_in)
                            
                        elif args.type == 'JetClass':
                            g = model.representation(v)
                            if (args.average_sim) or (i == unscaled_idx):
                                z = model(v)
                            
                        elif args.type == 'Delphes':
                            v_in = v.reshape(-1,19,3)
                            g = model.representation(v_in, v_in)
                            if (args.average_sim) or (i == unscaled_idx):
                                z = model(v_in, v_in)
                            

                        if args.average_sim:
                            sim_losses.append(criterion(z.unsqueeze(1), labels[i].reshape(-1).to(device)))
                            
                        elif i == unscaled_idx:
                            sim_loss = criterion(z.unsqueeze(1), labels[i].reshape(-1).to(device))
                            
                    gx_list.append(g)
                    

                X, true = torch_prepare_data_from_gx(gx_list, unscaled_idx, nv_lst, device)

                # para_model + losses
                fx = para_model(X, true)
                expo_loss = linear_criterion(true, fx)
                
                if args.average_sim:
                    sim_loss  = torch.stack(sim_losses).mean()
                
                total_loss = (lambda_ * expo_loss) + (lambda1_*sim_loss)

                # collect values
                total_val = float(total_loss.item())
                expo_val  = float(expo_loss.item())
                sim_val   = float(sim_loss.item())

                epoch_total_sum += total_val
                epoch_expo_sum  += expo_val
                epoch_sim_sum   += sim_val
                processed_batches += 1

        # epoch averages (divide by actually processed batches)
        if processed_batches > 0:
            epoch_total_avg = epoch_total_sum / processed_batches
            epoch_expo_avg  = epoch_expo_sum  / processed_batches
            epoch_sim_avg   = epoch_sim_sum   / processed_batches
        else:
            epoch_total_avg = epoch_expo_avg = epoch_sim_avg = float('nan')

        print(f"[Validation Epoch {epoch_index}] total={epoch_total_avg:.6f} | expo={epoch_expo_avg:.6f} | sim={epoch_sim_avg:.6f}\n",flush=True)
        """tb_writer.add_scalar('Loss/val_total_epoch', epoch_total_avg, epoch_index)
        tb_writer.add_scalar('Loss/val_expo_epoch',  epoch_expo_avg,  epoch_index)
        tb_writer.add_scalar('Loss/val_sim_epoch',   epoch_sim_avg,   epoch_index)"""

        return epoch_total_avg, epoch_expo_avg, epoch_sim_avg

    
    summary_dir = os.path.join(args.output_path,"results")
    os.makedirs(summary_dir, exist_ok=True)
    writer = SummaryWriter(summary_dir, comment="Similarity with LR=1e-3", flush_secs=5)
    
    model_ckpt_dir = os.path.join(args.output_path, "model_checkpoints")
    os.makedirs(model_ckpt_dir, exist_ok=True)
    
    linear_model_ckpt_dir = os.path.join(args.output_path, "linear_model_checkpoints")
    os.makedirs(linear_model_ckpt_dir, exist_ok=True)
    
    if args.train:
        
        train_total_losses = []
        train_expo_losses = []
        train_sim_losses = []
        
        val_total_losses = []
        val_expo_losses = []
        val_sim_losses = []
        
        start_time = time.time()

        for epoch in range(1, args.epochs+1):
            print(f'EPOCH {epoch}',flush=True)
            temp_time= time.time()
            
            #Adjust the learning rate with Version 2 schedule (see OneNote)
            lr = adjust_learning_rate(args, 10, epoch, optimizer, base_lr=args.base_lr)
            print("current Learning rate: ", lr,flush=True)
            writer.add_scalar('Learning_rate', lr, epoch)
            
            # Gradient tracking
            model.train(True)
            para_model.train(True)
            
            train_total_loss,train_expo_loss,train_sim_loss = train_one_epoch(epoch, writer)
            
            train_total_losses.append(train_total_loss)
            train_expo_losses.append(train_expo_loss)
            train_sim_losses.append(train_sim_loss)

            # no gradient tracking, for validation
            model.train(False)
            para_model.train(False)
            
            val_total_loss,val_expo_loss,val_sim_loss = val_one_epoch(epoch, writer)
            
            val_total_losses.append(val_total_loss)
            val_expo_losses.append(val_expo_loss)
            val_sim_losses.append(val_sim_loss)

            temp_time = time.time()-temp_time
            print(f"Train/Val Sim Loss after epoch: {train_total_loss:.4f}/{val_total_loss:.4f}",flush=True)
            print(f"taking {temp_time:.1f}s to complete",flush=True)

            #Save checkpoints
            
            torch.save(model.state_dict(), os.path.join(model_ckpt_dir,f"{val_idx}_vae_ep_{epoch}.pth"))
            torch.save(para_model.state_dict(), os.path.join(linear_model_ckpt_dir,f"{val_idx}_vae_ep_{epoch}.pth"))
            
        writer.flush()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str),flush=True)
        
        final_dir = os.path.join(args.output_path, "final_models")
        os.makedirs(final_dir, exist_ok=True)
        
        #Saving model 
        torch.save(model.state_dict(),      os.path.join(final_dir, f"{args.model_name}_{val_idx}_final.pth"))
        torch.save(para_model.state_dict(), os.path.join(final_dir, f"{args.para_model_name}_{val_idx}_final.pth"))
        torch.save(optimizer.state_dict(), os.path.join(final_dir, f"{args.model_name}_{args.para_model_name}_{val_idx}_optimizer_final.pth"))

        # single combined checkpoint that also stores metadata
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "para_model_state": para_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),}, os.path.join(final_dir, f"{args.model_name}_{args.para_model_name}_{val_idx}_FULL_FINAL.pt"))

        fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

        # --- 1. Total Loss ---
        axs[0].plot(train_total_losses, label='train total loss', color='saddlebrown')
        axs[0].plot(val_total_losses, label='val total loss', color='darkslategrey')
        axs[0].set_ylabel('Total Loss')
        axs[0].legend()
        axs[0].grid(True, linestyle='--', alpha=0.4)

        # --- 2. Expo Loss ---
        axs[1].plot([i * lambda_ for i in train_expo_losses], label='train expo loss × λ', color='tan')
        axs[1].plot([i * lambda_ for i in val_expo_losses], label='val expo loss × λ', color='lightskyblue')
        axs[1].set_ylabel('Expo Loss × λ')
        axs[1].legend()
        axs[1].grid(True, linestyle='--', alpha=0.4)

        # --- 3. Sim Loss ---
        axs[2].plot(train_sim_losses, label='train sim loss', color='peru')
        axs[2].plot(val_sim_losses, label='val sim loss', color='lightslategrey')
        axs[2].set_xlabel('Iterations')
        axs[2].set_ylabel('Sim Loss')
        axs[2].legend()
        axs[2].grid(True, linestyle='--', alpha=0.4)

        # Adjust spacing and save
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_path, f'loss_valfold_{val_idx}.pdf'))
        plt.close()
        
        writer.close()
        
    else:
        #To-Do: add input path
        
        model.load_state_dict(torch.load(args.model_name, map_location=torch.device(device)))
        para_model.load_state_dict(torch.load(args.para_model_name, map_location=torch.device(device)))
        
        model.eval()
        para_model.eval()


def adjust_learning_rate(args, warmup_epochs ,epoch, optimizer, base_lr):
        max_epochs = args.epochs
        base_lr = base_lr * args.batch_size / 256 #Scale like suggested by VICReg for base_lr = 0.2 (SimCLR has base_lr = 0.3)
        if epoch <= warmup_epochs:
            lr = base_lr * epoch / warmup_epochs #Linear warmup
        else:
            epoch -= warmup_epochs
            max_epochs -= warmup_epochs
            q = 0.5 * (1 + math.cos(math.pi * epoch / max_epochs))
            end_lr = base_lr * 0.001
            lr = base_lr * q + end_lr * (1-q)

        for g in optimizer.param_groups:
            if g.get("apply_schedule", True):   # <-- only update tagged groups
                g["lr"] = lr
        return lr
            

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
    parser.add_argument('--nvs', nargs='+', type=float)
    parser.add_argument('--train-num',type=int,default=300000)
    parser.add_argument('--val-num',type=int,default=200000)
    parser.add_argument('--test-num',type=int,default=200000)

    parser.add_argument('--output-path',type=str,default = "/n/home07/tshelley2002/Uncertainty_Project/cl4ad_new/test_run/")
    
    
    parser.add_argument('--load-initial-model', action='store_true')
    parser.add_argument('--initial-model',type=str, default='/n/home07/tshelley2002/Uncertainty_Project/cl4ad_new/runs409/vae4.pth')
    parser.add_argument('--freeze-initial-model', action='store_true')
    
    parser.add_argument('--linear-model-simple',action='store_true')
    parser.add_argument('--quadratic-model',action='store_true')
    parser.add_argument('--zero-init',action='store_true')
    
    parser.add_argument('--load-initial-linear-model', action='store_true')
    parser.add_argument('--initial-linear-model',type=str,default='')
    parser.add_argument('--freeze-initial-linear-model', action='store_true')
    
    
    parser.add_argument('--freeze-linear-model', action='store_true')
    parser.add_argument('--freeze-model', action='store_true')
    
    parser.add_argument('--lambda',dest='lambda_',type = float,default=1)
    parser.add_argument('--lambda1',dest='lambda1_',type = float,default=1)
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--loss-temp', type=float, default=0.1)
    
    parser.add_argument('--model-regularization',type=float, default=1e-6)
    parser.add_argument('--linear-model-regularization',type=float,default=1e-7)
    
    parser.add_argument('--base-lr', type=float, default=0.00025)
    parser.add_argument('--linear-model-lr', type=float, default=2e-4)
    parser.add_argument('--average-sim', action='store_true')
    
    #Loading later models for validations
    #To-Do: add a input file path
    parser.add_argument('--model-name', type=str, default='test409') #"models/runs409/vae4.pth"
    parser.add_argument('--para-model-name', type=str, default='testlinear') 

    parser.add_argument('--output-filename', type=str, default='embedding.npz')
    
    parser.add_argument('--latent-dim', type=int, default=4)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--type', choices=('Delphes', 'JetClass', 'JetClass_Transformer'))
    parser.add_argument('--supervision', choices=('selfsupervised', 'supervised'))
    
    parser.add_argument('--k-fold', action='store_true')
    parser.add_argument('--k-fold-num',type=int,default=5)

    args = parser.parse_args()
    
    #Do the k-folding (runs the main loop five times)
    """if args.k_fold:
        for i in range(5):
            train_idx = [0,1,2,3,4]
            train_idx.remove(i)
            val_idx = i
            main(args, train_idx=train_idx, val_idx=val_idx)"""
            
    if args.k_fold:
        for i in range(args.k_fold_num):
            train_idx = [j for j in range(0,args.k_fold_num)]
            train_idx.remove(i)
            val_idx = i
            main(args, train_idx=train_idx, val_idx=val_idx)
    else:
        #To test just set the trainset to fold0 and valset to fold1
        main(args, train_idx=[0], val_idx=1)
    
    
    """if args.load_initial_model:
        if not args.initial_model:
            parser.error("--initial-model must be provided if --load-initial-model is True")
    else:
        args.initial_model = None  # ignore it if not loading
        args.freeze_initial_model = False
        
    if args.load_initial_linear_model:
        if not args.initial_linear_model:
            parser.error("--initial-linear-model must be provided if --load-initial-linear-model is True")
    else:
        args.initial_linear_model = None  # ignore it if not loading
        args.freeze_initial_linear_model = False"""