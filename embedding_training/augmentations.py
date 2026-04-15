import numpy as np
import torch
# AUGMENTATIONS introduced in https://arxiv.org/pdf/2301.04660
# code adapted from https://github.com/bmdillon/AnomalyCLR/blob/main/EventLevelAnomalyAugmentations.py

#################################################
#################################################
###  PHYSICAL AUGMENTATIONS
#################################################
#################################################

def rotate_events(input_batch, device=None, feat_dim=57):
    '''
    Args:
        input_batch: (batch_size, 57) flattened input
        device: cuda or cpu depending on input
    Returns:
        permutated_batch: (batch_size, 57) permutated output
    '''
    if feat_dim == 57:
        input_numpy = input_batch.numpy().reshape(-1,19,3).copy()
    elif feat_dim == 512:
        input_numpy = input_batch.numpy().reshape(-1,128,4).copy()

    #Rotate the whole thing around the beamline (phi) at angle
    #Angles are stored from [-pi,pi]
    angle = np.random.uniform(input_numpy.shape[0])*2*np.pi - np.pi
    phi = input_numpy[:,:,2] # [N, 19]
    ones  = np.ones_like(phi)# [N, 19]
    is_0pad = np.sum(input_numpy, axis=2) == 0  # [N, 19]
    angle_matrix = np.einsum('ij,i->ij', ones, angle)  # [N, 19]
    angle_matrix[is_0pad == True] = 0
    phi = (phi+ angle_matrix)
    phi = np.where(phi>np.pi, phi - 2*np.pi, phi)
    phi = np.where(phi<-np.pi, phi+2*np.pi, phi)
    input_numpy[:,:,2]=phi
    
    #Return a torch tensor on the given device and correct shape (-1,57)
    rotated_batch = torch.from_numpy(input_numpy.reshape(-1,feat_dim)).to(dtype=torch.float32)
    return rotated_batch
        
def energy_smear_jets(input_batch, device=None, feat_dim=57, strength=1.0):
    if feat_dim == 57:
        input_numpy = input_batch.numpy().reshape(-1,19,3).copy()
    elif feat_dim == 512:
        input_numpy = input_batch.numpy().reshape(-1,128,4).copy()
    
    pT = input_numpy[:, 9:, 0]
    mask = pT > 0
    pTfunc = np.sqrt( 0.052*pT**2 + 1.502*pT )
    shift_pt = np.nan_to_num( strength * pTfunc * np.random.randn( pT.shape[0], pT.shape[1] ), posinf = 0., neginf = 0.) * mask
    input_numpy[:, 9:, 0] += shift_pt
    # clip to zero
    lz = input_numpy[:, 9:, 0] < 0.
    input_numpy[:, 9:, 0][lz] = 0.0
    #Return a torch tensor on the given device and correct shape (-1,57)
    esmeared_batch = torch.from_numpy(input_numpy.reshape(-1,feat_dim)).to(dtype=torch.float32)
    return esmeared_batch

def get_std_rivet(pTs, scaler_pt, A=0.028, B=25, C=0.1):
    #  standard deviation for the Rivet detector simulation
    mask = (pTs > 0)
    np_sett_dict = np.seterr(over = 'ignore')
    if scaler_pt != None:
        std_rivet  = A/(1+np.exp( ( (pTs *scaler_pt) -B)/C) )
    else: 
        std_rivet  = A/(1+np.exp( ( pTs -B)/C) )
    std_rivet[~mask] = 0
    np.seterr(over = np_sett_dict['over'])
    return std_rivet

def etaphi_smear_events(input_batch, scaler_pt=1.0, strength=1.0, ):
    if feat_dim == 57:
        input_numpy = input_batch.numpy().reshape(-1,19,3).copy()
    elif feat_dim == 512:
        input_numpy = input_batch.numpy().reshape(-1,128,4).copy()
        
    std = get_std_rivet(input_numpy[:,1:,0], scaler_pt )
    noise_eta = np.random.normal( loc=0.0, scale=strength*std )
    noise_phi = np.random.normal( loc=0.0, scale=strength*std )
    noise     = np.stack( [noise_eta, noise_phi], axis=1 )
    input_numpy[:,1:, 1:3] += noise
    # adjust phi to be in [-pi,pi]
    input_numpy[:,1:, 2] = np.where(input_numpy[:, 1:, 2]>np.pi, batch_distorted[:, 2, 1:]-np.pi, batch_distorted[:, 1:, 2]) 
    input_numpy[:, 1:, 2] = np.where(input_numpy[:, 1:, 2]< -np.pi, batch_distorted[:, 2, 1:]+np.pi, batch_distorted[:, 1:, 2])
    
    crosses_upper_bound_e = input_numpy[:, 1:5, 1] > 3.
    crosses_lower_bound_e = input_numpyd[:,1:5, 1] < -3.
    crosses_e = crosses_lower_bound_e | crosses_upper_bound_e
    crosses_upper_bound_mu = input_numpy[:, 5:9, 1] > 2.1
    crosses_lower_bound_mu = input_numpy[:, 5:9, 1] < -2.1
    crosses_mu = crosses_lower_bound_mu | crosses_upper_bound_mu
    crosses_upper_bound_jet = input_numpy[:, 9:, 1] > 4.
    crosses_lower_bound_jet = input_numpy[:, 9:, 1] < -4.
    crosses_jet = crosses_lower_bound_jet | crosses_upper_bound_jet
    for i in range( np.shape(input_numpy)[2] ):
        input_numpy[:, 1:5, i][crosses_e] = 0.
        input_numpy[:, 5:9, i][crosses_mu] = 0.
        input_numpy[:, 9:, i][crosses_jet] = 0.
        
    #Return a torch tensor on the given device and correct shape (-1,57)
    distorted_batch = torch.from_numpy(input_numpy.reshape(-1,feat_dim)).to(dtype=torch.float32)
    return batch_distorted
'''
def apply_sin(batch_inp, scale_angle):
    # returns an array with shape (batch_size, 8, 19), where phi is split into sin(phi) [2] and cos(phi) [3], the rest is left unchanged 
    batch = batch_inp.copy()
    batch_size = len(batch)
    new_batch = np.ones( (batch_size, 8, 19) )
    splitt = np.split(batch, [2, 3], axis=1)
    new_batch[:, :2, : ] = splitt[0]
    phi = splitt[1]
    phi = phi.reshape((batch_size, 19))
    if scale_angle:
        phi = phi * np.pi
    new_batch[:, 4:, :] = splitt[2]
    no_phi = phi == 0.
    phi_sin = np.sin(phi)
    phi_cos = np.cos(phi)
    phi_sin[no_phi] = 0.
    phi_cos[no_phi] = 0.
    new_batch[:, 2, :] = phi_sin
    new_batch[:, 3, :] = phi_cos
    return new_batch 
'''
#################################################
#################################################
###  ANOMALY AUGMENTATIONS
#################################################
#################################################

def collinear_fill_e_mu(input_batch, device=None, feat_dim=57):
    if feat_dim == 57:
        input_numpy = input_batch.numpy().reshape(-1,19,3).copy()
    elif feat_dim == 512:
        input_numpy = input_batch.numpy().reshape(-1,128,4).copy()

    # ELECTRONS
    n_constit = 4
    n_nonzero = np.count_nonzero(np.sum(input_numpy[:, 1:5, :], axis=2), axis=2)
    n_split = np.minimum(n_nonzero, n_constit-n_nonzero)
    idx_flip = np.where(n_nonzero != n_split)
    mask_split = np.sum(input_numpy[:, 1:5, :], axis=2) != 0
    mask_split [idx_flip] = np.flip(mask_split[idx_flip], axis=1)
    mask_split[idx_flip] = np.invert(mask_split[idx_flip])
    r_split = np.random.uniform(size=mask_split.shape)
    a =       r_split * mask_split*input_numpy[:, 1:5, 0]
    b = (1.0-r_split) * mask_split*input_numpy[:, 1:5, 0]
    c =                ~mask_split*input_numpy[:, 1:5, 0]
    input_numpy[:, 1:5, 0] = a + c + np.flip(b, axis=1)
    input_numpy[:, 1:5, 1] += np.flip(mask_split*input_numpy[:, 1:5, 1], axis=1)
    input_numpy[:, 1:5, 2] += np.flip(mask_split*input_numpy[:, 1:5, 1], axis=1)

    # MUONS
    n_constit = 4
    n_nonzero = np.count_nonzero(np.sum(input_numpy[:, 5:9, :], axis=2), axis=1)
    n_split = np.minimum(n_nonzero, n_constit-n_nonzero)
    idx_flip = np.where(n_nonzero != n_split)
    mask_split = np.sum(input_numpy[:, 5:9, :], axis=2) != 0
    mask_split [idx_flip] = np.flip(mask_split[idx_flip], axis=1)
    mask_split[idx_flip] = np.invert(mask_split[idx_flip])
    r_split = np.random.uniform(size=mask_split.shape)
    a =       r_split * mask_split*input_numpy[:, 5:9, 0]
    b = (1.0-r_split) * mask_split*input_numpy[:, 5:9, 0]
    c =                ~mask_split*input_numpy[:, 5:9, 0]
    input_numpy[:, 5:9, 0] = a + c + np.flip(b, axis=1)
    input_numpy[:, 5:9, 1] += np.flip(mask_split*input_numpy[:, 5:9, 1], axis=1)
    input_numpy[:, 5:9, 2] += np.flip(mask_split*input_numpy[:, 5:9, 2], axis=1)
    
    #Return a torch tensor on the given device and correct shape (-1,57)
    filled_batch = torch.from_numpy(input_numpy.reshape(-1,feat_dim)).to(dtype=torch.float32)
    return filled_batch


def collinear_fill_jets(input_batch, device=None, feat_dim=57):
    if feat_dim == 57:
        input_numpy = input_batch.numpy().reshape(-1,19,3).copy()
    elif feat_dim == 512:
        input_numpy = input_batch.numpy().reshape(-1,128,4).copy()

    n_constit = 10
    n_nonzero = np.count_nonzero(np.sum(input_numpy[:, 9:, :], axis=2), axis=1)
    n_split = np.minimum(n_nonzero, n_constit-n_nonzero)
    idx_flip = np.where(n_nonzero != n_split)
    mask_split = np.sum(input_numpy[:, 9:, :], axis=2) != 0
    mask_split [idx_flip] = np.flip(mask_split[idx_flip], axis=1)
    mask_split[idx_flip] = np.invert(mask_split[idx_flip])
    r_split = np.random.uniform(size=mask_split.shape)
    a =       r_split * mask_split*input_numpy[:, 9:, 0]
    b = (1.0-r_split) * mask_split*input_numpy[:, 9:, 0]
    c =                ~mask_split*input_numpy[:, 9:, 0]
    input_numpy[:, 9:, 0] = a + c + np.flip(b, axis=1)
    input_numpy[:, 9:, 1] += np.flip(mask_split*input_numpy[:, 9:, 1], axis=1)
    input_numpy[:, 9:, 2] += np.flip(mask_split*input_numpy[:, 9:, 2], axis=1)

    #Return a torch tensor on the given device and correct shape (-1,57)
    filled_batch = torch.from_numpy(input_numpy.reshape(-1,feat_dim)).to(dtype=torch.float32)
    return filled_batch


def add_objects(input_batch, device=None, feat_dim=57):
    if feat_dim == 57:
        input_numpy = input_batch.numpy().reshape(-1,19,3).copy()
    elif feat_dim == 512:
        input_numpy = input_batch.numpy().reshape(-1,128,4).copy()
    n_els = 4
    n_mus = 4
    n_jets = 10
    n_nonzero_els = np.count_nonzero(np.sum(input_numpy[:, 1:5, :], axis=2), axis=1)
    n_nonzero_mus = np.count_nonzero(np.sum(input_numpy[:, 5:9, :], axis=2), axis=1)
    n_nonzero_jets = np.count_nonzero(np.sum(input_numpy[:, 9:, :], axis=2), axis=1)
    n_new_els = np.random.randint( 0, high=4-n_nonzero_els+1 )
    n_new_mus = np.random.randint( 0, high=4-n_nonzero_mus+1 )
    n_new_jets = np.random.randint( 0, high=10-n_nonzero_jets+1 )
    maxpts = np.max( batchinput_batch[:,:,0], axis=-1 )
    for n in range( batch_filled.shape[0] ):
        # electrons
        el_pts = np.expand_dims( 3.0 + (maxpts[n]-3.0) * np.random.rand( n_new_els[n] ), axis=1 )
        el_phis = np.expand_dims( 2*np.pi * ( np.random.rand(n_new_els[n]) - 0.5 ), axis=1 )
        el_etas = np.expand_dims( 2*3 * ( np.random.rand(n_new_els[n]) - 0.5 ), axis=1 )
        #el_one_hot = np.concatenate( [np.zeros(shape=(n_new_els[n],1)), np.ones(shape=(n_new_els[n],1)), np.zeros(shape=(n_new_els[n],1)), np.zeros(shape=(n_new_els[n],1))], axis=1 ) 
        els = np.concatenate( [el_pts, el_etas, el_phis], axis=1 )
        el_start = 1 + n_nonzero_els[n]
        el_end = 1 + n_nonzero_els[n] + n_new_els[n]
        input_numpy[n,el_start:el_end,:] = els 
        # muons
        mu_pts = np.expand_dims( 3.0 + (maxpts[n]-3.0) * np.random.rand( n_new_mus[n] ), axis=1 )
        mu_phis = np.expand_dims( 2*np.pi * ( np.random.rand(n_new_mus[n]) - 0.5 ), axis=1 )
        mu_etas = np.expand_dims( 2*2.1 * ( np.random.rand(n_new_mus[n]) - 0.5 ), axis=1 )
        #mu_one_hot = np.concatenate( [np.zeros(shape=(n_new_mus[n],1)), np.zeros(shape=(n_new_mus[n],1)), np.ones(shape=(n_new_mus[n],1)), np.zeros(shape=(n_new_mus[n],1))], axis=1 ) 
        mus = np.concatenate( [mu_pts, mu_etas, mu_phis], axis=1 )
        mu_start = 5 + n_nonzero_mus[n]
        mu_end = 5 + n_nonzero_mus[n] + n_new_mus[n]
        input_numpy[n,mu_start:mu_end, :] = mus
        # jets
        jet_pts = np.expand_dims( 15.0 + (maxpts[n]-15.0) * np.random.rand( n_new_jets[n] ), axis=1 )
        jet_phis = np.expand_dims( 2*np.pi * ( np.random.rand(n_new_jets[n]) - 0.5 ), axis=1 )
        jet_etas = np.expand_dims( 2*4 * ( np.random.rand(n_new_jets[n]) - 0.5 ), axis=1 )
        #jet_one_hot = np.concatenate( [np.zeros(shape=(n_new_jets[n],1)), np.zeros(shape=(n_new_jets[n],1)), np.zeros(shape=(n_new_jets[n],1)), np.ones(shape=(n_new_jets[n],1))], axis=1 ) 
        jets = np.concatenate( [jet_pts, jet_etas, jet_phis], axis=1 )
        jet_start = 9 + n_nonzero_jets[n]
        jet_end = 9 + n_nonzero_jets[n] + n_new_jets[n]
        input_batch[n,jet_start:jet_end,:] =  jets
        # MET
        old_met_pt = input_batch[n,0,0]
        old_met_phi = input_batch[n,0,2]
        old_met = np.array( [ old_met_pt * np.sin(old_met_phi), old_met_pt * np.cos(old_met_phi) ] )
        new_obj = np.concatenate( [ els[:,0:3], mus[:,0:3], jets[:,0:3] ], axis=0 )
        new_met = old_met - np.array( [ new_obj[:,0] * np.sin(new_obj[:,2]), new_obj[:,0] * np.cos(new_obj[:,2]) ] ).sum(axis=-1)
        new_met_pt = np.sqrt( new_met[0]**2 + new_met[1]**2 )
        if new_met[1]<0. and new_met[0]>0.:
            new_met_phi =  np.pi - np.arcsin( new_met[0]/new_met_pt )
        elif new_met[1]<0. and new_met[0]<0.:
            new_met_phi = -np.pi - np.arcsin( new_met[0]/new_met_pt )
        else:
            new_met_phi = np.arcsin( new_met[0]/new_met_pt )
        input_batch[n,0,0] = new_met_pt 
        
    #Return a torch tensor on the given device and correct shape (-1,57)
    filled_batch = torch.from_numpy(input_numpy.reshape(-1,feat_dim)).to(dtype=torch.float32)
    return filled_batch

def add_objects_wsmear(input_batch, scaler_pt, etaphi_smear_strength, device=None, feat_dim=57):
    input_batch = add_objects(input_batch, device, feat_dim)
    input_batch = etaphi_smear_events(input_batch,scaler_pt=scaler_pt,strength=etaphi_smear_strength,device=device,feat_dim=feat_dim)
    return input_batch
'''
def add_objects_constptmet( batch, scaler_pt, scale_angle, etaphi_smear_strength ):
    batch_filled = batch.copy()
    batch_filled = collinear_fill_jets_v2( batch_filled, scaler_pt)
    batch_filled = collinear_fill_e_mu_v2( batch_filled, scaler_pt )
    batch_filled = etaphi_smear_events( batch_filled, scaler_pt, scale_angle, strength=etaphi_smear_strength )
    return batch_filled
'''

def shift_met_or_pt(input_batch, device=None, feat_dim=57 ):
    if feat_dim == 57:
        input_numpy = input_batch.numpy().reshape(-1,19,3).copy()
    elif feat_dim == 512:
        input_numpy = input_batch.numpy().reshape(-1,128,4).copy()
    
    rands = np.random.randint( low=0, high=3, size=input_numpy.shape[0] )
    shifts =  1.0 + np.random.rand( input_numpy.shape[0] ) * 4.0
    shifts_met =  0.5 + np.random.rand( input_numpy.shape[0] ) * 4.5
    input_numpy[np.where(rands==0),0, 0] *= shifts_met[np.where(rands==0)]
    input_numpy[np.where(rands==1),1:,0] *= np.expand_dims( shifts[np.where(rands==1)], axis=-1 )
    input_numpy[np.where(rands==2),:, 0 ] *= np.expand_dims( shifts[np.where(rands==2)], axis=-1 )
    #Return a torch tensor on the given device and correct shape (-1,57)
    shifted_batch = torch.from_numpy(input_numpy.reshape(-1,feat_dim)).to(dtype=torch.float32)
    return shifted_batch


def neg_augs(input_batch, scaler_pt, scale_angle, etaphi_smear_strength, addobj=True, addobj_wcpm=True, shpt=True, shmet=True, shporm=False, device=None, feat_dim=57):
    if feat_dim == 57:
        input_numpy = input_batch.numpy().reshape(-1,19,3).copy()
    elif feat_dim == 512:
        input_numpy = input_batch.numpy().reshape(-1,128,4).copy()
        
    n_augs = 0
    aug_list = []
    if addobj: n_augs+=1; aug_list.append("ao")
    if addobj_wcpm: n_augs+=1; aug_list.append("aowcpm")
    if shporm: n_augs+=1; aug_list.append("spm")
    rands = np.random.randint( low=0, high=n_augs, size=batch_aug.shape[0] )
    rand_opts = range( n_augs )
    for j in range( n_augs ):
        aug = aug_list[j]
        n = rand_opts[j]
        if aug=="ao":
            input_batch[ np.where(rands==n) ] = add_objects_wsmear( input_batch[ np.where(rands==n) ], scaler_pt, scale_angle, etaphi_smear_strength)
        if aug=="aowcpm":
            input_batch[ np.where(rands==n) ] = add_objects_constptmet( input_batch[ np.where(rands==n) ], scaler_pt, scale_angle, etaphi_smear_strength )
        if aug=="spm":
            input_batch[ np.where(rands==n) ] = shift_met_or_pt( input_batch[ np.where(rands==n) ] )
    #Return a torch tensor on the given device and correct shape (-1,57)
    aug_batch = torch.from_numpy(input_numpy.reshape(-1,feat_dim)).to(dtype=torch.float32)
    return aug_batch

#################################################
#################################################

def permutation(input_batch, device=None, rand_number=0, same_particle=False, feat_dim=57):
    '''
    Applies the augmentation "permutation" to events in a batch (torch tensor) and outputs a permutated torch tensor
    Permute constituents in the DELPHES dataset w/ structure: MET, 4x electron, 4x muon, 10x jet.
    Each constituent has 3x features: transverse mom. pT, pseudorapidity eta, azimuthal angle phi.
    Args:
        input_batch: (batch_size, 57) flattened input
        device: cuda or cpu depending on input
        same_particle: (bool) if true only shuffles particles of the same type, otherwise shuffles all constituents
    Returns:
        permutated_batch: (batch_size, 57) permutated output
    '''
    if feat_dim == 57:
        input_numpy = input_batch.numpy().reshape(-1,19,3).copy()
    elif feat_dim == 512:
        input_numpy = input_batch.numpy().reshape(-1,128,4).copy()
    #Permute the electrons, muons and jets
    if same_particle: #Does not work for JetClass
        [np.random.shuffle(x[1:5]) for x in input_numpy] #electrons
        [np.random.shuffle(x[5:9]) for x in input_numpy] #muons
        [np.random.shuffle(x[9:19]) for x in input_numpy] #jets
    else:
        [np.random.shuffle(x[:]) for x in input_numpy] #all constituents
    #Return a torch tensor on the given device and correct shape (-1,57)
    permutated_batch = torch.from_numpy(input_numpy.reshape(-1,feat_dim)).to(dtype=torch.float32)

    return permutated_batch

def rot_around_beamline(input_batch, device=None, rand_number=0, feat_dim=57):
    '''
    Applies the augmentation "rotation around beamline" to events in a batch (torch tensor) and outputs a permutated torch tensor
    Rotates each event in the batch w/ structure: MET, 4x electron, 4x muon, 10x jet around a random angle.
    Angle around beamline is given by phi in features: transverse mom. pT, pseudorapidity eta, azimuthal angle phi.
    Args:
        input_batch: (batch_size, 57) flattened input
        device: cuda or cpu depending on input
    Returns:
        permutated_batch: (batch_size, 57) permutated output
    '''
    if feat_dim == 57:
        input_numpy = input_batch.numpy().reshape(-1,19,3).copy()
    elif feat_dim == 512:
        input_numpy = input_batch.numpy().reshape(-1,128,4).copy()
    #Rotate the whole thing around the beamline (phi) at angle
    #np.random.seed(rand_number)
    #Iterate through the batch
    for x in input_numpy:
        #Angles are stored from [-pi,pi]
        angle = np.random.uniform(0,2)*np.pi
        x[:,2] = (((x[:,2]+np.pi) + angle)%(2*np.pi))-np.pi

    #Return a torch tensor on the given device and correct shape (-1,57)
    rotated_batch = torch.from_numpy(input_numpy.reshape(-1,feat_dim)).to(dtype=torch.float32)

    return rotated_batch

def gaussian_resampling_pT(input_batch, device=None, rand_number=0, std_scale=2.0, feat_dim=57):
    '''
    Applies the augmentation "gaussian resampling of pT" to events in a batch (torch tensor) and outputs a torch tensor with pT values rescaled within std.
    Resample pT of constituents with mu=pT, std=pT*std_scale in the DELPHES dataset w/ structure: MET, 4x electron, 4x muon, 10x jet.
    Each constituent has 3x features: transverse mom. pT, pseudorapidity eta, azimuthal angle phi.
    Args:
        input_batch: (batch_size, 57) flattened input
        device: cuda or cpu depending on input
        std_scale: scale multiplier for standard deviation std = pT * std_scale (default: 0.1)
    Returns:
        resampled_batch: (batch_size, 57) permutated output
    '''
    if feat_dim == 57:
        input_numpy = input_batch.numpy().reshape(-1,19,3).copy()
    elif feat_dim == 512:
        input_numpy = input_batch.numpy().reshape(-1,128,4).copy()
    #Guassian resample the pT's of each constituent with mu=pT and std = pT * std_scale
    #np.random.seed(rand_number) #not sure if I should seed
    for x in input_numpy:
        x[:,0] = np.random.normal(loc=x[:,0], scale=np.absolute(x[:,0])*std_scale)

    #Return a torch tensor on the given device and correct shape (-1,57)
    resampled_batch = torch.from_numpy(input_numpy.reshape(-1,feat_dim)).to(dtype=torch.float32)

    return resampled_batch

def gaussian_resampling(input_batch, device=None, rand_number=0, std_scale=1.5, feat_dim=57):
    '''
    Applies the augmentation "gaussian resampling of pT" to events in a batch (torch tensor) and outputs a torch tensor with pT values rescaled within std.
    Resample pT of constituents with mu=pT, std=pT*std_scale in the DELPHES dataset w/ structure: MET, 4x electron, 4x muon, 10x jet.
    Each constituent has 3x features: transverse mom. pT, pseudorapidity eta, azimuthal angle phi.
    Args:
        input_batch: (batch_size, 57) flattened input
        device: cuda or cpu depending on input
        std_scale: scale multiplier for standard deviation std = pT * std_scale (default: 0.1)
    Returns:
        resampled_batch: (batch_size, 57) permutated output
    '''
    if feat_dim == 57:
        input_numpy = input_batch.numpy().reshape(-1,19,3).copy()
    elif feat_dim == 512:
        input_numpy = input_batch.numpy().reshape(-1,128,4).copy()
    #Guassian resample the pT's of each constituent with mu=pT and std = pT * std_scale
    #np.random.seed(rand_number) #not sure if I should seed
    for x in input_numpy:
        x[:,:] = np.random.normal(loc=x[:,:], scale=np.absolute(x[:,:])*std_scale)

    #Return a torch tensor on the given device and correct shape (-1,57)
    resampled_batch = torch.from_numpy(input_numpy.reshape(-1,feat_dim)).to(dtype=torch.float32)

    return resampled_batch

def naive_masking(input_batch, device=None, rand_number=0, p=0.5, mask_full_particle=False, feat_dim=57):
    '''
    Applies the augmentation "naive_masking" to events in a batch (torch tensor) and outputs a torch tensor values randomly masked with probability p.
    Args:
        input_batch: (batch_size, 57) flattened input
        device: cuda or cpu depending on input
        p: probability of bernoulli trial (default: p=0.2)
        mask_full_particle: Whether to randomly mask full particles or randomly mask constistuents individually (default: False)
    Returns:
        resampled_batch: (batch_size, 57) permutated output
    '''
    #np.random.seed(rand_number)
    #input_numpy = input_batch.cpu().detach().numpy().reshape(-1)
    input_numpy = input_batch.numpy().reshape(-1).copy()
    #Randomly (with prob. p) set parts of the input to 0.0 (mask/crop)
    if mask_full_particle:
        if feat_dim == 57:
            mask = np.random.choice([True, False], size=int(input_numpy.shape[0]/3), replace=True, p=[p, 1-p]).reshape(-1,19)
            input_numpy.reshape(-1,19,3)[mask] = 0.0
        elif feat_dim == 512:
            mask = np.random.choice([True, False], size=int(input_numpy.shape[0]/4), replace=True, p=[p, 1-p]).reshape(-1,128)
            input_numpy.reshape(-1,128,4)[mask] = 0.0
    else:
        mask = np.random.choice([True, False], size=input_numpy.shape[0], replace=True, p=[p, 1-p])
        #print(f'Input numpy: {input_numpy}')
        #print(f'Mask: {mask}')
        input_numpy[mask] = 0.0
    resampled_batch = torch.from_numpy(input_numpy.reshape(-1,feat_dim)).to(dtype=torch.float32)
    #print(f'Resampled batch: {resampled_batch}')
    return resampled_batch

def hardjet_masking(input_batch, device=None):
    '''
    Crops/masks the area of deltaR < 3.0 around the first hard jet of the event and outputs a torch tensor where the rest has been zero padded.
    The input has structure: MET, 4x electron, 4x muon, 10x jet with each constituent described by 3 features: transverse mom. pT, pseudorapidity eta, azimuthal angle phi.
        input_batch: (batch_size, 57) flattened input
        device: cuda or cpu depending on input
    Returns:
        resampled_batch: (batch_size, 57) masked output
    '''
    input_numpy = input_batch.cpu().detach().numpy().reshape(-1,19,3)
    #Calculate deltaR = (delta_phi**2 + delta_eta**2)**(1/2) from first jet
    deltaR = np.sqrt((input_numpy[:,:,1] - input_numpy[:,9,1])**2 + (input_numpy[:,:,2]-input_numpy[:,9,2])**2 + 1e-4)
    mask = np.where(deltaR <= 3.0, True, False)
    input_numpy[~mask] = 0.0
    
    resampled_batch = torch.from_numpy(input_numpy.reshape(-1,57)).to(dtype=torch.float32, device=device)
    return resampled_batch

def hardlepton_masking(input_batch, device=None):
    '''
    Crops/masks the area of deltaR < 3.0 around the first hard lepton of the event and outputs a torch tensor where the rest has been zero padded.
    The input has structure: MET, 4x electron, 4x muon, 10x jet with each constituent described by 3 features: transverse mom. pT, pseudorapidity eta, azimuthal angle phi.
        input_batch: (batch_size, 57) flattened input
        device: cuda or cpu depending on input
    Returns:
        resampled_batch: (batch_size, 57) masked output
    '''
    input_numpy = input_batch.cpu().detach().numpy().reshape(-1,19,3).copy()
    #As there is a guaranteed lepton in each event, we have to find the highest pT one first (either first electron or muon)
    lepton_pT = np.concatenate((input_numpy[:, 0:4, 0], input_numpy[:, 4:8, 0]))  # Concatenate electron and muon pT values
    highest_pT_index = np.argmax(lepton_pT, axis=1)

    #Calculate deltaR = (delta_phi**2 + delta_eta**2)**(1/2) from first jet
    deltaR = np.sqrt((input_numpy[:,:,1] - input_numpy[:,highest_pT_index,1])**2 + (input_numpy[:,:,2]-input_numpy[:,highest_pT_index,2])**2 + 1e-4)
    mask = np.where(deltaR <= 3.0, True, False)
    input_numpy[~mask] = 0.0
    
    resampled_batch = torch.from_numpy(input_numpy.reshape(-1,57)).to(dtype=torch.float32, device=device)
    return resampled_batch

def detector_crop(input_batch,feat_dim, device=None, rand_number=0, crop_size = 2.5):
    '''
    Applies the augmentation "detector_crop" to events in a batch (torch tensor) and outputs a torch tensor.
    It randomly crops an area of deltaR <= crop_size of the detector and masks it by zero padding the rest.
    Args:
        input_batch: (batch_size, 57) flattened input
        device: cuda or cpu depending on input
        crop_size: region of the detector crop (default: deltaR <= 3.0)
    Returns:
        resampled_batch: (batch_size, 57) masked output
    '''
    if feat_dim == 57:
        input_numpy = input_batch.numpy().reshape(-1,19,3).copy()
    elif feat_dim == 512:
        input_numpy = input_batch.numpy().reshape(-1,128,4).copy()
    #Randomly crop a region of the detector with size deltaR
    #First find a random point in the angular space of the detector by uniform sampling on the unit sphere using the inverse transform method
    theta = np.arccos(1-2*np.random.rand(input_numpy.shape[0]))
    phi = 2*np.pi*np.random.rand(input_numpy.shape[0])
    eta = np.log(np.tan(theta/2+1e-4))
    #Then crop within size deltaR by setting the outside to 0.0
    deltaR = np.sqrt((input_numpy[:,:,1] - eta[:,None])**2 + (input_numpy[:,:,2]-phi[:,None])**2 + 1e-4)
    mask = deltaR <= crop_size
    input_numpy[~mask] = 0.0
    resampled_batch = torch.from_numpy(input_numpy.reshape(-1,feat_dim)).to(dtype=torch.float32)
    return resampled_batch

def corruption(input_batch, feat_low, feat_high, feat_mean, feat_std, mode=None, device=None, rand_number=0, corruption_rate=0.6):
    '''
    Applies the augmentation "corruption" to events in a batch (torch tensor) and outputs a torch tensor values randomly masked with probability p.
    From: https://arxiv.org/pdf/2106.15147 SCARF corruption method
    Args:
        input_batch: (batch_size, 57) flattened input
        mode: choice('uniform', 'gaussian') which determines which distribution to sample from
        device: cuda or cpu depending on input
        corruption_rate: probability of corruption (default: 0.6)
        feat_low, feat_high: (57,) Expects input of min, max of all the features in the training dataset
    Returns:
        resampled_batch: (batch_size, 57) permutated output
    '''
    np.random.seed(rand_number)
    input_numpy = input_batch.cpu().detach().numpy().reshape(-1)
    #Make mask with given feature corruption rate, treat each of the 57 features seperately (no full_particle_corruption)
    mask = np.random.choice([True, False], size=input_numpy.shape[0], replace=True, p=[corruption_rate, 1-corruption_rate])
    mask = mask.reshape(-1,57)

    if mode=='uniform' or mode==None:
        #Sample the batch by drawing from a uniform distribution given by the min, max values of the training dataset
        marginals = np.random.uniform(feat_low, feat_high, size=mask.shape)
    elif mode=='gaussian':
        marginals = np.random.normal(feat_mean, feat_std, size=mask.shape)
    
    resampled_batch = np.where(mask, marginals, input_numpy.reshape(-1,57))

    resampled_batch = torch.from_numpy(input_numpy.reshape(-1,57)).to(dtype=torch.float32, device=device)
    return resampled_batch

class Transform():
    def __init__(self, augmentations, feat_dim):
        self.augmentations = []
        self.feat_dim = feat_dim
        print(f"Using the following augments:")
        for augment in augmentations:
            print(f"{augment}")
            if augment == "naive_masking":
                self.augmentations.append(naive_masking)
            elif augment == "gaussian_resampling_pT":
                self.augmentations.append(gaussian_resampling_pT)
            elif augment == "rot_around_beamline":
                self.augmentations.append(rot_around_beamline)
            elif augment == 'permutation':
                self.augmentations.append(permutation)
            elif augment == 'gaussian_resampling':
                self.augmentations.append(gaussian_resampling)
            elif augment == 'detector_crop':
                self.augmentations.append(detector_crop)
            else:
                assert False

    def __call__(self, input):
        for augmentation in self.augmentations:
            input = augmentation(input, feat_dim=self.feat_dim)
        return input

    
