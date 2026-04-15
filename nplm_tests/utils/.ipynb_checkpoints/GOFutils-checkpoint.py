import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter


class novelty_finder(nn.Module):
    """
    Fully-connected NN with hard weight clipping.
    """
    def __init__(
        self,
        input_dim,
        architecture=(1, 4, 1),
        activation="sigmoid",
        weight_clipping=None,
        trainable=True,
        initializer="xavier",
    ):
        super().__init__()
        assert architecture[0] == input_dim
        self.weight_clipping = weight_clipping
        # Activation map
        activations = {
            "sigmoid": nn.Sigmoid(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
        }
        act = activations[activation]
        
        # Hidden network
        layers = []
        for in_dim, out_dim in zip(architecture[:-2], architecture[1:-1]):
            layers += [nn.Linear(in_dim, out_dim), act]
        self.hidden_net = nn.Sequential(*layers)
        self.output_layer = nn.Linear(architecture[-2], architecture[-1])

        self.initialize_weights(initializer)
        self.is_trainable(trainable)
            
    def forward(self, x):
        x = self.hidden_net(x)
        return self.output_layer(x)

    def clip_weights(self):
        """
        Hard projection of parameters onto [-c, c].
        Call after optimizer.step().
        """
        if self.weight_clipping is None:
            return

        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.weight.clamp_(-self.weight_clipping, self.weight_clipping)
                    if m.bias is not None:
                        m.bias.clamp_(-self.weight_clipping, self.weight_clipping)

    def initialize_weights(self, initializer):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if initializer == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif initializer == "normal":
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                elif initializer == "zeros":
                    nn.init.zeros_(m.weight)
                else:
                    raise ValueError(f"Unknown initializer: {initializer}")
                nn.init.zeros_(m.bias)

    def is_trainable(self, bool):
        for p in self.parameters():
                p.requires_grad_(bool)


class norm_syst_finder(nn.Module):
    """                              
    """
    def __init__(self, input_shape, nu, nu_central, nu_ref, nu_sigma, polyn_degrees=1, 
                 trainable=True, device='cpu', name='syst_finder', **kwargs):
        super(norm_syst_finder, self).__init__()
        self.nu = Parameter(nu.reshape((-1, )).type(torch.double), requires_grad=trainable)
        self.nu_central = nu_central.reshape((-1, )).clone().detach().type(torch.double)
        self.nu_ref = nu_ref.reshape((-1, )).clone().detach().type(torch.double)
        self.nu_sigma = nu_sigma.reshape((-1, )).clone().detach().type(torch.double)
        self.to(device)

    def LikEvidence(self):#, x):
        return (self.nu-self.nu_ref).squeeze()# [1,] #*torch.ones_like(x[:, 0:1]) # [N, 1]
        
    def LikAux(self):
        return -0.5*((self.nu-self.nu_central)**2 - (self.nu_ref-self.nu_central)**2)/self.nu_sigma**2 # [1, 1]
    
    def is_trainable(self, bool):
        self.nu.requires_grad = bool

class shape_syst_finder(nn.Module):
    """                              
    """
    def __init__(self, input_shape, nu, nu_central, nu_ref, nu_sigma, nu_std, polyn_degrees=1,
                 trainable=True, device='cpu', name='syst_finder', **kwargs):
        super(shape_syst_finder, self).__init__()
        self.nu = Parameter(nu.reshape((1, 1)).type(torch.double), requires_grad=trainable)
        self.nu_central = nu_central.reshape((1, 1)).clone().detach().type(torch.double)
        self.nu_ref = nu_ref.reshape((1, 1)).clone().detach().type(torch.double)
        self.nu_sigma = nu_sigma.reshape((1, 1)).clone().detach().type(torch.double)
        self.nu_std = nu_std.reshape((1, 1)).clone().detach().type(torch.double)
        self.polyn_degrees = polyn_degrees 
        self.to(device)

    def LikEvidence(self, x):
        """
        x[:, 0]: lin coeffs at x (a_1(x))
        x[:, 1]: quad coeffs at x (a_2(x))
        ...
        """
        assert x.shape[1] >= self.polyn_degrees
        out = [x[:, k:k+1]* (self.nu/self.nu_std)**(k+1)  for k in range(self.polyn_degrees)]
        out = torch.stack(out, dim=0) # [polyn_degrees, N, 1]
        return torch.sum(out, dim=0) # [N, 1]
        
    def LikAux(self,):
        return -0.5*((self.nu-self.nu_central)**2 - (self.nu_ref-self.nu_central)**2)/self.nu_sigma**2 # [1, 1]
    
    def is_trainable(self, bool):
        self.nu.requires_grad = bool
        
class NP_GOF_sys(nn.Module):
    """  
    x: list of tensors
    x[0]: input data
    x[>0]: coeffs of the shape_syst_finder
    """
    def __init__(self, input_shape, novelty_finder=None, shape_syst_finder_list=None, norm_syst_finder=None,
                 train_novelty_finder=True, train_norm_syst_finder=True, train_shape_syst_finder=True,
                 shape_syst_finder_id_list=None,
                 device='cpu', name='NP_GOF_sys', **kwargs):
        super(NP_GOF_sys, self).__init__()
        self.device=device
        if shape_syst_finder_list==None: 
            self.shape_syst_finder_list=None
            self.shape_syst_number = 0
            self.shape_syst_finder_id_list = None
        else:
            self.shape_syst_finder_list = nn.ModuleList(shape_syst_finder_list)
            self.shape_syst_number = len(shape_syst_finder_list)
            self.shape_syst_finder_id_list = shape_syst_finder_id_list
            for i in range(self.shape_syst_number):
                self.shape_syst_finder_list[i].is_trainable(train_shape_syst_finder)
            
            
        if norm_syst_finder==None: 
            self.norm_syst_finder=None
        else: 
            self.norm_syst_finder = norm_syst_finder
            self.norm_syst_finder.is_trainable(train_norm_syst_finder)

        if novelty_finder==None:
            self.novelty_finder=novelty_finder
        else:
            self.novelty_finder = novelty_finder
            #self.novelty_finder.is_trainable(train_novelty_finder)

        self.to(device)

    def forward(self, x):
        '''
        x: list of tensors
        x[0]: input data
        x[>0]: coeffs of the shape_syst_finder
        '''
        Levid = 0
        
        
        if self.novelty_finder!=None:
            Levid += self.novelty_finder(x[0]) # [N, 1]
            
        if self.shape_syst_number:
            shape_out = [self.shape_syst_finder_list[i].LikEvidence(x[i+1]) 
                         for i in range(self.shape_syst_number)]
            shape_out = torch.stack(shape_out, dim=0) # [shape_syst_number, N, 1]
            Levid += torch.sum(shape_out, dim=0) # [N, 1]
            
            
        if self.norm_syst_finder!=None:
            norm_out = self.norm_syst_finder.LikEvidence() # [1,]
            Levid += norm_out # [N, 1]
            
        return Levid # [N, 1]

    def AuxiliaryLikelihood(self,):
        Laux = torch.zeros(1, 1, device=self.device)
        if self.shape_syst_number:
            # update Laux
            shape_aux = [self.shape_syst_finder_list[i].LikAux() 
                         for i in range(self.shape_syst_number)]
            shape_aux = torch.stack(shape_aux, dim=0) # [shape_syst_number, 1]
            Laux += torch.sum(shape_aux) # [1, 1]
        if self.norm_syst_finder!=None:
            # update Laux
            Laux += self.norm_syst_finder.LikAux() # [1, 1]
        return  Laux #[1, 1]
        
    def loss_auxiliary(self):
        Laux = self.AuxiliaryLikelihood()
        return -torch.sum(Laux)
    
    def loss_evidence(self, pred, true):
        f   = pred[:, 0]
        y   = true[:, 0]
        w   = true[:, 1]
        return torch.sum((1-y)*w*(torch.exp(f)-1) - y*w*(f)) #- torch.sum(Laux) # [1, 1]


    def get_nu_shape_list(self):
       return [self.shape_syst_finder_list[i].nu for i in range(self.shape_syst_number)]

    def get_nu_norm(self):
        return self.norm_syst_finder.nu

    def set_initialization_nu_shapes(self, nu_shapes_init):
        for i in range(self.shape_syst_number):
            self.shape_syst_finder_list[i].nu.data = nu_shapes_init[i].reshape((1, 1)).clone().detach().type(torch.double)
        return

    def set_initialization_nu_norm(self, nu_norm_init):
        self.norm_syst_finder.nu.data = nu_norm_init.reshape((1, )).clone().detach().type(torch.double)

            
            
        