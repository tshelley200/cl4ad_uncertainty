import torch, math
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

def pairwise_dist(X, P):
    X2 = (X ** 2).sum(dim=1, keepdim=True) # (n x 1)                                                            
    P2 = (P ** 2).sum(dim=1, keepdim=True) # (n' x 1)                                                            
    XP = X @ P.T # (n x n')                                                                                      
    return X2 + P2.T - 2 * XP # (n x n')  

def Annealing_Linear(t, ini, fin, t_fin):
    if t < t_fin:
        return ini + (fin - ini) * (t + 1) / t_fin
    else:
        return torch.tensor([fin])

def Annealing(t, ini, fin, t_fin):
    if t<t_fin: return ini*((fin/ini)**((t+1)/t_fin))
    else: return torch.tensor([fin])


class SparKer(nn.Module):
    '''                              
    return exp(-0.5(x-mu_i)**2/scale**2) * exp( -0.5(x-mu_i)**2/scale**2 )/sum_j[exp( -0.5(x-mu_j)**2/scale**2 )] 
    '''
    def __init__(self, input_shape, centroids, width, coeffs, coeffs_clip,
                 train_centroids=False, train_width=False, train_coeffs=True,
                 positive_coeffs=False,
                 name=None, **kwargs):
        super(SparKer, self).__init__()
        self.epsilon=1e-10
        self.coeffs_clip = False
        if coeffs_clip!=None:
            self.coeffs_clip = True
        if positive_coeffs:
            self.cmin=0
            self.cmax=coeffs_clip
        else:
            self.cmin=-coeffs_clip
            self.cmax=coeffs_clip
        self.train_coeffs=train_coeffs
        self.coeffs = nn.Parameter(coeffs.reshape((-1, 1)).type(torch.float32),
                               requires_grad=train_coeffs) # [M, 1]                                                                                  
        self.kernel_layer = KernelLayer(input_shape=input_shape, 
                                        centroids=centroids, 
                                        width=width,
                                        train_centroids=train_centroids, 
                                        train_width=train_width,
                                        name='kernel_layer')

    def forward(self, x):
        K_x, _ = self.kernel_layer(x) # [n, M]    
        Z = torch.sum(K_x, dim=1, keepdim=True) +self.epsilon # [n, 1]  
        out = torch.tensordot(torch.mul(K_x,K_x), self.coeffs, dims=([1], [0])) # [n, 1] 
        out = torch.divide(out, Z) # [n, 1]  
        return out

    def get_softmax(self, x):
        K_x, _ = self.kernel_layer(x) # [n, M]    
        Z = torch.sum(K_x, dim=1, keepdim=True) +self.epsilon # [n]  
        return K_x/Z # [n, M] 
        
    def get_kernel(self, x):
        K_x, _ = self.kernel_layer(x) # [n, M]  
        return K_x
    
    def clip_coeffs(self):
        if self.coeffs_clip:
            self.coeffs.data = self.coeffs.data.clamp(self.cmin,self.cmax)
        return

    def get_coeffs(self):
        return self.coeffs

    def get_centroids(self):
        return self.kernel_layer.get_centroids()

    def get_width(self):
        return self.kernel_layer.get_width()

    def set_width(self, width):
        self.kernel_layer.set_width(width)
        return

class KernelLayer(nn.Module):
    '''
    layer of M gaussians: [K_1, ..., K_m]
    input data: x [shape: (N,d)]
    output: K_1(x), ..., K_m(x) [shape: (N, m)]
    '''
    def __init__(self, centroids, width,
                 train_centroids=False, 
                 train_width=False,
                 name=None, **kwargs):
        super(KernelLayer, self).__init__()

        self.M = centroids.shape[0]
        self.d = centroids.shape[1]
        self.width = nn.Parameter(width.type(torch.float32), requires_grad=train_width)
        self.centroids = nn.Parameter(centroids.type(torch.float32), requires_grad=train_centroids)

    def forward(self, x):
        out, arg = self.Kernel(x)
        return out, arg

    def get_width(self):
        return self.width

    def set_width(self, widths):
        self.width.data = widths
        return

    def get_centroids(self):
        return self.centroids #[M, d]

    def norm_const(self):
        """ 
        # widths.shape = [M, d]  
        Returns the normalization constant for a gaussian     
        # return.shape = [M,] 
        """
        return (1.0 / ((2*math.pi)**(self.d/2) * (self.width**self.d)))

    def Kernel(self,x):
        """  
        # x.shape = [N, d] 
        # widths.shape = [M, d] 
        # centroids.shape = [M, d] 
        Returns exp(-0.5*(x-mu)^2/scale^2)
        # return.shape = [N,M]   
        """
        diff = x.unsqueeze(1) - self.centroids.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=2)
        arg = -0.5 * dist_sq / (self.width ** 2)
        kernel = self.norm_const() * torch.exp(arg)  # [N, M]
        return kernel, arg # [N, M] 