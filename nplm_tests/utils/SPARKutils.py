import torch
import torch.nn as nn
from torch.autograd import Variable
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
        return fin

def Annealing(t, ini, fin, t_fin):
    if t<t_fin: return ini*((fin/ini)**((t+1)/t_fin))
    else: return fin


class TAU(nn.Module):
    """
    model performing GOF with uncertainties using the profiled Likelihood-Ratio-test
    domain shifts have polinomial behaviors: log(r)= a0 * v + a1 * v^2 + ...  
    """
    def __init__(self, input_shape, ADmodule, NUcoeffs=[], NU_init=0, NU_ref=0, NU_err=):
        

class SoftSparKer(nn.Module):
    '''                              
    return exp(-0.5(x-mu_i)**2/scale**2) * exp( -0.5(x-mu_i)**2/scale**2 )/sum_j[exp( -0.5(x-mu_j)**2/scale**2 )] 
    '''
    def __init__(self, input_shape, centroids, widths, coeffs, resolution_const, resolution_scale, coeffs_clip,
                 train_centroids=False, train_widths=False, train_coeffs=True,
                 name=None, **kwargs):
        super(SoftSparKer, self).__init__()
        self.epsilon=1e-10
        if positive_coeffs:
            self.cmin=0
            self.cmax=coeffs_clip
        else:
            self.cmin=-coeffs_clip
            self.cmax=coeffs_clip
        self.coeffs = Variable(coeffs.reshape((-1, 1)).type(torch.double),
                               requires_grad=train_coeffs) # [M, 1]                                                                                  
        self.kernel_layer = KernelLayer(input_shape=input_shape, centroids=centroids, width=width,
                                        train_centroids=train_centroids, train_widths=train_widths,
                                        name='kernel_layer')

    def call(self, x):
        K_x, _ = self.kernel_layer.call(x) # [n, M]    
        Z = torch.sum(K_x, dim=1, keepdim=True) +self.epsilon # [n, 1]  
        out = torch.tensordot(torch.mul(K_x,K_x), self.coeffs, dims=([1], [0])) # [n, 1] 
        return torch.divide(out, Z) # [n, 1]  

    def get_softmax(self, x):
        K_x, _ = self.kernel_layer.call(x) # [n, M]    
        Z = torch.sum(K_x, dim=1, keepdim=True) +self.epsilon # [n]  
        return K_x/Z # [n, M] 
        
    def get_kernel(self, x):
        K_x, _ = self.kernel_layer.call(x) # [n, M]  
        return K_x
    
    def clip_coeffs(self):
        self.coeffs.data = self.coeffs.data.clamp(self.cmin,self.cmax)
        return

    def get_centroids_entropy(self):
        return self.kernel_layer.get_centroids_entropy()

    def get_coeffs(self):
        return self.coeffs

    def get_centroids(self):
        return self.kernel_layer.get_centroids()

    def get_widths(self):
        return self.kernel_layer.get_widths()

    def clip_centroids(self):
        self.kernel_layer.clip_centroids()

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
                 cmin=None, cmax=None,
                 train_centroids=True, train_widths=False,
                 name=None, **kwargs):
        super(KernelLayer, self).__init__()
        self.cmin=cmin
        self.cmax=cmax
        self.M = centroids.shape[0]
        self.d = centroids.shape[1]
        self.width = Variable(width.type(torch.float32), requires_grad=train_widths) #1,
        self.centroids = Variable(centroids.type(torch.float32), requires_grad=train_centroids) #M, d

    def call(self, x):
        out, arg = self.Kernel(x)
        return out, arg

    def get_widths(self):
        return self.width.data*torch.ones((self.M, self.d))

    def set_width(self, width):
        self.width.data = width
        self.compute_cov_diag()
        return

    def get_centroids(self):
        return self.centroids #[M, d]

    def clip_centroids(self):
        if (not self.cmin==None) and (not self.cmax==None):
            self.centroids.data = self.centroids.data.clamp(self.cmin,self.cmax)
        return

    def get_centroids_entropy(self):
        """  
        sum_j(sum_i(K_i(mu_j))*log(sum_i(K_i(mu_j))))  
        return: scalar 
        """
        K_mu, _ = self.call(self.centroids) #[M, M]    
        K_mu = torch.mean(K_mu, axis=1) # [M,]  
        return torch.sum(torch.multiply(K_mu, torch.log(K_mu)))

    def Kernel(self,x):
        """  
        # x.shape = [N, d] 
        # widths.shape = [M, d] 
        # centroids.shape = [M, d] 
        Returns exp(-0.5*(x-mu)^2/scale^2)
        # return.shape = [N,M]   
        """
        dist_sq  = torch.subtract(x[:, None, :], self.centroids[None, :, :])**2 # [N, M, d]  
        arg = -0.5*torch.sum(dist_sq/self.cov_diag,axis=2) # [N, M]                                                      
        kernel = torch.exp(arg)
        return kernel, arg # [N, M] 
