import torch 

# regularizers
def L2Regularizer(pred):
    return torch.sum(torch.multiply(pred,pred))

def L1Regularizer(pred):
    return torch.sum(torch.abs(pred))

def CentroidsEntropyRegularizer(entropy):
    return entropy
