import torch
import torch.nn as nn
import torch.nn.functional as F


# Simplified implemented from https://github.com/violatingcp/codec/blob/main/losses.py
class SimCLRLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.logical_not(mask).float()

        logits = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        exp_logits += torch.exp(logits) * mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - self.temperature * mean_log_prob_pos

        loss = loss.view(1, batch_size).float().mean()

        return loss


class VICRegLoss(torch.nn.Module):
    def __init__(self, inv_weight, var_weight, cov_weight):
        super().__init__()
        self.inv_weight = inv_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight

    def forward(self, x, y):
        repr_loss = F.mse_loss(x, y)
        
        x_mu = x.mean(dim=0)
        x_std = torch.sqrt(x.var(dim=0)+1e-4)
        y_mu = y.mean(dim=0)
        y_std = torch.sqrt(y.var(dim=0)+1e-4)

        x = (x - x_mu)
        y = (y - y_mu)

        N = x.size(0)
        D = x.size(-1)

        std_loss = torch.mean(F.relu(1 - x_std)) / 2
        std_loss += torch.mean(F.relu(1 - y_std)) / 2

        cov_x = (x.T.contiguous() @ x) / (N - 1)
        cov_y = (y.T.contiguous() @ y) / (N - 1)

        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(D)
        cov_loss += self.off_diagonal(cov_y).pow_(2).sum().div(D)

        weighted_inv = repr_loss * self.inv_weight # * self.hparams.invariance_loss_weight
        weighted_var = std_loss * self.var_weight # self.hparams.variance_loss_weight
        weighted_cov = cov_loss * self.cov_weight #self.hparams.covariance_loss_weight

        loss = weighted_inv + weighted_var + weighted_cov

        return loss

    def off_diagonal(self, x):
        #num_batch, n, m = x.shape
        n, m = x.shape
        assert n == m
        # All off diagonal elements from complete batch flattened
        #return x.flatten(start_dim=1)[...,:-1].view(num_batch, n - 1, n + 1)[...,1:].flatten()
        return x.flatten()[...,:-1].view(n - 1, n + 1)[...,1:].flatten()

    
class SimCLRloss_nolabels_fast(torch.nn.Module):
    """
    Implement (hopefully) faster version of the unsupervised loss in SimCLR.
    see https://arxiv.org/pdf/2002.05709.pdf
    Implementation from https://github.com/HobbitLong/SupContrast/blob/master/losses.py.
    """
    def __init__(self, temperature=0.07,contrast_mode='all',
                 base_temperature=0.07):
        super(SimCLRloss_nolabels_fast, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        #Normalize features
        features = F.normalize(features, p=2, dim=2)
        
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



class ExpoLoss_Linear(nn.Module):
    """
    Gaia's expo loss function:
        L = mean( y * c^2 + (1 - y) * (1 - c)^2 ),
    where c = sigmoid(-a1 * nu)
    """

    def __init__(self):
        super().__init__()  # initializes base nn.Module

    def forward(self, true: torch.Tensor, fx: torch.Tensor) -> torch.Tensor:
        
        y  = true[:, 0]
        c = torch.sigmoid(-fx)

        loss = torch.mean((y * c**2) + ((1 - y) * (1 - c)**2))
        
        return loss

class NPLMLoss_Linear(nn.Module):
    """Gaia's NPLM loss function:
        L = sum ((1 - y) * w * (torch.exp(f) - 1) - y * w * f))"""
    
    def __init__(self):
        super().__init__()

    def forward(self, true: torch.Tensor, fx: torch.Tensor) -> torch.Tensor:
        
        y  = true[:, 0]
        loss = torch.sum((1 - y) * (torch.exp(fx) - 1) - y * fx)
        
        return loss

 