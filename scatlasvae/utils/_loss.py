import torch
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence as kldiv
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import rbf_kernel
from torch.autograd import Variable

from ._logger import *
from ._distributions import *
from ._compat import *
from ..externals._trvae_mmd_loss import mmd_loss_calc

# Reference: https://github.com/tim-learn/ATDOC/blob/main/loss.py
class MMD(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X - f_of_Y
        loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            torch.cuda.empty_cache()
            return loss

class LossFunction:
    @staticmethod
    def mse(recon_x:   torch.tensor, 
            x:         torch.tensor, 
            reduction: str = "sum"):
        """
        The reconstruction error in the form of mse
        """
        return F.mse_loss(recon_x, x, reduction = reduction)

    @staticmethod
    def bce(recon_x:   torch.tensor, 
            x:         torch.tensor, 
            reduction: str = "sum"):
        """
        The error in the form of bce
        """
        return F.binary_cross_entropy(recon_x, x, reduction=reduction)

    @staticmethod
    def vae_mse(recon_x: torch.tensor, 
                x:       torch.tensor, 
                mu:      torch.tensor, 
                var:     torch.tensor, 
        ):
        """
        The KL-divergence of the latent probability z
        KL(q || p) = -∫ q(z) log [ p(z) / q(z) ] 
                = -E[log p(z) - log q(z)] 
        """
        MSE = F.mse_loss(recon_x, x, reduction = "sum")
        KLD = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim = 1), dim = 0)

        return MSE + KLD
        
    @staticmethod
    def mmd_loss(x: torch.tensor, y: torch.tensor, dim = -1):
        if dim == -1:
            return MMD()(x,y)
        elif dim == 1:
            result = torch.tensor(0.)
            for i in range(x.shape[1]):
                result += MMD()(x[:,i],y[:,i])
            return result
        else:
            raise ValueError("dim must be -1 or 1")
    
    @staticmethod 
    def mmd_loss_trvae(x: torch.tensor, y: torch.tensor):
        return mmd_loss_calc(x,y)
    
    @staticmethod
    def zinb_reconstruction_loss(X:            torch.tensor, 
                                 total_counts: torch.tensor = None,
                                 logits:       torch.tensor = None,
                                 mu:           torch.tensor = None,
                                 theta:        torch.tensor = None,
                                 gate_logits:  torch.tensor = None,
                                 reduction:    str = "sum"):
        if ((total_counts == None) and (logits == None)):
            if ((mu == None) and (theta == None )):
                raise ValueError
            logits = (mu / theta).log()
            total_counts = theta + 1e-6
            znb = ZeroInflatedNegativeBinomial(
                total_count=total_counts, 
                logits=logits,
                gate_logits=gate_logits
            )   
        else: 
            znb = ZeroInflatedNegativeBinomial(
                total_count=total_counts, 
                logits=logits, 
                gate_logits=gate_logits
            )
        if reduction == "sum":
            reconst_loss = -znb.log_prob(X).sum(dim = 1)
        elif reduction == "mean":
            reconst_loss = -znb.log_prob(X).mean(dim = 1)
        elif reduction == "none":
            reconst_loss = -znb.log_prob(X)
        return reconst_loss

    @staticmethod
    def nb_reconstruction_loss(X:            torch.tensor, 
                               total_counts: torch.tensor = None,
                               logits:       torch.tensor = None,
                               mu:           torch.tensor = None,
                               theta:        torch.tensor = None,
                               reduction:    str = "sum"):
        if ((total_counts == None) and (logits == None)):
            if ((mu == None) and (theta == None )):
                raise ValueError
            logits = (mu + 1e-6) - (theta + 1e-6).log()
            total_counts = theta 
        
        nb = NegativeBinomial(
            total_count=total_counts, 
            logits=logits, 
        )
        if reduction == "sum":
            reconst_loss = -nb.log_prob(X).sum(dim = 1)
        elif reduction == "mean":
            reconst_loss = -nb.log_prob(X).mean(dim = 1)
        elif reduction == "none":
            reconst_loss = -nb.log_prob(X)
        return reconst_loss

    @staticmethod 
    def zi_gaussian_reconstruction_loss(
        X,
        mean,
        variance,
        gate_logits,
        reduction: Literal['sum','mean'] = 'sum'
    ):
        zg = ZeroInflatedGaussian(
            mean=mean,
            variance=variance,
            gate_logits=gate_logits
        )   
        if reduction == "sum":
            reconst_loss = -zg.log_prob(X).sum(dim = 1)
        elif reduction == "mean":
            reconst_loss = -zg.log_prob(X).mean(dim = 1)
        return reconst_loss 

    @staticmethod
    def kld(q: torch.tensor,
            p: torch.tensor):
        kl_loss = kldiv(q.log(), p, reduction="sum", log_target=False)
        kl_loss.requires_grad_(True)
        return kl_loss

    @staticmethod
    def kl1(mu:  torch.tensor, 
            var: torch.tensor):
        return kldiv(Normal(mu, torch.sqrt(var)), Normal(0, 1)).sum(dim=1)

    @staticmethod
    def kl2(mu1: torch.tensor, 
           var1: torch.tensor, 
           mu2:  torch.tensor, 
           var2: torch.tensor):
        return kldiv(Normal(mu1, var1.sqrt()), Normal(mu2, var2.sqrt()))

    @staticmethod
    def soft_cross_entropy(pred:         torch.tensor, 
                           soft_targets: torch.tensor):
        # use nn.CrossEntropyLoss if not using soft labels
        logsoftmax = nn.LogSoftmax(dim=1)
        return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
