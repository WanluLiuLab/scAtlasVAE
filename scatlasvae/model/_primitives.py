import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Union, Mapping, Iterable, List, Optional, Callable
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torch import nn, einsum
import warnings

from einops import rearrange

import numpy as np

from functools import partial

from ..utils._loss import LossFunction
from ..utils._logger import  mt
from ..utils._definitions import FCDEF
from ..utils._tensor_utils import one_hot
from ..utils._compat import Literal

import math
from scipy.stats import truncnorm
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    >>> early_stopping = EarlyStopping(patience=patience, verbose=True)
    >>> early_stopping(valid_loss, model)
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss




####################################################
#                    Linear                        #
####################################################

def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out

def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f

def _trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))
        
def lecun_normal_init_(weights):
    _trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    _trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")
    
class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: Literal["default","final","gating","glorot","normal","relu"] = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        with torch.no_grad():
            if init_fn is not None:
                init_fn(self.weight, self.bias)
            elif init == "default":
                lecun_normal_init_(self.weight)
            elif init == "final":
                final_init_(self.weight)
            elif init == "gating":
                gating_init_(self.weight)
                if bias:
                    self.bias.fill_(1.0)
            elif init == "glorot":
                glorot_uniform_init_(self.weight)
            elif init == "normal":
                normal_init_(self.weight)
            elif init == "relu":
                he_normal_init_(self.weight)
            else:
                raise ValueError("Invalid init string.")
            

class Sparsemax(nn.Module):
    """Sparsemax activation function.

    Pytorch implementation of Sparsemax function from:
    -- "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
    -- André F. T. Martins, Ramón Fernandez Astudillo (http://arxiv.org/abs/1602.02068)
    """
    def __init__(self, dim=None):
        """Initialize sparsemax activation
        
        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.

        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size

        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor

        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=input.device, dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input

        
class ReparameterizeLayerBase(nn.Module):
    """Base layer for VAE reparameterization"""
    def __init__(self) -> None:
        """
        @brief ReparameterizeLayer Defining methods for reparameterize tricks
        """
        super(ReparameterizeLayerBase, self).__init__()

    def reparameterize(
        self, 
        mu: torch.Tensor, 
        var: torch.Tensor
    ) -> torch.Tensor :
        untran_z = Normal(mu, var.sqrt()).rsample()
        return untran_z

    def reparameterize_transformation(
        self, 
        transfunc, 
        mu: torch.Tensor, 
        var: torch.Tensor
    ) -> torch.Tensor:
        z = self.reparameterize(mu, var)
        ztrans = transfunc(z)
        return ztrans, z

class MMDLayerBase:
    """Base layer for Maximum-mean descrepancy calculation"""
    
    def mmd_loss(self, 
        z: torch.Tensor, 
        cat: np.array, 
        dim=-1,
        min_n_samples: int = 2,
        flavor: Literal['trvae','default'] = 'trvae'
    ) -> torch.Tensor:
        zs = []
        loss = torch.tensor(0.0, device=z.device)
        for i in np.unique(cat):
            idx = list(map(lambda z:z[0], filter(lambda x:x[1] == i, enumerate(cat))))
            zs.append(z[idx])
        for i in range(len(np.unique(cat))):
            for j in range(i+1,len(np.unique(cat))):
                if zs[i].shape[0] > min_n_samples and zs[j].shape[0] > min_n_samples:
                    if flavor == 'trvae':
                        loss += LossFunction.mmd_loss_trvae(
                            zs[i], zs[j]
                        )
                    else:
                        loss += LossFunction.mmd_loss(
                            zs[i], zs[j], dim=-1
                        ) / len(zs)
        return loss

    def hierarchical_mmd_loss_1(
        self, 
        z: torch.Tensor, 
        cat: np.array, 
        hierarchical_weight: Iterable[float]
    ) -> torch.Tensor:
        if len(cat.shape) <= 1:
            raise ValueError("Illegal category array")
        if len(z) != cat.shape[0]:
            raise ValueError("Dimension of z {} should be equal to dimension of category {}".format(len(z), cat.shape[0]))
        if len(hierarchical_weight) != cat.shape[1]:
            raise ValueError("Dimension of hierarchical_weight {} should be equal to dimension of category {}".format(len(hierarchical_weight), cat.shape[1]))

        if cat.shape[1] < 2:
            cat = cat.flatten()
            return self.MMDLoss(z, cat)
        loss = 0
        zs = []
        for i in np.unique(cat[:, 0]):
            idx = list(map(lambda t:t[0], filter(lambda x:x[1] == i, enumerate(cat[:,0]))))
            loss += self.HierarchicalMMDLoss(z[idx], cat[idx,1:], hierarchical_weight[1:])
            zs.append(z[idx])
        for i in range(len(np.unique(cat[:,0]))):
            for j in range(i+1, len(np.unique(cat[:,0]))):
                loss += LossFunction.mmd_loss(
                    zs[i], zs[j]
                )
        return loss

    def hierarchical_mmd_loss_2(self, 
        z: torch.Tensor, 
        cat1: np.array, 
        cat2: np.array,
        dim=-1,
        min_n_samples: int = 2,
        flavor: Literal['trvae','default'] = 'trvae'
    ) -> torch.Tensor:
        """
        Hierachical mmd loss with independent categories

        :param z: torch.Tensor: Latent space
        :param cat1: np.array: Categories
        :param cat2: np.array: Categories
        """
        loss = torch.tensor(0.0, device=z.device)

        for i in np.unique(cat1):
            zs = []
            idx = list(map(lambda z:z[0], filter(lambda x:x[1] == i, enumerate(cat1))))
            zz = z[idx]
            for j in np.unique(cat2[idx]):
                idx2 = list(map(lambda z: z[0], filter(lambda x:x[1] == j, enumerate(cat2[idx]))))
                zzz = zz[idx2]
                zs.append(zzz)
            for i in range(len(np.unique(cat2[idx]))):
                for j in range(i+1,len(np.unique(cat2[idx]))):
                    if zs[i].shape[0] > min_n_samples and zs[j].shape[0] > min_n_samples:
                        if flavor == 'trvae':
                            loss += LossFunction.mmd_loss_trvae(
                                zs[i], zs[j]
                            ) / (len(zs) * len(np.unique(cat1)))
                        else:
                            loss += LossFunction.mmd_loss(
                                zs[i], zs[j], dim=-1
                            ) / (len(zs) * len(np.unique(cat1)))
        return loss


class FCLayer(nn.Module):
    """FCLayer Fully-Connected Layers for a neural network """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_cat_list: Iterable[int] = None,
        cat_dim: int = 8,
        cat_embedding: Literal["embedding", "onehot"] = "onehot",
        bias: bool = True,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        activation_fn: nn.Module = nn.ReLU,
        activation_dim: int = None,
        device: str = "cuda"
    ):
        super(FCLayer, self).__init__()
        if n_cat_list is not None:
            # Categories
            if not all(map(lambda x: x > 1, n_cat_list)):
                warnings.warn("category list contains values less than 1")
            self.n_category = len(n_cat_list)
            self._cat_dim = cat_dim
            self._cat_embedding = cat_embedding
            if cat_embedding == "embedding":
                self.cat_dimension = self.n_category * cat_dim # Total dimension of categories using embedding
            else:
                self.cat_dimension = sum(n_cat_list) # Total dimension of categories using one-hot
            self.n_cat_list = n_cat_list
            if cat_embedding == "embedding":
                self.cat_embedding = nn.ModuleList(
                    [nn.Embedding(n, cat_dim) for n in n_cat_list]
                )
            else: 
                self.cat_embedding = [
                    partial(one_hot, n_cat=n) for n in n_cat_list
                ]

        else:
            # No categories will be included
            self.n_category = 0
            self.n_cat_list = None
        
        self._fclayer = nn.Sequential(
                *list(filter(lambda x:x, 
                        [
                            nn.Linear(in_dim, out_dim, bias=bias) 
                            if self.n_category == 0 
                            else nn.Linear(in_dim + self.cat_dimension, out_dim, bias=bias),
                            nn.BatchNorm1d(out_dim, momentum=0.01, eps=0.001) if use_batch_norm else None,
                            nn.LayerNorm(out_dim, elementwise_affine=False) if use_layer_norm else None,
                            activation_fn(dim=activation_dim) if activation_dim else activation_fn() if activation_fn else None,
                            nn.Dropout(p = dropout_rate) if dropout_rate > 0 else None
                        ]
                    )
                )
            )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device 

    def forward(self, X: torch.Tensor, cat_list: torch.Tensor =  None, return_category_embedding: bool = False):
        category_embedding = []
        if self.n_category > 0:
            if cat_list != None:
                if (len(cat_list) != self.n_category):
                    raise ValueError("Number of category list should be equal to {}".format(self.n_category))

                for i, (n_cat, cat) in enumerate(zip(self.n_cat_list, cat_list)):
                    assert(n_cat > 1)
                    if self._cat_embedding == "embedding":
                        category_embedding.append(self.cat_embedding[i](cat))
                    else: 
                        category_embedding.append(self.cat_embedding[i](cat.unsqueeze(0).T))
            else:
                if X.shape[1] != self.in_dim + self.n_category:
                    raise ValueError("Dimension of X should be equal to {} + {} but found {} if cat_list is provided".format(self.in_dim, self.n_category, X.shape[1]))
                cat_list = X[:, -self.n_category:].type(torch.LongTensor).T.to(self.device)
                for i, (n_cat, cat) in enumerate(zip(self.n_cat_list, cat_list)):
                    if self._cat_embedding == "embedding":
                        category_embedding.append(self.cat_embedding[i](cat))
                    else: 
                        category_embedding.append(self.cat_embedding[i](cat.unsqueeze(0).T))
               
            category_embedding = torch.hstack(category_embedding).to(self.device)
            if return_category_embedding:
                return self._fclayer(torch.hstack([X[:,:self.in_dim], category_embedding])), category_embedding
            else: 
                return self._fclayer(torch.hstack([X[:,:self.in_dim], category_embedding]))
        else:
            return self._fclayer(X)

    def to(self, device:str):
        super(FCLayer, self).to(device)
        self.device=device 
        return self

class PredictionLayerBase(nn.Module):
    """Prediction layer base """
    def __init__(self, *, in_dim:int, n_pred_category: int):
        super(PredictionLayerBase, self).__init__()
        self.in_dim = in_dim
        self.n_pred_category = n_pred_category
        self.decoder = FCLayer(
            in_dim = in_dim,
            out_dim = n_pred_category,
            bias = False,
            dropout_rate = 0,
            use_batch_norm = False,
            use_layer_norm = False,
            activation_fn = nn.ReLU,
        )

    def forward(self, X: torch.Tensor):
        return nn.Softmax(-1)( self.decoder(X) )

class SAE(nn.Module):
    ''' Stacked Autoencoders. 
        Fitting includes stacked fitting and fine-tuning:
            Fine-tuning step removes the decoder and use clustering method
            to fine-tune the encoder.
        parameters:
            dim:    int 
            stacks: Iterable[int]
            n_cat_list: Iterable[int]
            cat_dim: int
    '''
    def __init__(
            self, 
            dim:int, 
            stacks:Iterable[int] = [512, 128, 64], 
            n_cat_list: Iterable[int] = None,
            cat_dim: int = 8,
            bias: bool = True,
            dropout_rate: float = 0.1,
            use_batch_norm: bool = True,
            use_layer_norm: bool = False,
            cat_embedding: Literal["embedding", "onthot"] = "onehot",
            activation_fn: nn.Module = nn.ReLU,
            encode_only: bool = False,
            decode_only: bool = False,
            device="cuda"
    ):
        super(SAE, self).__init__()
        fcargs = dict(
            bias=bias, 
            dropout_rate=dropout_rate, 
            use_batch_norm=use_batch_norm, 
            use_layer_norm=use_layer_norm,
            activation_fn=activation_fn,
            device=device,
            cat_embedding = cat_embedding
        )
        self.dim = dim
        self.num_layers = len(stacks)
        self.n_cat_list = n_cat_list
        self.cat_dim = cat_dim
        self.n_category = len(n_cat_list) if n_cat_list != None else 0
        self.stacks = stacks
        self.out_dim = stacks[-1]
        layers = [None] * len(stacks)
        self.n_layers = len(stacks)
        if (encode_only & decode_only):
            raise ValueError("SAE instance cannot be both encode and decode only")
        for i,j in enumerate(stacks):
            if i == 0:
                layers[i] = [FCLayer(dim, 
                             stacks[i], 
                             n_cat_list, 
                             cat_dim,
                             **fcargs)
                             if not decode_only 
                             else None, 
                             FCLayer(stacks[i], dim, **fcargs) 
                             if not encode_only 
                             else None]
            else:
                layers[i] = [FCLayer(stacks[i-1], stacks[i], **fcargs)
                             if not decode_only 
                             else None, 
                             FCLayer(stacks[i], stacks[i-1], **fcargs) 
                             if not encode_only 
                             else None ]
        layers = [i for s in layers for i in s]
        self.layers = nn.ModuleList(layers)
        self.device = device
        self.loss = []
        self.encode_only = encode_only
        self.decode_only = decode_only

    def get_layer(self, codec:str, layer:int):
        i = 0 if codec == FCDEF.ENCODER else 1
        return self.layers[layer * 2 + i]

    def encode(self, x: torch.Tensor):
        '''
        encode features in the nth layer 
        '''
        if self.decode_only:
            raise TypeError("This is an decoder-only SAE instance")
        h = None
        for i in range(self.num_layers):
            layer = self.get_layer(FCDEF.ENCODER, i)
            if i == self.num_layers - 1:
                if i == 0:
                    h = layer(x)
                else:
                    h = layer(h)
            else:
                if i == 0: 
                    h = layer(x)
                else:
                    h = layer(h)
        return h
    
    def decode(self, z: torch.Tensor):
        '''
        decode features in the nth layer 
        '''
        if self.encode_only:
            raise TypeError("This is an encoder-only SAE instance")
        h = None
        for i in range(self.num_layers):
            layer = self.get_layer(FCDEF.DECODER, self.num_layers - 1 - i)
            if i == self.num_layers - 1:
                if i == 0:
                    h = layer(z)
                else:
                    h = layer(h)
            else:
                if i == 0:
                    h = layer(z)
                else:
                    h = layer(h)
        return h

    def forward(self, x: torch.Tensor):
        z = self.encode(x.view(-1, self.dim))
        return self.decode(z), z

    def fit(self, X_train, max_epoch, minibatch = True, lr = 1e-3):
        optimizer = optim.Adam(self.parameters(), lr, weight_decay = 1e-3);
        scheduler =  ReduceLROnPlateau(optimizer, mode="min", patience=10)
        if minibatch:
            for epoch in range(1, max_epoch+1):
                epoch_total_loss = 0
                for batch_index, X in enumerate(X_train):
                    if X.device.type != self.device:
                        X = X.to(self.device)
                    recon_batch, hidden_batch = self.forward(X)
                    loss = LossFunction.mse(recon_batch, X)
                    epoch_total_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step(loss)
                    del(X) # Reduce memory 
                    torch.cuda.empty_cache() 

                mt('Epoch: {} Average loss: {:.8f}'.format(epoch, epoch_total_loss))
                self.loss.append(epoch_total_loss)
        else:
            for epoch in range(1, max_epoch+1):
                epoch_total_loss = 0
                if X_train.device.type != self.device:
                    X_train = X_train.to(self.device)
                recon_batch, hidden_batch = self.forward(X_train)
                loss = LossFunction.mse(recon_batch, X_train)
                epoch_total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step(loss)
                torch.cuda.empty_cache() 
                mt('Epoch: {} Average loss: {:.8f}'.format(epoch, epoch_total_loss))
                self.loss.append(epoch_total_loss)
    def to(self, device:str):
        super(SAE, self).to(device)
        self.device=device 
        return self

class PositionalEncoding(nn.Module):
    def __init__(self, *,
        n_hiddens: int, 
        dropout: float, 
        max_len:int=1000
    ) -> None:
        super(PositionalEncoding, self).__init__()
        assert(n_hiddens % 2 == 0)
        self.n_hiddens = n_hiddens
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, n_hiddens))
        X = rearrange(torch.arange(max_len, dtype=torch.float32),
            '(n d) -> n d', d = 1) / torch.pow(1e4, torch.arange(
            0, n_hiddens, 2, dtype=torch.float32) / n_hiddens)
        self.P[:,:,0::2] = torch.sin(X)
        self.P[:,:,1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :X.shape[2]].to(X.device)
        return self.dropout(X)

class FCSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head,
        n_heads
    ):
        super().__init__()
        inner_dim = dim_head * n_heads
        self.n_heads = n_heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, X, mask = None):
        """ X.shape = (batch, tokens, dim)
        """
        h = self.n_heads
        q, k, v = self.to_qkv(X).chunk(3, dim = -1)
        # q.shape = (batch, head, tokens, inner_dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        # attention.shape = (batch, head, tokens, tokens)
        attention = einsum('b h n d, b h m d -> b h n m', k, q) * self.scale
        if mask != None:
            if mask.shape != attention.shape:
                mask = rearrange(mask, 'n h -> n () h ()')
            attention = attention.masked_fill(mask, -torch.finfo(attention.dtype).max)
        attention = torch.softmax(attention, dim=-1)
        # out.shape = (batch, head, inner_dim, tokens)
        out = einsum('b h n e, b h e d -> b h d n', attention, v)
        # out.shape = (batch, tokens, head * inner_dim)
        out = rearrange(out, 'b h d n -> b n (h d)')
        return self.to_out(out)

class AddNorm(nn.Module):
    """Residual connection followed by layer normalization.""" 
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs) 
        self.dropout = nn.Dropout(dropout) 
        self.ln = nn.LayerNorm(normalized_shape)
    
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

# reference: https://github.com/daiquocnguyen/Graph-Transformer/blob/master/UGformerV2_PyTorch/UGformerV2.py
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_dim, out_dim, activation = torch.relu, bias:bool = False) -> None:
        super(GraphConvolution, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim 
        self.weight = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        self.bn = nn.BatchNorm1d(out_dim)
        self.has_bias = bias

    def forward(self, X, A):
        Z = self.weight(X)
        output = torch.mm(A, Z)
        output = self.bn(output)
        return self.activation(output)

# reference: https://github.com/daiquocnguyen/Graph-Transformer/blob/master/UGformerV2_PyTorch/UGformerV2.py    
class FullyConnectedGraphTransformer(nn.Module):
    def __init__(self, feature_dim,
                       ff_hidden_size,
                       n_self_att_layers,
                       dropout,
                       n_GNN_layers,
                       n_head, device = "cuda"
                       ) -> None:
        super(FullyConnectedGraphTransformer, self).__init__()
        self.feature_dim = feature_dim
        self.ff_hidden_size = ff_hidden_size
        self.n_self_att_layers = n_self_att_layers
        self.n_GNN_layers = n_GNN_layers
        self.n_head = n_head
        self.GNN_layers = nn.ModuleList()
        self.selfAtt_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.node_embedding = nn.Linear(feature_dim * n_GNN_layers, feature_dim, bias = True)
        self.device = device
        for _ in range(self.n_GNN_layers):
            encoder_layer = nn.TransformerEncoderLayer(self.feature_dim, 
                nhead=self.n_head, 
                dim_feedforward=self.ff_hidden_size,
                dropout=0.5)
            self.selfAtt_layers.append(
                nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=self.n_self_att_layers).to(device))
            self.GNN_layers.append(GraphConvolution(self.feature_dim, self.feature_dim, torch.relu, True).to(device))
            self.dropouts.append(nn.Dropout(dropout))
    
    def reset_parameters(self):
        for i in self.selfAtt_layers:
            i.reset_parameters()
        self.prediction.reset_parameters()
        for i in self.dropouts:
            i.reset_parameters()

    def forward(self, X, A):
        Xs = []
        for i in range(self.n_GNN_layers):
            # self attention over all nodes
            X = X.unsqueeze(1)
            X = self.selfAtt_layers[i](X)
            X = X.squeeze(1)
            X = self.GNN_layers[i](X, A)
            Xs.append(X)
        X = torch.hstack(Xs)
        X = self.node_embedding(X)
        return X

    def to(self, device:str):
        super(FullyConnectedGraphTransformer, self).to(device)
        self.device=device 
        return self