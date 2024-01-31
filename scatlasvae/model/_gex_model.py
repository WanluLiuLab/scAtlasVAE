# Pytorch
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.distributions import kl_divergence as kld
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Third Party
import numpy as np
from pathlib import Path
import pandas as pd
import scanpy as sc

# Built-in
import time
import numpy as np
from scipy.sparse import issparse
from sklearn.utils import class_weight
from collections import Counter
from itertools import chain
from copy import deepcopy
import json
from typing import Mapping, Union, Iterable, Tuple, Optional, Mapping, Dict
import os
import warnings


# Package
from ._primitives import *
from ..utils._tensor_utils import one_hot, get_k_elements, get_last_k_elements, get_elements
from ..utils._decorators import typed
from ..utils._loss import LossFunction
from ..utils._logger import mt, mw, Colors, get_tqdm

from ..utils._utilities import random_subset_by_key_fast, euclidean
from ..utils._compat import Literal
from ..tools._umap import umap_alignment
from ..utils._utilities import get_default_device
from ..preprocessing._preprocess import subset_adata_by_genes_fill_zeros

MODULE_PATH = Path(__file__).parent
warnings.filterwarnings("ignore")


class scAtlasVAE(ReparameterizeLayerBase, MMDLayerBase):
    """
    VAE model for atlas-level integration and label transfer

    :param adata: AnnData. If provided, initialize the model with the adata.
    :param hidden_stacks: List[int]. Number of hidden units in each layer. Default: [128] (one hidden layer with 128 units)
    :param n_latent: int. Number of latent dimensions. Default: 10
    :param n_batch: int. Number of batch. Default: 0
    :param n_label: int. Number of label. Default: 0
    :param n_additional_batch: Optional[Iterable[int]]. Number of categorical covariate. Default: None
    :param batch_key: str. Batch key in adata.obs. Default: None
    :param label_key: str. Label key in adata.obs. Default: None
    :param dispersion: Literal["gene", "gene-batch", "gene-cell"]. Dispersion method. Default: "gene-cell"
    :param log_variational: bool. If True, log the variational distribution. Default: True
    :param total_variational: bool. If True, normalize the counts with library size. Default: False
    :param bias: bool. If True, use bias in the linear layer. Default: True
    :param use_batch_norm: bool. If True, use batch normalization. Default: True
    :param use_layer_norm: bool. If True, use layer normalization. Default: False
    :param batch_hidden_dim: int. Number of hidden units in the batch embedding layer. Default: 8
    :param batch_embedding: Literal["embedding", "onehot"]. Batch embedding method. Default: "batch_embedding"
    :param constrain_latent_method: Literal['mse', 'normal']. Method to constrain the latent embedding. Default: 'mse'
    :param constrain_latent_embedding: bool. If True, constrain the latent embedding. Default: False
    :param constrain_latent_key: str. Key to the data to constrain the latent embedding. Default: 'X_gex'
    :param encode_libsize: bool. If True, encode the library size. Default: False
    :param decode_libsize: bool. If True, decode the library size. Default: True
    :param dropout_rate: float. Dropout rate. Default: 0.1
    :param activation_fn: nn.Module. Activation function. Default: nn.ReLU
    :param inject_batch: bool. If True, inject batch information. Default: True
    :param inject_label: bool. If True, inject label information. Default: False
    :param inject_additional_batch: bool. If True, inject categorical covariate information. Default: True
    :param unlabel_key: str. key for unlabeled cells. Default: "undefined"
    :param mmd_key: Optional[Literal['batch']]. If provided, use MMD loss. Default: None (do not use MMD loss)
    :param pretrained_state_dict: torch.device or str. Build the model loading the pretrained state dict
    :param device: Optional[Union[str, torch.device]]. Device to use. Default: determined by availablility of CUDA device

    :example:
        >>> import scatlasvae
        >>> model = scatlasvae.model.scAtlasVAE(
        >>>    adata,
        >>>    batch_key = ['sample_name','study_name'],
        >>>    label_key = ['cell_type', 'cell_subtype'],
        >>> )
    """
    def __init__(self, *,
       adata: Optional[sc.AnnData] = None,
       hidden_stacks: List[int] = [128],
       n_latent: int = 10,
       n_batch: int = 0,
       n_label: int = 0,
       n_additional_batch: Optional[Iterable[int]] = None,
       n_additional_label: Optional[Iterable[int]] = None,
       batch_key = None,
       label_key = None,
       dispersion:  Literal["gene", "gene-batch", "gene-cell"] = "gene-cell",
       log_variational: bool = True,
       total_variational: bool = False,
       bias: bool = True,
       use_batch_norm: bool = True,
       use_layer_norm: bool = False,
       batch_hidden_dim: int = 8,
       batch_embedding: Literal["embedding", "onehot"] = "embedding",
       reconstruction_method: Literal['mse', 'zg', 'zinb'] = 'zinb',
       constrain_n_label: bool = True,
       constrain_n_batch: bool = True,
       constrain_latent_method: Literal['mse', 'normal'] = 'mse',
       constrain_latent_embedding: bool = False,
       constrain_latent_key: str = 'X_gex',
       encode_libsize: bool = False,
       decode_libsize: bool = True,
       dropout_rate: float = 0.1,
       activation_fn: nn.Module = nn.ReLU,
       inject_batch: bool = True,
       inject_label: bool = False,
       inject_additional_batch: bool = True,
       mmd_key: Optional[Literal['batch','additional_batch','both']] = None,
       unlabel_key: str = 'undefined',
       device: Optional[Union[str, torch.device]] = None,
       pretrained_state_dict: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> None:
        if device is None:
            device = get_default_device()
        
        super(scAtlasVAE, self).__init__()
        if adata.X.dtype != np.int32 and reconstruction_method in ['zinb', 'nb']:
            mw("adata.X is not of type np.int32. \n" + \
                " "*40 + "\tCheck whether you are using raw count matrix.")
            # adata.X = adata.X.astype(np.int32)

        self.adata = adata
        self.in_dim = adata.shape[1] if adata else -1
        self.n_hidden = hidden_stacks[-1]
        self.n_latent = n_latent
        self.n_additional_batch = n_additional_batch
        self.n_additional_label = n_additional_label
        self._hidden_stacks = hidden_stacks

        if n_batch > 0 and not batch_key:
            raise ValueError("Please provide a batch key if n_batch is greater than 0")
        if n_label > 0 and not label_key:
            raise ValueError("Please provide a label key if n_batch is greater than 0")

        self.label_key = label_key if isinstance(label_key, str) else label_key[0] if label_key is not None and isinstance(label_key, Iterable) else None
        self.label_category = None 
        self.label_category_summary = None 
        self.batch_key = batch_key if isinstance(batch_key, str) else batch_key[0] if batch_key is not None and isinstance(batch_key, Iterable) else None
        self.batch_category = None 
        self.batch_category_summary = None 
        self.additional_batch_keys = None if isinstance(batch_key, str) or (isinstance(batch_key, Iterable) and len(batch_key) == 1) else batch_key[1:] if batch_key is not None else None
        self.additional_batch_category = None 
        self.additional_batch_category_summary = None 
        self.additional_label_keys = None if isinstance(label_key, str) or (isinstance(label_key, Iterable) and len(label_key) == 1) else label_key[1:] if label_key is not None else None
        self.additional_label_category = None 
        self.additional_label_category_summary = None 


        self.n_batch = n_batch
        self.n_label = n_label
        self.unlabel_key = unlabel_key
        self.new_adata_code = None
        self.log_variational = log_variational
        self.total_variational = total_variational
        self.mmd_key = mmd_key
        self.reconstruction_method = reconstruction_method
        self.constrain_latent_embedding = constrain_latent_embedding
        self.constrain_latent_method = constrain_latent_method
        self.constrain_latent_key = constrain_latent_key
        self.constrain_n_label = constrain_n_label
        self.constrain_n_batch = constrain_n_batch

        self.device=device

        self.initialize_dataset()

        self.batch_embedding = batch_embedding
        if batch_embedding == "onehot":
            batch_hidden_dim = self.n_batch
        self.batch_hidden_dim = batch_hidden_dim
        self.inject_batch = inject_batch
        self.inject_label = inject_label
        self.inject_additional_batch = inject_additional_batch
        self.encode_libsize = encode_libsize
        self.decode_libsize = decode_libsize
        self.dispersion = dispersion


        self.fcargs = dict(
            bias           = bias,
            dropout_rate   = dropout_rate,
            use_batch_norm = use_batch_norm,
            use_layer_norm = use_layer_norm,
            activation_fn  = activation_fn,
            device         = device
        )



        #############################
        # Model Trainable Variables #
        #############################

        if self.dispersion == "gene":
            self.px_rate = torch.nn.Parameter(torch.randn(self.in_dim))
        elif self.dispersion == "gene-batch":
            self.px_rate = torch.nn.Parameter(torch.randn(self.in_dim, self.n_batch))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )

        ############
        # ENCODERS #
        ############

        self.encoder = SAE(
            self.in_dim if not self.encode_libsize else self.in_dim + 1,
            stacks = hidden_stacks,
            # n_cat_list = [self.n_batch] if self.n_batch > 0 else None,
            cat_dim = batch_hidden_dim,
            cat_embedding = batch_embedding,
            encode_only = True,
            **self.fcargs
        )

        # The latent cell representation z ~ Logisticnormal(0, I)
        self.z_mean_fc = nn.Linear(self.n_hidden, self.n_latent)
        self.z_var_fc = nn.Linear(self.n_hidden, self.n_latent)
        self.z_transformation = nn.Softmax(dim=-1)

        ############
        # DECODERS #
        ############

        if self.n_additional_batch_ is not None and self.inject_additional_batch:
            if self.n_batch > 0 and self.n_label > 0 and inject_batch and inject_label:
                decoder_n_cat_list = [self.n_batch, self.n_label, *self.n_additional_batch]
            elif self.n_batch > 0 and inject_batch:
                decoder_n_cat_list = [self.n_batch, *self.n_additional_batch]
            elif self.n_label > 0 and inject_label:
                decoder_n_cat_list = [self.n_label, *self.n_additional_batch]
            else:
                decoder_n_cat_list = None
        else:
            if self.n_batch > 0 and self.n_label > 0 and inject_batch and inject_label:
                decoder_n_cat_list = [self.n_batch, self.n_label]
            elif self.n_batch > 0 and inject_batch:
                decoder_n_cat_list = [self.n_batch]
            elif self.n_label > 0 and inject_label:
                decoder_n_cat_list = [self.n_label]
            else:
                decoder_n_cat_list = None

        self.decoder_n_cat_list = decoder_n_cat_list

        self.decoder = FCLayer(
            in_dim = self.n_latent,
            out_dim = self.n_hidden,
            n_cat_list = decoder_n_cat_list,
            cat_dim = batch_hidden_dim,
            cat_embedding = batch_embedding,
            use_layer_norm=False,
            use_batch_norm=True,
            dropout_rate=0,
            device=device
        )

        self.px_rna_rate_decoder = nn.Linear(self.n_hidden, self.in_dim)
        self.px_rna_scale_decoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.in_dim),
            nn.Softmax(dim=-1)
        )
        self.px_rna_dropout_decoder = nn.Linear(self.n_hidden, self.in_dim)

        if self.n_label > 0:
            self.fc = nn.Sequential(
                nn.Linear(self.n_latent, self.n_label)
            )

        if self.n_additional_label is not None:
            self.additional_fc = nn.ModuleList([
                nn.Linear(self.n_latent, x) for x in self.n_additional_label
            ])

        self._trained = False

        self.to(device)

        if pretrained_state_dict is not None:
            self.partial_load_state_dict(pretrained_state_dict)


    def __repr__(self):
        return f'{Colors.ORANGE}VAEModel{Colors.NC} object containing:\n' + \
            f'    {Colors.GREEN}adata{Colors.NC}: {self.adata}\n' + \
            f'    {Colors.GREEN}in_dim{Colors.NC}: {Colors.CYAN}{self.in_dim}{Colors.NC}\n' + \
            f'    {Colors.GREEN}n_hidden{Colors.NC}: {Colors.CYAN}{self.n_hidden}{Colors.NC}\n' + \
            f'    {Colors.GREEN}labels{Colors.NC}: {self.label_key} of {Colors.CYAN}{self.n_label}{Colors.NC}\n' if self.batch_key else '' + \
            f'    {Colors.GREEN}batches{Colors.NC}: {self.batch_key} of {Colors.CYAN}{self.n_batch}{Colors.NC}\n' if self.label_key else '' + \
            f'    {Colors.GREEN}additional_batches{Colors.NC}: {self.additional_batch_keys} of {Colors.CYAN}{self.n_additional_batch}{Colors.NC}\n' if self.additional_batch_keys else ''

    def partial_load_state_dict(self, state_dict: Mapping[str, torch.Tensor]):
        """
        Partially load the state dict

        :param state_dict: Mapping[str, torch.Tensor]. State dict to load
        """
        original_state_dict = self.state_dict()
        warned = False
        ignored_keys = []
        for k,v in state_dict.items():
            if k not in original_state_dict.keys():
                mt(f"Warning: {k} not found in the model. Ignoring {k} in the provided state dict.")
                ignored_keys.append(k)
            elif v.shape != original_state_dict[k].shape:
                mw(f"Warning: shape of {k} does not match. \n" + \
                    ' '*40 + "\tOriginal:" + f" {original_state_dict[k].shape},\n" + \
                    ' '*40 + f"\tNew: {v.shape}")
                state_dict[k] = original_state_dict[k]
        for k,v in original_state_dict.items():
            if k not in state_dict.keys():
                mw(f"Warning: {k} not found in the provided state dict. " + \
                     f"Using {k} in the original state dict.")
                state_dict[k] = v
        for i in ignored_keys:
            state_dict.pop(i)
        self.load_state_dict(state_dict)

    def get_config(self):
        """
        Get the model config

        :return: dict. Model config dictionary
        """
        return {
            'hidden_stacks': self._hidden_stacks,
            'n_latent': self.n_latent,
            'n_batch': self.n_batch,
            'n_label': self.n_label,
            'n_additional_batch': self.n_additional_batch,
            'n_additional_label': self.n_additional_label,
            'batch_key': self.batch_key if self.additional_batch_keys is None else [self.batch_key] + self.additional_batch_keys,
            'label_key': self.label_key if self.additional_label_keys is None else [self.label_key] + self.additional_label_keys,
            'dispersion': self.dispersion,
            'log_variational': self.log_variational,
            'bias': self.fcargs['bias'],
            'use_batch_norm': self.fcargs['use_batch_norm'],
            'use_layer_norm': self.fcargs['use_layer_norm'],
            'batch_hidden_dim': self.batch_hidden_dim,
            'batch_embedding': self.batch_embedding,
            'reconstruction_method': self.reconstruction_method,
            'encode_libsize': self.encode_libsize,
            'decode_libsize': self.decode_libsize,
            'dropout_rate': self.fcargs['dropout_rate'],
            'activation_fn': self.fcargs['activation_fn'],
            'inject_batch': self.inject_batch,
            'inject_label': self.inject_label,
            'inject_additional_batch': self.inject_additional_batch,
            'mmd_key': self.mmd_key,
            'unlabel_key': self.unlabel_key,
        }

    def save_to_disk(self, path_to_state_dict: Union[str, Path]):
        """
        Save the model to disk

        :param path_to_state_dict: str or Path. Path to save the model
        """
        model_state_dict = self.state_dict()
        model_var_index = self.adata.var.index
        state_dict = {
            "model_state_dict": model_state_dict,
            "model_var_index": model_var_index,
            "model_config": self.get_config(),
            "batch_category": self.batch_category,
            "batch_category_summary": self.batch_category_summary,
            "label_category": self.label_category,
            "label_category_summary": self.label_category_summary,
            "additional_label_category": self.additional_label_category,
            "additional_label_category_summary": self.additional_label_category_summary,
            "additional_batch_category": self.additional_batch_category,
            "additional_batch_category_summary": self.additional_batch_category_summary,
        }
        torch.save(state_dict, path_to_state_dict)

    def load_from_disk(self, path_to_state_dict: Union[str, Path]):
        """
        Load the model from disk

        :param path_to_state_dict: str or Path. Path to load the model
        """
        state_dict = torch.load(path_to_state_dict)
        self.partial_load_state_dict(state_dict["model_state_dict"])

    @staticmethod
    def setup_anndata(
        adata: sc.AnnData, 
        path_to_state_dict: Optional[Union[str, Path]] = None, 
        unlabel_key: str = 'undefined'
    ):
        """
        Setup the model with adata

        :param adata: AnnData. AnnData to setup the model
        :param path_to_state_dict: Optional[str, Path]. Path to the state dict to load
        :param unlabeled_key: str. Default: Undefined

        """
        state_dict = torch.load(path_to_state_dict)
        if 'model_var_index' in state_dict.keys():
            model_var_index = state_dict['model_var_index']
            if any(list(map(lambda x: x not in model_var_index, adata.var.index))) or any(list(map(lambda x: x not in adata.var.index, model_var_index))):
                mw("the provided adata contains variables not in the state dict.")
                mt("         The model will be initialized with the variables in the state dict.")
                adata = subset_adata_by_genes_fill_zeros(adata, list(model_var_index))


        if state_dict["batch_category"] is not None:
            if state_dict['model_config']['batch_key'] not in adata.obs.keys():
                adata.obs[state_dict['model_config']['batch_key']] = unlabel_key
                adata.obs[state_dict['model_config']['batch_key']] = pd.Categorical(
                    list(adata.obs[state_dict['model_config']['batch_key']] ),
                    categories = pd.Categorical(
                        state_dict["batch_category"].categories
                    ).add_categories(unlabel_key).categories
                )
            else:
                adata.obs[state_dict['model_config']['batch_key']] = pd.Categorical(
                    list(adata.obs[state_dict['model_config']['batch_key']] ),
                    categories=state_dict["batch_category"].categories
                )


        if state_dict["label_category"] is not None:
            if state_dict['model_config']['label_key'] not in adata.obs.keys():
                adata.obs[state_dict['model_config']['label_key']] = unlabel_key
                adata.obs[state_dict['model_config']['label_key']] = pd.Categorical(
                    list(adata.obs[state_dict['model_config']['label_key']] ),
                    categories = pd.Categorical(
                        state_dict["label_category"].categories
                    ).add_categories(unlabel_key).categories if \
                    unlabel_key not in state_dict["label_category"].categories else \
                    state_dict["label_category"].categories
                )
            else:
                adata.obs[state_dict['model_config']['label_key']] = pd.Categorical(
                    list(adata.obs[state_dict['model_config']['label_key']] ),
                    categories=state_dict["label_category"].categories
                )

        if state_dict["additional_batch_category"] is not None:
            for i,k in enumerate(state_dict['model_config']['additional_batch_keys']):
                if k not in adata.obs.keys():
                    adata.obs[k] = unlabel_key
                    adata.obs[k] = pd.Categorical(
                        list(adata.obs[k] ),
                        categories = pd.Categorical(
                            state_dict["additional_batch_category"][i].categories
                        ).add_categories(unlabel_key).categories if \
                        unlabel_key not in state_dict["additional_batch_category"][i].categories else \
                        state_dict["additional_batch_category"][i].categories
                    )

                else:
                    adata.obs[k] = pd.Categorical(
                        list(adata.obs[k] ),
                        categories=state_dict["additional_batch_category"][i].categories
                    )

        if state_dict["additional_label_category"] is not None:
            for i,k in enumerate(state_dict['model_config']['additional_label_keys']):
                if k not in adata.obs.keys():
                    adata.obs[k] = unlabel_key
                    adata.obs[k] = pd.Categorical(
                        list(adata.obs[k] ),
                        categories = pd.Categorical(
                            state_dict["additional_label_category"][i].categories
                        ).add_categories(unlabel_key).categories if \
                        unlabel_key not in state_dict["additional_label_category"][i].categories else \
                        state_dict["additional_label_category"][i].categories
                    )
                else:
                    adata.obs[k] = pd.Categorical(
                        list(adata.obs[k] ),
                        categories=state_dict["additional_label_category"][i].categories
                    )

        return adata

    def initialize_dataset(self):
        mt("Initializing dataset into memory")
        if self.batch_key is not None:
            n_batch_ = len(np.unique(self.adata.obs[self.batch_key]))
            if self.n_batch != n_batch_:
                mt(f"warning: the provided n_batch={self.n_batch} does not match the number of batch in the adata.")
                if self.constrain_n_batch:
                    mt(f"         setting n_batch to {n_batch_}")
                    self.n_batch = n_batch_
            if not isinstance(self.adata.obs[self.batch_key], pd.Categorical):
                self.batch_category = pd.Categorical(self.adata.obs[self.batch_key])
            self.batch_category_summary = dict(Counter(self.batch_category))
            for k in self.batch_category.categories:
                if k not in self.batch_category_summary.keys():
                    self.batch_category_summary[k] = 0


        if self.label_key is not None:
            n_label_ = len(np.unique(list(filter(lambda x: x != self.unlabel_key, pd.Categorical(self.adata.obs[self.label_key]).categories))))

            if self.n_label != n_label_:
                mt(f"warning: the provided n_label={self.n_label} does not match the number of batch in the adata.")
                if self.constrain_n_label:
                    mt(f"         setting n_label to {n_label_}")
                    self.n_label = n_label_

            if not isinstance(self.adata.obs[self.label_key], pd.Categorical):
                self.adata.obs[self.label_key] = list(self.adata.obs[self.label_key])
                self.adata.obs[self.label_key] = pd.Categorical(self.adata.obs[self.label_key].fillna(self.unlabel_key))

            self.label_category = pd.Categorical(self.adata.obs[self.label_key])
            self.label_category_summary = dict(Counter(list(filter(lambda x: x != self.unlabel_key, self.label_category))))
            for k in self.label_category.categories:
                if k not in self.label_category_summary.keys() and k != self.unlabel_key:
                    self.label_category_summary[k] = 0

            self.label_category_weight = len(self.label_category) / torch.tensor([
                self.label_category_summary[x] for x in list(filter(lambda x:
                    x != self.unlabel_key,
                    self.label_category.categories
                ))], dtype=torch.float64).to(self.device)

            if self.unlabel_key in self.label_category.categories:
                self.new_adata_code = list(self.label_category.categories).index(self.unlabel_key)

        self.n_additional_label_ = None
        self.additional_label_category = None
        self.additional_label_category_summary = None
        if self.additional_label_keys is not None:
            self.n_cell_additional_label = [len(list(filter(lambda x: x != self.unlabel_key,self.adata.obs[x]))) for x in [self.label_key] + self.additional_label_keys]

            self.n_additional_label_ = [len(np.unique(list(filter(lambda x: x != self.unlabel_key,pd.Categorical(self.adata.obs[x]).categories)))) for x in self.additional_label_keys]

            # self.additional_label_weight = sum(self.n_cell_additional_label) / torch.tensor(self.n_cell_additional_label)
            self.additional_label_weight = torch.tensor([1] * len(self.n_cell_additional_label), dtype=torch.float64).to(self.device)

            if self.n_additional_label == None or len(self.n_additional_label_) != len(self.n_additional_label):
                mt(f"warning: the provided n_additional_label={self.n_additional_label} does not match the number of additional label in the adata.")
                if self.constrain_n_label:
                    mt(f"         setting n_additional_label to {self.n_additional_label_}")
                    self.n_additional_label = self.n_additional_label_
            else:
                for e,(i,j) in enumerate(zip(self.n_additional_label_, self.n_additional_label)):
                    if i != j:
                        mt(f"n_additional_label {self.additional_label_keys[e]} does not match the number in the adata.")
                        if self.constrain_n_label:
                            mt(f"         setting n_additional_label {e} to {i}")
                            self.n_additional_label[e] = i
            self.additional_label_category = [
                pd.Categorical(self.adata.obs[x]) for x in self.additional_label_keys
            ]
            self.additional_label_category_summary = [dict(Counter(x)) for x in self.additional_label_category]
            for i in range(len(self.additional_label_category_summary)):
                for k in self.additional_label_category[i].categories:
                    if k not in self.additional_label_category_summary[i].keys() and k != self.unlabel_key:
                        self.additional_label_category_summary[i][k] = 0

            self.additional_label_category_weight = [len(label_category) / torch.tensor([
                self.additional_label_category_summary[e][x] for x in list(filter(lambda x:
                    x != self.unlabel_key,
                    label_category.categories
                ))], dtype=torch.float64).to(self.device) for e,label_category in enumerate(self.additional_label_category)]



            self.additional_new_adata_code = [list(x.categories).index(self.unlabel_key) if self.unlabel_key in x.categories else -1 for x in self.additional_label_category]

        self.n_additional_batch_ = None
        if self.additional_batch_keys is not None:
            self.n_additional_batch_ = [len(np.unique(self.adata.obs[x])) for x in self.additional_batch_keys]
            if self.n_additional_batch == None or len(self.n_additional_batch_) != len(self.n_additional_batch):
                mt(f"warning: the provided n_additional_batch={self.n_additional_batch} does not match the number of categorical covariate in the adata.")

                if self.constrain_n_batch:
                    mt(f"         setting n_additional_batch to {self.n_additional_batch_}")
                    self.n_additional_batch = self.n_additional_batch_
            else:
                for e,(i,j) in enumerate(zip(self.n_additional_batch_, self.n_additional_batch)):
                    if i != j:
                        mt(f"n_additional_batch {self.additional_batch_keys[e]} does not match the number in the adata.")

                        if self.constrain_n_batch:
                            mt(f"         setting n_additional_batch {e} to {i}")
                            self.n_additional_batch[e] = i
            self.additional_batch_category = [pd.Categorical(self.adata.obs[x]) for x in self.additional_batch_keys]
            self.additional_batch_category_summary = [dict(Counter(x)) for x in self.additional_batch_category]
            for i in range(len(self.additional_batch_category_summary)):
                for k in self.additional_batch_category[i].categories:
                    if k not in self.additional_batch_category_summary[i].keys():
                        self.additional_batch_category_summary[i][k] = 0

        X = self.adata.X

        self._n_record = X.shape[0]
        self._indices = np.array(list(range(self._n_record)))
        batch_categories, label_categories = None, None
        additional_label_categories = None
        additional_batch_categories = None

        if self.batch_key is not None:
            if self.batch_key not in self.adata.obs.columns:
                raise ValueError(f"batch_key {self.batch_key} is not found in AnnData obs")
            batch_categories = np.array(self.batch_category.codes)
        if self.label_key is not None:
            if self.label_key not in self.adata.obs.columns:
                raise ValueError(f"label_key {self.label_key} is not found in AnnData obs")
            label_categories = np.array(self.label_category.codes)
        if self.additional_label_keys is not None:
            for e,i in enumerate(self.additional_label_keys):
                if i not in self.adata.obs.columns:
                    raise ValueError(f"additional_label_keys {i} is not found in AnnData obs")
            additional_label_categories = [np.array(x.codes) for x in self.additional_label_category]
        if self.additional_batch_keys is not None:
            for e,i in enumerate(self.additional_batch_keys):
                if i not in self.adata.obs.columns:
                    raise ValueError(f"additional_batch_keys {i} is not found in AnnData obs")
            additional_batch_categories = [np.array(x.codes) for x in self.additional_batch_category]


        if self.constrain_latent_embedding and self.constrain_latent_key in self.adata.obsm.keys():
            P = self.adata.obsm[self.constrain_latent_key]
            if additional_batch_categories is not None:
                if batch_categories is not None and label_categories is not None and additional_label_categories is not None:
                    _dataset = list(zip(X, P, batch_categories, label_categories, *additional_label_categories, *additional_batch_categories))
                elif batch_categories is not None and label_categories is not None:
                    _dataset = list(zip(X, P, batch_categories, label_categories, *additional_batch_categories))
                elif batch_categories is not None:
                    _dataset = list(zip(X, P, batch_categories, *additional_batch_categories))
                elif label_categories is not None:
                    _dataset = list(zip(X, P, label_categories, *additional_batch_categories))
                else:
                    _dataset = list(zip(X, P, *additional_batch_categories))
            else:
                if batch_categories is not None and label_categories is not None and additional_label_categories is not None:
                    _dataset = list(zip(X, P, batch_categories, label_categories, *additional_label_categories))
                elif batch_categories is not None and label_categories is not None:
                    _dataset = list(zip(X, P, batch_categories, label_categories))
                elif batch_categories is not None:
                    _dataset = list(zip(X, P, batch_categories))
                elif label_categories is not None:
                    _dataset = list(zip(X, P, label_categories))
                else:
                    _dataset = list(zip(X, P))
        else:
            if additional_batch_categories is not None:
                if batch_categories is not None and label_categories is not None and additional_label_categories is not None:
                    _dataset = list(zip(X, batch_categories, label_categories, *additional_label_categories, *additional_batch_categories))
                elif batch_categories is not None and label_categories is not None:
                    _dataset = list(zip(X, batch_categories, label_categories, *additional_batch_categories))
                elif batch_categories is not None:
                    _dataset = list(zip(X, batch_categories, *additional_batch_categories))
                elif label_categories is not None:
                    _dataset = list(zip(X, label_categories, *additional_batch_categories))
                else:
                    _dataset = list(zip(X, *additional_batch_categories))
            else:
                if batch_categories is not None and label_categories is not None and additional_label_categories is not None:
                    _dataset = list(zip(X, batch_categories, label_categories, *additional_label_categories))
                elif batch_categories is not None and label_categories is not None:
                    _dataset = list(zip(X, batch_categories, label_categories))
                elif batch_categories is not None:
                    _dataset = list(zip(X, batch_categories))
                elif label_categories is not None:
                    _dataset = list(zip(X, label_categories))
                else:
                    _dataset = list(X)


        _shuffle_indices = list(range(len(_dataset)))
        np.random.shuffle(_shuffle_indices)
        self._dataset = np.array([_dataset[i] for i in _shuffle_indices])

        self._shuffle_indices = np.array(
            [x for x, _ in sorted(zip(range(len(_dataset)), _shuffle_indices), key=lambda x: x[1])]
        )

        self._shuffled_indices_inverse = _shuffle_indices


    def as_dataloader(
        self,
        subset_indices: Union[torch.tensor, np.ndarray] = None,
        n_per_batch: int = 128,
        train_test_split: bool = False,
        random_seed: bool = 42,
        validation_split: bool = .2,
        shuffle: bool = True,
    ):
        indices = subset_indices if subset_indices is not None else self._indices
        np.random.seed(random_seed)
        if shuffle:
            np.random.shuffle(indices)
        if train_test_split:
            split = int(np.floor(validation_split * self._n_record))
            if split % n_per_batch == 1:
                n_per_batch -= 1
            elif (self._n_record - split) % n_per_batch == 1:
                n_per_batch += 1
            train_indices, val_indices = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)
            return DataLoader(indices, n_per_batch,  sampler = train_sampler), DataLoader(indices, n_per_batch, sampler = valid_sampler)
        if len(indices) % n_per_batch == 1:
            n_per_batch -= 1
        return DataLoader(indices, n_per_batch, shuffle = shuffle)

    def _normalize_data(self, X, after=None, copy=True):
        X = X.clone() if copy else X
        X = X.to(torch.float32)  # Check if torch.float64 should be used
        counts = X.sum(axis=1)
        counts_greater_than_zero = counts[counts > 0]
        after = torch.median(counts_greater_than_zero, dim=0).values if after is None else after
        counts += counts == 0
        counts = counts / after
        X /= counts.unsqueeze(1)
        return X
    

    def encode(self, X: torch.Tensor, batch_index: torch.Tensor = None, eps: float = 1e-4):
        # Encode for hidden space
        # if batch_index is not None and self.inject_batch:
        #    X = torch.hstack([X, batch_index])
        libsize = torch.log(X.sum(1))
        if self.reconstruction_method == 'zinb':
            if self.total_variational:
                X = self._normalize_data(X, after=1e4, copy=True)
            if self.log_variational:
                X = torch.log(1+X)
        q = self.encoder.encode(torch.hstack([X,libsize.unsqueeze(1)])) if self.encode_libsize else self.encoder.encode(X)
        q_mu = self.z_mean_fc(q)
        q_var = torch.exp(self.z_var_fc(q)) + eps
        z = Normal(q_mu, q_var.sqrt()).rsample()
        H = dict(
            q = q,
            q_mu = q_mu,
            q_var = q_var,
            z = z
        )

        return H

    def decode(self,
        H: Mapping[str, torch.tensor],
        lib_size:torch.tensor,
        batch_index: torch.Tensor = None,
        label_index: torch.Tensor = None,
        additional_batch_index: torch.Tensor = None,
        eps: float = 1e-4
    ):
        z = H["z"] # cell latent representation

        if additional_batch_index is not None and self.inject_additional_batch:
            if batch_index is not None and label_index is not None and self.inject_batch and self.inject_label:
                z = torch.hstack([z, batch_index, label_index, *additional_batch_index])
            elif batch_index is not None and self.inject_batch:
                z = torch.hstack([z, batch_index, *additional_batch_index])
            elif label_index is not None and self.inject_label:
                z = torch.hstack([z, label_index, *additional_batch_index])
        else:
            if batch_index is not None and label_index is not None and self.inject_batch and self.inject_label:
                z = torch.hstack([z, batch_index, label_index])
            elif batch_index is not None and self.inject_batch:
                z = torch.hstack([z, batch_index])
            elif label_index is not None and self.inject_label:
                z = torch.hstack([z, label_index])

        # eps to prevent numerical overflow and NaN gradient
        px = self.decoder(z)
        h = None

        px_rna_scale = self.px_rna_scale_decoder(px)
        if self.decode_libsize and not self.reconstruction_method == 'mse':
            px_rna_scale_final = px_rna_scale * lib_size.unsqueeze(1)
        elif self.reconstruction_method == 'mse':
            px_rna_scale_final = torch.log(px_rna_scale * 1e4 + 1)
        else:
            px_rna_scale_final = px_rna_scale

        if self.dispersion == "gene-cell":
            px_rna_rate = self.px_rna_rate_decoder(px) ## In logits
        elif self.dispersion == "gene-batch":
            px_rna_rate = F.linear(one_hot(batch_index, self.n_batch), self.px_rate)
        elif self.dispersion == "gene":
            px_rna_rate = self.px_rate

        px_rna_dropout = self.px_rna_dropout_decoder(px)  ## In logits

        R = dict(
            h = h,
            px = px,
            px_rna_scale_orig = px_rna_scale,
            px_rna_scale = px_rna_scale_final,
            px_rna_rate = px_rna_rate,
            px_rna_dropout = px_rna_dropout
        )
        return R

    def forward(
        self,
        X: torch.Tensor,
        lib_size: torch.Tensor,
        batch_index: torch.Tensor = None,
        label_index: torch.Tensor = None,
        additional_label_index: torch.Tensor = None,
        additional_batch_index: torch.Tensor = None,
        P: torch.Tensor = None,
        reduction: str = "sum",
        compute_mmd: bool = False
    ):
        H = self.encode(X, batch_index)
        q_mu = H["q_mu"]
        q_var = H["q_var"]
        mean = torch.zeros_like(q_mu)
        scale = torch.ones_like(q_var)
        kldiv_loss = kld(Normal(q_mu, q_var.sqrt()), Normal(mean, scale)).sum(dim = 1)
        prediction_loss = torch.tensor(0., device=self.device)
        additional_prediction_loss = torch.tensor(0., device=self.device)
        R = self.decode(H, lib_size, batch_index, label_index, additional_batch_index)

        if self.reconstruction_method == 'zinb':
            reconstruction_loss = LossFunction.zinb_reconstruction_loss(
                X,
                mu = R['px_rna_scale'],
                theta = R['px_rna_rate'].exp(),
                gate_logits = R['px_rna_dropout'],
                reduction = reduction
            )
        elif self.reconstruction_method == 'zg':
            reconstruction_loss = LossFunction.zi_gaussian_reconstruction_loss(
                X,
                mean=R['px_rna_scale'],
                variance=R['px_rna_rate'].exp(),
                gate_logits=R['px_rna_dropout'],
                reduction=reduction
            )
        elif self.reconstruction_method == 'mse':
            X_norm = self._normalize_data(X, after=1e4)
            X_norm = torch.log(X_norm + 1)
            reconstruction_loss = nn.functional.mse_loss(
                X_norm,
                R['px_rna_scale'],
                reduction=reduction
            )

        if self.n_label > 0:
            criterion = nn.CrossEntropyLoss(weight=self.label_category_weight)

            prediction = self.fc(H['z'])

            if self.new_adata_code and self.new_adata_code in label_index:
                prediction_index = (label_index != self.new_adata_code).squeeze()
                prediction_loss = criterion(prediction[prediction_index], one_hot(label_index[prediction_index], self.n_label))
            else:
                prediction_loss = criterion(prediction, one_hot(label_index, self.n_label))

        if self.n_additional_label is not None:
            prediction_loss = prediction_loss * self.additional_label_weight[0]
            for e,i in enumerate(self.n_additional_label):
                criterion = nn.CrossEntropyLoss(weight=self.additional_label_category_weight[e])
                additional_prediction = self.additional_fc[e](H['z'])
                if self.additional_new_adata_code[e] and self.additional_new_adata_code[e] in additional_label_index[e]:
                    additional_prediction_index = (additional_label_index[e] != self.additional_new_adata_code[e]).squeeze()

                    additional_prediction_loss += criterion(
                        additional_prediction[additional_prediction_index],
                        one_hot(additional_label_index[e][additional_prediction_index], i) * self.additional_label_weight[e+1]
                    )
                else:
                    additional_prediction_loss += criterion(additional_prediction, one_hot(additional_label_index[e], i)) * self.additional_label_weight[e+1]

        latent_constrain = torch.tensor(0.)
        if self.constrain_latent_embedding and P is not None:
            # Constrains on cells with no PCA information will be ignored
            latent_constrain_mask = P.mean(1) != 0
            if self.constrain_latent_method == 'mse':
                latent_constrain = (
                    nn.MSELoss(reduction='none')(P, q_mu).sum(1) * latent_constrain_mask
                ).sum() / len(list(filter(lambda x: x != 0, P.detach().cpu().numpy().mean(1))))
            elif self.constrain_latent_method == 'normal':
                latent_constrain = (
                    kld(Normal(q_mu, q_var.sqrt()), Normal(P, torch.ones_like(P))).sum(1) * latent_constrain_mask
                ).sum() / len(list(filter(lambda x: x != 0, P.detach().cpu().numpy().mean(1))))

        mmd_loss = torch.tensor(0.)
        if self.mmd_key is not None and compute_mmd:
            if self.mmd_key == 'batch':
                mmd_loss = self.mmd_loss(
                    H['q_mu'],
                    batch_index.detach().cpu().numpy(),
                    dim=1
                )
            elif self.mmd_key == 'additional_batch':
                for i in range(len(self.additional_batch_keys)):
                    mmd_loss_st += self.mmd_loss(
                        H['q_mu'], 
                        additional_batch_index[i].detach().cpu().numpy(),
                        dim=1
                    )
            elif self.mmd_key == 'both':
                mmd_loss_st = self.mmd_loss(
                    H['q_mu'], 
                    batch_index.detach().cpu().numpy(),
                    dim=1
                )
                for i in range(len(self.additional_batch_keys)):
                    mmd_loss_st += self.mmd_loss(
                        H['q_mu'], 
                        additional_batch_index[i].detach().cpu().numpy(),
                        dim=1
                    )

        loss_record = {
            "reconstruction_loss": reconstruction_loss,
            "prediction_loss": prediction_loss,
            "additional_prediction_loss": additional_prediction_loss,
            "kldiv_loss": kldiv_loss,
            "mmd_loss": mmd_loss,
            "latent_constrain_loss": latent_constrain
        }
        return H, R, loss_record

    def calculate_metric(self, X_test, kl_weight, pred_weight, mmd_weight, reconstruction_reduction):
        epoch_total_loss = 0
        epoch_reconstruction_loss = 0
        epoch_kldiv_loss = 0
        epoch_prediction_loss = 0
        epoch_mmd_loss = 0
        b = 0
        with torch.no_grad():
            for b, X in enumerate(X_test):
                X, P, batch_index, label_index, additional_label_index, additional_batch_index, lib_size = self._prepare_batch(X)

                H, R, L = self.forward(
                    X,
                    lib_size,
                    batch_index,
                    label_index,
                    additional_label_index,
                    additional_batch_index,
                    P,
                    reduction=reconstruction_reduction,
                    compute_mmd = mmd_weight > 0
                )
                reconstruction_loss = L['reconstruction_loss']
                prediction_loss = pred_weight * L['prediction_loss']
                additional_prediction_loss = pred_weight * L['additional_prediction_loss']
                kldiv_loss = kl_weight * L['kldiv_loss']
                mmd_loss = mmd_weight * L['mmd_loss']

                avg_reconstruction_loss = reconstruction_loss.sum()
                avg_kldiv_loss = kldiv_loss.sum()
                avg_mmd_loss = mmd_loss

                epoch_reconstruction_loss += avg_reconstruction_loss.item()
                epoch_kldiv_loss += avg_kldiv_loss.item()
                if self.n_label > 0:
                    epoch_prediction_loss += prediction_loss.sum().item()
                if self.n_additional_label is not None:
                    epoch_prediction_loss += additional_prediction_loss.sum().item()

                epoch_mmd_loss += avg_mmd_loss
                epoch_total_loss += (avg_reconstruction_loss + avg_kldiv_loss + avg_mmd_loss).item()
        return {
            "epoch_reconstruction_loss":  epoch_reconstruction_loss / (b+1),
            "epoch_kldiv_loss": epoch_kldiv_loss / (b+1),
            "epoch_mmd_loss": epoch_mmd_loss / (b+1),
            "epoch_prediction_loss": epoch_prediction_loss / (b+1),
            "epoch_total_loss": epoch_total_loss / (b+1),
        }


    def fit(self,
        max_epoch: Optional[int] = None,
        n_per_batch:int = 128,
        kl_weight: float = 1.,
        pred_weight: float = 1.,
        mmd_weight: float = 1.,
        constrain_weight: float = 1.,
        optimizer_parameters: Iterable = None,
        validation_split: float = .2,
        lr: bool = 5e-5,
        lr_schedule: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_min: float = 1e-6,
        n_epochs_kl_warmup: Union[int, None] = 400,
        weight_decay: float = 1e-6,
        random_seed: int = 12,
        subset_indices: Union[torch.tensor, np.ndarray] = None,
        pred_last_n_epoch: int = 10,
        pred_last_n_epoch_fconly: bool = False,
        reconstruction_reduction: str = 'sum',
    ):
        """
        Fit the model.
        
        :param max_epoch: int. Maximum number of epoch to train the model. If not provided, the model will be trained for 400 epochs or 20000 / n_record * 400 epochs as default.
        :param n_per_batch: int. Number of cells per batch.
        :param kl_weight: float. (Maximum) weight of the KL divergence loss.
        :param pred_weight: float. weight of the prediction loss.
        :param mmd_weight: float. weight of the mmd loss. ignored if mmd_key is None
        :param constrain_weight: float. weight of the constrain loss. ignored if constrain_latent_embedding is False. 
        :param optimizer_parameters: Iterable. Parameters to be optimized. If not provided, all parameters will be optimized.
        :param validation_split: float. Percentage of data to be used as validation set.
        :param lr: float. Learning rate.
        :param lr_schedule: bool. Whether to use learning rate scheduler.
        :param lr_factor: float. Factor to reduce learning rate.
        :param lr_patience: int. Number of epoch to wait before reducing learning rate.
        :param lr_threshold: float. Threshold to trigger learning rate reduction.
        :param lr_min: float. Minimum learning rate.
        :param n_epochs_kl_warmup: int. Number of epoch to warmup the KL divergence loss (deterministic warm-up
        of the KL-term).
        :param weight_decay: float. Weight decay (L2 penalty).
        :param random_seed: int. Random seed.
        :param subset_indices: Union[torch.tensor, np.ndarray]. Indices of cells to be used for training. If not provided, all cells will be used.
        :param pred_last_n_epoch: int. Number of epoch to train the prediction layer only.
        :param pred_last_n_epoch_fconly: bool. Whether to train the prediction layer only.
        :param reconstruction_reduction: str. Reduction method for reconstruction loss. Can be 'sum' or 'mean'.
        """
        self.train()
        if max_epoch is None:
            max_epoch = np.min([round((20000 / self._n_record ) * 400), 400])
            mt(f"max_epoch is not provided, setting max_epoch to {max_epoch}")
        if n_epochs_kl_warmup:
            n_epochs_kl_warmup = min(max_epoch, n_epochs_kl_warmup)
            kl_warmup_gradient = kl_weight / n_epochs_kl_warmup
            kl_weight_max = kl_weight
            kl_weight = 0.

        if optimizer_parameters is None:
            optimizer = optim.AdamW(self.parameters(), lr, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(optimizer_parameters, lr, weight_decay=weight_decay)

        scheduler = ReduceLROnPlateau(
            optimizer,
            patience=lr_patience,
            factor=lr_factor,
            threshold=lr_threshold,
            min_lr=lr_min,
            threshold_mode="abs",
            verbose=True,
        ) if lr_schedule else None

        labels=None

        best_state_dict = None
        best_score = 0
        current_score = 0
        pbar = get_tqdm()(range(max_epoch), desc="Epoch", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        loss_record = {
            "epoch_reconstruction_loss": 0,
            "epoch_kldiv_loss": 0,
            "epoch_prediction_loss": 0,
            "epoch_mmd_loss": 0,
            "epoch_total_loss": 0
        }

        epoch_total_loss_list = []
        epoch_reconstruction_loss_list = []
        epoch_kldiv_loss_list = []
        epoch_prediction_loss_list = []
        epoch_mmd_loss_list = []


        for epoch in range(1, max_epoch+1):
            self._trained = True
            pbar.desc = "Epoch {}".format(epoch)
            epoch_total_loss = 0
            epoch_reconstruction_loss = 0
            epoch_kldiv_loss = 0
            epoch_prediction_loss = 0
            epoch_mmd_loss = 0
            X_train, X_test = self.as_dataloader(
                n_per_batch=n_per_batch,
                train_test_split = True,
                validation_split = validation_split,
                random_seed=random_seed,
                subset_indices=subset_indices
            )
            if self.n_label > 0 and epoch+1 == max_epoch - pred_last_n_epoch:
                mt("saving transcriptome only state dict")
                self.gene_only_state_dict = deepcopy(self.state_dict())
                if  pred_last_n_epoch_fconly:
                    optimizer = optim.AdamW(chain(self.att.parameters(), self.fc.parameters()), lr, weight_decay=weight_decay)

            for b, X in enumerate(X_train):
                X, P, batch_index, label_index, additional_label_index, additional_batch_index, lib_size = self._prepare_batch(X)

                H, R, L = self.forward(
                    X,
                    lib_size,
                    batch_index,
                    label_index,
                    additional_label_index,
                    additional_batch_index,
                    P,
                    reduction=reconstruction_reduction,
                    compute_mmd = mmd_weight > 0
                )

                reconstruction_loss = L['reconstruction_loss']
                prediction_loss = pred_weight * L['prediction_loss']
                additional_prediction_loss = pred_weight * L['additional_prediction_loss']
                kldiv_loss = kl_weight * L['kldiv_loss']
                mmd_loss = mmd_weight * L['mmd_loss']

                avg_reconstruction_loss = reconstruction_loss.sum()  / n_per_batch
                avg_kldiv_loss = kldiv_loss.sum()  / n_per_batch
                avg_mmd_loss = mmd_loss / n_per_batch

                epoch_reconstruction_loss += avg_reconstruction_loss.item()
                epoch_kldiv_loss += avg_kldiv_loss.item()
                if self.n_label > 0:
                    epoch_prediction_loss += prediction_loss.sum().item()

                if epoch > max_epoch - pred_last_n_epoch:
                    loss = avg_reconstruction_loss + avg_kldiv_loss + avg_mmd_loss + prediction_loss.sum() + additional_prediction_loss.sum()
                else:
                    loss = avg_reconstruction_loss + avg_kldiv_loss + avg_mmd_loss

                if self.constrain_latent_embedding:
                    loss += constrain_weight * L['latent_constrain_loss']

                epoch_total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix({
                    'reconst': '{:.2e}'.format(loss_record["epoch_reconstruction_loss"]),
                    'kldiv': '{:.2e}'.format(loss_record["epoch_kldiv_loss"]),
                    'pred': '{:.2e}'.format(loss_record["epoch_prediction_loss"]),
                    'mmd': '{:.2e}'.format(loss_record["epoch_mmd_loss"]),
                    'step': f'{b} / {len(X_train)}'
                })
            loss_record = self.calculate_metric(X_test, kl_weight, pred_weight, mmd_weight, reconstruction_reduction)
            if lr_schedule:
                scheduler.step(loss_record["epoch_total_loss"])
            pbar.set_postfix({
                'reconst': '{:.2e}'.format(loss_record["epoch_reconstruction_loss"]),
                'kldiv': '{:.2e}'.format(loss_record["epoch_kldiv_loss"]),
                'pred': '{:.2e}'.format(loss_record["epoch_prediction_loss"]),
                'mmd': '{:.2e}'.format(loss_record["epoch_mmd_loss"]),
            })
            epoch_total_loss_list.append(epoch_total_loss)
            epoch_reconstruction_loss_list.append(epoch_reconstruction_loss)
            epoch_kldiv_loss_list.append(epoch_kldiv_loss)
            epoch_prediction_loss_list.append(epoch_prediction_loss)
            epoch_mmd_loss_list.append(epoch_mmd_loss)
            pbar.update(1)
            if n_epochs_kl_warmup:
                kl_weight = min( kl_weight + kl_warmup_gradient, kl_weight_max)
            random_seed += 1
        if current_score < best_score:
            mt("restoring state dict with best performance")
            self.load_state_dict(best_state_dict)
        pbar.close()
        self.trained_state_dict = deepcopy(self.state_dict())
        return dict(
            epoch_total_loss_list=epoch_total_loss_list,   
            epoch_reconstruction_loss_list=epoch_reconstruction_loss_list,
            epoch_kldiv_loss_list=epoch_kldiv_loss_list,
            epoch_prediction_loss_list=epoch_prediction_loss_list,
            epoch_mmd_loss_list=epoch_mmd_loss_list
        )
    

    @torch.no_grad()
    def predict_labels(
        self,
        n_per_batch: int = 128,
        return_pandas: bool = False,
        show_progress: bool = True
    ) -> List:
        """
        Predict labels from trained model. 

        :param n_per_batch: int. Number of cells for each mini-batch during inference.
        :param return_pandas: bool. return a pandas DataFrame if True else return a pytorch tensor.
        :param show_progress: bool. Show progress bar of total progress.
        """
        self.eval()
        X = self.as_dataloader(subset_indices = list(range(len(self._dataset))), shuffle=False, n_per_batch=n_per_batch)
        predictions = []
        additional_predictions = []
        if show_progress:
            pbar = get_tqdm()(X, desc="Predicting Labels", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        for x in X:
            x = self._dataset[x.cpu().numpy()]
            batch_index = None
            label_index = None
            if self.n_batch > 0 or self.n_label > 0:
                if not isinstance(x, Iterable) and len(x) > 1:
                    raise ValueError()
                if self.n_batch > 0 and self.n_label > 0:
                    X, batch_index, label_index = get_k_elements(x,0), get_k_elements(x,1), get_k_elements(X,2)
                elif self.n_batch > 0:
                    X, batch_index = get_k_elements(x,0), get_k_elements(x,1)
                elif self.n_label > 0:
                    X, label_index = get_k_elements(x,0), get_k_elements(x,1)

            if self.n_label > 0:
                label_index = torch.tensor(label_index)
                if not isinstance(label_index, torch.FloatTensor):
                    label_index = label_index.type(torch.FloatTensor)
                label_index = label_index.to(self.device).unsqueeze(1)
            if self.n_batch > 0:
                batch_index = torch.tensor(batch_index)
                if not isinstance(batch_index, torch.FloatTensor):
                    batch_index = batch_index.type(torch.FloatTensor)
                batch_index = batch_index.to(self.device).unsqueeze(1)

            X = torch.tensor(np.vstack(list(map(lambda x: x.toarray() if issparse(x) else x, X))))
            if not isinstance(X, torch.FloatTensor):
                X = X.type(torch.FloatTensor)
            X = X.to(self.device)

            H = self.encode(X, batch_index if batch_index != None else None)
            prediction = self.fc(H['z'])
            predictions.append(prediction.detach().cpu())

            if self.n_additional_label is not None:
                additional_prediction = [None] * len(self.n_additional_label)
                for i in range(len(self.n_additional_label)):
                    additional_prediction[i] = self.additional_fc[i](H['z']).detach().cpu()
                additional_predictions.append(additional_prediction)

            if show_progress:
                pbar.update(1)
        if show_progress:
            pbar.close()

        predictions = torch.vstack(predictions)[self._shuffle_indices]
        predictions_argmax = torch.argmax(predictions, dim=1)
        predictions_argmax = list(map(lambda x:
            self.label_category.categories[x],
            predictions_argmax.detach().cpu().numpy()
        ))

        if return_pandas:
            predictions_argmax = pd.DataFrame(predictions_argmax)
            predictions_argmax.columns = [self.label_key]
            return predictions_argmax

        if self.n_additional_label is not None:
            additional_predictions_result = [None] * len(self.n_additional_label)
            additional_predictions_result_argmax = [None] * len(self.n_additional_label)
            for i in range(len(self.n_additional_label)):
                additional_predictions_ = torch.vstack([additional_predictions[x][i] for x in range(len(additional_predictions))]) [self._shuffle_indices]
                additional_predictions_result_argmax[i] = np.argmax(additional_predictions_, axis=1)
                additional_predictions_result_argmax[i] = list(map(lambda x:
                    self.additional_label_category[i].categories[x],
                    additional_predictions_result_argmax[i].numpy()
                ))
            if return_pandas:
                additional_predictions_result_argmax = pd.DataFrame(additional_predictions_result_argmax).T
                additional_predictions_result_argmax.columns = self.additional_label_keys
                return pd.concat([predictions_argmax, additional_predictions_result_argmax], axis=1)

            return predictions, additional_predictions

        return predictions


    @torch.no_grad()
    def get_latent_embedding(
        self,
        latent_key: Literal["z", "q_mu"] = "q_mu",
        n_per_batch: int = 128,
        show_progress: bool = True
    ) -> np.ndarray:
        self.eval()
        X = self.as_dataloader(
            subset_indices = list(range(len(self._dataset))), 
            shuffle=False, 
            n_per_batch=n_per_batch
        )
        if isinstance(latent_key, str):
            Zs = []
        elif isinstance(latent_key, Iterable):
            Zs = [[] for _ in range(len(latent_key))]
        if show_progress:
            pbar = get_tqdm()(X, desc="Latent Embedding", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        for x in X:
            X, P, batch_index, label_index, additional_label_index, additional_batch_index, lib_size = self._prepare_batch(x)

            H = self.encode(X, batch_index if batch_index != None else None)
            if isinstance(latent_key, str):
                Zs.append(H[latent_key].detach().cpu().numpy())
            elif isinstance(latent_key, Iterable):
                for i in range(len(latent_key)):
                    Zs[i].append(H[latent_key[i]].detach().cpu().numpy())
            if show_progress:
                pbar.update(1)
        if show_progress:
            pbar.close()
        if isinstance(latent_key, str):
            return np.vstack(Zs)[self._shuffle_indices]
        elif isinstance(latent_key, Iterable):
            return [np.vstack(Z)[self._shuffle_indices] for Z in Zs]

    @torch.no_grad()
    def get_reconstructed_expression(self, k = 'px_rna_scale_orig', n_per_batch=256,show_progress=True) -> np.ndarray:
        self.eval()
        Zs = []
        X = self.as_dataloader(subset_indices = list(range(len(self._dataset))), shuffle=False, n_per_batch=n_per_batch)
        predictions = []
        additional_predictions = []
        if show_progress:
            pbar = get_tqdm()(X, desc="Predicting Labels", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        for x in X:
            X, P, batch_index, label_index, additional_label_index, additional_batch_index, lib_size = self._prepare_batch(x)

            _,R,_ = self.forward(
                X,
                lib_size,
                batch_index,
                label_index,
                additional_label_index,
                additional_batch_index,
                P,
            )
            Zs.append(R[k].detach().cpu().numpy())
            if show_progress:
                pbar.update(1)
        return np.vstack(Zs)[self._shuffle_indices]


    def to(self, device:str):
        super(scAtlasVAE, self).to(device)
        self.device=device
        return self

    def transfer(self,
        new_adata: sc.AnnData,
        batch_key: str,
        concat_with_original: bool = True,
        fraction_of_original: Optional[float] = None,
        times_of_new: Optional[float] = None
    ):
        new_batch_category = new_adata.obs[batch_key]
        original_batch_dim = self.batch_hidden_dim
        new_n_batch = len(np.unique(new_batch_category))

        if self.batch_embedding == "embedding":
            original_embedding_weight = self.decoder.cat_embedding[0].weight

        new_adata.obs[self.batch_key] = new_adata.obs[batch_key]

        original_batch_categories = self.batch_category.categories

        if fraction_of_original is not None:
            old_adata = random_subset_by_key_fast(
                self.adata,
                key = batch_key,
                n = int(len(self.adata) * fraction_of_original)
            )
        elif times_of_new is not None:
            old_adata = random_subset_by_key_fast(
                self.adata,
                key = batch_key,
                n = int(len(new_adata) * times_of_new)
            )
        else:
            old_adata = self.adata

        old_adata.obs['_transfer_label'] = 'reference'
        new_adata.obs['_transfer_label'] = 'query'

        if concat_with_original:
            self.adata = sc.concat([old_adata, new_adata])
        else:
            self.adata = new_adata

        self.initialize_dataset()

        if self.batch_embedding == "onehot":
            self.batch_hidden_dim = self.n_batch

        if self.n_additional_batch_ is not None and self.inject_additional_batch:
            if self.n_batch > 0 and self.n_label > 0 and self.inject_batch and self.inject_label:
                decoder_n_cat_list = [self.n_batch, self.n_label, *self.n_additional_batch]
            elif self.n_batch > 0 and self.inject_batch:
                decoder_n_cat_list = [self.n_batch, *self.n_additional_batch]
            elif self.n_label > 0 and self.inject_label:
                decoder_n_cat_list = [self.n_label, *self.n_additional_batch]
            else:
                decoder_n_cat_list = None
        else:
            if self.n_batch > 0 and self.n_label > 0 and self.inject_batch and self.inject_label:
                decoder_n_cat_list = [self.n_batch, self.n_label]
            elif self.n_batch > 0 and self.inject_batch:
                decoder_n_cat_list = [self.n_batch]
            elif self.n_label > 0 and self.inject_label:
                decoder_n_cat_list = [self.n_label]
            else:
                decoder_n_cat_list = None


        self.decoder_n_cat_list = decoder_n_cat_list

        original_weight = torch.tensor(self.decoder._fclayer[0].weight)

        self.decoder = FCLayer(
            in_dim = self.n_latent,
            out_dim = self.n_hidden,
            n_cat_list = self.decoder_n_cat_list,
            cat_dim = self.batch_hidden_dim,
            cat_embedding = self.batch_embedding,
            use_layer_norm=False,
            use_batch_norm=True,
            dropout_rate=0,
            device=self.device
        )


        if self.batch_embedding == 'embedding':
            new_embedding = nn.Embedding(self.n_batch + new_n_batch, self.batch_hidden_dim).to(self.device)
            original_category_index = [list(self.batch_category.categories).index(x) for x in original_batch_categories]
            new_embedding_weight = new_embedding.weight.detach()
            new_embedding_weight[original_category_index] = original_embedding_weight.detach()
            new_embedding.weight = nn.Parameter(new_embedding_weight)
            new_embedding = new_embedding.to(self.device)
            self.decoder.cat_embedding[0] = new_embedding

        new_weight = torch.tensor(self.decoder._fclayer[0].weight)
        new_weight[:,:(self.n_latent + original_batch_dim)] = original_weight[:,:(self.n_latent + original_batch_dim)]
        self.decoder._fclayer[0].weight = nn.Parameter(new_weight)
        self.to(self.device)


    def transfer_label(
        self,
        reference_adata: sc.AnnData,
        label_key: str,
        method: Literal['knn'] = 'knn',
        use_rep: str = 'X_gex',
        **method_kwargs
    ):
        """
        Transfer label from reference_adata to self.adata

        :param reference_adata: sc.AnnData
        :param label_key: str
        """
        s = set(reference_adata.obs.index)
        s = list(filter(lambda x: x in s, self.adata.obs.index))

        self.adata.obs[label_key] = np.nan
        ss = set(s)
        indices = list(map(lambda x: x in ss, self.adata.obs.index))
        self.adata.obs[label_key][indices] = reference_adata[s].obs[label_key]

        if 'X_gex' not in self.adata.obsm.keys():
            Z = self.get_latent_embedding()
            self.adata.obsm[use_rep] = Z

        if method == 'knn':
            from sklearn.neighbors import KNeighborsClassifier
            knn = KNeighborsClassifier(**method_kwargs)
            knn.fit(
                self.adata.obsm[use_rep][indices], self.adata.obs.loc[indices, label_key]
            )
            self.adata.obs.loc[self.adata.obs['_transfer_label'] == 'query', label_key] = knn.predict(
                self.adata.obsm[use_rep][self.adata.obs['_transfer_label'] == 'query']
            )
        else:
            raise NotImplementedError()

    def umap_alignment(
        self,
        reference_adata: sc.AnnData,
        label_key: str,
        method: Literal['retrain','knn'] = 'knn',
        use_rep: str = 'X_gex',
        **method_kwargs
    ):
        umap_alignment(
            reference_adata.obsm[use_rep],
            reference_adata.obsm['X_umap'],
            reference_adata.obsm[use_rep],
            method=method,

        )

    def _prepare_batch(self, X):
                P = None
                X = self._dataset[X.cpu().numpy()]
                batch_index, label_index, additional_label_index, additional_batch_index = None, None, None, None
                if self.n_batch > 0 or self.n_label > 0:
                    if not isinstance(X, Iterable) and len(X) > 1:
                        raise ValueError()
                    if self.n_additional_batch_ is not None:
                        if self.n_batch > 0 and self.n_label > 0 and self.n_additional_label is not None:
                            if self.constrain_latent_embedding:
                                X, P, batch_index, label_index, additional_label_index, additional_batch_index = (
                                    get_k_elements(X,0),
                                    get_k_elements(X,1),
                                    get_k_elements(X,2),
                                    get_k_elements(X,3),
                                    get_elements(X,4, len(self.n_additional_label)),
                                    get_last_k_elements(X,4+len(self.n_additional_label))
                                )
                            else:
                                X, batch_index, label_index, additional_label_index, additional_batch_index = (
                                    get_k_elements(X,0),
                                    get_k_elements(X,1),
                                    get_k_elements(X,2),
                                    get_elements(X,3, len(self.n_additional_label)),
                                    get_last_k_elements(X,3+len(self.n_additional_label))
                                )
                            additional_label_index = list(np.vstack(additional_label_index).T.astype(int))
                        elif self.n_batch > 0 and self.n_label > 0:
                            if self.constrain_latent_embedding:
                                X, P, batch_index, label_index, additional_batch_index = (
                                    get_k_elements(X,0),
                                    get_k_elements(X,1),
                                    get_k_elements(X,2),
                                    get_k_elements(X,3),
                                    get_last_k_elements(X,4)
                                )
                            else:
                                X, batch_index, label_index, additional_batch_index = (
                                    get_k_elements(X,0),
                                    get_k_elements(X,1),
                                    get_k_elements(X,2),
                                    get_last_k_elements(X,3)
                                )
                        elif self.n_batch > 0:
                            if self.constrain_latent_embedding:
                                X, P, batch_index, additional_batch_index = (
                                    get_k_elements(X,0),
                                    get_k_elements(X,1),
                                    get_k_elements(X,2),
                                    get_last_k_elements(X,3)
                                )
                            else:
                                X, batch_index, additional_batch_index = get_k_elements(X,0), get_k_elements(X,1),  get_last_k_elements(X,2)
                        elif self.n_label > 0:
                            if self.constrain_latent_embedding:
                                X, P, label_index, additional_batch_index = get_k_elements(X,0), get_k_elements(X,1), get_k_elements(X,2), get_last_k_elements(X,3)
                            else:
                                X, label_index, additional_batch_index = get_k_elements(X,0), get_k_elements(X,1), get_last_k_elements(X,2)
                        additional_batch_index = list(np.vstack(additional_batch_index).T.astype(int))
                    else:
                        if self.n_batch > 0 and self.n_label > 0 and self.n_additional_label is not None:
                            if self.constrain_latent_embedding:
                                X, P, batch_index, label_index, additional_label_index = (
                                    get_k_elements(X,0),
                                    get_k_elements(X,1),
                                    get_k_elements(X,2),
                                    get_k_elements(X,3),
                                    get_last_k_elements(X,4)
                                )
                            else:
                                X, batch_index, label_index, additional_label_index = (
                                    get_k_elements(X,0),
                                    get_k_elements(X,1),
                                    get_k_elements(X,2),
                                    get_last_k_elements(X,3)
                                )
                            additional_label_index = list(np.vstack(additional_label_index).T.astype(int))
                        elif self.n_batch > 0 and self.n_label > 0:
                            if self.constrain_latent_embedding:
                                X, P, batch_index, label_index = get_k_elements(X,0), get_k_elements(X,1), get_k_elements(X,2), get_k_elements(X,3)
                            else:
                                X, batch_index, label_index = get_k_elements(X,0), get_k_elements(X,1), get_k_elements(X,2)
                        elif self.n_batch > 0:
                            if self.constrain_latent_embedding:
                                X, P, batch_index = get_k_elements(X,0), get_k_elements(X,1),  get_k_elements(X,2)
                            else:
                                X, batch_index = get_k_elements(X,0), get_k_elements(X,1)
                        elif self.n_label > 0:
                            if self.constrain_latent_embedding:
                                X, P, label_index = get_k_elements(X,0), get_k_elements(X,1), get_k_elements(X,2)
                            else:
                                X, label_index = get_k_elements(X,0), get_k_elements(X,1)


                X = torch.tensor(np.vstack(list(map(lambda x: x.toarray() if issparse(x) else x, X))))
                if self.constrain_latent_embedding:
                    P = torch.tensor(np.vstack(P)).type(torch.FloatTensor).to(self.device)

                if self.n_label > 0:
                    label_index = torch.tensor(label_index)
                    if not isinstance(label_index, torch.FloatTensor):
                        label_index = label_index.type(torch.FloatTensor)
                    label_index = label_index.to(self.device).unsqueeze(1)

                if self.n_batch > 0:
                    batch_index = torch.tensor(batch_index)
                    if not isinstance(batch_index, torch.FloatTensor):
                        batch_index = batch_index.type(torch.FloatTensor)
                    batch_index = batch_index.to(self.device).unsqueeze(1)

                if self.n_additional_label is not None:
                    for i in range(len(additional_label_index)):
                        additional_label_index[i] = torch.tensor(additional_label_index[i])
                        if not isinstance(additional_label_index[i], torch.FloatTensor):
                            additional_label_index[i] = additional_label_index[i].type(torch.FloatTensor)
                        additional_label_index[i] = additional_label_index[i].to(self.device).unsqueeze(1)

                if self.n_additional_batch_ is not None:
                    for i in range(len(additional_batch_index)):
                        additional_batch_index[i] = torch.tensor(additional_batch_index[i])
                        if not isinstance(additional_batch_index[i], torch.FloatTensor):
                            additional_batch_index[i] = additional_batch_index[i].type(torch.FloatTensor)
                        additional_batch_index[i] = additional_batch_index[i].to(self.device).unsqueeze(1)

                if not isinstance(X, torch.FloatTensor):
                    X = X.type(torch.FloatTensor)
                if P is not None and not isinstance(P, torch.FloatTensor):
                    P = P.type(torch.FloatTensor)
                    P = P.to(self.device)
                X = X.to(self.device)

                lib_size = X.sum(1).to(self.device)
                return X, P, batch_index, label_index, additional_label_index, additional_batch_index, lib_size