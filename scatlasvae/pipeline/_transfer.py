import scanpy as sc
import pandas as pd
import numpy as np
import torch
from ..model import scAtlasVAE
from ..tools import umap_alignment

def run_transfer(
    adata_reference: sc.AnnData,
    adata_query: sc.AnnData,
    path_to_state_dict: str,
    label_key: str = 'cell_type',
    device = 'cpu'
):
    """
    Transfer cell type labels from reference to query dataset

    :param adata_reference: Reference dataset
    :param adata_query: Query dataset
    :param path_to_state_dict: Path to the state dict file
    :param label_key: Key to store cell type labels
    :param device: Device to use. Default is 'cpu'

    """
    state_dict = torch.load(path_to_state_dict, map_location=device)
    # compatible with scatlasvae version 0.0.1
    
    scAtlasVAE.setup_anndata(
        adata_query,
        path_to_state_dict=path_to_state_dict,
    )

    vae_model_transfer = scAtlasVAE(
      adata=adata_query,
       pretrained_state_dict=state_dict['model_state_dict'],
       device=device,
       **state_dict['model_config']
    )
    adata_query.obsm['X_gex'] = vae_model_transfer.get_latent_embedding()
    adata_query.obsm['X_umap'] = umap_alignment(
        adata_reference.obsm['X_gex'],
        adata_reference.obsm['X_umap'],
        adata_query.obsm['X_gex'],
        method='knn',
        n_neighbors=3
    )['embedding']
    df = vae_model_transfer.predict_labels(return_pandas=True)
    adata_query.obs[label_key] = list(df[label_key])

    return adata_query  