import torch
import numpy as np

from ._compat import Literal
from ._logger import mt
umap_is_installed = False
try:
    from umap import UMAP as cpuUMAP
    umap_is_installed = True
except ImportError:
    pass


cuml_is_installed = False

try:
    import cuml
    from cuml.manifold import UMAP as cumlUMAP
    cuml_is_installed = True
except ImportError:
    pass


def get_default_umap_reducer(**kwargs):
    if 'min_dist' not in kwargs:
        kwargs['min_dist'] = 0.1
    if 'spread' not in kwargs:
        kwargs['spread'] = 1.5
    if 'local_connectivity' not in kwargs:
        kwargs['local_connectivity'] = 1.5
    if 'verbose' not in kwargs:
        kwargs['verbose'] = True
    if 'repulsion_strength' not in kwargs:
        kwargs['repulsion_strength'] = 1.2
    return cpuUMAP(
        **kwargs
    )

def get_constrained_umap_reducer(**kwargs):
    if 'min_dist' not in kwargs:
        kwargs['min_dist'] = 0.1
    if 'spread' not in kwargs:
        kwargs['spread'] = 1.5
    if 'local_connectivity' not in kwargs:
        kwargs['local_connectivity'] = 1.5
    if 'verbose' not in kwargs:
        kwargs['verbose'] = True
    if 'repulsion_strength' not in kwargs:
        kwargs['repulsion_strength'] = 1.2
    return cpuUMAP(
        target_metric='euclidean',
        **kwargs
    )

def get_default_cuml_reducer(**kwargs):
    if 'min_dist' not in kwargs:
        kwargs['min_dist'] = 0.1
    if 'spread' not in kwargs:
        kwargs['spread'] = 1.5
    if 'local_connectivity' not in kwargs:
        kwargs['local_connectivity'] = 1.5
    if 'verbose' not in kwargs:
        kwargs['verbose'] = True
    if 'repulsion_strength' not in kwargs:
        kwargs['repulsion_strength'] = 1.2

    if torch.cuda.is_available():
        return cumlUMAP(
            **kwargs
        )
    else:
        raise Exception('CUDA is not available. Please install CUDA and cuml to use cumlUMAP or use get_default_umap_reducer')

def get_default_reducer():
    if torch.cuda.is_available() and cuml_is_installed:
        return get_default_cuml_reducer()
    else:
        return get_default_umap_reducer()
    

def transfer_umap(
    reference_embedding,
    reference_umap,
    query_embedding,
    method: Literal['retrain','knn'] = 'knn', 
    subsample: int = 100000,
    return_subsampled_indices: bool = False,
    return_subsampled_reference_umap: bool = False,
    n_epochs: int = 10,
    use_cuml_umap: bool = False,
    return_reducer: bool = False,
    **kwargs
):
    """
    Transfer UMAP from reference_embedding to query_embedding

    :param reference_embedding: Reference embedding
    :param reference_umap: Reference UMAP
    :param query_embedding: Query embedding
    :param method: Method to use. Either 'retrain' or 'knn'. 
        If 'retrain', use a new umap reducer. 
        If 'knn', use reference_umap to find nearest neighbors and average their UMAP coordinates
    :param subsample: Number of cells to subsample from reference_embedding.
    :param return_subsampled_indices: Return subsampled indices.
    :param return_subsampled_reference_umap: Return subsampled reference UMAP
    :param n_epochs: Number of epochs to use for retraining. Ignore if method is 'knn'
    :param return_reducer: Return reducer. Ignore if method is 'knn'.
    :param kwargs: Additional arguments to pass to UMAP. Ignore if method is 'knn'.

    :example:
        >>> import scatlasvae
        >>> import scanpy as sc
        >>> import numpy as np
        >>> adata_query.obsm['X_umap'] = scatlasvae.utils.transfer_umap(
        >>>     adata_reference.obsm['X_gex'],
        >>>     adata_reference.obsm['X_umap'],
        >>>     adata_query.obsm['X_gex']
        >>> )["embedding]
    """
    indices = np.arange(len(reference_embedding))
    if subsample < len(indices):
        indices = np.random.choice(indices, size=subsample, replace=False)

    if method == 'retrain':
        mt("Warning: retrain using UMAP reducer is not yet fully developed. Using knn instead.")
        if use_cuml_umap:
            raise NotImplementedError()
        else:
            reducer = get_default_umap_reducer(
                init = reference_umap[indices],
                target_metric = 'euclidean', 
                n_epochs = n_epochs,
                **kwargs
            )

        reducer.fit(reference_embedding[indices], y=reference_umap[indices])
        x = reducer.transform(np.vstack([reference_embedding,query_embedding]))
        z = x[len(reference_embedding):]
        return {
            'embedding': z,
            'reducer': reducer if return_reducer else None,
            'subsampled_indices': indices if return_subsampled_indices else None,
            'subsampled_reference_umap': x[:len(reference_embedding)] if return_subsampled_reference_umap else None
        }

    else:
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=5)
        knn.fit(reference_embedding[indices])
        D, I = knn.kneighbors(query_embedding)
        z = reference_umap[indices][I].mean(1)
        return {
            'embedding': z,
            'reducer': reducer if return_reducer else None,
            'subsampled_indices': indices if return_subsampled_indices else None,
            'subsampled_reference_umap': reference_umap[indices] if return_subsampled_reference_umap else None
        }