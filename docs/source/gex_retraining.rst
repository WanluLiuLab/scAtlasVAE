Re-Training  Gene Expression (GEX)
==============================================

This is a repository for the code for retraining multi-source gene expression (GEX) data using the VAE model from the TCR-DeepInsight package.

.. code-block:: python
  :linenos:

  import scatlasvae

  # Load the data
  reference_adata = scatlasvae.read_h5ad("reference_adata.h5ad")
  query_adata = scatlasvae.read_h5ad("query_adata.h5ad")
  assert(reference_adata.shape[1] != query_adata.shape[1])
  assert("X_gex" in reference_adata.obsm.keys())

The :code:`reference_adata` and :code:`query_adata` are :class:`anndata.AnnData` objects with raw GEX count matrix stored in adata.X, with different number of genes
To enable transfer between the two datasets with different number of genes, we need to first re-train a VAE model on the reference dataset using the shared genes between the two datasets.    
The :code:`X_gex` is the VAE embedding of the GEX data obtained from previous training. The :code:`constrain_latent_embedding` and :code:`constrain_latent_key` arguments constrain the VAE embedding to be close to the :code:`X_gex` embedding. This is useful when the VAE model is trained on a different subset of genes (e.g. highly variable genes) and we want to use the VAE embedding of the full set of genes.


.. code-block:: python
  :linenos:

  shared_genes = set(reference_adata.var_names).intersection(set(query_adata.var_names))

  reference_adata = reference_adata[:, list(shared_genes)]
  query_adata = query_adata[:, list(shared_genes)]

  # Retrain the VAE model
  vae_model = scatlasvae.model.scAtlasVAE(
    adata=reference_adata,
    batch_key="sample_name", 
    batch_embedding='embedding', 
    device='cuda:0', 
    batch_hidden_dim=10,
    constrain_latent_embedding=True,
    constrain_latent_key='X_gex'
  )
  vae_model.fit(max_epoch=8)
  vae_model.save_to_disk("retrained_vae_model.pt")

  # Get the VAE embedding of the query dataset
  vae_model = tdi.model.VAEModel(
    adata=query,
    batch_key="sample_name", 
    batch_embedding='embedding', 
    device='cuda:0', 
    batch_hidden_dim=10,
    pretrained_state_dict=torch.load("retrained_vae_model.pt")['model_state_dict']
  )
  query_adata.obsm['X_gex'] = vae_model.get_latent_embedding()


