Integrating Multi-source Gene Expression (GEX)
==============================================

This is a repository for the code for integrating multi-source gene expression (GEX) data using the VAE model from the TCR-DeepInsight package.


.. code-block:: python
  :linenos:

  import scatlasvae

  # Load the data
  adata = tdi.data.human_gex_reference_v2()


The `adata` is a :class:`anndata.AnnData` object with raw GEX count matrix stored in adata.X.


Training the VAE model using batch key
--------------------------------------

The following use `sample_name` as the batch key. The batch index is converted to **one-hot encoding** for the decoder part of the model to remove the batch effect.


.. code-block:: python
  :linenos:

  vae_model = scatlasvae.model.scAtlasVAE(
    adata=adata,
    batch_key="sample_name", 
    batch_embedding='onehot',
    device='cuda:0', 
  )
  loss_record = vae_model.fit()
  adata.obsm['X_gex'] = vae_model.get_latent_embedding()


The following use `sample_name` as the batch key. The batch index is converted to **64-dimensional embedding** for the decoder part of the model to remove the batch effect.


.. code-block:: python
  :linenos:

  vae_model = scatlasvae.model.scAtlasVAE(
    adata=adata,
    batch_key="sample_name", 
    batch_embedding='embedding', 
    device='cuda:0', 
    batch_hidden_dim=64,
  )
  loss_record = vae_model.fit()
  adata.obsm['X_gex'] = vae_model.get_latent_embedding()

Training the VAE model using batch key and categorical covariates (e.g. `study_name`)
-------------------------------------------------------------------------------------


.. code-block:: python
  :linenos:

  vae_model = scatlasvae.model.scAtlasVAE(
    adata=adata,
    batch_key="sample_name", 
    additional_batch_keys=['study_name'],
    batch_embedding='embedding', 
    device='cuda:0', 
    batch_hidden_dim=64
  )
  loss_record = vae_model.fit()
  adata.obsm['X_gex'] = vae_model.get_latent_embedding()

Training the VAE model using batch key and label key (e.g. `cell_type`)
-----------------------------------------------------------------------


.. code-block:: python
  :linenos:

  vae_model = scatlasvae.model.scAtlasVAE(
    adata=adata,
    batch_key="sample_name", 
    label_key='cell_type',
    batch_embedding='embedding', 
    device='cuda:0', 
    batch_hidden_dim=64,
  )
  loss_record = vae_model.fit()
  adata.obsm['X_gex'] = vae_model.get_latent_embedding()


Training the VAE model using multiple batch keys and mutiple label keys
-----------------------------------------------------------------------


.. code-block:: python
  :linenos:
  
  vae_model = scatlasvae.model.scAtlasVAE(
    adata=adata,
    batch_key="sample_name", 
    additional_batch_keys=['study_name'],
    label_key='cell_type_1',
    additional_label_keys=['cell_type_2'],
    batch_embedding='embedding', 
    device='cuda:0', 
    batch_hidden_dim=64,
  )
  loss_record = vae_model.fit()
  adata.obsm['X_gex'] = vae_model.get_latent_embedding()

Saving the VAE model
--------------------

The `save_to_disk` method saves the VAE model to the `path`.

.. code-block:: python
  :linenos:

  vae_model.save_to_disk(path)