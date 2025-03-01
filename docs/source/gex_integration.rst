Integrating Multi-source Gene Expression (GEX)
==============================================

This is a repository for the code for integrating multi-source gene expression (GEX) data using the :py:class:`scatlasvae.model.scAtlasVAE` from the scAtlasVAE package.


.. code-block:: python
  :linenos:

  import scatlasvae

  adata = scatlasvae.read_h5ad("path/to/adata.h5ad")

The `adata` is a :py:class:`anndata.AnnData` object with raw GEX count matrix stored in adata.X.


Training the VAE model using batch key
--------------------------------------


The following use `sample_name` as the batch key. The batch index is converted to **10-dimensional embedding** for the decoder part of the model to remove the batch effect. 


.. code-block:: python
  :linenos:

  vae_model = scatlasvae.model.scAtlasVAE(
    adata=adata,
    batch_key="sample_name", 
    batch_embedding='embedding', 
    device='cuda:0', 
    batch_hidden_dim=10,
  )
  loss_record = vae_model.fit()
  adata.obsm['X_gex'] = vae_model.get_latent_embedding()



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




Training the VAE model using batch key and categorical covariates (e.g. `study_name`)
-------------------------------------------------------------------------------------


.. code-block:: python
  :linenos:

  vae_model = scatlasvae.model.scAtlasVAE(
    adata=adata,
    batch_key=["sample_name","study_name"],
    batch_embedding='embedding', 
    device='cuda:0', 
    batch_hidden_dim=10
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
    batch_hidden_dim=10,
  )
  loss_record = vae_model.fit()
  adata.obsm['X_gex'] = vae_model.get_latent_embedding()


Training the VAE model using multiple batch keys and mutiple label keys
-----------------------------------------------------------------------

When integrating datasets from different atlas with different batch keys and cell type annotations, the `batch_key` and `label_key` can be a list of keys, and value in `label_key` can be 'undefined' if the cell type annotation is not available in the dataset. These information will not be used in the model training.

After training the multi-batch and multi-label model, one can use :py:method:`scatlasvae.ut.cell_type_alignment` to visualize the alignment of cell types.

.. code-block:: python
  :linenos:
  
  vae_model = scatlasvae.model.scAtlasVAE(
    adata=adata,
    batch_key=["sample_name", "study_name"],
    label_key=["cell_type_1","cell_type_2"],
    batch_embedding='embedding', 
    device='cuda:0', 
    batch_hidden_dim=10,
  )
  loss_record = vae_model.fit()
  adata.obsm['X_gex'] = vae_model.get_latent_embedding()

  predictions = vae_model.predict_labels(return_pandas=True)
  predictions.columns = list(map(lambda x: 'predicted_'+x, predictions.columns))
  adata.obs = adata.obs.join(predictions)

  predictions_logits = vae_model.predict_labels(return_pandas=False)
  adata.uns['predictions_logits'] = predictions_logits


.. code-block:: python
  :linenos:

  count, fig = scatlasvae.ut.cell_type_alignment(
    adata, 
    obs_1='predicted_cell_type_1', 
    obs_2='predicted_cell_type_2', 
    return_fig=True
  )
  fig.show() 

  
Saving the VAE model
--------------------

The `save_to_disk` method saves the VAE model to the `path`.

.. code-block:: python
  :linenos:

  vae_model.save_to_disk(path)