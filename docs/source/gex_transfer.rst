Transfering Multi-source Gene Expression (GEX)
==============================================

This is a repository for the code for transfering multi-source gene expression (GEX) data using the VAE model from the TCR-DeepInsight package.


Transfer **without** training query datasets with reference datasets
--------------------------------------------------------------------

.. code-block:: python
  :linenos:

  import scatlasvae

  # Load the data
  query_adata = scatlasvae.read_h5ad("query_adata.h5ad")

The `adata` is a :class:`anndata.AnnData` object with raw GEX count matrix stored in adata.X.
To transfer the GEX data, we first need to build a VAE model with previously trained model parameters weights.


.. code-block:: python
  :linenos:

  reference_adata = scatlasvae.read_h5ad("reference_adata.h5d")

  reference_model = scatlasvae.model.scAtlasVAE(
    adata=reference_adata,
    batch_key="sample_name", 
    batch_embedding='embedding', 
    device='cuda:0', 
    batch_hidden_dim=64
  )

  reference_model.fit()
  reference_model.save_to_disk("model.pt")


We need to make sure that the number of genes in the new data is the same as the number of genes in the training data. If not, please see the `Retraining Multi-source GEX Data <gex_retraining.html>`_ tutorial for how to transfer GEX data with different number of genes.

.. code-block:: python
  :linenos:

  scatlasvae.model.scAtlasVAE.setup_anndata(query_adata, "model.pt")
  model_state_dict = torch.load("model.pt")
  query_model = scatlasvae.model.scAtlasVAE(
    adata=query_adata,
    batch_key="sample_name", 
    batch_embedding='embedding', 
    device='cuda:0', 
    batch_hidden_dim=64,
    pretrained_state_dict=model_state_dict['model_state_dict'],
  )
  predictions = query_model.predict_labels(
    return_pandas=True,
    show_progress=True
  )
  predictions.columns = list(map(lambda x: 'predicted_'+x, predictions.columns))
  query_adata.obs = query_adata.obs.join(predictions)

  predictions_logits = query_model.predict_batch(return_pandas=False)
  query_adata.uns['predictions_logits'] = predictions_logits

  count, fig = scatlasvae.ut.cell_type_alignment(
    query_adata,
    obs_1='original_celltype', 
    obs_2='predicted_cell_type, 
    return_fig=True
  )
  fig.show() 


Getting the transfered latent embedding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
  :linenos:

  query_adata.obsm['X_gex'] = query_model.get_latent_embedding()


Mapping the UMAP representation to the reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
  :linenos:
  
  query_adata.obsm['X_umap'] = tdi.ut.transfer_umap(
    reference_adata.obsm['X_gex'],
    reference_adata.obsm['X_umap'],
    query_adata.obsm['X_gex']
    method = 'knn'
  ) 

Optionally, if the `label_key` or `additional_label_keys` is setted in the reference 
model, one can use `query_model.predict_labels()` to get the transfered cell types. 



Transfer by training query datasets with reference datasets
-----------------------------------------------------------

The more accurate way to project query data to reference data is by co-training the 
reference and query datasets. This would results in more accurate prediction of cell types.

.. code-block:: python
  :linenos:

  import scatlasvae
  import scanpy as sc 

  query_adata.obs['cell_type'] = 'undefined'
  merged_adata = sc.concat([reference_adata, query_adata])
  
  model = scatlasvae.model.scAtlasVAE(
    adata=merged_adata,
    batch_key="sample_name", 
    batch_embedding='embedding', 
    label_key="cell_type",
    device='cuda:0', 
    batch_hidden_dim=64
  )

  predictions = model.predict_labels(
    return_pandas=True,
    show_progress=True
  )

  predictions.columns = list(map(lambda x: 'predicted_'+x, predictions.columns))
  merged_adata.obs = merged_adata.obs.join(predictions)

  predictions_logits = model.predict_batch(return_pandas=False)
  merged_adata.uns['predictions_logits'] = predictions_logits

  count, fig = scatlasvae.ut.cell_type_alignment(
    merged_adata[query_adata.obs.index], 
    obs_1='original_celltype', 
    obs_2='predicted_cell_type, 
    return_fig=True
  )
  fig.show() 