Automatic transfer to pan-disease CD8+ T cell atlas
===================================================

We can use the trained model to transfer the cell type information from the reference to the query dataset using `scatlasvae.pipeline.run_transfer`.

Step1: Load the reference and query data
----------------------------------------
.. code-block:: python
  :linenos:

  import scatlasvae
  # available at https://zenodo.org/records/12542577/files/huARdb_v2_GEX.CD8.hvg4k.h5ad
  adata_reference = scatlasvae.read_h5ad("huARdb_v2_GEX.CD8.hvg4k.h5ad.h5ad")
  
  # Load the pre-computed cell latent representation
  # available at https://zenodo.org/records/13382785/files/huARdb_v2_GEX.CD8.hvg4k.X_gex.npy
  adata_reference.obsm['X_gex'] = np.load("huARdb_v2_GEX.CD8.hvg4k.X_gex.npy")

  # your query data
  adata_query = scatlasvae.read_h5ad("query_adata.h5ad")

  # If the number of genes in the query data is different from the reference data:

  adata_query = scatlasvae.pp._preprocess.subset_adata_by_genes_fill_zeros(
    adata_query, adata_reference.var_names
  )

Step2: Transfer the cell type annotation
----------------------------------------

.. code-block:: python
  :linenos:

  scatlasvae.pipeline.run_transfer(
    adata_reference,
    adata_query,
    "huARdb_v2_GEX.CD8.hvg4k.supervised.model", # available at https://zenodo.org/records/12542577/files/huARdb_v2_GEX.CD8.hvg4k.supervised.model
    label_key = 'cell_subtype_3'
  )


Step3: Transfer the higher-resolution cell type annotation for Tex
------------------------------------------------------------------

.. code-block:: python
  :linenos:
  
   # available at https://zenodo.org/records/13382785/files/huARdb_v2_GEX.CD8.Tex.hvg4k.h5ad?download=1
  adata_reference_tex = scatlasvae.read_h5ad("huARdb_v2_GEX.CD8.Tex.hvg4k.h5ad")
  adata_query_tex = adata_query[list(map(lambda x: 'Tex' in x, adata_query.obs['cell_subtype_3']))]

  scatlasvae.pipeline.run_transfer(
    adata_reference_tex,
    adata_query_tex,
    "huARdb_v2_GEX.CD8.Tex.hvg4k.supervised.model", # available at https://zenodo.org/records/12542577/files/huARdb_v2_GEX.CD8.Tex.hvg4k.supervised.model
    label_key = 'cell_subtype_4'
  )