API
===

Preprocessing
-------------

**VDJPreprocessingV1Human** and **VDJPreprocessingV1Mouse** are the main 
preprocessing classes for human and mouse data, respectively. They are
used to preprocess raw data and update the AnnData object with the
preprocessed data from CellRanger.


.. autoclass:: scatlasvae.pp.VDJPreprocessingV1Human
    :members:
    :undoc-members:
    :show-inheritance:


.. autoclass:: scatlasvae.pp.VDJPreprocessingV1Mouse
    :members:
    :undoc-members:
    :show-inheritance:


Model
-----

.. currentmodule:: scatlasvae.model._gex_model

.. autoclass:: scatlasvae.model.scAtlasVAE
    :undoc-members:
    :show-inheritance:


scAtlasVAE's Methods table
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

    ~scatlasvae.model._gex_model.scAtlasVAE.partial_load_state_dict
    ~scatlasvae.model._gex_model.scAtlasVAE.get_config
    ~scatlasvae.model._gex_model.scAtlasVAE.save_to_disk
    ~scatlasvae.model._gex_model.scAtlasVAE.load_from_disk
    ~scatlasvae.model._gex_model.scAtlasVAE.setup_anndata
    ~scatlasvae.model._gex_model.scAtlasVAE.fit
    ~scatlasvae.model._gex_model.predict_labels

scAtlasVAE's Methods
~~~~~~~~~~~~~~~~~~~~
.. automethod:: scAtlasVAE.partial_load_state_dict
.. automethod:: scAtlasVAE.get_config
.. automethod:: scAtlasVAE.save_to_disk
.. automethod:: scAtlasVAE.load_from_disk
.. automethod:: scAtlasVAE.setup_anndata
.. automethod:: scAtlasVAE.fit


Utilities
---------

.. autofunction:: scatlasvae.utils.transfer_umap
.. autofunction:: scatlasvae.utils.cell_type_alignment
.. autofunction:: scatlasvae.utils.get_default_device