# scAtlasVAE

scAtlasVAE is a method for rapid atlas-level integration of large-scale scRNA-seq datasets and accurate data transfer for query datasets. 

## Installation

### Using PyPI

```shell
pip3 install scatlasvae
```

## Usage


```python
import scatlasvae

adata = scatlasvae.read_h5ad("path_to_adata")
vae_model = scatlasvae.model.scAtlasVAE(
    adata,
    batch_key="sample_name"
)
vae_model.fit()
```

