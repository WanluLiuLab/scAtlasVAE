# scAtlasVAE

scAtlasVAE is a method for rapid atlas-level integration of large-scale scRNA-seq datasets and data transfer for query datasets. 

<img src="./docs/source/_static/imgs/scAtlasVAE.png" alt="TCRDeepInsight" style="zoom:150%;" />


## Documentation

[Documentation](https://scatlasvae.readthedocs.io/en/latest/)

## Installation

### Create a new environment

```shell
# This will create a new environment named scatlasvae
conda env create -f environment.yml 
conda activate scatlasvae
```

### Installing via PyPI

If you are using GPU, please install the GPU version of scAtlasVAE.
```shell
pip3 install scatlasvae[gpu]
```

If you are using CPU, please install the CPU version of scAtlasVAE.
```shell
pip3 install scatlasvae[cpu]
```


### Install PyTorch 

Please see the [PyTorch official website](https://pytorch.org/) for installing GPU-enabled version of PyTorch.

```python
# Testing if CUDA is available
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

scAtlasVAE was tested on NVIDIA RTX2080Ti, RTX3090Ti, A10, A100, and A800 device on Ubuntu 20.04.

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

See complete usage guidance at [Integration tutorial](https://scatlasvae.readthedocs.io/en/latest/gex_integration.html)

## Change Log

### 1.0.3a2 (2024-04-08)

- update the enviorment configuration and dependencies list
- update the model initialization for the scAtlasVAE model

### 1.0.2 (2024-03-21)

- Bug fixed for transfer learning
- Specify the version of several dependencies, including `umap-learn`, `numba`, and `numpy`.
