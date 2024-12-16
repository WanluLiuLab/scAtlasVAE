[![Stars](https://img.shields.io/github/stars/WanluLiuLab/scAtlasVAE?style=flat&logo=GitHub&color=blue)](https://github.com/WanluLiuLab/scAtlasVAE/stargazers)
[![PyPI](https://img.shields.io/pypi/v/scatlasvae?logo=PyPI)](https://pypi.org/project/rapids-singlecell)
[![Documentation Status](https://readthedocs.org/projects/scatlasvae/badge/?version=latest)](https://scatlasvae.readthedocs.io?badge=latest)

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

If you are using GPU, please install additional dependencies for GPU (`pynvml` and `nvidia-ml-py3`.)
```shell
pip3 install "scatlasvae[gpu]"
```



### Install PyTorch 

Please see the [PyTorch official website](https://pytorch.org/) for installing GPU-enabled version of PyTorch.

```python
# Testing if CUDA is available
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

If the above code returns `True`, the GPU is available.
Else, please manually install the GPU version of PyTorch via
```shell
pip3 install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```
Where `cu117` is your GPU version. You can find the version of your GPU from [NVIDIA official website](https://developer.nvidia.com/cuda-gpus).


scAtlasVAE was tested on NVIDIA RTX2080Ti, RTX3090Ti, A10, A100, and A800 device on Ubuntu 20.04.

## Usage

### Basic Usage
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

### Common Issues
1. During `model.fit()`, the following is are reported.
```
RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling `cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)
```
**Solution**: The cublas library is incompatible with your environment. Please use `pip uninstall nvidia-cublas-cu11` to uninstall the cublas library

2. `nan` reported during `model.fit()`
Please ensure all cells in your adata have non-zero total-count by checking `any(np.array(adata.X.sum(1) > 0).flatten())`. Also try to use a smaller learning rate during `model.fit`



## Cite

If you find scAtlasVAE useful for your research, please cite our paper:

Xue, Z., Wu, L., Tian, R. *et al.* (2024). Integrative mapping of human CD8+ T cells in inflammation and cancer. *Nature Methods*. [DOI: 10.1038/s41592-024-02530-0](https://doi.org/10.1038/s41592-024-02530-0)

## Link

Please visit our <a href='https://huarc.net/v2/atlas/'>huARdb website</a>

## Change Log

### 1.0.6a2 (2024-10-31)

- Add support for anndata-tensorstore

### 1.0.6a1 (2024-10-30)

- Update layer specification for the scAtlasVAE model
- Bug fixed for progress bar in jupyter notebook enviorment

### 1.0.5 (2024-10-17)

- Bug fixed for transfer learning

### 1.0.4 (2024-04-08)

- bug fixed for existing problems

### 1.0.3a2 (2024-04-08)

- update the enviorment configuration and dependencies list
- update the model initialization for the scAtlasVAE model

### 1.0.2 (2024-03-21)

- Bug fixed for transfer learning
- Specify the version of several dependencies, including `umap-learn`, `numba`, and `numpy`.
