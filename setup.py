import setuptools
from scatlasvae._version import version

version = version
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name="scAtlasVAE",
    version=version,
    url="https://github.com/WanluLiuLab/scAtlasVAE",
    home_page="https://github.com/WanluLiuLab/scAtlasVAE",
    download_url="https://github.com/WanluLiuLab/scAtlasVAE",
    author="Ziwei Xue",
    author_email="xueziweisz@gmail.com",
    description="scAtlasVAE: a deep learning framework for atlas-scale scRNA-seq integration and analysis",
    long_description="scAtlasVAE: a deep learning framework for atlas-scale scRNA-seq integration and analysis",
    long_description_content_type='text/plain',
    packages=setuptools.find_packages(exclude=[
        "*reference*",
        "*pretrained_weights*",
        "*docs*",
    ]),
    install_requires=[
        'anndata==0.8.0',
        'numba==0.57.1',
        'numpy>=1.21,<1.22',
        'scanpy==1.8.1',
        'scikit-learn==0.24.1',
        'matplotlib>=3.3.4,<=3.6.3',
        'einops>=0.4.1 ',
        'biopython==1.79',
        'seaborn==0.12.2',
        'scirpy==0.10.1',
        'pandas==1.4.2',
        'tabulate>=0.8.9',
        'umap-learn==0.5.1',
        "torch==1.13.1",
        "torchvision==0.14.1",
        "torchaudio==0.13.1",
        'plotly>=5.10.0',
        'anndata-tensorstore'
    ],
    extras_require=dict(
        gpu=[
            "pynvml==11.5.0",
            "nvidia-ml-py3==7.352.0", 
            "numpy==1.26.4",
            "numba==0.59.1",
            "cudf-cu12==23.*",
            "dask-cudf-cu12==23.*",
            "cuml-cu12==23.*", 
            "cugraph-cu12==23.*",
            "nvidia-cuda-runtime-cu12==12.4.*"
        ]
        
    ),
    dependency_links=[
        'https://download.pytorch.org/whl/torch_stable.html',
        'https://pypi.nvidia.com'
    ],
    include_package_data=False,
)
