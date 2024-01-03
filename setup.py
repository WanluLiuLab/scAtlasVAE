import setuptools 

setuptools.setup(
    name="scAtlasVAE",
    version="0.0.5",
    author="Ziwei Xue",
    author_email="xueziweisz@gmail.com",
    description="scAtlasVAE: a deep learning framework for atlas-scale scRNA-seq integration and analysis",
    long_description="scAtlasVAE: a deep learning framework for atlas-scale scRNA-seq integration and analysis",
    package_dir = {'': 'scatlasvae'},
    packages=setuptools.find_packages("scatlasvae", exclude=[
        "*reference*",
        "*pretrained_weights*",
        "*docs*",
        "data"
    ]),
    install_requires=[
        'anndata==0.8.0',
        'scanpy==1.8.1',
        'scikit-learn==0.24.1',
        'matplotlib==3.3.4',
        'einops==0.4.1 ',
        'biopython==1.79',
        'seaborn==0.12.2',
        'scirpy==0.10.1',
        'pandas==1.4.2',
        'tabulate==0.8.9',
        'umap-learn',
    ],
    dependency_links=['https://download.pytorch.org/whl/cu117'],
    include_package_data=False,
)
