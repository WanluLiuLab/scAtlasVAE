import os
from pathlib import Path
import scanpy as sc
import warnings
MODULE_PATH = Path(__file__).parent
warnings.filterwarnings("ignore")

zenodo_accession = '10472914'


################
#     Human    #
################


def human_gex_reference_v2_cd8():
    """
    Load the human gex reference v2
    """
    if not os.path.exists(os.path.join(MODULE_PATH, '../data/datasets')):
        os.makedirs(os.path.join(MODULE_PATH, '../data/datasets'))

    default_path = os.path.join(MODULE_PATH, '../data/datasets/huARdb_v2_GEX.CD8.hvg4k.h5ad')
    if os.path.exists(
        default_path
    ):
        return sc.read_h5ad(default_path)
    else:
        os.system(f"curl -o {default_path} https://zenodo.org/record/{zenodo_accession}/files/huARdb_v2_GEX.CD8.hvg4k.h5ad?download=1")
    
    return sc.read_h5ad(default_path)


################
#     Model    #
################


def download_model_weights():
    if not os.path.exists(os.path.join(MODULE_PATH, '../data/pretrained_weights')):
        os.makedirs(os.path.join(MODULE_PATH, '../data/pretrained_weights'))

    os.system(f"curl -o {os.path.join(MODULE_PATH, '../data/pretrained_weights', 'huARdb_v2_GEX.CD8.hvg4k.supervised.model')} https://zenodo.org/record/{zenodo_accession}/files/huARdb_v2_GEX.CD8.hvg4k.supervised.model?download=1")
    os.system(f"curl -o {os.path.join(MODULE_PATH, '../data/pretrained_weights', 'huARdb_v2_GEX.CD8.hvg4k.Tex.supervised.model')} https://zenodo.org/record/{zenodo_accession}/files/huARdb_v2_GEX.CD8.hvg4k.Tex.supervised.model?download=1")

