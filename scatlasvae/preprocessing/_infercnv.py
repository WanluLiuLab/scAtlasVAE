import scanpy as sc
import infercnvpy as cnv
import pandas as pd
from typing import Callable, List, Optional, Union, Iterable
from pathlib import Path
import json
import os
import numpy as np
import uuid
import tqdm
from ..utils._logger import mt
from ..utils._compat import Literal
from ..utils._utilities import writeMM

MODULE_PATH = Path(__file__).parent

HSAP_GENE_ORDER = MODULE_PATH / '../data/refdata/human/hsap_gene_order.txt'



def infercnv_R(
    adata:sc.AnnData,
    obs_key:str,
    r_path: str,
    ref_group_names: Optional[List[str]] = None,
    platform: Literal['10x', 'smartseq2'] = '10x',
):
    if len(np.unique(adata.obs.index)) != adata.shape[0]:
        adata.obs_names_make_unique()
    tempfile = '/tmp/infercnv/' + str(uuid.uuid4())
    if not os.path.exists(tempfile):
        os.makedirs(tempfile)
    X = adata.X
    mt(f"Write raw_counts_matrix.mat and annotations_file.tsv to {tempfile}")
    writeMM(X, tempfile + '/raw_counts_matrix.mat')

    pd.DataFrame(adata.obs.index).to_csv(
        tempfile + '/obsnames.tsv',
        sep='\t',
        header=None,
        index=False
    )
    pd.DataFrame(adata.var.index).to_csv(
        tempfile + '/varnames.tsv',
        sep='\t',
        header=None,
        index=False
    )
    mt(f"Write annotations_file.tsv to {tempfile}")
    adata.obs.loc[:,obs_key].to_csv(
        tempfile + '/annotations_file.tsv',
        sep='\t',
        header=None
    )
    r_script = f"""
library(infercnv)
raw_counts_matrix <- Matrix::readMM("{tempfile + '/raw_counts_matrix.mat'}")
raw_counts_matrix <- as(raw_counts_matrix, "dgCMatrix")
rownames(raw_counts_matrix) <- read.csv("{tempfile + '/varnames.tsv'}",header=FALSE)[,1]
colnames(raw_counts_matrix) <- read.csv("{tempfile + '/obsnames.tsv'}",header=FALSE)[,1]
infercnv_obj = CreateInfercnvObject(
    # raw_counts_matrix="{tempfile + '/raw_counts_matrix.tsv.gz'}",
    raw_counts_matrix=raw_counts_matrix,
    annotations_file="{tempfile + '/annotations_file.tsv'}",
    delim="\t",
    gene_order_file="{HSAP_GENE_ORDER}",
    ref_group_names = c({json.dumps(ref_group_names)[1:-1]})) 

infercnv_obj = infercnv::run(
    infercnv_obj,
    cutoff={1 if platform == '10x' else 0.1},
    out_dir="{tempfile}", 
    cluster_by_groups=TRUE, 
)
    """
    mt(f"Write rscript to {tempfile}")
    with open(os.path.join(tempfile, "rscript.R"), "w+") as f:
        f.write(r_script)
    mt(f"Run rscript")
    os.system(f'{r_path} {os.path.join(tempfile, "rscript.R")}')


def infercnv_py(
    adata:sc.AnnData,
    obs_key:str,
    ref_group_names: Optional[List[str]] = None,
    platform: Literal['10x', 'smartseq2'] = '10x',
    window_size: int = 250,
    n_jobs: int = 1,
):
    gene_order = pd.read_csv(HSAP_GENE_ORDER, sep='\t', header=None, index_col=0)
    gene_order.columns = ["chromosome", "start", "end"]
    adata = adata[:, list(filter(lambda x: x in gene_order.index, adata.var.index))]
    adata.var = adata.var.join(gene_order, how='left')
    cnv.tl.infercnv(
        adata,
        reference_key=obs_key,
        reference_cat=ref_group_names,
        window_size=window_size,
        n_jobs=n_jobs,
    )
    return adata

#scores=apply(infercnv_obj@expr.data,2,function(x){ sum(x < 0.95 | x > 1.05)/length(x) })