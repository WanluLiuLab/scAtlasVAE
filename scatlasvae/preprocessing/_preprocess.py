import multiprocessing
import os
import scirpy as ir
import scanpy as sc
import scipy
import numpy as np
import pandas as pd
from scipy import sparse
from collections import Counter
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Union, Iterable
from functools import partial
from pathlib import Path
from ..utils._logger import mt
from ..utils._decorators import typed
from ..utils._definitions import (
    TRA_DEFINITION_ORIG,
    TRAB_DEFINITION_ORIG,
    TRB_DEFINITION_ORIG,
    TRA_DEFINITION,
    TRAB_DEFINITION,
    TRB_DEFINITION,
)
from ..utils._logger import mt, Colors, get_tqdm


MODULE_PATH = Path(__file__).parent


plt.ioff()

"""
Reference human T/B cell annotation data for singleR. Originated from
Monaco, et al. (2019). RNA-Seq Signatures Normalized by mRNA Abundance Allow Absolute Deconvolution of Human Immune Cell Types. 
Cell Reports 26, 1627-1640.e7. 10.1016/j.celrep.2019.01.041.
The reference data is subsetted to only contain T or B cells, removing other immune cell types.

:attr:`Tonly`: T cell only reference data
:attr:`TBonly`: T/B cell reference data
"""
HSAP_REF_DATA = {
    "Tonly": MODULE_PATH / "./data/refdata/human/reft_name.Rdata",
    "TBonly": MODULE_PATH / "./data/refdata/human/reft_name_tb.Rdata"
}



MMUS_REF_DATA = {
    "Tonly_no_thymus": MODULE_PATH / "./data/refdata/mouse/reft_name_mouse_nothymus.rds",
    "Tonly_with_thymus":  MODULE_PATH / "./data/refdata/mouse/reft_name_mouse.rds"
}
"""
Reference muose T/B cell annotation data for singleR
Monaco, et al. (2019). RNA-Seq Signatures Normalized by mRNA Abundance Allow Absolute Deconvolution of Human Immune Cell Types. 
Cell Reports 26, 1627-1640.e7. 10.1016/j.celrep.2019.01.041.
The reference data is subsetted to only contain T cells, removing other immune cell types. For pro-T and pre-T cells, 
use `Tonly_with_thymus`. For mature T cells, use `Tonly_no_thymus`.

:attr:`Tonly_no_thymus`: T cell only reference data without thymus derived T cells
:attr:`Tonly_with_thymus`: T cell only reference data with thymus derived T cells
"""


class VDJPreprocessingV1Human:
    def __init__(
        self,
        *,
        cellranger_gex_output_path: str,
        cellranger_vdj_output_path: str,
        output_path: str,
        check_existing_files: bool = False,
        check_valid_vdj: bool = True,
        vdj_all_contig: bool = False,
        cellranger_gex_output_path_opt: Optional[str] = None,
        cellranger_vdj_output_path_opt: Optional[Union[str, Iterable[str]]] = None,
        sample_name: Optional[str] = None,
        study_name: Optional[str] = None
    ):
        """
        Preprocess the output of cellranger vdj and cellranger gex.

        :param cellranger_gex_output_path: Path to the cellranger gex output folder.
        :param cellranger_vdj_output_path: Path to the cellranger vdj output folder.
        :param output_path: Path to the output folder.
        :param check_existing_files: If True, check if the output files already exist and skip the computation.
        :param check_valid_vdj: If True, check if the vdj data is valid.
        :param vdj_all_contig: If True, use the all_contig_annotations.json file instead of filtered_contig_annotations.csv.
        :param cellranger_gex_output_path_opt: Path to the cellranger gex output folder for the optional sample.
        :param cellranger_vdj_output_path_opt: Path to the cellranger vdj output folder for the optional sample.
        :param sample_name: Name of the sample.
        :param study_name: Name of the study.

        :example:
        >>> # Preprocess the output of Human sample cellranger vdj and cellranger gex.
        >>> import scatlasvae
        >>> pp = scatlasvae.pp.VDJPreprocessingV1Human(
        ...     cellranger_gex_output_path = "./cellranger_gex_output",
        ...     cellranger_vdj_output_path = "./cellranger_vdj_output",
        ...     output_path = "./output",
        ... )
        >>> pp.process(
        ...     r_path = "/opt/anaconda3/envs/r403/bin/Rscript",
        ...     ref_data_path = scatlasvae.pp.HSAP_REF_DATA["Tonly"]
        ... )
        """
        self.filtered_10x_feature_bc_matrix_path = os.path.join(
            os.path.abspath(cellranger_gex_output_path), "filtered_feature_bc_matrix"
        )
        self.filtered_10x_feature_bc_matrix_path_opt = None
        if cellranger_gex_output_path_opt is not None:
            self.filtered_10x_feature_bc_matrix_path_opt = os.path.join(
                os.path.abspath(cellranger_gex_output_path_opt), "filtered_feature_bc_matrix"
            )

        self.raw_10x_feature_bc_matrix_path = os.path.join(
            os.path.abspath(cellranger_gex_output_path), "raw_feature_bc_matrix"
        )
        self.output_path = os.path.abspath(output_path)
        self.vdj_all_contig = vdj_all_contig
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        self.vdj_data_path = os.path.join(cellranger_vdj_output_path, 'filtered_contig_annotations.csv')
        self.vdj_data_path_all_contig = os.path.join(cellranger_vdj_output_path, 'all_contig_annotations.json')
        
        self.check_valid_vdj = check_valid_vdj

        if cellranger_vdj_output_path_opt:
            if isinstance(cellranger_vdj_output_path_opt, str):
                self.vdj_data_path_opt = os.path.join(cellranger_vdj_output_path_opt, 'filtered_contig_annotations.csv')
                self.vdj_data_path_opt_all_contig = os.path.join(cellranger_vdj_output_path, 'all_contig_annotations.json')

            elif isinstance(cellranger_vdj_output_path_opt, Iterable):
                self.vdj_data_path_opt = list(map(lambda x: os.path.join(x, 'filtered_contig_annotations.csv'), cellranger_vdj_output_path_opt))
                self.vdj_data_path_opt_all_contig = list(map(lambda x: os.path.join(x, 'all_contig_annotations.json'), cellranger_vdj_output_path_opt))
        else: 
            self.vdj_data_path_opt = None
            os.system(f"cp {self.vdj_data_path_all_contig} {os.path.join(self.output_path, 'all_contig_annotations.json')}")

        self.sample_name = sample_name
        self.study_name = study_name
        self.statistics = {}


    def process(self, 
        r_path: str = "/opt/anaconda3/envs/r403/bin/Rscript", 
        ref_data_path: str = HSAP_REF_DATA["Tonly"] 
    ):
        """
        Preprocess the output of cellranger vdj and cellranger gex.
        
        :param r_path: Path to the Rscript executable. 
        :param ref_data_path: Path to the reference data. Available references are defined in `scatlasvae.pp.HSAP_REF_DATA` and `scatlasvae.pp.MMUS_REF_DATA`.
        """
        self._initialize_filtered_adata()
        self._initialize_vdj_data()
        self._merge_adata_with_vdj()
        if self._filtered_adata.shape[0] == 0:
            print("No cells with IR information found. Exiting...")
            return
        self._cell_annotation(r_path = r_path, ref_data_path = ref_data_path)
        self._tcr_preprocessing()
        self._gex_preprocessing()
        
        
    def _check_output_path(self):
        """
        Check if the output files already exist. If not, create the output folder.

        :return: True if the output files already exist, False otherwise.
        """
        if os.path.exists(os.path.join(self.output_path, "results_raw.h5ad")) and os.path.exists(os.path.join(self.output_path, "results_single_chain_raw.h5ad")):
            self._filtered_adata = sc.read_h5ad(os.path.join(self.output_path, "results_raw.h5ad"))
            self._filtered_adata_single_chain = sc.read_h5ad(os.path.join(self.output_path, "results_single_chain_raw.h5ad"))
            return True 
        else:
            return False

    def _initialize_filtered_adata(self):
        """
        Initialize the filtered adata object.

        :return: None
        """
        self._filtered_adata = sc.read_10x_mtx(self.filtered_10x_feature_bc_matrix_path)

        if self.filtered_10x_feature_bc_matrix_path_opt is not None:
            _filtered_adata_opt = sc.read_10x_mtx(self.filtered_10x_feature_bc_matrix_path_opt)
            barcode_1 = list(self._filtered_adata.obs.index)
            barcode_2 = list(_filtered_adata_opt.obs.index)
            all_barcodes = sorted(list(set(barcode_1).union(set(barcode_2))))
            all_barcodes_index = dict(zip(all_barcodes, list(range(len(all_barcodes)))))
            new_X = np.zeros((len(all_barcodes), self._filtered_adata.shape[1]))
            indices_1 = list(map(lambda x: all_barcodes_index[x], barcode_1))
            indices_2 = list(map(lambda x: all_barcodes_index[x], barcode_2))
            new_X[indices_1] += self._filtered_adata.X.toarray()
            new_X[indices_2] += _filtered_adata_opt.X.toarray()
            self._filtered_adata = sc.AnnData(X = sparse.csr_matrix(new_X), obs = pd.DataFrame(index=all_barcodes), var = self._filtered_adata.var)

        self._raw_filtered_adata = self._filtered_adata.copy()
        self._filtered_adata.var_names_make_unique()
        self.statistics["filtered_adata_original"] = {
            "cell_number": self._filtered_adata.shape[0],
            "gene_number": self._filtered_adata.shape[1]
        }

    def _initialize_vdj_data(self):
        """
        Initialize the vdj data.

        :return: None
        """
        if self.vdj_all_contig:
            self._vdj = ir.io.read_10x_vdj(self.vdj_data_path_all_contig)
        else:
            self._vdj = ir.io.read_10x_vdj(self.vdj_data_path)
        if isinstance(self.vdj_data_path_opt, str):
            self._vdj_opt = ir.io.read_10x_vdj(self.vdj_data_path_opt)
        elif isinstance(self.vdj_data_path_opt, Iterable):
            self._vdj_opt = list(map(ir.io.read_10x_vdj, self.vdj_data_path_opt))
        else: 
            self._vdj_opt = None 

        if self.vdj_data_path_opt is not None:
            if sorted(Counter(list(map(lambda z: z[:2], filter(lambda x: type(x) == str, self._vdj.obs['IR_VJ_1_v_call'])))).items(), key=lambda x: -x[1])[0][0] == 'TR':
                os.system(f"cp {self.vdj_data_path_all_contig} {os.path.join(self.output_path, 'all_contig_annotations_TCR.json')}")
            else:
                os.system(f"cp {self.vdj_data_path_all_contig} {os.path.join(self.output_path, 'all_contig_annotations_BCR.json')}")


    def _merge_adata_with_vdj(self):
        """
        Merge the filtered adata with the vdj data.

        :return: None
        """
        ir.pp.merge_with_ir(self._filtered_adata, self._vdj)
        if isinstance(self._vdj_opt, sc.AnnData):
            self._filtered_adata.obs['is_cell'] = 'None'
            self._filtered_adata.obs['high_confidence'] = 'None'
            self._vdj_opt.obs['is_cell'] = 'None'
            self._vdj_opt.obs['high_confidence'] = 'None'
            ir.pp.merge_airr_chains(self._filtered_adata, self._vdj_opt)
        elif isinstance(self._vdj_opt, Iterable):
            for i in self._vdj_opt:
                self._filtered_adata.obs['is_cell'] = 'None'
                self._filtered_adata.obs['high_confidence'] = 'None'
                i.obs['is_cell'] = 'None'
                i.obs['high_confidence'] = 'None'
                ir.pp.merge_airr_chains(self._filtered_adata, i)
        ir.tl.chain_qc(self._filtered_adata)
        if self.check_valid_vdj:
            self._filtered_adata = self._filtered_adata[
                list(map(lambda x: x in ["single pair", "extra VJ", "extra VDJ"], self._filtered_adata.obs["chain_pairing"]))
            ].copy()
        self._filtered_adata_single_chain = self._filtered_adata[self._filtered_adata.obs["chain_pairing"] == "single pair"]
        self._filtered_adata.write_h5ad(os.path.join(self.output_path, "results_raw.h5ad"))
        self._filtered_adata_single_chain.write_h5ad(os.path.join(self.output_path, "results_single_chain_raw.h5ad"))

    def _gex_preprocessing(self):
        pipeline = lambda adata: [f(adata) for f in [
            partial(sc.pp.normalize_total, target_sum = 1e6),
            sc.pp.log1p,
            partial(
                sc.pp.highly_variable_genes, 
                min_mean=0.0125, 
                max_mean=np.max(adata.X.toarray()), 
                min_disp=0.5
            ),
            partial(sc.pp.pca, svd_solver="arpack"),
            sc.pp.neighbors,
            partial(sc.tl.tsne, n_jobs=int(multiprocessing.cpu_count() / 2)),
            partial(sc.tl.umap),
            sc.tl.leiden
        ]]
        self._filtered_adata_preprocessed = self._filtered_adata.copy()
        self._filtered_adata_single_chain_preprocessed = self._filtered_adata_single_chain.copy()
        pipeline(self._filtered_adata_preprocessed)
        pipeline(self._filtered_adata_single_chain_preprocessed)
        self._filtered_adata_preprocessed.write_h5ad(os.path.join(self.output_path, "results_preprocessed.h5ad"))
        self._filtered_adata_single_chain_preprocessed.write_h5ad(os.path.join(self.output_path, "results_single_chain_preprocessed.h5ad"))

    def _tcr_preprocessing(self):
        pipeline = lambda adata: [f(adata) for f in [
            ir.pp.ir_dist,
            partial(ir.tl.define_clonotypes, receptor_arms="all", dual_ir="any"),
            partial(ir.pp.ir_dist,  metric='hamming', sequence='aa', cutoff=2, n_jobs=4),
            partial(ir.tl.define_clonotype_clusters, partitions="connected",sequence='aa', metric='hamming'),
            partial(ir.tl.clonotype_network, min_cells=1, sequence='aa', metric='hamming')
        ]]
        pipeline(self._filtered_adata)
        pipeline(self._filtered_adata_single_chain)

    def _update_annotated(self, _filtered_adata_annotated: sc.AnnData):
        """
        This method adapts `20210820UpdatePrediction.py`
        """
        ## choose T helper cells
        Th_cells = _filtered_adata_annotated[_filtered_adata_annotated.obs.predictions.str.contains('Th|Follicular'),:]
        ## CD8A expression
        if "CD8A" in Th_cells.var.index:
            Th_cells_CD8A_np_matrix = Th_cells[:,['CD8A']].X.todense()
        else: 
            Th_cells_CD8A_np_matrix = self._raw_filtered_adata[Th_cells.obs.index, ['CD8A']].X.todense()
        ## Flatten
        Th_cells_CD8A_expression = [y for x in Th_cells_CD8A_np_matrix.tolist() for y in x]
        Th_cells_CD8A = pd.DataFrame({"barcode":Th_cells.obs.index.to_list(),
                                    'expression':Th_cells_CD8A_expression})
        ## choose cells whose expression of CD8A is larger than 0.
        Th_cells_CD8A_filter_barcodes = Th_cells_CD8A[Th_cells_CD8A['expression']>0]['barcode'].to_list()
        ## choose CD8 cells
        CD8_cells = _filtered_adata_annotated[_filtered_adata_annotated.obs.predictions.str.contains('CD8'),:]
        ## CD4 expression
        if "CD4" in CD8_cells.var.index:
            CD8_cells_np_matrix = CD8_cells[:,['CD4']].X.todense()
        else: 
            CD8_cells_np_matrix = self._raw_filtered_adata[CD8_cells.obs.index, ['CD4']].X.todense()
        ## Flatten
        CD8_cells_CD4_expression = [y for x in CD8_cells_np_matrix.tolist() for y in x]
        CD8_cells_CD4 = pd.DataFrame({"barcode":CD8_cells.obs.index.to_list(),
                                    'expression':CD8_cells_CD4_expression})
        ## choose cells whose expression of CD4 is larger than 0.
        CD8_cells_CD4_filter_barcodes = CD8_cells_CD4[CD8_cells_CD4['expression']>0]['barcode'].to_list()

        ### update predictions
        predictions = _filtered_adata_annotated.obs.predictions
        ## add Upredictive category before update

        # TODO: Deal with the predictions
        try:
            predictions = predictions.cat.add_categories(['Unpredictive'])
        except:
            pass 

        for i in (CD8_cells_CD4_filter_barcodes + Th_cells_CD8A_filter_barcodes):
            predictions[i] = 'Unpredictive'
        _filtered_adata_annotated.obs.predictions = predictions

    def _cell_annotation(self, r_path: str, ref_data_path: str = HSAP_REF_DATA["Tonly"]):
        r_script = (
            """
            library(Seurat)
            library(DoubletFinder)
            library(dplyr)
            library(SingleR)
            library(zellkonverter)
            library(SeuratDisk)
            RhpcBLASctl::blas_set_num_threads(6)
            """
            f"rawdata = Read10X('{self.filtered_10x_feature_bc_matrix_path}')\n"
            """
            pdf(NULL)
            seurat_object = CreateSeuratObject(rawdata, min.cells = 3, min.features = 200)
            seurat_object[["percent.mt"]] = PercentageFeatureSet(seurat_object,pattern = '^MT-')
            seurat_object = subset(seurat_object, subset = nFeature_RNA > 200 & percent.mt < 20)
            seurat_object = NormalizeData(seurat_object)
            seurat_object = FindVariableFeatures(seurat_object,selection.method = 'vst',nfeatures = 2000)
            seurat_object = ScaleData(seurat_object,verbose = FALSE, features = rownames(seurat_object))
            seurat_object = RunPCA(seurat_object, verbose = FALSE)
            sweep.res.list = paramSweep_v3(seurat_object, PCs = 1:50)
            sweep.stats = summarizeSweep(sweep.res.list, GT = FALSE)
            bcmvn = find.pK(sweep.stats)
            pk_bcmvn = as.numeric(as.vector(bcmvn$pK[which.max(bcmvn$BCmetric)]))
            cellnumber = ncol(seurat_object)
            multiplet_rate = 0
            if (cellnumber <= 500) {
            multiplet_rate = 0.004
            } else if ( cellnumber > 500 & cellnumber <= 1000) {
            multiplet_rate = 0.008
            } else if ( cellnumber > 100 & cellnumber <= 2000) {
            multiplet_rate = 0.016
            } else if ( cellnumber > 2000 & cellnumber <= 3000) {
            multiplet_rate = 0.023
            } else if ( cellnumber > 3000 & cellnumber <= 4000) {
            multiplet_rate = 0.031
            } else if ( cellnumber > 4000 & cellnumber <= 5000) { 
            multiplet_rate = 0.039
            } else if ( cellnumber > 5000 & cellnumber <= 6000) {
            multiplet_rate = 0.046
            } else if ( cellnumber > 6000 & cellnumber <= 7000) {
            multiplet_rate = 0.054
            } else if ( cellnumber > 7000 & cellnumber <= 8000) {
            multiplet_rate = 0.061
            } else if ( cellnumber > 8000 & cellnumber <= 9000) {
            multiplet_rate = 0.069
            } else {
            multiplet_rate = 0.076
            }

            annotations = seurat_object@meta.data$seurat_clusters
            homotypic.prop = modelHomotypic(annotations)  
            nExp_poi = round(multiplet_rate*cellnumber)  
            nExp_poi.adj = round(nExp_poi*(1-homotypic.prop))

            seurat_object_filterDoublet = doubletFinder_v3(
                seurat_object, 
                PCs = 1:50, pN = 0.25, 
                pK = pk_bcmvn, nExp = nExp_poi, 
                reuse.pANN = FALSE
            )
            seurat_object_filterDoublet = subset(seurat_object_filterDoublet,
                cells = rownames(seurat_object_filterDoublet[[paste("DF.classifications_0.25",pk_bcmvn,nExp_poi,sep = '_')]])
            )
            """
            f"load('{ref_data_path}')\n"
            """
            sce <- as.SingleCellExperiment(seurat_object_filterDoublet)
            predictions <- SingleR(test = sce, ref = reft, labels = reft$label.fine)
            sce[["predictions"]] <- predictions$labels
            """ 
            f"SeuratDisk::SaveH5Seurat(as.Seurat(sce), '{os.path.join(self.output_path, 'filterDoublet_prediction_sce.h5seurat')}')\n"
            f"""SeuratDisk::Convert("{os.path.join(self.output_path, 'filterDoublet_prediction_sce.h5seurat')}", dest="h5ad")\n"""
        )
        with open(os.path.join(self.output_path, "rscript.R"), "w+") as f:
            f.write(r_script)
        if os.path.exists(os.path.join(self.output_path, 'filterDoublet_prediction_sce.h5seurat')):
            os.system(f"rm {os.path.join(self.output_path, 'filterDoublet_prediction_sce.h5seurat')}")
        os.system(f'{r_path} {os.path.join(self.output_path, "rscript.R")}')
        os.system(f"rm {os.path.join(self.output_path, 'filterDoublet_prediction_sce.h5seurat')}")
        _filtered_adata_annotated = sc.read_h5ad(os.path.join(self.output_path, 'filterDoublet_prediction_sce.h5ad'))
        self._update_annotated(_filtered_adata_annotated)
        self._filtered_adata = self._filtered_adata[list(map(lambda x: x in  _filtered_adata_annotated.obs.index, self._filtered_adata.obs.index))].copy()
        self._filtered_adata.obs["predictions"] = _filtered_adata_annotated.obs.loc[self._filtered_adata.obs.index, "predictions"]
        self._filtered_adata_single_chain = self._filtered_adata_single_chain[list(map(lambda x: x in _filtered_adata_annotated.obs.index, self._filtered_adata_single_chain.obs.index))].copy()
        self._filtered_adata_single_chain.obs["predictions"] = _filtered_adata_annotated.obs.loc[self._filtered_adata_single_chain.obs.index, "predictions"]
    

class VDJPreprocessingV1Mouse(VDJPreprocessingV1Human):
    def process(self, r_path: str = "/opt/anaconda3/envs/r403/bin/Rscript", ref_data_path: str = MMUS_REF_DATA["Tonly_no_thymus"] ):
        self._initialize_filtered_adata()
        self._initialize_vdj_data()
        self._merge_adata_with_vdj()
        self._cell_annotation(r_path = r_path, ref_data_path = ref_data_path)
        self._tcr_preprocessing()
        self._gex_preprocessing()

    def _update_annotated(self, _filtered_adata_annotated: sc.AnnData):
        """
        This method adapts `20210820UpdatePrediction.py`
        Mouse version adapting human genes
        """
        ## choose CD4 T cells
        Th_cells = _filtered_adata_annotated[
            np.array(_filtered_adata_annotated.obs.predictions_new.str.contains('CD4')) |
            np.array(_filtered_adata_annotated.obs.predictions_new.str.contains('T.4'))
        ,:]
        ## CD8A expression
        if "Cd8a" in Th_cells.var.index:
            Th_cells_CD8A_np_matrix = Th_cells[:,['Cd8a']].X.todense()
        else: 
            Th_cells_CD8A_np_matrix = self._raw_filtered_adata[Th_cells.obs.index, ['Cd8a']].X.todense()
        ## Flatten
        Th_cells_CD8A_expression = [y for x in Th_cells_CD8A_np_matrix.tolist() for y in x]
        Th_cells_CD8A = pd.DataFrame({"barcode":Th_cells.obs.index.to_list(),
                                    'expression':Th_cells_CD8A_expression})
        ## choose cells whose expression of CD8A is larger than 0.
        Th_cells_CD8A_filter_barcodes = Th_cells_CD8A[Th_cells_CD8A['expression']>0]['barcode'].to_list()
        ## choose CD8 cells
        CD8_cells = _filtered_adata_annotated[
            np.array(_filtered_adata_annotated.obs.predictions_new.str.contains('CD8')) | 
            np.array(_filtered_adata_annotated.obs.predictions_new.str.contains('T.8'))
        ,:]
        ## CD4 expression
        if "Cd4" in CD8_cells.var.index:
            CD8_cells_np_matrix = CD8_cells[:,['Cd4']].X.todense()
        else: 
            CD8_cells_np_matrix = self._raw_filtered_adata[CD8_cells.obs.index, ['Cd4']].X.todense()
        ## Flatten
        CD8_cells_CD4_expression = [y for x in CD8_cells_np_matrix.tolist() for y in x]
        CD8_cells_CD4 = pd.DataFrame({"barcode":CD8_cells.obs.index.to_list(),
                                    'expression':CD8_cells_CD4_expression})
        ## choose cells whose expression of CD4 is larger than 0.
        CD8_cells_CD4_filter_barcodes = CD8_cells_CD4[CD8_cells_CD4['expression']>0]['barcode'].to_list()

        ### update predictions
        predictions_new = _filtered_adata_annotated.obs.predictions_new
        ## add Upredictive category before update

        # TODO: Deal with the predictions
        try:
            predictions_new = predictions_new.cat.add_categories(['Unpredictive'])
        except:
            pass 

        for i in (CD8_cells_CD4_filter_barcodes + Th_cells_CD8A_filter_barcodes):
            predictions_new[i] = 'Unpredictive'
        _filtered_adata_annotated.obs.predictions_new = predictions_new

        
    def _cell_annotation(self, r_path: str, ref_data_path: str = MMUS_REF_DATA["Tonly_no_thymus"]):
        r_script = (
            """
            library(Seurat)
            library(DoubletFinder)
            library(dplyr)
            library(SingleR)
            library(zellkonverter)
            library(SeuratDisk)
            RhpcBLASctl::blas_set_num_threads(6)
            """
            f"rawdata = Read10X('{self.filtered_10x_feature_bc_matrix_path}')\n"
            """
            pdf(NULL)
            seurat_object = CreateSeuratObject(rawdata, min.cells = 3, min.features = 200)
            seurat_object[["percent.mt"]] = PercentageFeatureSet(seurat_object,pattern = '^MT-')
            seurat_object = subset(seurat_object, subset = nFeature_RNA > 200 & percent.mt < 20)
            seurat_object = NormalizeData(seurat_object)
            seurat_object = FindVariableFeatures(seurat_object,selection.method = 'vst',nfeatures = 2000)
            seurat_object = ScaleData(seurat_object,verbose = FALSE, features = rownames(seurat_object))
            seurat_object = RunPCA(seurat_object, verbose = FALSE)
            sweep.res.list = paramSweep_v3(seurat_object, PCs = 1:50)
            sweep.stats = summarizeSweep(sweep.res.list, GT = FALSE)
            bcmvn = find.pK(sweep.stats)
            pk_bcmvn = as.numeric(as.vector(bcmvn$pK[which.max(bcmvn$BCmetric)]))
            cellnumber = ncol(seurat_object)
            multiplet_rate = 0
            if (cellnumber <= 500) {
            multiplet_rate = 0.004
            } else if ( cellnumber > 500 & cellnumber <= 1000) {
            multiplet_rate = 0.008
            } else if ( cellnumber > 100 & cellnumber <= 2000) {
            multiplet_rate = 0.016
            } else if ( cellnumber > 2000 & cellnumber <= 3000) {
            multiplet_rate = 0.023
            } else if ( cellnumber > 3000 & cellnumber <= 4000) {
            multiplet_rate = 0.031
            } else if ( cellnumber > 4000 & cellnumber <= 5000) { 
            multiplet_rate = 0.039
            } else if ( cellnumber > 5000 & cellnumber <= 6000) {
            multiplet_rate = 0.046
            } else if ( cellnumber > 6000 & cellnumber <= 7000) {
            multiplet_rate = 0.054
            } else if ( cellnumber > 7000 & cellnumber <= 8000) {
            multiplet_rate = 0.061
            } else if ( cellnumber > 8000 & cellnumber <= 9000) {
            multiplet_rate = 0.069
            } else {
            multiplet_rate = 0.076
            }

            annotations = seurat_object@meta.data$seurat_clusters
            homotypic.prop = modelHomotypic(annotations)  
            nExp_poi = round(multiplet_rate*cellnumber)  
            nExp_poi.adj = round(nExp_poi*(1-homotypic.prop))

            seurat_object_filterDoublet = doubletFinder_v3(
                seurat_object, 
                PCs = 1:50, pN = 0.25, 
                pK = pk_bcmvn, nExp = nExp_poi, 
                reuse.pANN = FALSE
            )
            seurat_object_filterDoublet = subset(seurat_object_filterDoublet,
                cells = rownames(seurat_object_filterDoublet[[paste("DF.classifications_0.25",pk_bcmvn,nExp_poi,sep = '_')]])
            )
            """
            f"reft <- readRDS('{ref_data_path}')\n"
            """
            sce <- as.SingleCellExperiment(seurat_object_filterDoublet)
            predictions <- SingleR(test = sce, ref = reft, labels = reft$label.new)
            sce[["predictions_new"]] <- predictions$labels
            predictions <- SingleR(test = sce, ref = reft, labels = reft$label.fine)
            sce[["predictions_fine"]] <- predictions$labels
            """ 
            f"SeuratDisk::SaveH5Seurat(as.Seurat(sce), '{os.path.join(self.output_path, 'filterDoublet_prediction_sce.h5seurat')}')\n"
            f"""SeuratDisk::Convert("{os.path.join(self.output_path, 'filterDoublet_prediction_sce.h5seurat')}", dest="h5ad")\n"""
        )
        with open(os.path.join(self.output_path, "rscript.R"), "w+") as f:
            f.write(r_script)
        if os.path.exists(os.path.join(self.output_path, 'filterDoublet_prediction_sce.h5seurat')):
            os.system(f"rm {os.path.join(self.output_path, 'filterDoublet_prediction_sce.h5seurat')}")
        os.system(f'{r_path} {os.path.join(self.output_path, "rscript.R")}')
        os.system(f"rm {os.path.join(self.output_path, 'filterDoublet_prediction_sce.h5seurat')}")
        _filtered_adata_annotated = sc.read_h5ad(os.path.join(self.output_path, 'filterDoublet_prediction_sce.h5ad'))
        self._update_annotated(_filtered_adata_annotated)

        self._filtered_adata = self._filtered_adata[list(map(lambda x: x in  _filtered_adata_annotated.obs.index, self._filtered_adata.obs.index))].copy()
        self._filtered_adata.obs["predictions_fine"] = _filtered_adata_annotated.obs.loc[self._filtered_adata.obs.index, "predictions_fine"]
        self._filtered_adata_single_chain = self._filtered_adata_single_chain[list(map(lambda x: x in _filtered_adata_annotated.obs.index, self._filtered_adata_single_chain.obs.index))].copy()
        self._filtered_adata_single_chain.obs["predictions_fine"] = _filtered_adata_annotated.obs.loc[self._filtered_adata_single_chain.obs.index, "predictions_fine"]

        self._filtered_adata = self._filtered_adata[list(map(lambda x: x in  _filtered_adata_annotated.obs.index, self._filtered_adata.obs.index))].copy()
        self._filtered_adata.obs["predictions_new"] = _filtered_adata_annotated.obs.loc[self._filtered_adata.obs.index, "predictions_new"]
        self._filtered_adata_single_chain = self._filtered_adata_single_chain[list(map(lambda x: x in _filtered_adata_annotated.obs.index, self._filtered_adata_single_chain.obs.index))].copy()
        self._filtered_adata_single_chain.obs["predictions_new"] = _filtered_adata_annotated.obs.loc[self._filtered_adata_single_chain.obs.index, "predictions_new"]


@typed({
    'gex_adata': sc.AnnData, 
    'gex_embedding_key': str,
    'tcr_embedding_key': str,
    'joint_embedding_key': str,
})
def update_anndata(
    gex_adata: sc.AnnData,
    gex_embedding_key: str = 'X_gex',
    tcr_embedding_key: str = 'X_tcr',
    joint_embedding_key: str = 'X_gex_tcr',
):
    """
    Update the adata with the embedding keys

    .. note::
        This method modifies the `gex_adata` inplace. 
        added columns in .obs: `tcr`, `CDR3a`, `CDR3b`, `TRAV`, `TRAJ`, `TRBV`, `TRBJ`

    .. note::
        `"tcr"` should be in `gex_adata.obs.columns`.


    :param gex_adata: AnnData object
    :param gex_embedding_key: embedding key for gex
    :param tcr_embedding_key: embedding key for tcr
    :param joint_embedding_key: embedding key for joint
    """
    gex_adata.uns['embedding_keys'] = {
        "gex": gex_embedding_key,
        "tcr": tcr_embedding_key,
        "joint": joint_embedding_key,
    }
    for i,j in zip([
        'IR_VJ_1_junction_aa',
        'IR_VDJ_1_junction_aa',
        'IR_VJ_1_v_call',
        'IR_VJ_1_j_call',
        'IR_VDJ_1_v_call',
        'IR_VDJ_1_j_call'
    ], ['CDR3a', 
        'CDR3b', 
        'TRAV', 
        'TRAJ', 
        'TRBV', 
        'TRBJ'
    ]):
        if not (i in gex_adata.obs.columns or j in gex_adata.obs.columns):
            raise ValueError(f"{i} or {j} is not in adata.obs.columns")
        if i in gex_adata.obs.columns:
            gex_adata.obs[j] = gex_adata.obs[i]

    if not 'individual' in gex_adata.obs.columns:
        raise ValueError("individual is not in adata.obs.columns.")
    mt("TCRDeepInsight: initializing dataset")
    mt("TCRDeepInsight: adding 'tcr' to adata.obs")
    gex_adata.obs['tcr'] = None
    gex_adata.obs.iloc[:, list(gex_adata.obs.columns).index("tcr")] = list(map(lambda x: '='.join(x), gex_adata.obs.loc[:,TRAB_DEFINITION + ['individual']].to_numpy()))

@typed({
    'gex_adata': sc.AnnData,
    'gex_embedding_key': str,
    'agg_index': pd.DataFrame,
})
def aggregated_gex_embedding_by_tcr(
        gex_adata: sc.AnnData,
        gex_embedding_key: str,
        agg_index: pd.DataFrame,
    ):
    """
    Aggregate GEX embedding by TCR

    :param gex_adata: AnnData object
    :param agg_index: Aggregated index

    :return: aggregated GEX embedding
    """
    all_gex_embedding = []
    for i in agg_index['index']:
        all_gex_embedding.append(
            gex_adata.obsm[gex_embedding_key][i].mean(0)
        )
    all_gex_embedding = np.vstack(all_gex_embedding)
    return all_gex_embedding

def annotate_t_cell_cd4_cd8(adata: sc.AnnData, use_rep='X_gex'):
    adata.obs["cd4_cd8_class"] = None
    adata.obs.loc[
        ((adata.X[:,list(adata.var_names).index("CD8A")].toarray()  == 0) & \
         (adata.X[:,list(adata.var_names).index("CD8B")].toarray()  == 0) & \
         (adata.X[:,list(adata.var_names).index("CD4")].toarray() > 0) ).flatten()
    ,"cd4_cd8_class"] = "CD4"
    adata.obs.loc[
        (((adata.X[:,list(adata.var_names).index("CD8A")].toarray() > 0) | \
        (adata.X[:,list(adata.var_names).index("CD8B")].toarray() > 0)) & \
        (adata.X[:,list(adata.var_names).index("CD4")].toarray() == 0) ).flatten()
    ,"cd4_cd8_class"] = "CD8"
    adata.obs.loc[
        ((adata.X[:,list(adata.var_names).index("CD8A")].toarray() == 0) & \
         (adata.X[:,list(adata.var_names).index("CD8B")].toarray() == 0) & \
        (adata.X[:,list(adata.var_names).index("CD4")].toarray() == 0) ).flatten()
    ,"cd4_cd8_class"] = "double_negative"
    adata.obs.loc[
        (((adata.X[:,list(adata.var_names).index("CD8A")].toarray() > 0) | \
        (adata.X[:,list(adata.var_names).index("CD8B")].toarray() > 0)) & \
        (adata.X[:,list(adata.var_names).index("CD4")].toarray() > 0) ).flatten()
    ,"cd4_cd8_class"] = "double_positive"
    from sklearn.neighbors import KNeighborsClassifier
    nn = KNeighborsClassifier(n_neighbors=13)
    nn.fit(adata[list(map(lambda x: 
        x in ["CD8","CD4"], adata.obs['cd4_cd8_class']))].obsm[use_rep], 
        adata[list(map(lambda x: x in ["CD8","CD4"], adata.obs['cd4_cd8_class']))].obs["cd4_cd8_class"]
    )
    Y = nn.predict(adata.obsm[use_rep])
    adata.obs["cd4_cd8_annotation"] = Y

def subset_adata_by_genes_fill_zeros(
    adata: sc.AnnData, 
    gene_list: List[str]
):
    """
    Subset adata by genes and fill zeros for missing genes

    :param adata: AnnData object
    :param gene_list: List of genes to subset

    :return: Subsetted AnnData object
    """
    low_memory = False
    if adata.shape[0] > 100000:
        low_memory = True
    X = np.zeros((adata.shape[0], len(gene_list)), dtype=np.int32)
    var = list(adata.var.index)
    pbar = get_tqdm()(range(len(gene_list)), desc='Subset by genes', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    original_X = adata.X 

    if not low_memory and scipy.sparse.issparse(original_X):
        original_X = original_X.toarray()

    for i,j in enumerate(gene_list):
        if j in var:
            if low_memory:
                X[:,i] = original_X[:, var.index(j)].toarray().flatten()
            else:
                X[:,i] = original_X[:, var.index(j)].flatten()
        pbar.update(1)
    pbar.close()
    result_adata = sc.AnnData(
        X = scipy.sparse.csr_matrix(X),
        obs = adata.obs.copy(),
        var = pd.DataFrame(
            index = gene_list
        ),
        uns = adata.uns.copy(),
        obsm = adata.obsm.copy()
    )
    return result_adata