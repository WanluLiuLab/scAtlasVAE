from typing import Any
from scanpy import AnnData
import torch
import gc
import pandas as pd 
import os 
from ..model import scAtlasVAE
from ..preprocessing._preprocess import annotate_t_cell_cd4_cd8
from ..utils._logger import mw, mt

class scAtlasVAEPipeline:
    def __init__(
        self,
        cd4cd8_model_checkpoint_path: str,
        cd8_model_checkpoint_path: str,
        cd4_model_checkpoint_path: str,
    ):
        """
        Pipeline for scAtlasVAE
        :param cd4cd8_model_checkpoint_path: str. Path to the checkpoint file of trained VAE model for CD4/CD8 datasets
        :param cd8_model_checkpoint_path: str. Path to the checkpoint file of trained VAE model for CD8 datasets
        :param cd4_model_checkpoint_path: str. Path to the checkpoint file of trained VAE model for CD4 datasets
        """
        if not os.path.exists(cd4cd8_model_checkpoint_path):
            raise ValueError(f"cd4cd8_model_checkpoint_path {cd4cd8_model_checkpoint_path} does not exist.")
        self.cd4cd8_model_checkpoint_path = cd4cd8_model_checkpoint_path
        if not os.path.exists(cd8_model_checkpoint_path):
            mw(f"cd8_model_checkpoint_path {cd8_model_checkpoint_path} does not exist.")
        self.cd8_model_checkpoint_path = cd8_model_checkpoint_path
        if not os.path.exists(cd4_model_checkpoint_path):
            mw(f"cd4_model_checkpoint_path {cd4_model_checkpoint_path} does not exist.")
        self.cd4_model_checkpoint_path = cd4_model_checkpoint_path

    def _single_thread_call(self, adata: AnnData, predict_label: bool = False, **kwargs: Any) -> AnnData:
        # CD4/CD8 model
        adata_cd4cd8 = scAtlasVAE.setup_anndata(adata, self.cd4cd8_model_checkpoint_path)
        state_dict = torch.load(self.cd4cd8_model_checkpoint_path, map_location="cpu")
        config = state_dict['model_config']
        config.update(kwargs)
        vae_model_cd4cd8 = scAtlasVAE(
            adata=adata_cd4cd8,
            pretrained_state_dict=state_dict['model_state_dict'],
            **config,
        )
        Z = vae_model_cd4cd8.get_latent_embedding()
        adata_cd4cd8.obsm["X_gex_CD4CD8"] = Z
        annotate_t_cell_cd4_cd8(adata_cd4cd8, use_rep="X_gex_CD4CD8")

        # CD4/CD8 Annotation
        adata_cd8 = adata_cd4cd8[adata_cd4cd8.obs["cd4_cd8_annotation"] == "CD8"]
        adata_cd4 = adata_cd4cd8[adata_cd4cd8.obs["cd4_cd8_annotation"] == "CD4"]
        
        # CD8 only model
        adata_cd8 = scAtlasVAE.setup_anndata(adata_cd8, self.cd8_model_checkpoint_path)
        state_dict = torch.load(self.cd8_model_checkpoint_path, map_location="cpu")
        config = state_dict['model_config']
        config.update(kwargs)
        vae_model_cd8 = scAtlasVAE(
            adata=adata_cd8,
            pretrained_state_dict=state_dict['model_state_dict'],
            **config,
        )
        Z = vae_model_cd8.get_latent_embedding()
        if (config['label_key'] is not None or config['additional_label_key'] is not None) and predict_label:
            predicted_labels = vae_model_cd8.predict_labels(return_pandas=True)
            predicted_labels.columns = list(map(lambda x: 'predicted_' + x, predicted_labels.columns))
            predicted_labels.index = adata_cd8.obs.index
            adata_cd8.obs = adata_cd8.obs.join(predicted_labels)
        adata_cd8.obsm["X_gex"] = Z

        # CD4 only model
        adata_cd4 = scAtlasVAE.setup_anndata(adata_cd4, self.cd4_model_checkpoint_path)
        state_dict = torch.load(self.cd4_model_checkpoint_path, map_location="cpu")
        config = state_dict['model_config']
        config.update(kwargs)
        vae_model_cd4 = scAtlasVAE(
            adata=adata_cd4,
            pretrained_state_dict=state_dict['model_state_dict'],
            **config,
        )
        Z = vae_model_cd4.get_latent_embedding()
        if (config['label_key'] is not None or config['additional_label_key'] is not None) and predict_label:
            predicted_labels = vae_model_cd4.predict_labels(return_pandas=True)
            predicted_labels.columns = list(map(lambda x: 'predicted_' + x, predicted_labels.columns))
            predicted_labels.index = adata_cd4.obs.index
            adata_cd4.obs = adata_cd4.obs.join(predicted_labels)
        adata_cd4.obsm["X_gex"] = Z
        
        return adata_cd4, adata_cd8
    
    def _multi_thread_call(self, adata: AnnData, predict_label: bool = False, n_per_block: int = 100000, **kwargs: Any) -> AnnData:
        # CD4/CD8 model
        _adata_cd4cd8 = []
        for i in range(0, adata.shape[0], n_per_block):
            adata_cd4cd8 = scAtlasVAE.setup_anndata(adata[i:i+n_per_block], self.cd4cd8_model_checkpoint_path)
            state_dict = torch.load(self.cd4cd8_model_checkpoint_path, map_location="cpu")
            config = state_dict['model_config']
            config.update(kwargs)
            vae_model_cd4cd8 = scAtlasVAE(
                adata=adata_cd4cd8,
                pretrained_state_dict=state_dict['model_state_dict'],
                **config,
            )
            Z = vae_model_cd4cd8.get_latent_embedding()
            adata_cd4cd8.obsm["X_gex_CD4CD8"] = Z
            _adata_cd4cd8.append(adata_cd4cd8)
            del vae_model_cd4cd8
            gc.collect()
            torch.cuda.empty_cache()

        adata_cd4cd8 = AnnData.concatenate(*_adata_cd4cd8)
        del _adata_cd4cd8
        gc.collect()
        torch.cuda.empty_cache()
        annotate_t_cell_cd4_cd8(adata_cd4cd8, use_rep="X_gex_CD4CD8")

        # CD4/CD8 Annotation
        adata_cd8_merged = adata_cd4cd8[adata_cd4cd8.obs["cd4_cd8_annotation"] == "CD8"]
        adata_cd4_merged = adata_cd4cd8[adata_cd4cd8.obs["cd4_cd8_annotation"] == "CD4"]
        
        # CD8 only model
        _adata_cd8 = []
        for i in range(0, adata_cd8_merged.shape[0], n_per_block):
            adata_cd8 = scAtlasVAE.setup_anndata(adata_cd8_merged[i:i+n_per_block], self.cd8_model_checkpoint_path)
            
            state_dict = torch.load(self.cd8_model_checkpoint_path, map_location="cpu")
            config = state_dict['model_config']
            config.update(kwargs)
            vae_model_cd8 = scAtlasVAE(
                adata=adata_cd8,
                pretrained_state_dict=state_dict['model_state_dict'],
                **config,
            )
            Z = vae_model_cd8.get_latent_embedding()
            if config['label_key'] is not None or config['additional_label_key'] is not None:
                predicted_labels = vae_model_cd8.predict_labels(return_pandas=True)
                predicted_labels.columns = list(map(lambda x: 'predicted_' + x, predicted_labels.columns))
                predicted_labels.index = adata_cd8.obs.index
            adata_cd8.obs = pd.concat([adata_cd8.obs, predicted_labels], 1)
            adata_cd8.obsm["X_gex"] = Z
            _adata_cd8.append(adata_cd8)
            del vae_model_cd8
            gc.collect()
            torch.cuda.empty_cache()

        adata_cd8 = AnnData.concatenate(*_adata_cd8)
        del _adata_cd8
        gc.collect()

        # CD4 only model
        _adata_cd4 = []
        for i in range(0, adata_cd4_merged.shape[0], n_per_block):
            adata_cd4 = scAtlasVAE.setup_anndata(adata_cd4_merged[i:i+n_per_block], self.cd4_model_checkpoint_path)
            state_dict = torch.load(self.cd4_model_checkpoint_path, map_location="cpu")
            config = state_dict['model_config']
            config.update(kwargs)
            vae_model_cd4 = scAtlasVAE(
                adata=adata_cd4,
                pretrained_state_dict=state_dict['model_state_dict'],
                **config,
            )
            Z = vae_model_cd4.get_latent_embedding()
            if config['label_key'] is not None or config['additional_label_key'] is not None:
                predicted_labels = vae_model_cd4.predict_labels(return_pandas=True)
                predicted_labels.columns = list(map(lambda x: 'predicted_' + x, predicted_labels.columns))
                predicted_labels.index = adata_cd4.obs.index
            adata_cd4.obs = pd.concat([adata_cd4.obs, predicted_labels], 1)
            adata_cd4.obsm["X_gex"] = Z
            _adata_cd4.append(adata_cd4)
            del vae_model_cd4
            gc.collect()
            torch.cuda.empty_cache()
        adata_cd4 = AnnData.concatenate(*_adata_cd4)
        del _adata_cd4
        gc.collect()
        torch.cuda.empty_cache()
        
        return adata_cd4, adata_cd8


    def __call__(self, adata: AnnData, predict_label: bool = False, **kwargs: Any ) -> AnnData:
        if adata.shape[0] <= 100000:
            return self._single_thread_call(adata, predict_label, **kwargs)
        else:
            return self._multi_thread_call(adata, predict_label, **kwargs)