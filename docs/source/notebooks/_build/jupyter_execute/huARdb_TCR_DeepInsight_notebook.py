#!/usr/bin/env python
# coding: utf-8

# # TCR-DeepInsight Vignette

# The emergence of single-cell immune profiling technology has led to the production of a large amount of data on single-cell gene expression (GEX) and T cell receptor (TCR), which has great potential for studying TCR biology and identifying effective TCRs. However, one of the major challenges is the lack of a reference atlas that provides easy access to these datasets. On the other hand, the use of TCR engineering in disease immunotherapy is rapidly advancing, and single-cell immune profiling data can be a valuable resource for identifying functional TCRs. Nevertheless, the lack of efficient computational tools to integrate and identify functional TCRs is a significant obstacle in this field.

# The TCR-DeepInsight is a module that can perform **GEX** / **TCR** joint analysis.

# <div class="alert alert-info">
# 
# **Note**
#     
# In Jupyter notebooks and lab, you can see the documentation for a python function by hitting ``SHIFT + TAB``. Hit it twice to expand the view.
# 
# </div>

# In[8]:


import tcr_deep_insight as tdi


# # Training new reference datasets

# ## Training GEX reference

# The GEX data is stored in an .h5ad file, which contains **raw gene expression matrix** and meta informations including **study_name**, **sample_name**, and other annotations

# In[8]:


gex_reference_adata = tdi.data.human_gex_reference_v1()


# In[31]:


gex_reference_adata


# Here we use the reference GEX data from the paper, which contains 1,017,877 T cells from various studies and diseases

# In[5]:


gex_reference_adata.obs


# The VAEModel for GEX can be easily set up by the following code. The **sample_name** is set for the batch key for correcting batch effects.

# In[7]:


vae_model = tdi.model.VAEModel(
    adata=gex_reference_adata, 
    batch_key="sample_name", 
    device='cuda:1'
)


# Fitting the VAE model is ultra-fast with GPU support. We use only 16 epoch to train the VAE model as the number of cells is very large.

# In[8]:


vae_model.fit(
    max_epoch=16,
    lr=1e-4,
    n_per_batch=256
)


# In[ ]:


The latent embedding from the VAE model can be easily obtained and projected to 2D dimension.


# In[9]:


gex_reference_adata.obsm["X_embedding"] = X_embedding = vae_model.get_latent_embedding()


# In[10]:


import umap
gex_reference_adata.obsm["X_umap"] = tdi.tl.umap.UMAP(min_dist=0.5).fit_transform(X_embedding)


# In[23]:


fig, ax = plt.subplots(figsize=(4, 4))
tdi.pl.umap(
    gex_reference_adata, 
    color='cell_type_1', 
    palette=tdi.pl.palette.reannotated_prediction_palette,
    ax=ax
)


# ## Training TCR reference

# Similar to the GEX data, we store the TCR data in a .h5ad file, which contains the full-length 
# TCR sequence. We aggregate the TCR sequence by **individual** to obtain unique clonotypes.

# In[17]:


tcr_reference_adata = tdi.read_h5ad("./tcr_deep_insight/data/reference/human_tcr_reference_v1.h5ad")


# In[4]:


tcr_reference_adata.obs


# We implemented a tokenizer for the full-length TCR sequence.

# In[4]:


tcr_tokenizer = tdi.model.TCRabTokenizer(
    tra_max_length=48, 
    trb_max_length=48,
    species='human' # or 'mouse'
)
tcr_dataset = tcr_tokenizer.to_dataset(
   ids=tcr_reference_adata.obs.index,
   alpha_v_genes=list(tcr_reference_adata.obs['TRAV']),
   alpha_j_genes=list(tcr_reference_adata.obs['TRAJ']),
   beta_v_genes=list(tcr_reference_adata.obs['TRBV']),
   beta_j_genes=list(tcr_reference_adata.obs['TRBJ']),
   alpha_chains=list(tcr_reference_adata.obs['CDR3a']),
   beta_chains=list(tcr_reference_adata.obs['CDR3b']),
)


# If the dataset is for training, we can split the dataset into train and test set.

# In[5]:


# Train Test Split
tcr_dataset = tcr_dataset['train'].train_test_split(0.05)


# The dataset can be saved to disk

# In[ ]:


tcr_dataset.save_to_disk("./tcr_deep_insight/data/datasets/human_tcr_v1")


# In[3]:


import datasets
tcr_dataset = datasets.load_from_disk("./tcr_deep_insight/data/datasets/human_tcr_v1")


# In[4]:


tcr_model = tdi.model.TCRabModel(
    tdi.model.config.get_human_config(),
    labels_number=1
).to("cuda:1")

tcr_collator = tdi.model.TCRabCollator(48, 48, mlm_probability=0.15)
tcr_model_optimizers = (torch.optim.AdamW(tcr_model.parameters(), lr=1e-4), None)
tcr_model_trainer = tdi.model.TCRabTrainer(
    tcr_model, 
    collator=tcr_collator, 
    train_dataset=tcr_dataset['train'], 
    test_dataset=tcr_dataset['test'], 
    optimizers=tcr_model_optimizers, 
    device='cuda:1'
)


# In[7]:


tcr_model_trainer.fit(max_epoch=3, show_progress_bar=True)


# In[ ]:


torch.save(tcr_model.state_dict(), '/path/to/saved/checkpoints')


# In[5]:


tcr_model.load_state_dict(torch.load("./tcr_deep_insight/data/pretrained_weights/human_bert_tcr_768_v1.ckpt"))


# In[6]:


tcr_model_trainer.attach_train_dataset(tcr_dataset['train'])
tcr_model_trainer.attach_test_dataset(tcr_dataset['test'])


# In[ ]:


all_train_result, all_test_result = tcr_model_trainer.evaluate(n_per_batch=64, show_progress=True)


# In[20]:


from collections import Counter

def FLATTEN(x): 
    return [i for s in x for i in s]

print("==== Train ====")
print('masked amino acids prediction {:.3f}'.format(Counter(FLATTEN(all_train_result['aa']))[True] / len(FLATTEN(all_train_result['aa']))))
print('masked TRAV prediction {:.3f}'.format(Counter(FLATTEN(all_train_result['av']))[True] / len(FLATTEN(all_train_result['av']))))
print('masked TRBV prediction {:.3f}'.format(Counter(FLATTEN(all_train_result['bv']))[True] / len(FLATTEN(all_train_result['bv']))))
print("==== Test ====")
print('masked amino acids prediction {:.3f}'.format(Counter(FLATTEN(all_test_result['aa']))[True] / len(FLATTEN(all_test_result['aa']))))
print('masked TRAV prediction {:.3f}'.format(Counter(FLATTEN(all_test_result['av']))[True] / len(FLATTEN(all_test_result['av']))))
print('masked TRBV prediction {:.3f}'.format(Counter(FLATTEN(all_test_result['bv']))[True] / len(FLATTEN(all_test_result['bv']))))


# In[46]:


import sklearn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

labels = np.unique(FLATTEN(all_train_result['aa_gt']) + FLATTEN(all_train_result['aa_pred']))
cm = sklearn.metrics.confusion_matrix(
    FLATTEN(all_train_result['aa_gt']), 
    FLATTEN(all_train_result['aa_pred']), 
    labels=labels
)
cm = cm[:20,:][:,:20]
cm = pd.DataFrame(
    cm, 
    index=list(map(lambda x: _AMINO_ACIDS_INDEX_REVERSE[x], labels[:20])),
    columns=list(map(lambda x: _AMINO_ACIDS_INDEX_REVERSE[x], labels[:20]))
)
from tcr_deep_insight.utils._amino_acids import _AMINO_ACIDS_INDEX_REVERSE
sns.clustermap(cm, row_cluster=False, col_cluster=False, standard_scale='var')
plt.show()


# ## Cluster TCRs in reference datasets

# In[84]:


tdi.pp.update_anndata(gex_reference_adata)
tcr_reference_adata = tdi.pp.unique_tcr_by_individual(gex_reference_adata, label_key='cell_type_1')


# In[85]:


tdi.tl.get_pretrained_tcr_embedding(
    tcr_adata=tcr_reference_adata,
    bert_config=tdi.model.config.get_human_config(),
    checkpoint_path='./tcr_deep_insight/data/pretrained_weights/human_bert_tcr_768_v1.ckpt',
    pca_path='./tcr_deep_insight/data/pretrained_weights/human_bert_tcr_pca_v1.pkl',
    use_pca=True
)


# In[37]:


tcr_reference_adata.write_h5ad('./tcr_deep_insight/data/reference/human_tcr_reference_v1.h5ad')


# <div class="alert alert-info">
# <h2> Note </h2>
#     
# You should have faiss-gpu installed to set `gpu=1`. This is much faster than the CPU version. For more information, please see https://pypi.org/project/faiss-gpu/.
# 
# </div>

# In[61]:


tcr_cluster_result = tdi.tl.cluster_tcr(
    tcr_reference_adata,
    label_key='disease_type_1',
    gpu=1
)


# In[26]:


import matplotlib.pyplot as plt
import numpy as np


result_tcr = tcr_cluster_result.obs[
    np.array(tcr_cluster_result.obs['disease_type_1'] == 'Solid tumor') & 
    np.array(tcr_cluster_result.obs['count'] > 2) &
    np.array(tcr_cluster_result.obs['number_of_individuals'] > 3)
]


fig,ax = tdi.pl.createFig((5,5))
ax.scatter(
    result_tcr['tcr_similarity_score'],
    result_tcr['disease_specificity_score'],
    s=result_tcr['count'] * 6, 
    linewidths=0
)
plt.show()


# <div class="alert alert-info">
# <h2> Note </h2>
#     
# You should have `mafft` installed in your system to produce the logoplot below.
# 
# In **Ubuntu**, 
# 
# `sudo apt update`
# `sudo apt install mafft`
# 
# In **MacOS**, please use HomeBrew
# 
# `brew install mafft`
# 
#     
# In CentOS or other operating systems, please check In **CentOS**, please check https://mafft.cbrc.jp/alignment/software/linuxportable.html for manual installation.
#     
#     
# </div>
# 
# 

# In[43]:


tcrs = list(filter(lambda x: x != '-', result_tcr.sort_values('disease_specificity_score', ascending=False).iloc[0,1:41])) 

tdi.pl.plot_gex_selected_tcrs(
    gex_reference_adata,
    tcrs,
    color='cell_type_1',
    palette=tdi.pl.palettes.reannotated_prediction_palette,
)


# # Cluster TCRs in querying datasets from reference datasets

# Once the single-cell immune profiling datasets are processed by CellRanger and the GEX and TCR information are integrated by Scanpy, and Scirpy, you would get a datasets including:
# 1) The raw gene expression matrix
# 2) The Full-length TCR sequence for each single-cell
# 
# 
# And you would provide the sample name as well as the individual name to the dataset

# We use an example datasets from Sun et al., 2022 of Gastric Cancer ([Sun, K. et al. (2022)](https://doi.org/10.1038/s41467-022-32627-z)). 
# 
# 
# 

# In[ ]:


gex_query_adata = tdi.read_h5ad("./tcr_deep_insight/data/GEX.GC.h5ad")


# In[4]:


gex_query_adata.obs


# In[5]:


gex_query_adata.obs['individual'] = list(map(lambda x: x.split("-")[-1], gex_query_adata.obs.index))


# In[6]:


tdi.pp.update_anndata(gex_query_adata)


# We wrapped our TCR-DeepInsight model into few lines of Python code here

# In[9]:


_ = tdi.tl.get_pretrained_gex_embedding(
    gex_query_adata,
    gex_reference_adata=gex_reference_adata,
    transfer_label='cell_type_1',
    checkpoint_path='./tcr_deep_insight/data/pretrained_weights/human_vae_gex_all_cd4_cd8_v1.ckpt',
    device='cuda:0'
)


# We get the joint representation of the reference GEX and the query new dataset GEX and project then to 2D using UMAP

# In[10]:


import matplotlib.pyplot as plt
fig,ax=tdi.pl.createFig((5,5))
tdi.pl.umap(gex_query_adata, color='cell_type_1', palette=tdi.pl.palette.reannotated_prediction_palette,ax=ax)


# In[11]:


import matplotlib.pyplot as plt
ig,ax=tdi.pl.createFig((5,5))
tdi.pl.umap(gex_query_adata, color='sample_name', palette=tdi.pl.palettes.godsnot_102,ax=ax)


# In[12]:


tcr_query_adata = tdi.pp.unique_tcr_by_individual(gex_query_adata, label_key='cell_type_1')


# In[87]:


tcr_query_adata.obs['disease_type_2'] = 'Gastric Cancer'
tcr_reference_adata.obs['disease_type_2'] = 'Others'


# In[69]:


tdi.tl.get_pretrained_tcr_embedding(
    tcr_adata=tcr_query_adata,
    bert_config=tdi.model.config.get_human_config(),
    checkpoint_path='./tcr_deep_insight/data/pretrained_weights/human_bert_tcr_768_v1.ckpt',
    pca_path='./tcr_deep_insight/data/pretrained_weights/human_bert_tcr_pca_v1.pkl',
    use_pca=True
)


# In[88]:


tcr_cluster_result = tdi.tl.cluster_tcr_from_reference(
    tcr_query_adata,
    tcr_reference_adata,
    label_key='disease_type_2',
    gpu=1
)


# In[89]:


tdi.tl.inject_tcr_cluster(tcr_query_adata, tcr_cluster_result, 'cell_type_1')


# In[5]:


import numpy as np
result_tcr = tcr_cluster_result.obs[
    np.array(tcr_cluster_result.obs['count'] > 3) & 
    np.array(tcr_cluster_result.obs['number_of_individuals'] > 1) & 
    np.array(tcr_cluster_result.obs['cell_type_1'] != 'MAIT')
]


# In[6]:


gex_query_adata=tdi.read_h5ad("/Users/snow/Downloads/gex_query_adata.h5ad")


# In[7]:


tcrs = list(filter(lambda x: x != '-', result_tcr.sort_values('tcr_similarity_score', ascending=False).iloc[0,1:41])) 

tdi.pl.plot_gex_tcr_selected_tcrs(
    gex_adata = gex_query_adata,
    color='cell_type',
    tcrs=tcrs,
    palette=tdi.pl.palette.reannotated_prediction_palette,
)

