import os
import argparse
import sys
import numpy as np
import random
import pandas as pd
from typing import Iterable
import torch
import torch.nn.functional as F
from collections import Counter
import scanpy as sc
import numba
from umap.distances import euclidean
import scipy 
import math 
import tabulate
import warnings
import tqdm
from ._logger import Colors, mw
from ._parallelizer import Parallelizer
from ._decorators import deprecated

is_nvidia_smi_warned = False

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

def print_gpu_mem(i):
    global is_nvidia_smi_warned
    try:
        import nvidia_smi
    except:
        if not is_nvidia_smi_warned:
            mw("install nvidia_smi with pip install nvidia-ml-py3 for automatically select cuda device by memory usage.")
            is_nvidia_smi_warned = True
        return "0", "0", "0 %"
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    return convert_size(info.used), convert_size(info.total), '%.2f' % ((info.used / info.total) * 100) + ' %'


def get_default_device():
    """
    Get the default device for training
    """
    devices = {}
    n_cuda_device = torch.cuda.device_count()
    if n_cuda_device > 0:
        for i in range(n_cuda_device):
            used, total, perc = print_gpu_mem(i)
            devices['cuda:' + str(i)] = {
                'name': torch.cuda.get_device_name(i),
                'used': used,
                'total': total,
                'used %': perc
            }
    else:
        return 'cpu'
    selected_device = sorted(devices.items(), key=lambda x: float(x[1]['used %'].split(" ")[0]))[0][0]
    for k,v in devices.items():
        if k == selected_device:
            v['selected'] = '*'
        else:
            v['selected'] = ' '
    _df = pd.DataFrame(devices).T
    print(tabulate.tabulate(_df, headers=_df.columns))
    return selected_device


def FLATTEN(x): 
    return [i for s in x for i in s]


def multi_values_dict(keys, values):
    ret = {}
    for k,v in zip(keys, values):
        if k not in ret.keys():
            ret[k] = [v]
        else:
            ret[k].append(v)
    return ret


def random_subset_by_key_fast(adata, key, n):
    from collections import Counter
    counts = {k:v/len(adata) for k,v in Counter(adata.obs[key]).items()}
    ns = [(k,int(v*n)) for k,v in counts.items()]
    all_indices = []
    for k,v in ns:
        indices = np.argwhere(np.array(adata.obs[key] == k)).flatten()
        if len(indices) > 0:
            indices = np.random.choice(indices, v, replace=False)
            all_indices.append(indices)
    all_indices = np.hstack(all_indices)
    return adata[all_indices]


def exists(x):
    return x is not None

def absent(x):
    return x is None


def print_version():
    print(Colors.YELLOW)
    print('Python VERSION:{}\n'.format(Colors.NC), sys.version)
    print(Colors.YELLOW)
    print('PyTorch VERSION:{}\n'.format(Colors.NC), torch.__version__)
    print(Colors.YELLOW)
    print('CUDA VERSION{}\n'.format(Colors.NC))
    from subprocess import call
    try: call(["nvcc", "--version"])
    except: pass
    print(Colors.YELLOW)
    print('CUDNN VERSION:{}\n'.format(Colors.NC), torch.backends.cudnn.version())
    print(Colors.YELLOW)
    print('Number CUDA Devices:{}\n'.format(Colors.NC), torch.cuda.device_count())
    try:
        print('Devices             ')
        call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    except FileNotFoundError:
        # There is no nvidia-smi in this machine
        pass
    if torch.cuda.is_available():
        print('Active CUDA Device: GPU', torch.cuda.current_device())
        print ('Available devices     ', torch.cuda.device_count())
        print ('Current cuda device   ', torch.cuda.current_device())
    else:
        # cuda not available
        pass

def readMM(path):
    f = open(path)
    while 1:
        line = f.readline()
        if not line.startswith("%"):
            break 

    n,m,total=line.strip().split(" ")
    n = int(n)
    m = int(m)
    total = int(total)
    z = np.zeros((n,m),dtype=np.int32)
    pbar = tqdm.tqdm(total=total)
    while 1:
        line = f.readline()
        if not line:
            break
        a,b,c = line.split(' ')
        a = int(a)
        b = int(b)
        c = int(c)
        z[a-1,b-1]=c
        pbar.update(1)
    pbar.close()
    return scipy.sparse.csr_matrix(z.T)

def iter_spmatrix(matrix):
    """ Iterator for iterating the elements in a ``scipy.sparse.*_matrix`` 

    This will always return:
    >>> (row, column, matrix-element)

    Currently this can iterate `coo`, `csc`, `lil` and `csr`, others may easily be added.

    Parameters
    ----------
    matrix : ``scipy.sparse.sp_matrix``
      the sparse matrix to iterate non-zero elements
    """
    from scipy.sparse import isspmatrix_coo, isspmatrix_csc, isspmatrix_csr, isspmatrix_lil
    if isspmatrix_coo(matrix):
        for r, c, m in zip(matrix.row, matrix.col, matrix.data):
            yield r, c, m

    elif isspmatrix_csc(matrix):
        for c in range(matrix.shape[1]):
            for ind in range(matrix.indptr[c], matrix.indptr[c+1]):
                yield matrix.indices[ind], c, matrix.data[ind]

    elif isspmatrix_csr(matrix):
        for r in range(matrix.shape[0]):
            for ind in range(matrix.indptr[r], matrix.indptr[r+1]):
                yield r, matrix.indices[ind], matrix.data[ind]

    elif isspmatrix_lil(matrix):
        for r in range(matrix.shape[0]):
            for c, d in zip(matrix.rows[r], matrix.data[r]):
                yield r, c, d

    else:
        raise NotImplementedError("The iterator for this sparse matrix has not been implemented")

def writeMM(mat, path):
    with open(path, 'w') as f:
        f.write("%%MatrixMarket matrix coordinate integer general\n")
        f.write("% Generated by Snowxue\n")
        f.write("% \n")
        f.write(str(mat.shape[1]) + " " + str(mat.shape[0]) + " " + str(int(mat.nnz)) + "\n")
        pbar = tqdm.tqdm(total=mat.nnz,bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',desc="writing matrix")
        for i,j,v in iter_spmatrix(mat):
            f.write(str(j+1) + " " + str(i+1) + " " + str(v) + "\n")
            pbar.update(1)
        pbar.close()

def next_unicode_char(char):
    return chr(ord(char) + 1)
