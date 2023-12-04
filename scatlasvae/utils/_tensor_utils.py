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

def get_k_elements(arr: Iterable, k:int):
    return list(map(lambda x: x[k], arr))

def get_last_k_elements(arr: Iterable, k:int):
    return list(map(lambda x: x[k:], arr))

def get_elements(arr: Iterable, a:int, b:int):
    return list(map(lambda x: x[a:a+b], arr))

def one_hot_(labels, return_dict = False):
    n_labels_ = np.unique(labels)
    n_labels = dict(zip(n_labels_, range(len(n_labels_))))
    if return_dict:
        return {"one_hot": F.one_hot( torch.tensor(list(map(lambda x: n_labels[x], labels)))), "labels": n_labels}
    return F.one_hot( torch.tensor(list(map(lambda x: n_labels[x], labels))))

def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    """One hot a tensor of categories."""
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)

    