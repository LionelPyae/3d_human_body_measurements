
import scipy
# from scipy.sparse import csc_matrix
import numpy as np
import scipy.sparse as sp
import torch

def to_tensor(array):
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array).float()
    elif sp.issparse(array):
        array = sp.coo_matrix(array)
        indices = np.vstack((array.row, array.col)).astype(np.int64)
        values = array.data.astype(np.float32)
        size = array.shape
        return torch.sparse_coo_tensor(indices, values, size).to_dense()
    else:
        raise ValueError("Input should be a numpy array or scipy sparse matrix")

