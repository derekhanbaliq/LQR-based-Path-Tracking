import cvxpy
import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags

A_block = []
A = np.array([[1, 2, 3, 4],
              [0, 0, 0, 0],
              [5, 6, 7, 8],
              [0, 0, 0, 9]])

A_block.append(A)
A_block.append(A)
print(A_block)  # length = 2, each element's type is ndarray
print("----------------")

A_block = block_diag(tuple(A_block))  # from list to coo_matrix only! ndarray failed!
print(type(A_block))
print(A_block)
m, n = A_block.shape  # 8 x 8 sparse diagonal matrix
print("----------------")

Annz_k = cvxpy.Parameter(A_block.nnz)
# print(A_block.nnz)  # 32
# print(Annz_k)  # param1
# print("----------------")

data = np.ones(Annz_k.size)  # 32 x 1, all elements are 1, Annz_k.size = 32
# print(data)
# print(A_block.row)
# print(A_block.col)

rows = A_block.row * n + A_block.col
# print(rows)  # No. ? element in 8 x 8 matrix
cols = np.arange(Annz_k.size)
# print(cols)  #  0, 1, 2, ... 30, 31 - we have 32 elements that need to be care - diagonal & nonzero
print("----------------")

Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, Annz_k.size))
print(Indexer)  # (rows, cols)	data
print("----------------")

Annz_k.value = A_block.data  # list, length = 32
print(A_block.data)
print("----------------")

# https://www.cvxpy.org/api_reference/cvxpy.atoms.affine.html#cvxpy.reshape
Ak_ = cvxpy.reshape(Indexer @ Annz_k, (m, n), order="C")
# print(Ak_)  # Indexer @ param11, (8, 8), language C
print("----------------")