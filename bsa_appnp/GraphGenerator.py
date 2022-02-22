import scipy.sparse

import tensorflow as tf
import scipy.sparse as sp
import numpy as np
rs = np.random.RandomState(seed=42)

def save_blocks(ED_mat, all_batches, name_edmat, name_batches  ):
    with open(name_edmat, 'wb') as f:
        np.save(f, ED_mat)
    with open(name_batches, 'wb') as f:
        np.save(f,  all_batches)


def load_blocks(name_edmat, name_batches):
    with open(name_edmat, 'rb') as f:
        Edmat = np.load(f)
    with open(name_batches, 'rb') as f:
        all_batches = np.load(f, allow_pickle=True)
    return Edmat, all_batches

def extract_submatrix(A, rows, cols ):
    return A[rows,:][:,cols]

def calc_A_hatmod(adj_matrix: sp.spmatrix, sigma) -> sp.spmatrix:
    A = adj_matrix
    D_vec = np.sum(A, axis=1).A1
    lsigma = sigma - 1
    rsigma = - sigma
    D_l = sp.diags(np.power(D_vec, lsigma))
    D_r = sp.diags(np.power(D_vec, rsigma))
    return D_l @ A @ D_r

def calc_A_hat( adj_matrix: sp.spmatrix , ) -> sp.spmatrix:
    D_vec = np.sum(adj_matrix, axis=1).A1
    D_vec = sp.diags(1/np.sqrt(D_vec))
    return D_vec @ adj_matrix @ D_vec

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.sparse.SparseTensor(indices, np.array(coo.data, dtype=np.float32), coo.shape)

def iid_divide(l, g):
    """
    divide list l among g groups
    each group has either int(len(l)/g) or int(len(l)/g)+1 elements
    returns a list of groups
    """
    num_elems = len(l)
    group_size = int(len(l)/g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    columns_indices = []
    global_counter = 0
    for i in range(num_small_groups):
        columns_indices.append(np.repeat(global_counter, group_size))
        global_counter += 1
        glist.append(l[group_size * i : group_size * (i + 1)])
    bi = group_size*num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        columns_indices.append(np.repeat(global_counter, group_size))
        global_counter += 1
        glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
    return glist, columns_indices

def _precompute_block(batch_size, adj_matrix, seed):
    nnodes = adj_matrix.shape[0]
    nbatches = nnodes / batch_size
    rs = np.random.RandomState(seed=seed)
    list_nodes = np.arange(adj_matrix.shape[0], dtype=np.int32)
    rs.shuffle(list_nodes)
    data = np.ones(nnodes)
    partitions, columns_indices = iid_divide(list_nodes, int(nbatches))
    columns_indices = np.hstack(columns_indices)
    B = scipy.sparse.csr_matrix((data, (columns_indices, np.hstack(partitions))))
    ED_mat = B @adj_matrix@B.T
    return ED_mat.todense(), partitions

def _batch_generation_uniform(bt, A, tr_id_temp, attr_matrix, Y, sparse):
    #if np.sum(A[tr_id_temp,:][:,bt]) == 0:
    #    bt = [bt[0]]
    y_gather = Y[tr_id_temp]
    rows_ = np.concatenate([tr_id_temp, bt])
    attr_rows = convert_sparse_matrix_to_sparse_tensor(attr_matrix[rows_]) if sparse else attr_matrix[rows_]
    A_batch = convert_sparse_matrix_to_sparse_tensor(extract_submatrix(A, rows_, rows_))
    return A_batch, attr_rows, None, y_gather
