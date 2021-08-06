import numpy as np
import networkx as nx
import scipy.sparse as sp
import tracemalloc
import linecache
import os


def calc_A_hat(adj_matrix: sp.spmatrix,  sigma) -> sp.spmatrix:
    nnodes =  adj_matrix.shape[0]
    #print(adj_matrix)
    A = adj_matrix + sp.eye(nnodes)
    D_vec = np.sum(A, axis=1).A1
    lsigma = sigma - 1
    rsigma = - sigma
    D_l = sp.diags(np.power(D_vec, lsigma))
    D_r = sp.diags(np.power(D_vec, rsigma))
    return  D_l @ A @ D_r, D_vec


def calc_A_hatmod(adj_matrix: sp.spmatrix, sigma) -> sp.spmatrix:
    nnodes = adj_matrix.shape[0]
    A = adj_matrix
    D_vec = np.sum(A, axis=1).A1
    lsigma = sigma - 1
    rsigma = - sigma

    D_l = sp.diags(np.power(D_vec, lsigma))
    D_r = sp.diags(np.power(D_vec, rsigma))

    return D_l @ A @ D_r, D_l

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def lc(graph):
    A = graph.adj_matrix
    GA_ = nx.from_numpy_array(A.toarray())  #
    GAund = nx.to_undirected(GA_)
    Aund = nx.to_numpy_array(GAund)
    graphs = list(nx.connected_components(GAund))
    large_cc = list(graphs[0])
    Af = Aund[large_cc, :]
    Af = Af[:, large_cc]
    Af = sp.csr_matrix(Af)
    return Af , large_cc

def exact_pr(A,b):
    return np.linalg.inv(A)@ b