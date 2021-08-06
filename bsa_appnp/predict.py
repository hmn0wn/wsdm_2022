import numpy as np
import copy
from memory_profiler import profile
from timeit import default_timer as timer

#@profile(precision=10)
def BSA(A, b, x, all_batches, P, niter=3, seed=0, epsilon=0.1, gamma=0.5):
    rs = np.random.RandomState(seed=seed)
    n_butches = len(all_batches)
    Q = epsilon / n_butches + (1 - epsilon) * P
    list_batches = np.arange(n_butches)
    rows_id = 1
    batch_id = 1
    rows_ = all_batches[rows_id]
    cols_ = all_batches[batch_id]
    start = timer()
    for i in range(niter):
        jump = P[rows_id, batch_id]
        qjump = Q[rows_id, batch_id]
        x[rows_] = x[rows_]  + \
                   1/(1+i)**gamma * \
                   jump/qjump *\
                   (1 / jump * A[rows_, :][:, cols_] @ x[cols_] -
                      x[rows_] +
                      b[rows_])
        rows_id = copy.copy(batch_id)
        batch_id = rs.choice(list_batches, 1, p=Q[rows_id])[0]
        rows_ = np.copy(cols_)
        cols_ = all_batches[batch_id]
    end = timer()
    return x, end - start

