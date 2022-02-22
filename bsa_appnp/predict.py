import numpy as np
import copy
from memory_profiler import profile
from timeit import default_timer as timer

#@profile(precision=10)
def BSA(A, b, x, all_batches, P, Q, niter=3, seed=0, epsilon=0.1, gamma=0.5):
    rs = np.random.RandomState(seed=seed)
    n_butches = len(all_batches)
    
    list_batches = np.arange(n_butches)
    random_jump = False
    if not random_jump:
        batch_i = 0
    
    rows_id = 1
    batch_id = 0
    rows_ = all_batches[rows_id]
    cols_ = all_batches[batch_id]

    with open("py.log", 'w') as f:
        #f.write("rows_: ", " ".join(map(str, rows_.tolist())))
        for batch in all_batches:
            f.write("batch: ")
            f.write(" ".join(map(str, batch.tolist())))
            f.write("\n")

    for i in range(10):
        el = rows_[i]
        print(el, end=" ")
    print()
    for i in range(10):
        el = cols_[i]
        print(el, end=" ")
    print()
    start = timer()

    for i in range(niter):
        print(f"batch_id: {batch_id}")
        jump = P[rows_id, batch_id]
        qjump = Q[rows_id, batch_id]
        x[rows_] = x[rows_]  + \
                1/(1+i)**gamma * \
                jump/qjump *\
                (1 / jump * A[rows_, :][:, cols_] @ x[cols_] -
                    x[rows_] +
                    b[rows_])
        rows_id = copy.copy(batch_id)
        if random_jump:
            batch_id = rs.choice(list_batches, 1, p=Q[rows_id])[0]
        else:
            batch_i = (batch_i + 1)%n_butches
            batch_id = list_batches[batch_i]
        rows_ = np.copy(cols_)
        cols_ = all_batches[batch_id]
    end = timer()
    return x, end - start

