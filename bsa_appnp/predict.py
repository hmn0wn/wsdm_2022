import numpy as np
import copy
from memory_profiler import profile
from timeit import default_timer as timer
from bsa_appnp.read_utils import utils

def BSA(A, b, x, all_batches, rows_id_seq, P, Q, niter=3, seed=0, epsilon=0.1, gamma=0.5):
    print("BSA python")

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

    start = timer()
    last_batch_id = batch_id
    for i in range(niter):
        # print("="*25)
        jump = P[rows_id, batch_id]
        qjump = Q[rows_id, batch_id]
        print(f"batch_id: {batch_id}")
        if False:
            print(f"jump: {jump}")
            print(f"qjump: {qjump}")
            utils.print_vec("./logs/loops/", f"rows{i}", rows_)
            utils.print_vec("./logs/loops/", f"cols{i}", cols_)
            utils.print_mat("./logs/loops", f"x_rows{i}", x[rows_])
            utils.print_matsp("./logs/loops", f"A_{i}", A[rows_, :][:, cols_])
            utils.print_mat("./logs/loops/", f"x_cols{i}", x[cols_])
            utils.print_mat("./logs/loops/", f"b_rows{i}", b[rows_])

        res = x[rows_] + \
            1/(1+i)**gamma * jump/qjump *\
            (1 / jump * A[rows_, :][:, cols_] @ x[cols_] -
             x[rows_] + b[rows_])
        #print_mat("./logs", f"res{i}", res, True)

        x[rows_] = res

        rows_id = copy.copy(batch_id)
        if random_jump:
            batch_id = rs.choice(list_batches, 1, p=Q[rows_id])[0]
        else:
            batch_i = (batch_i + 1) % n_butches
            batch_id = list_batches[batch_i]
        rows_ = all_batches[last_batch_id]
        last_batch_id = batch_id

        cols_ = all_batches[batch_id]
    utils.print_mat("./logs", f"x_res", x, False)
    end = timer()
    print("="*50)
    print(f"py BSA time: {end-start} s")
    return x, end - start
