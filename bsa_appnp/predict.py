import numpy as np
import copy
from memory_profiler import profile
from timeit import default_timer as timer
from bsa_appnp.read_utils import utils

def BSA(A, b, x_prev,x, all_batches, rows_id_seq, P, Q, niter=3, seed=0, epsilon=0.1, gamma=0.5):
    print("BSA python")

    n_butches = len(all_batches)
    start = timer()
    if(False):
        for worker_index in range(rows_id_seq.shape[0]):
            for i in range(rows_id_seq.shape[1]-1):
                rows_id = rows_id_seq[worker_index, i]
                batch_id = rows_id_seq[worker_index, i+1]
                rows_ = all_batches[rows_id]
                cols_ = all_batches[batch_id]

                # print("="*25)
                jump = P[rows_id, batch_id]
                qjump = Q[rows_id, batch_id]
                #print(f"update: {rows_id}<-{batch_id}")
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
    else:
        for work_index in range(rows_id_seq.shape[1]):
            for worker_index in range(rows_id_seq.shape[0]):
                rows_id = worker_index
                batch_id = rows_id_seq[worker_index, work_index]
                #print(f"id: {batch_id} --> {rows_id}")
                rows_ = all_batches[rows_id]
                cols_ = all_batches[batch_id]
                jump = P[rows_id, batch_id]
                qjump = Q[rows_id, batch_id]
                if work_index % 2:
                    #print("x --> x_prev")
                    x_prev[rows_] = x[rows_] + \
                        1/(1+work_index)**gamma * jump/qjump *\
                        (1 / jump * A[rows_, :][:, cols_] @ x[cols_] -
                        x[rows_] + b[rows_])
                else:
                    #print("x_prev --> x")
                    x[rows_] = x_prev[rows_] + \
                        1/(1+work_index)**gamma * jump/qjump *\
                        (1 / jump * A[rows_, :][:, cols_] @ x_prev[cols_] -
                        x_prev[rows_] + b[rows_])
                if True:
                    #print(f"jump: {jump}")
                    #print(f"qjump: {qjump}")
                    findex = f"{work_index}_{worker_index}_{batch_id}->{rows_id}"

                    utils.print_matsp("./logs/loops/",  findex + f"_A"    , A[rows_, :][:, cols_])
                    utils.print_vec("./logs/loops/",    findex + f"_rows"   , rows_)
                    utils.print_vec("./logs/loops/",    findex + f"_cols"   , cols_)

                    utils.print_mat("./logs/loops/",    findex + f"_x_rows" , x[rows_])
                    utils.print_mat("./logs/loops/",    findex + f"_x_cols" , x[cols_])
                    utils.print_mat("./logs/loops/",    findex + f"_b_rows" , b[rows_])

                    utils.print_mat("./logs/loops/",    findex + f"_x"      , x, False)
                    utils.print_mat("./logs/loops/",    findex + f"_x_prev" , x_prev, False)

    end = timer()
    utils.print_mat("./logs", f"x_res", x, False)
    utils.print_mat("./logs", f"x_prev_res", x_prev, False)
    print("="*50)
    print(f"py BSA time: {end-start} s")
    return x, end - start
