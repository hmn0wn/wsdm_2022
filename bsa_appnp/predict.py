import numpy as np
import copy
from memory_profiler import profile
from timeit import default_timer as timer

def print_mat(dir_name, mat_name, mat, to_print=False):
    import os
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    nmax = 10
    if to_print: print(mat_name)
    with open(f"{dir_name}/{mat_name}_mat.py.log", 'w') as f:
        for i,eli in enumerate(mat):
            for j,elj in enumerate(eli):
                f.write(f"{elj:0.5f} ")
                if to_print: print(f"{elj:0.5f} ",end="")
                if j > nmax: break
            f.write("\n")
            if to_print: print()
            if i > nmax: break

def print_matsp(dir_name, mat_name, mat, to_print=False):
    import os
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    nmax = 10
    if to_print: print(mat_name)
    with open(f"{dir_name}/{mat_name}_mat.py.log", 'w') as f:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                f.write(f"{mat[i,j]:0.5f} ")
                if to_print: print(f"{mat[i,j]:0.5f} ", end='')
                if j > nmax: break
            f.write("\n")
            if to_print: print()
            if i > nmax: break

def print_matsp_i(dir_name, mat_name, mat):
    import os
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    nmax = 10
    with open(f"{dir_name}/{mat_name}_mat.py.log", 'w') as f:
        for i in range(mat.shape[0]):
            for j in mat[i].nonzero()[1]:
                f.write(f"{str(i)}\t{str(j)}\t\t: {str(mat[i,j])}\n")

def print_vec(vec_name, vec):
    nmax = 10
    print(f"{vec_name}: ", end='')
    for i,el in enumerate(vec):
        print(el, end=" ")
        if i > nmax: break
    print()

#@profile(precision=10)
def BSA(A, b, x, all_batches, P, Q, niter=3, seed=0, epsilon=0.1, gamma=0.5):
    print("BSA python")
    print_matsp_i("./logs", f"A", A)
    print_mat("./logs", "b", b)
    print_mat("./logs", "x", x)
    print_mat("./logs", "P", P)
    print_mat("./logs", "Q", Q)
    print_mat("./logs", "all_batches", all_batches)


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
        print("="*25)
        jump = P[rows_id, batch_id]
        qjump = Q[rows_id, batch_id]
        print(f"batch_id: {batch_id}")
        #print(f"jump: {jump}")
        #print(f"qjump: {qjump}")
        print_vec("rows_", rows_)
        print_vec("cols_", cols_)
        print_mat("./logs", f"x_rows{i}", x[rows_])
        print_matsp("./logs", f"A_{i}", A[rows_, :][:, cols_], True)
        print_mat("./logs", f"x_cols{i}", x[cols_], True)
        print_mat("./logs", f"b_rows{i}", b[rows_])
        res = A[rows_, :][:, cols_] @ x[cols_]
        print_mat("./logs", f"res{i}", res)

        #print_mat("./logs", f"x_rows{i}1", x[rows_])

        #x[rows_] = x[rows_]  + \
                #1/(1+i)**gamma * jump/qjump *\
                #(1 / jump * A[rows_, :][:, cols_] @ x[cols_] -\
                #x[rows_] + b[rows_])
        rows_id = copy.copy(batch_id)
        if random_jump:
            batch_id = rs.choice(list_batches, 1, p=Q[rows_id])[0]
        else:
            batch_i = (batch_i + 1)%n_butches
            batch_id = list_batches[batch_i]
        rows_ = all_batches[last_batch_id]
        last_batch_id = batch_id

        cols_ = all_batches[batch_id]
    end = timer()
    return x, end - start

