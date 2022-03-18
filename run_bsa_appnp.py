from bsa_appnp.read_utils import utils_cpp, utils, sparse_serialize
from bsa_appnp.bsa_appnp import bsann
from bsa_appnp.GraphGenerator import  _precompute_block,  calc_A_hat, load_blocks, save_blocks, convert_sparse_matrix_to_sparse_tensor, calc_A_hatmod
import os
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import copy
from timeit import default_timer as timer
import numpy as np
from sklearn.metrics import accuracy_score
from bsa_appnp.training import traininng
from bsa_appnp.predict import BSA
import gc
from memory_profiler import profile, memory_usage
import scipy.sparse as sp
import struct

main_seed = 42
tf.random.set_seed(0)

#@profile(precision=10)
def run(seed, batch_size, btl_, niter, gamma, data_name,  load_check, dim, \
    alpha, maxepochs, sparse=True, tau=100, bsa_type_cpp=False, thread_num=1,\
         extra_logs=False, persent=100):

    if bsa_type_cpp:
        from predictc import BSAcpp
    else:
    #vscode cant import cython modules in debug mode
        class BSAcpp:
            pass
    
    if "test" not in data_name:
        rs = np.random.RandomState(seed=main_seed)
        batch_size_logits = 10000
        compute_ED = 0
        num_edges = []
        indexes_dict = {}
        if data_name == 'reddit':
            sparse = False

        file_path = f'{data_name}.npz'
        A, attr_matrix, labels, train_idx, val_idx, test_idx =\
            utils.get_data(
                f"bsa_appnp/data/{file_path}",
                seed=seed,
                ntrain_div_classes=20,
                normalize_attr=None)

        nnodes = A.shape[0]
        all_n = np.arange(nnodes)
        D_vec = np.sum(A, axis=1).A1
        Ah = calc_A_hat(A)
        nclasses = len(np.unique(labels))
        optimal_batch_size = int(nnodes / np.median(D_vec)) #средняя степень вершин
        #optimal_batch_size = batch_size
        batch_all = np.array(list(set(all_n) - set(train_idx)))
        labels_test = labels[test_idx]
        labels = np.array(labels)

        if btl_ > len(train_idx):
            btl = len(train_idx) - 1
        else:
            btl = copy.copy(btl_)

        rs.shuffle(batch_all)
        start = timer()
        #print('batch size', batch_size)
        partitions = np.array_split(batch_all, batch_size)

        tr_id_temp = rs.choice(train_idx, btl, False)
        for ii, val in enumerate(partitions):
            indexes_dict[ii] = val
            #temp_ = np.concatenate([tr_id_temp, val ])
            #print(A[tr_id_temp,:][:, val].count_nonzero())
            #num_edges.append(A[tr_id_temp,:][:, val].count_nonzero())
        #print('edges in batch', num_edges)

        time_batch_generation = timer() - start
        model = \
            bsann(nnodes=nnodes,
                nclasses=nclasses,
                ndim=dim,
                niterations=niter,
                gamma=gamma,
                batch=batch_size,
                alpha=alpha)

        ed_name = f'./cache/ED_'+data_name+"_"+str(optimal_batch_size)+'test.npy'
        all_name =f'./cache/all_batches_'+data_name+"_"+str(optimal_batch_size)+'test.npy'

        if load_check:
            ED_mat,  all_batches = \
                load_blocks(ed_name, all_name)
            print(ed_name)
            print(all_name)
        else:
            start = timer()
            ED_mat, all_batches = \
                _precompute_block(batch_size=optimal_batch_size,
                                adj_matrix=A,
                                seed=main_seed)
            compute_ED = timer() - start
            ED_mat = np.array(calc_A_hatmod(ED_mat, sigma=0))

            save_blocks(ED_mat, all_batches, ed_name, all_name)
            #print("saved!!!")

        parameters = dict()
        parameters['Ah'] = Ah
        parameters['device'] = '/GPU:0'
        parameters['model'] = model
        parameters['train_idx'] = train_idx
        parameters['maxepochs'] = maxepochs
        parameters['dataset'] = indexes_dict
        parameters['attr_matrix'] = attr_matrix# utils.normalize_attributes()
        parameters['btl'] = btl
        parameters['val_idx'] = val_idx
        parameters['Y'] = labels
        parameters['sparse'] = sparse
        parameters['data_name'] = data_name
        vars_, training_time = traininng(parameters=parameters)
        model.trainable_variables[0].assign(vars_[0])
        model.trainable_variables[1].assign(vars_[1])
        model.save_weights(f'./weights/my_model_{data_name}')
        del vars_
        gc.collect()

        Z = []
        start = timer()

        for i in range(0, nnodes, batch_size_logits):
            attr_rows = convert_sparse_matrix_to_sparse_tensor(parameters['attr_matrix'][i:i+batch_size_logits]) if \
                sparse else\
                parameters['attr_matrix'][i:i+batch_size_logits]
            zt = tf.nn.softmax(model.predict(attr_rows)).numpy()
            Z.append(zt)
        end = timer()
        Z = np.row_stack(Z)

        inference_time_batch = end - start
        Ah[Ah.nonzero()] = Ah[Ah.nonzero()] * alpha

    print("="*100)
    if data_name == "test":
        indptr = np.array([0, 2, 3, 6])
        indices = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
        Ah = sp.csr_matrix((data, indices, indptr), shape=(3, 3))
    
    if data_name == "test1":
        row = np.array([0,0,0,1,2,2,2,2,3,4,4,4,5,5,5])
        col = np.array([0,2,4,2,0,1,2,4,2,1,3,5,0,2,5])
        data = np.array([0.01, 0.02, 0.01, 0.03, 0.04, 0.05, 0.06, 0.02, 0.05, 0.04, 0.03, 0.01, 0.02, 0.01, 0.01])
        Ah = sp.csr_matrix((data, (row, col)), shape=(6, 6))
        #print(Ah.toarray())

        Z = np.array([\
            #np.array([0.011, 0.012, 0.013, 0.014, 0.015, 0.016], dtype=np.float32),\
            #np.array([0.021, 0.022, 0.023, 0.024, 0.025, 0.026], dtype=np.float32),\
            #np.array([0.031, 0.032, 0.033, 0.034, 0.035, 0.036], dtype=np.float32),\
            #np.array([0.041, 0.042, 0.043, 0.044, 0.045, 0.046], dtype=np.float32),\
            #np.array([0.051, 0.052, 0.053, 0.054, 0.055, 0.056], dtype=np.float32),\
            #np.array([0.061, 0.062, 0.063, 0.064, 0.065, 0.066], dtype=np.float32)])
            np.array([11, 12], dtype=np.float32),\
            np.array([21, 22], dtype=np.float32),\
            np.array([31, 32], dtype=np.float32),\
            np.array([41, 42], dtype=np.float32),\
            np.array([51, 52], dtype=np.float32),\
            np.array([61, 62], dtype=np.float32)])

        #all_batches = [\
        # np.array([0], dtype=np.int32), \
        # np.array([1,2], dtype=np.int32), \
        # np.array([3,4,5], dtype=np.int32)]
        
        all_batches = [\
            np.array([0, 1], dtype=np.int32), \
            np.array([2, 3], dtype=np.int32), \
            np.array([4, 5], dtype=np.int32)]
        
        np.random.seed(main_seed)
        ED_mat = np.random.rand(3,3)
        row_sums = ED_mat.sum(axis=1)
        ED_mat = ED_mat / row_sums[:, np.newaxis]

        test_idx = [4,5]
        labels_test = [1,1]
    
    if False:
        adj_matrix_save_path =  f"bsa_appnp/data/{data_name}.npz"
        sp.save_npz(adj_matrix_save_path, Ah, compressed=False)
    
    if False:
        Ah = Ah.tocsc()
        sparse_serialize.serialize_sparse(f"bsa_appnp/data/{data_name}_A", Ah)
    

    n_butches = len(all_batches)
    max_len = len(max(all_batches, key=len))
    all_batches_squared = []
    all_batches_persent = all_batches[ : int(len(all_batches) * persent/100)]
    for b in all_batches_persent:
        diff = np.empty(max_len - len(b), dtype=np.int32)
        diff.fill(-1)
        all_batches_squared.append(np.concatenate((b, diff), axis=0))

    all_batches_squared = np.array(all_batches_squared)
    all_batches_squared = all_batches_squared.ravel(order='C')\
    .reshape(all_batches_squared.shape[0], all_batches_squared.shape[1], order='C')
    
    epsilon=0.1 
    gamma=0.3
    ED_mat = ED_mat.ravel(order='F').reshape(ED_mat.shape[0], ED_mat.shape[1], order='F').astype('float32')
    Q = epsilon / n_butches + (1 - epsilon) * ED_mat
    
    Z = Z.ravel(order='F').reshape(Z.shape[0], Z.shape[1], order='F').astype('float32')

    b = (1 - alpha) * Z
    A = Ah.astype('float32')
    x = Z
    P = ED_mat
    x_prev = copy.deepcopy(x)

    list_batches = np.arange(n_butches)
    rs = np.random.RandomState(seed=seed)
    if not bsa_type_cpp:
        if False:
            rows_id_seq = [list(range(thread_num)),]
            
            assert(thread_num < n_butches * 2)
            for i in range(tau//thread_num):
                cur_bathces = []
                for j in range(thread_num):
                    b = rows_id_seq[i][0]
                    while b in cur_bathces: # or b in rows_id_seq[i]:
                        b = rs.choice(list_batches, 1, p=Q[rows_id_seq[i][j]])[0]
                        
                    cur_bathces.append(b)
                rows_id_seq.append(cur_bathces)
                print(f"{i}: {rows_id_seq[i+1]} --> {rows_id_seq[i]}")
        
            rows_id_seq = np.array([np.array(el) for el in rows_id_seq])
            rows_id_seq = rows_id_seq.transpose()
        else:
            assert(tau % 2)
            rows_id_seq = []
            for i in range(tau):
                cur_bathces = []
                for j in range(n_butches):
                    b_ = rs.choice(list_batches, 1, p=Q[j])[0]
                    cur_bathces.append(b_)
                rows_id_seq.append(cur_bathces)

            rows_id_seq = np.array([np.array(el) for el in rows_id_seq])
            rows_id_seq = rows_id_seq.transpose()
        
        with open("./logs/bsa_serialized.py.log", "w") as f:
            f.write(f"{data_name} {niter} {epsilon} {gamma} {thread_num} {tau}")

        #print_matsp_i("./logs", f"A", A)
        utils_cpp.print_mat("./logs", "b", b)
        utils_cpp.print_mat("./logs", "x", x)
        utils_cpp.print_mat("./logs", "x_prev", x_prev)
        utils_cpp.print_mat("./logs", "P", P)
        utils_cpp.print_mat("./logs", "Q", Q)
        #utils_cpp.print_mat("./logs", "all_batches", all_batches)
        utils_cpp.print_mat("./logs", "rows_id_seq", rows_id_seq)
        utils_cpp.print_mat("./logs", "all_batches_squared", all_batches_squared)

        #print(f"python: extra_logs={extra_logs}")
    else:
        rows_id_seq=utils_cpp.read_mat("./logs/rows_id_seq_mat.py.log")
        rows_id_seq = rows_id_seq.ravel(order='F').reshape(rows_id_seq.shape[0], rows_id_seq.shape[1], order='F').astype('int32')
        #rows_id_seq = rows_id_seq.astype('int32')



    linear_time = 0
    A_map = dict()
    for worker_index in range(rows_id_seq.shape[0]):
        rows_id = worker_index
        A_map[rows_id] = dict()
        for work_index in range(tau):
            batch_id = rows_id_seq[worker_index, work_index]
            rows_ = all_batches[rows_id]
            cols_ = all_batches[batch_id]
            A_map[rows_id][batch_id] = A[rows_, :][:, cols_]

    sparse_serialize.serialize_sparse_map(f"bsa_appnp/data/{data_name}_A_map", A_map)

    if bsa_type_cpp:
        py_bsa = BSAcpp(\
            b,\
            x_prev,
            x,
            P,
            Q,
            rows_id_seq,
            all_batches_squared,
            data_name,
            epsilon,
            gamma,
            niter, 
            thread_num,
            extra_logs,
            tau)
        py_bsa.bsa_operation()
        accuracy_prev = accuracy_score(labels_test, np.argmax(x_prev[test_idx], axis=1))
        accuracy_ = accuracy_score(labels_test, np.argmax(x[test_idx], axis=1))
        print(f"sum cpp : ", x.sum())
        print(f"accuracy{tau-1} cpp: {accuracy_prev}")
        print(f"accuracy{tau} cpp: {accuracy_}")


        print("="*50)
    else:
        Z, linear_time = \
            BSA(A_map=A_map,
                b=b,
                x_prev=x_prev,
                x=x,
                tau=tau,
                niter=niter,
                P=P,
                Q = Q,
                all_batches=all_batches,
                rows_id_seq=rows_id_seq,
                epsilon=epsilon,
                gamma=gamma,
                seed=main_seed,
                extra_logs=extra_logs)
        accuracy_prev = accuracy_score(labels_test, np.argmax(x_prev[test_idx], axis=1))
        accuracy_ = accuracy_score(labels_test, np.argmax(Z[test_idx], axis=1))
        
        print(f"sum py  : ", x.sum())
        print(f"accuracy{tau-1} py: {accuracy_prev}")
        print(f"accuracy{tau} py: {accuracy_}")
    
    #return accuracy_, \
    #       training_time, \
    #       inference_time_batch, \
    #       linear_time, training_time + \
    #       inference_time_batch + \
    #       linear_time + \
    #       time_batch_generation + \
    #       compute_ED, \
    #      num_edges

def check():

    res_py = utils_cpp.read_mat("./logs/x_res_mat.py.log")
    res_cpp = utils_cpp.read_mat("./logs/x_res_mat.cpp.log")
    print("="*50)
    print(f"sum cpp : ", res_py.sum())
    print(f"sum py  : ", res_cpp.sum())
    print(f"diff sum: ", (res_cpp - res_py).sum())


def accuracy_check(data_name, tau):
    
    file_path = f'{data_name}.npz'
    A, attr_matrix, labels, train_idx, val_idx, test_idx =\
        utils.get_data(
            f"bsa_appnp/data/{file_path}",
            seed=seed,
            ntrain_div_classes=20,
            normalize_attr=None)

    labels_test = labels[test_idx]

    res_py = utils_cpp.read_mat("./logs/x_res_mat.py.log")
    res_cpp = utils_cpp.read_mat("./logs/x_res_mat.cpp.log")
    
    accuracy_py = accuracy_score(labels_test, np.argmax(res_py[test_idx], axis=1))
    accuracy_cpp = accuracy_score(labels_test, np.argmax(res_cpp[test_idx], axis=1))

    print(f"accuracy py: {accuracy_py}")
    print(f"accuracy cpp: {accuracy_cpp}")



if __name__ == "__main__":
    import sys, getopt
    argv = (sys.argv[1:])
    try:
        opts, args = getopt.getopt(argv,"pckae",[])
    except getopt.GetoptError:
        sys.exit(2)
    
    bs = 512
    gamma = 0.3
    alpha = 0.9
    seed = 0
    tau = 1#100
    niter = 1
    dim = 64
    mepoch = 200#  200
    bsa_type_cpp = False
    thread_num = 6
    dataset_name = 'reddit'#'pubmed'
    load_check = False
    extra_logs = 0
    persent = 100
    
    if tau > 5: extra_logs = 0
    
    for opt, arg in opts:
        
        if opt == "-p":
            run(seed=seed,
                tau=tau,
                btl_=bs,
                batch_size=bs,
                niter=niter,
                gamma=gamma,
                data_name=dataset_name,
                load_check=load_check,
                dim=dim,
                alpha=alpha,
                maxepochs=mepoch,
                bsa_type_cpp=False,
                thread_num=thread_num,
                extra_logs=extra_logs,
                persent=persent
            )
        elif opt == "-c":
            run(seed=seed,
                tau=tau,
                btl_=bs,
                batch_size=bs,
                niter=niter,
                gamma=gamma,
                data_name=dataset_name,
                load_check=load_check,
                dim=dim,
                alpha=alpha,
                maxepochs=mepoch,
                bsa_type_cpp=True,
                thread_num=thread_num,
                extra_logs=extra_logs,
                persent=persent
            )
        elif opt == "-k":
            check()
        elif opt == "-a":
            run(seed=seed,
                tau=tau,
                btl_=bs,
                batch_size=bs,
                niter=niter,
                gamma=gamma,
                data_name=dataset_name,
                load_check=load_check,
                dim=dim,
                alpha=alpha,
                maxepochs=mepoch,
                bsa_type_cpp=False,
                thread_num=thread_num,
                extra_logs=extra_logs,
                persent=persent
            )
            run(seed=seed,
                tau=tau,
                btl_=bs,
                batch_size=bs,
                niter=niter,
                gamma=gamma,
                data_name=dataset_name,
                load_check=load_check,
                dim=dim,
                alpha=alpha,
                maxepochs=mepoch,
                bsa_type_cpp=True,
                thread_num=thread_num,
                extra_logs=extra_logs,
                persent=persent
            )
            check()
        elif opt == "-e":
            accuracy_check(dataset_name, tau=tau)
        else:
            assert(False and "WRONG CMD FLAG")

