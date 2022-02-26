from bsa_appnp.read_utils import utils
from bsa_appnp.bsa_appnp import bsann
from bsa_appnp.GraphGenerator import  _precompute_block,  calc_A_hat, load_blocks, save_blocks, convert_sparse_matrix_to_sparse_tensor, calc_A_hatmod
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

DEBUG_ = True

if not DEBUG_:
    from predictc import BSAcpp
else:
    #vscode cant import cython modules in debug mode
    class BSAcpp:
        pass


def graphsave(adj, dir, to_sort=False):
    if (sp.isspmatrix_csr(adj)):
        el = adj.indices
        pl = adj.indptr
        data = adj.data

        EL = np.array(el, dtype=np.uint32)
        PL = np.array(pl, dtype=np.uint32)

        EL_re = []
        if to_sort:
            for i in range(1, PL.shape[0]):
                #EL_re += sorted(EL[PL[i - 1]:PL[i]], key=lambda x: PL[x + 1] - PL[x])
                EL_re += sorted(EL[PL[i - 1]:PL[i]])
        else:
            EL_re = EL

        EL_re = np.asarray(EL_re, dtype=np.uint32)

        print("EL:", EL_re.shape, " size: ", EL_re.size)
        f1 = open(dir + 'el.txt', 'wb')
        for i in EL_re:
            m = struct.pack('I', i)
            f1.write(m)
        f1.close()

        print("PL:", PL.shape, " size: ", PL.size)
        f2 = open(dir + 'pl.txt', 'wb')
        for i in PL:
            m = struct.pack('I', i)
            f2.write(m)
        f2.close()

        print("DL:", data.shape, " size: ", data.size)
        f3 = open(dir + 'dl.txt', 'wb')
        for i in data:
            m = struct.pack('d', i)
            #print(m)
            f3.write(m)
        f3.close()

        return EL_re.size, PL.size
    else:
        print("Format Error!")


#@profile(precision=10)
def run(seed, batch_size, btl_, niter, gamma, data_name,  load_check, dim, \
    alpha, maxepochs, sparse=True, tau=100, bsa_type_cpp=False):
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
    #optimal_batch_size = int(nnodes / np.median(D_vec))
    optimal_batch_size = int(nnodes / 24)
    batch_all = np.array(list(set(all_n) - set(train_idx)))
    labels_test = labels[test_idx]
    labels = np.array(labels)

    if btl_ > len(train_idx):
        btl = len(train_idx) - 1
    else:
        btl = copy.copy(btl_)

    rs.shuffle(batch_all)
    start = timer()
    print('bs', batch_size)
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

    if load_check:
        ED_mat,  all_batches = \
            load_blocks(f'ED_'+data_name+"_"+str(optimal_batch_size)+'test.npy',
                        f'all_batches_'+data_name+"_"+str(optimal_batch_size)+'test.npy')
        print('ED_'+data_name+"_"+str(optimal_batch_size)+'test.npy')
        print('all_batches_'+data_name+"_"+str(optimal_batch_size)+'test.npy')
    else:
        start = timer()
        ED_mat, all_batches = \
            _precompute_block(batch_size=optimal_batch_size,
                              adj_matrix=A,
                              seed=main_seed)
        compute_ED = timer() - start
        ED_mat = np.array(calc_A_hatmod(ED_mat, sigma=0))

        save_blocks(ED_mat, all_batches, 'ED_'+data_name+"_"+str(optimal_batch_size)+'test.npy',
                    'all_batches_'+data_name+"_"+str(optimal_batch_size)+'test.npy')
        print("saved!!!")

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

    if False:
        data_name = "test"
        indptr = np.array([0, 2, 3, 6])
        indices = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
        Ah = sp.csr_matrix((data, indices, indptr), shape=(3, 3))
    if False:
        data_name = "test1"
        row = np.array([0,0,0,1,2,2,2,2,3,4,4,4,5,5,5])
        col = np.array([0,2,4,2,0,1,2,4,2,1,3,5,0,2,5])
        data = np.array([0.01, 0.02, 0.01, 0.03, 0.04, 0.05, 0.06, 0.02, 0.05, 0.04, 0.03, 0.01, 0.02, 0.01, 0.01])
        Ah = sp.csr_matrix((data, (row, col)), shape=(6, 6))
        print(Ah.toarray())

        Z = np.array([\
            #np.array([0.011, 0.012, 0.013, 0.014, 0.015, 0.016], dtype=np.float32),\
            #np.array([0.021, 0.022, 0.023, 0.024, 0.025, 0.026], dtype=np.float32),\
            #np.array([0.031, 0.032, 0.033, 0.034, 0.035, 0.036], dtype=np.float32),\
            #np.array([0.041, 0.042, 0.043, 0.044, 0.045, 0.046], dtype=np.float32),\
            #np.array([0.051, 0.052, 0.053, 0.054, 0.055, 0.056], dtype=np.float32),\
            #np.array([0.061, 0.062, 0.063, 0.064, 0.065, 0.066], dtype=np.float32)])
            np.array([0.011, 0.012], dtype=np.float32),\
            np.array([0.021, 0.022], dtype=np.float32),\
            np.array([0.031, 0.032], dtype=np.float32),\
            np.array([0.041, 0.042], dtype=np.float32),\
            np.array([0.051, 0.052], dtype=np.float32),\
            np.array([0.061, 0.062], dtype=np.float32)])

        all_batches = [np.array([0,1,2], dtype=np.int32), np.array([3,4,5], dtype=np.int32)]
        test_idx = [4,5]
        labels_test = [1,1]
    if False:
        adj_matrix_save_path =  f"bsa_appnp/data/adj_{data_name}.npz"
        sp.save_npz(adj_matrix_save_path, Ah, compressed=True)
    if False:
        m,n = graphsave(Ah, f"bsa_appnp/data/{data_name}_adj_", to_sort=False)
        n = n - 1 
    else:
        if data_name=="cora_full":
            n, m = 18800, 125370
        if data_name=="pubmed":
            n, m = 19717, 88648
        if data_name=='reddit':
            n, m = 232965, 23446803
        if data_name=='reddit':
            n, m = 232965, 114615892
        if data_name=='citeseer':
            n, m = 2110, 7388
        if data_name=='test':
            n,m = 3,6
        if data_name=='test1':
            n,m = 5,11

    all_batches = np.array(all_batches)
    n_butches = len(all_batches)
    epsilon=0.1
    gamma=0.3
    Q = epsilon / n_butches + (1 - epsilon) * ED_mat
    Q = np.concatenate([Q,Q])
    Z = Z.ravel(order='F').reshape(Z.shape[0], Z.shape[1], order='F').astype('float64')
    if bsa_type_cpp:
        if not DEBUG_:
            linear_time = 0
                #Z, linear_time = \
            py_bsa = BSAcpp()
            py_bsa.bsa_operation(data_name, Ah.shape[0], n, m, (1 - alpha) * Z, Z, tau, ED_mat, Q, all_batches, epsilon, gamma, main_seed)
    
        print("="*100)
    else:
        Z, linear_time = \
            BSA(A=Ah,
                b=(1 - alpha) * Z,
                x=Z,
                niter=tau,
                P=ED_mat,
                Q = Q,
                all_batches=all_batches,
                epsilon=epsilon,
                gamma=gamma,
                seed=main_seed)


    accuracy_ = accuracy_score(labels_test, np.argmax(Z[test_idx], axis=1))
    print('accuracy', accuracy_)
    return accuracy_, \
           training_time, \
           inference_time_batch, \
           linear_time, training_time + \
           inference_time_batch + \
           linear_time + \
           time_batch_generation + \
           compute_ED, \
           num_edges


dataset_name = 'cora_full'#'pubmed'
bs = 64#512
gamma = 0.3
alpha = 0.9
seed = 0
tau = 12#100
niter = 1
dim = 64
mepoch = 50#  200
bsa_type_cpp = True
acc_test, time_training, time_inference, time_inference_linear, time_total, num_edges = \
    run(seed=seed,
        tau=tau,
        btl_=bs,
        batch_size=bs,
        niter=niter,
        gamma=gamma,
        data_name=dataset_name,
        load_check=False,
        dim=dim,
        alpha=alpha,
        maxepochs=mepoch,
        #bsa_type_cpp=bsa_type_cpp
        )
