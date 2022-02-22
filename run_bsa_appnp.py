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

DEBUG_ = False

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
        return EL_re.size, PL.size
    else:
        print("Format Error!")


#@profile(precision=10)
def run(seed, batch_size, btl_, niter, gamma, data_name,  load_check, dim, alpha, maxepochs, sparse=True, tau=100):
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
    optimal_batch_size = int(nnodes / np.median(D_vec))
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
        adj_matrix_save_path =  f"bsa_appnp/data/adj_{data_name}.npz"
        sp.save_npz(adj_matrix_save_path, Ah, compressed=True)

    if False:
        n, m = graphsave(Ah, f"bsa_appnp/data/{data_name}_adj_", to_sort=False)
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

    all_batches = np.array(all_batches)
    n_butches = len(all_batches)
    epsilon=0.1
    gamma=0.3
    Q = epsilon / n_butches + (1 - epsilon) * ED_mat
    if not DEBUG_:
        linear_time = 0
            #Z, linear_time = \
        py_bsa = BSAcpp()
        py_bsa.bsa_operation(data_name, n, m, (1 - alpha) * Z, Z, tau, ED_mat, Q, all_batches, epsilon, gamma, main_seed)
    
    print("="*100)
    
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


dataset_name = 'citeseer'#'pubmed'
bs = 64#512
gamma = 0.3
alpha = 0.9
seed = 0
tau = 5#100
niter = 1
dim = 64
mepoch = 16#  200
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
        maxepochs=mepoch)
