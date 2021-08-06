import tensorflow as tf
from timeit import default_timer as timer
import numpy as np
from bsa_appnp.GraphGenerator import  _batch_generation_uniform, convert_sparse_matrix_to_sparse_tensor
from memory_profiler import profile
rs = np.random.RandomState(seed=42)
tf.random.set_seed(42)

#@profile(precision=10)
def traininng(parameters):
    sparse = parameters['sparse']
    train_idx = parameters['train_idx']
    Ah = parameters['Ah']
    device = parameters['device']
    model = parameters['model']
    maxepochs = parameters['maxepochs']
    dataset = parameters['dataset']
    attr_matrix = parameters['attr_matrix']
    btl = parameters['btl']
    Y = parameters['Y']
    val_idx = parameters['val_idx']
    max_val_ = np.inf


    with tf.device(device):
        batch_tr_idx = np.arange(btl)
        batch_val_idx = np.arange(len(val_idx))
        start = timer()
        for epoch in range(maxepochs):
            tr_id_temp = rs.choice(train_idx, btl, False)
            A_batch, attr_rows, _, y_gather = _batch_generation_uniform(A=Ah,
                                                                              attr_matrix=attr_matrix,
                                                                              bt=dataset[epoch],
                                                                              Y=Y,
                                                                              tr_id_temp=tr_id_temp,
                                                                              sparse=sparse)
            loss_, tr_var = model.network_learn(Y=y_gather,
                                                adj_matrix=A_batch,
                                                batch_idx=batch_tr_idx,
                                                Xr=attr_rows,
                                                i=epoch)

            if epoch % 20 == 0:
                val_loss = model.get_loss(Y=Y[val_idx],
                                          A=convert_sparse_matrix_to_sparse_tensor(Ah[val_idx,:][:,val_idx]),
                                          batch_idx=batch_val_idx,
                                          Xr=convert_sparse_matrix_to_sparse_tensor(attr_matrix[val_idx]) if sparse else attr_matrix[val_idx],
                                          i=epoch)
                loss = val_loss.numpy()
                if max_val_ > loss:
                    max_val_ = loss
                    best_trainables = tr_var
                    #model.save_weights('./weights/my_model_'+data_name)

        end = timer()
        return best_trainables,  end - start
