import tensorflow  as tf
import numpy as np
sparse_dot = tf.sparse.sparse_dense_matmul

tf.random.set_seed(0)

class bsann(tf.keras.Model):


    def _stochastic_approximation_layer(self, A, Z_rows,  i):
        Z_r = tf.identity(Z_rows)
        Ahat = tf.constant(self.alpha)*A
        for j in range(self.niterations):
            Z_r = Z_r +\
                  1 / (1 + i + j)**(self.gamma) *\
                  (self.nnodes/ (2*self.batch) *sparse_dot(Ahat, Z_r) -  Z_r + (1-self.alpha)*Z_rows)
        return tf.nn.softmax(Z_r)

    def __init__(self, nnodes, nclasses, ndim, niterations, alpha, batch, gamma):
        super(bsann, self).__init__()
        self.nnodes = nnodes
        self.nclasses = nclasses
        self.ndim = ndim
        self.niterations = niterations
        self.alpha = alpha
        self.batch = batch
        self.gamma = gamma

        xavier = tf.keras.initializers.glorot_uniform()
        self.out = tf.keras.layers.Dense(self.nclasses,
                                         kernel_initializer=xavier ,
                                         kernel_regularizer=tf.keras.regularizers.l2(5e-4))
        self.mean = tf.keras.layers.Dense(self.ndim, kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                                          kernel_initializer=xavier, activation=tf.nn.relu,)
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.train_op = tf.keras.optimizers.Adam(learning_rate=0.005)

    def call(self, A, Xr, i):
        dim_mean = self.mean(Xr)
        dim_mean = self.dropout(dim_mean)
        logit_row = self.out(dim_mean)
        return self._stochastic_approximation_layer(A=A,   Z_rows=logit_row,  i=i)

    def predict(self, X):
        dim_mean = self.mean(X)
        dim_mean = self.dropout(dim_mean)
        return self.out(dim_mean)

    # Custom loss fucntion
    def get_loss(self, Y, A, batch_idx,   Xr,  i):
        predictions_ = self.call(A=A, Xr=Xr,  i=i)
        predictions_ = tf.gather(predictions_, batch_idx, axis=0)
        cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(Y, predictions_)
        regularization_loss = tf.math.add_n(self.mean.losses)
        return tf.reduce_mean(cross_entropy) + regularization_loss

    # get gradients
    def get_grad(self,  Y, adj_matrix, batch_idx,  Xr,   i):
        with tf.GradientTape() as tape:
            tape.watch(self.mean.variables)
            tape.watch(self.out.variables)
            L = self.get_loss(Y, adj_matrix, batch_idx,  Xr,  i)
            g = tape.gradient(L, self.trainable_variables)
        return g, L

    # perform gradient descent
    def network_learn(self,  Y, adj_matrix, batch_idx,    Xr,   i):
        g, acc_ = self.get_grad(Y, adj_matrix, batch_idx,   Xr,   i)
        self.train_op.apply_gradients(zip(g, self.trainable_variables))
        return acc_, self.trainable_variables