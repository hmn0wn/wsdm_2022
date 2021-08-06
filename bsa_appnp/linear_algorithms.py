import bsa_appnp.utils_linear as ut
import numpy as np
import copy
import scipy.sparse as sp
#from memory_profiler import profile

class Linear_algorithms:
    def __init__(self, A, b, sigma, alpha, gamma, niter, batch_size, betta, epsilon, stats, algorithm, seed):
        self.alorithm = algorithm
        self.A = A
        self.stats = stats
        self.nnodes = A.shape[0]
        self.betta = betta
        self.batch_size = batch_size
        self.niter = niter
        self.alpha = alpha
        self.sigma = sigma
        self.epsilon = epsilon
        self.Ahat, dl = ut.calc_A_hat(A, sigma)
        self.Ahat = self.alpha * self.Ahat
        self.gamma = gamma
        self.x = np.copy(b)
        self.b = (1 - alpha) * b
        self.list_nodes = np.arange(self.nnodes)
        self.Asystem = sp.eye(self.nnodes) - self.Ahat
        # initialization
        if stats:
            self.error_ = {}
            self.error_['x'] = []
            self.error_['error_'] = []
        if self.alorithm == 'dsbgs':
            # initialization
            self.A_norm_frob = np.square(np.abs(self.Asystem))
            self.P, self.partitions = self._precompute_block(batch_size=self.batch_size,
                                                             adj_matrix=self.A_norm_frob,
                                                             seed=seed)
            self.P = np.array(self.P.flatten())[0]
            self.A_norm_global = np.sum(self.A_norm_frob)
            self.batch_index = np.arange(len(self.P))
            self.P /= self.A_norm_global
            self.number_batches = len(self.partitions)
        elif self.alorithm == 'rbgs':
            # initialization
            self.P, self.partitions = self._precompute_block(batch_size=self.batch_size,
                                                             adj_matrix=self.Ahat,
                                                             seed=seed)
            self.c = self.b.shape[1]
            self.resid = np.copy(self.b) / (1 - self.alpha)
            self.number_batches = len(self.partitions)
            self.batch_index = np.arange(self.number_batches)
        elif self.alorithm == 'bsa':
            self.P, self.partitions = self._precompute_block(self.batch_size, self.A, seed=seed)
            self.number_batches = len(self.partitions)
            self.P, Dl  = np.array(ut.calc_A_hatmod(self.P, sigma=0))
            self.Q = self.epsilon/self.number_batches + (1 - self.epsilon) * self.P
            self.batch_index = np.arange(self.number_batches)

    def run(self, seed):
        if self.alorithm == "rk":
            self.rk(seed)
        elif self.alorithm == "jor":
            self.jor(seed)
        elif self.alorithm == 'dsbgs':
            self.dsbgs(seed)
        elif self.alorithm == 'rbgs':
            self.rbgs(seed)
        elif self.alorithm == 'bsa':
            self.bsa(seed)

    def _collect_stats(self):
        self.error_['x'].append(np.copy(self.x))
        self.error_['error_'].append(np.linalg.norm(self.Asystem@self.x - self.b))

    def iid_divide(self, l, g):
        """
        divide list l among g groups
        each group has either int(len(l)/g) or int(len(l)/g)+1 elements
        returns a list of groups
        """
        num_elems = len(l)
        group_size = int(len(l)/g)
        num_big_groups = num_elems - g * group_size
        num_small_groups = g - num_big_groups
        glist = []
        columns_indices = []
        global_counter = 0
        for i in range(num_small_groups):
            columns_indices.append(np.repeat(global_counter, group_size))
            global_counter += 1
            glist.append(l[group_size * i : group_size * (i + 1)])
        bi = group_size*num_small_groups
        group_size += 1
        for i in range(num_big_groups):
            columns_indices.append(np.repeat(global_counter, group_size))
            global_counter += 1
            glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
        return glist, columns_indices
    def _precompute_block(self, batch_size, adj_matrix, seed):
        nnodes = adj_matrix.shape[0]
        nbatches = nnodes / batch_size
        rs = np.random.RandomState(seed=seed)
        list_nodes = np.arange(adj_matrix.shape[0], dtype=int)
        rs.shuffle(list_nodes)
        data = np.ones(nnodes)
        partitions, columns_indices = self.iid_divide(list_nodes, int(nbatches))
        columns_indices = np.hstack(columns_indices)
        B = sp.csr_matrix((data, (columns_indices, np.hstack(partitions))))
        ED_mat = B @adj_matrix@B.T
        return ED_mat.todense(), partitions

    def jor(self, seed):
        rs = np.random.RandomState(seed=seed)
        for i in range(self.niter):
            node = rs.choice(self.list_nodes, 1, replace=False)
            self.x[node] =  self.x[node] + \
                            self.gamma * (self.Ahat[node] @ self.x  -
                                          self.x[node] +
                                          self.b[node])
            if self.stats == True:
                self._collect_stats()
    def rk(self, seed):
        rs = np.random.RandomState(seed=seed)
        for i in range(self.niter):
            node = rs.choice(self.list_nodes, 1, replace=False)
            A_mid = self.Asystem[node]
            y = A_mid.indices
            data = np.array([A_mid.data]).T
            step_ = self.b[node] - A_mid @ self.x
            self.x[y]  += data * step_ / np.sum(np.square(np.abs(data)))
            if self.stats == True:
                self._collect_stats()
    def dsbgs(self, seed):
        rs = np.random.RandomState(seed=seed)
        for i in  range(self.niter):
            batch_id = rs.choice(self.batch_index, 1, p=self.P)[0]
            row_batch = self.partitions[batch_id // self.number_batches]
            column_batch = self.partitions[batch_id % self.number_batches]
            A_norm = self.P[batch_id] * self.A_norm_global
            A_ = self.Asystem[row_batch, :]
            A_I =  A_[:, column_batch]
            self.x[column_batch] = self.x[column_batch] -  \
                                   self.betta * (A_I.T / A_norm)  @ \
                                   (A_ @ self.x - self.b[row_batch])
            if self.stats == True:
                self._collect_stats()
    def rbgs(self, seed):
        rs = np.random.RandomState(seed=seed)
        for i in range(self.niter):
            batch_id = rs.choice(self.batch_index, 1, replace=False)[0]
            tau = self.partitions[batch_id]
            bs = len(tau)
            T = np.arange(bs)
            A_ = self.Asystem[:, tau]
            E = sp.csr_matrix((np.ones(bs), (tau, T)), shape=(self.nnodes, bs))
            # our sparse implementation
            diff = []
            for c_ in range(self.c):
                diff.append(sp.linalg.lsmr(A_, self.resid[:,c_])[0])
            diff = np.array(diff).T
            self.x += E@diff
            # default dense implementation
            #A_ = pinv(A_.todense())
            #self.x = self.x+ E@A_@self.resid
            self.resid = self.b - self.Asystem@self.x
            if self.stats == True:
                self._collect_stats()

    # @profile(precision=10)
    def bsa(self, seed):
        rs = np.random.RandomState(seed=seed)
        rows_id = 1
        batch_id = 1
        rows_ = self.partitions[rows_id]
        cols_ = self.partitions[batch_id]
        for i in range(self.niter):
            self.x[rows_] = self.x[rows_] + 1 / (i + 1)**self.gamma *  self.P[rows_id, batch_id]/self.Q[rows_id, batch_id]* \
                       (1/(self.P[rows_id, batch_id])*self.Ahat[rows_, :][:, cols_] @ self.x[cols_] -
                        self.x[rows_] +
                        self.b[rows_])


            rows_id = copy.copy(batch_id)
            batch_id = rs.choice(self.batch_index, 1, p=np.array(self.Q[rows_id,:])[0])[0]
            rows_ = np.copy(cols_)
            cols_ = self.partitions[batch_id]
            if self.stats == True:
                self._collect_stats()




