from predictc cimport Bsa
from libcpp cimport bool

cdef class BSAcpp:
    cdef Bsa c_bsa

    def __cinit__(self):
        self.c_bsa=Bsa()

    def bsa_operation(self, dataset_name, unsigned int size, unsigned int n_, unsigned int m_, \
    np.ndarray b, \
    np.ndarray x_prev, \
    np.ndarray x, unsigned int niter, \
    np.ndarray P, \
    np.ndarray Q, \
    np.ndarray all_batches, \
    np.ndarray rows_id_seq, \
    float epsilon, float gamma, unsigned int threads_num, uint extra_logs, uint tau):
        return self.c_bsa.bsa_operation(dataset_name.encode(),size, n_, m_, \
        Map[MatrixXd](b), \
        Map[MatrixXd](x_prev),\
        Map[MatrixXd](x), niter, \
        Map[MatrixXd](P), \
        Map[MatrixXd](Q), \
        FlattenedMapWithOrder[Array, int, Dynamic, Dynamic, RowMajor](all_batches), \
        Map[MatrixXi](rows_id_seq),\
        epsilon, gamma, threads_num, extra_logs, tau)