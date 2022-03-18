from predictc cimport Bsa
from libcpp cimport bool

cdef class BSAcpp:
    cdef Bsa *thisptr

    def __cinit__(self, \
        np.ndarray b, \
        np.ndarray x_prev, \
        np.ndarray x, \
        np.ndarray P, \
        np.ndarray Q, \
        np.ndarray rows_id_seq, \
        np.ndarray all_batches, \
        dataset_name, \
        float epsilon, \
        float gamma, \
        uint niter, \
        uint threads_num, \
        uint extra_logs, \
        uint tau):
        
        self.thisptr= new Bsa(\
        Map[MatrixXf](b), \
        Map[MatrixXf](x_prev), \
        Map[MatrixXf](x), \
        Map[MatrixXf](P), \
        Map[MatrixXf](Q), \
        Map[MatrixXi](rows_id_seq), \
        FlattenedMapWithOrder[Array, int, Dynamic, Dynamic, RowMajor](all_batches), \
        dataset_name.encode(), \
        epsilon, \
        gamma, \
        niter, \
        threads_num, \
        extra_logs, \
        tau)

    def bsa_operation(self):
        return self.thisptr.bsa_operation()