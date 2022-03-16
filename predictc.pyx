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
        uint size, \
        uint n, \
        uint m, \
        uint niter, \
        uint threads_num, \
        uint extra_logs, \
        uint tau):
        
        self.thisptr= new Bsa(\
        Map[MatrixXd](b), \
        Map[MatrixXd](x_prev), \
        Map[MatrixXd](x), \
        Map[MatrixXd](P), \
        Map[MatrixXd](Q), \
        Map[MatrixXi](rows_id_seq), \
        FlattenedMapWithOrder[Array, int, Dynamic, Dynamic, RowMajor](all_batches), \
        dataset_name.encode(), \
        epsilon, \
        gamma, \
        size, \
        n, \
        m, \
        niter, \
        threads_num, \
        extra_logs, \
        tau)

    def bsa_operation(self):
        return self.thisptr.bsa_operation()