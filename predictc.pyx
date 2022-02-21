from predictc cimport Bsa

cdef class BSAcpp:
    cdef Bsa c_bsa

    def __cinit__(self):
        self.c_bsa=Bsa()

    def bsa_operation(self, dataset_name, unsigned int n_, unsigned int m_, np.ndarray b, np.ndarray x, unsigned int niter, np.ndarray P, np.ndarray all_batches, float epsilon, float gamma, unsigned int seed):
        return self.c_bsa.bsa_operation(dataset_name.encode(), n_, m_, Map[MatrixXd](b), Map[MatrixXd](x), niter, Map[MatrixXd](P), Map[MatrixXd](all_batches), epsilon, gamma, seed)