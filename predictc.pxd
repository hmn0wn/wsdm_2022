from eigency.core cimport *
from libcpp.string cimport string


ctypedef unsigned int uint

cdef extern from "bsa.cpp":
    pass

cdef extern from "bsa.h" namespace "predictc":
    cdef cppclass Bsa:
        Bsa(\
        Map[MatrixXf] &, \
        Map[MatrixXf] &, \
        Map[MatrixXf] &, \
        Map[MatrixXf] &, \
        Map[MatrixXf] &, \
        Map[MatrixXi] &, \
        FlattenedMapWithOrder[Array, int, Dynamic, Dynamic, RowMajor] &, \
        string, \
        float, \
        float, \
        uint, \
        uint, \
        uint, \
        uint, \
        uint) except+

        float bsa_operation()
