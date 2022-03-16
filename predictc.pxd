from eigency.core cimport *
from libcpp.string cimport string


ctypedef unsigned int uint

cdef extern from "bsa.cpp":
    pass

cdef extern from "bsa.h" namespace "predictc":
    cdef cppclass Bsa:
        Bsa(\
        Map[MatrixXd] &, \
        Map[MatrixXd] &, \
        Map[MatrixXd] &, \
        Map[MatrixXd] &, \
        Map[MatrixXd] &, \
        Map[MatrixXi] &, \
        FlattenedMapWithOrder[Array, int, Dynamic, Dynamic, RowMajor] &, \
        string, \
        float, \
        float, \
        uint, \
        uint, \
        uint, \
        uint, \
        uint, \
        uint, \
        uint) except+

        double bsa_operation()
