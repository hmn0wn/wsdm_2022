from eigency.core cimport *
from libcpp.string cimport string


ctypedef unsigned int uint

cdef extern from "bsa.cpp":
    pass

cdef extern from "bsa.h" namespace "predictc":
    cdef cppclass Bsa:
        Bsa() except+
        double bsa_operation(string, uint, uint, uint, \
        Map[MatrixXd] &, \
        Map[MatrixXd] &, uint, \
        Map[MatrixXd] &, \
        Map[MatrixXd] &, \
        Map[MatrixXi] &, float, float, uint)
