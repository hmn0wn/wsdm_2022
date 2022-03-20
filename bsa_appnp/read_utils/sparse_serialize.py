import numpy as np
import scipy.sparse as sp
from bsa_appnp.read_utils import utils_cpp
import struct
from timeit import default_timer as timer

def serialize_sparse(file_path, mat, rewrite=True):
    indices = mat.indices
    indptr = mat.indptr
    data = mat.data

    indices_np = np.array(indices, dtype=np.uint32)
    indptr_np = np.array(indptr, dtype=np.uint32)

    #print("="*50)
    #print("type: ", "CSR" if sp.isspmatrix_csr(mat) else "CSC")
    #print("rows:", mat.shape[0])
    #print("cols:", mat.shape[1])
    #print("indices_size:", indices_np.size)
    #print("indptr_size :", indptr_np.size)

    with open(f"{file_path}_sp.pack", "wb" if rewrite else "ab") as f:

        f.write(struct.pack('I', 1 if sp.isspmatrix_csr(mat) else 0))
        f.write(struct.pack('I', mat.shape[0]))
        f.write(struct.pack('I', mat.shape[1]))
        f.write(struct.pack('I', indices_np.size))
        f.write(struct.pack('I', indptr_np.size))

        for el in indices_np:
            f.write(struct.pack('I', el))
            #print(el, end=' ')
        #print()

        for el in indptr_np:
            f.write(struct.pack('I', el))
            #print(el, end=' ')
        #print()

        for el in data:
            f.write(struct.pack('f', el))
            #print(el, end=' ')
        #print()

    return indices_np.size, indptr_np.size

def serialize_sparse_map(file_path, mat_map):
    start = timer()
    num = 0
    for batch_map in mat_map.values():
        num = num + len(batch_map)

    with open(f"{file_path}_sp.pack", "wb") as f:
        f.write(struct.pack("I", num))

    for row_id,batch_map in mat_map.items():
        for batch_id,value in batch_map.items():
            with open(f"{file_path}_sp.pack", "ab") as f:
                #print("\n\trow_id  : ", row_id)
                #print("\tbatch_id: ", batch_id)
                f.write(struct.pack("I", row_id))
                f.write(struct.pack("I", batch_id))
                csc_val=value.tocsc()
            serialize_sparse(file_path, csc_val, rewrite=False)
    end = timer()
    print(f"py serialize_sparse_map time: {end-start} s")

def sparse_matrix_test():
    row = np.array([0,0,0,1,2,2,2,2,3,4,4,4,5,5,5])
    col = np.array([0,2,4,2,0,1,2,4,2,1,3,5,0,2,5])
    data = np.array([0.01, 0.02, 0.01, 0.03, 0.04, 0.05, 0.06, 0.02, 0.05, 0.04, 0.03, 0.01, 0.02, 0.01, 0.01])
    Ar = sp.csr_matrix((data, (row, col)), shape=(6, 6))
    Ac = Ar.tocsc()

    utils_cpp.print_matsp("./logs/test", f"Ar", Ar)
    serialize_sparse("./logs/test/Ar", Ar)
    print("="*50)
    utils_cpp.print_matsp("./logs/test", f"Ac", Ac)
    serialize_sparse("./logs/test/Ac", Ac)

    row1 = np.array([1,2,0,2,4,2,1,4])
    col1 = np.array([0,0,1,1,2,3,4,4])
    data1 = np.array([22,7,3,5,14,1,17,8])
    Ar1 = sp.csr_matrix((data1, (row1, col1)), shape=(5, 5))
    Ac1 = Ar1.tocsc()

    utils_cpp.print_matsp("./logs/test", f"Ar1", Ar1)
    serialize_sparse("./logs/test/Ar1", Ar1)
    print("="*50)
    utils_cpp.print_matsp("./logs/test", f"Ac1", Ac1)
    serialize_sparse("./logs/test/Ac1", Ac1)

    mat_map = dict()
    mat_map[1] = dict()
    mat_map[1][2] = Ar
    mat_map[1][5] = Ac
    mat_map[5] = dict()
    mat_map[5][4] = Ar1
    mat_map[5][7] = Ac1
    serialize_sparse_map("./logs/test/Acr_map", mat_map)

   
if __name__ == "__main__":
    sparse_matrix_test()