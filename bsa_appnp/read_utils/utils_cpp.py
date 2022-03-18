
import numpy as np
import scipy.sparse as sp

nmax = 999999
def print_mat(dir_name, mat_name, mat, to_print=False):
    import os
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if to_print:
        print(mat_name)
    with open(f"{dir_name}/{mat_name}_mat.py.log", 'w') as f:
        f.write(f"{mat.shape[0]} {mat.shape[1]}\n")
        for i, eli in enumerate(mat):
            for j, elj in enumerate(eli):
                f.write(f"{elj:0.6f} ")
                if to_print:
                    print(f"{elj:0.6f} ", end="")
                if j == nmax-1:
                    break
            f.write("\n")
            if to_print:
                print()
            if i == nmax-1:
                break




nmax = 999999

def read_mat(mat_name):
    with open(f"{mat_name}", 'r') as f:
        w, h = [float(x) for x in next(f).split()]
        array = []
        for line in f:
            array.append([float(x) for x in line.split()])
        
    return np.array([np.array(el) for el in array])


def print_matsp(dir_name, mat_name, mat, to_print=False):
    import os
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if to_print:
        print(mat_name)
    with open(f"{dir_name}/{mat_name}_mat.py.log", 'w') as f:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                f.write(f"{mat[i,j]:0.5f} ")
                if to_print:
                    print(f"{mat[i,j]:0.6f} ", end='')
                if j == nmax-1:
                    break
            f.write("\n")
            if to_print:
                print()
            if i == nmax-1:
                break


def print_matsp_i(dir_name, mat_name, mat):
    import os
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(f"{dir_name}/{mat_name}_mat.py.log", 'w') as f:
        f.write(f"{mat.shape[0]} {mat.shape[1]}\n")
        for i in range(mat.shape[0]):
            for j in mat[i].nonzero()[1]:
                f.write(f"{str(i)}\t{str(j)}\t\t: {str(mat[i,j])}\n")


def print_vec(dir_name, vec_name, vec, to_print=False):
    import os
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if to_print:
        print(vec_name)
    with open(f"{dir_name}/{vec_name}_mat.py.log", 'w') as f:
        for i, el in enumerate(vec):
            f.write(f"{el:0.5f} ")
            if to_print:
                print(f"{el:0.6f} ", end='')
            if i == nmax-1:
                break
        if to_print:
            print()
