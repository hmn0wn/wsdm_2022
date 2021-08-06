#from memory_profiler import profile
import tracemalloc
import numpy as np
from bsa_appnp.data.io import load_dataset
from bsa_appnp.linear_algorithms import  Linear_algorithms
import bsa_appnp.utils_linear as ut

rs = np.random.RandomState(seed=42)
alpha = 0.9
sigma = 0.5
num_labeled_nodes = 20
graph = load_dataset('cora_ml')
labels_ = graph.labels
Af, large_cc = ut.lc(graph)
new_labels = labels_[large_cc]
num_classes = len(np.unique(new_labels))
nnodes = len(new_labels)

y_all = np.zeros((Af.shape[1], num_classes))
for i in range(len(new_labels)):
    c_ = int(new_labels[i] )
    y_all[i, c_] = 1

all_lab = []
for i in range(num_classes):
    set_labs = rs.choice(np.where(new_labels == i )[0], num_labeled_nodes, replace=False)
    all_lab += list(set_labs)
all_lab = np.array(all_lab)

y_train = np.zeros((Af.shape[1], num_classes))
for i in all_lab:
    y_train[i] =  y_all[i]

AHAT, DL = ut.calc_A_hat(Af, sigma=sigma)
rex = np.identity(nnodes) - alpha * AHAT

#exact_solution_dwd = ut.exact_pr(rex, (1 - alpha) * y_train)


la = Linear_algorithms(A=Af, b=y_train, sigma=sigma,
                       alpha=alpha, niter=1000,
                       gamma=0.3, batch_size=512, betta=15, epsilon=0.1,
                       stats=False, algorithm='bsa', seed=42)
tracemalloc.start()
rk = la.run( seed=42)
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
