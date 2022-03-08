#ifndef BSA_H
#define BSA_H

#include "string.h"
#include <atomic>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>

typedef unsigned int uint;
using SpMat = Eigen::SparseMatrix<double>;
using Trip = Eigen::Triplet<double>;

namespace predictc{
    class Bsa{
        public:
        Bsa();
        double bsa_operation(std::string dataset_name,uint size_, uint n_, uint m_, 
        Eigen::Map<Eigen::MatrixXd> &b, 
        Eigen::Map<Eigen::MatrixXd> &x, uint niter_, 
        Eigen::Map<Eigen::MatrixXd> &P,
        Eigen::Map<Eigen::MatrixXd> &Q, 
        Eigen::Map<Eigen::MatrixXi> &all_batches, 
        Eigen::Map<Eigen::MatrixXi> &rows_id_seq,
        float epsilon, float gamma_, uint threads_num_);
        
        void bsa_multithread(
            Eigen::Ref<Eigen::MatrixXd> b, 
            Eigen::Ref<Eigen::MatrixXd> x,
            Eigen::Ref<Eigen::MatrixXd> P,
            Eigen::Ref<Eigen::MatrixXd> Q, 
            Eigen::Ref<Eigen::MatrixXi> all_batches, 
            Eigen::Ref<Eigen::MatrixXi> rows_id_seq
        );

        void bsa_worker(
            Eigen::Ref<Eigen::MatrixXd> b, 
            Eigen::Ref<Eigen::MatrixXd> x,
            Eigen::Ref<Eigen::MatrixXd> P,
            Eigen::Ref<Eigen::MatrixXd> Q, 
            Eigen::Ref<Eigen::MatrixXi> all_batches, 
            Eigen::Ref<Eigen::MatrixXi> rows_id_seq,
            uint worker_index
        );

        static std::atomic<bool> worker_func_end_wall;
        static std::atomic<bool> worker_func_begin_wall;
        static std::atomic<bool> dispatcher_must_exit;
        static std::atomic<int> waiting_workers;
        static std::atomic<int> done_workers;
        static std::atomic<int> global_time;

        float gamma;
        uint niter;
        SpMat A;

        int threads_num;
    };
}
#endif // BSA_H

//TODO:
// sparse matrix serialize to bin https://gist.github.com/zishun/da277d30f4604108029d06db0e804773