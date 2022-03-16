#ifndef BSA_H
#define BSA_H

#include "string.h"
#include <atomic>
#include <unordered_map>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>

typedef unsigned int uint;
using SpMat = Eigen::SparseMatrix<double>;
using Trip = Eigen::Triplet<double>;
using MatrixXiRowMajor = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMajorArray = Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace predictc{
    class Bsa{
        public:
        Bsa(
            Eigen::Map<Eigen::MatrixXd> &b_,
            Eigen::Map<Eigen::MatrixXd> &x_prev_,
            Eigen::Map<Eigen::MatrixXd> &x_,
            Eigen::Map<Eigen::MatrixXd> &P_,
            Eigen::Map<Eigen::MatrixXd> &Q,
            Eigen::Map<Eigen::MatrixXi> &rows_id_seq_,
            Eigen::Map<RowMajorArray> &all_batches_,
            std::string dataset_name_,
            float epsilon_,
            float gamma_, 
            uint size_,
            uint n_,
            uint m_, 
            uint niter_,
            uint threads_num_,
            uint extra_logs_,
            uint tau_
            );

        double bsa_operation();
        
        void construct_sparse_blocks_vec();
        
        void construct_sparse_blocks_mat();

        void bsa();

        void bsa_multithread1();

        void bsa_worker1(uint worker_index, uint work_index);

        void bsa_multithread_all();

        void bsa_worker_all(uint worker_index, uint work_index);

        static std::atomic<bool> worker_func_end_wall;
        static std::atomic<bool> worker_func_begin_wall;
        static std::atomic<bool> dispatcher_must_exit;
        static std::atomic<int> waiting_workers;
        static std::atomic<int> done_workers;
        static std::atomic<int> global_time;

        SpMat A;

        Eigen::Ref<Eigen::MatrixXd> b;
        Eigen::Ref<Eigen::MatrixXd> x_prev;
        Eigen::Ref<Eigen::MatrixXd> x;
        Eigen::Ref<Eigen::MatrixXd> P;
        Eigen::Ref<Eigen::MatrixXd> Q;
        Eigen::Ref<Eigen::MatrixXi> rows_id_seq;
        std::vector<Eigen::Map<Eigen::VectorXi>> all_batches;
        
        std::string dataset_name;
        float epsilon;
        float gamma; 
        uint size;
        uint n;
        uint m; 
        uint niter;
        uint threads_num;
        uint extra_logs;
        uint tau;

        std::unordered_map<int64_t, double> A_map;
        std::unordered_map<int32_t, std::unordered_map<int32_t, double>> Af_map;

        std::vector<std::vector<SpMat>> A_blocks_vec;
        std::unordered_map<int32_t, std::unordered_map<int32_t, SpMat>> A_blocksf_map;
    };
}
#endif // BSA_H

//TODO:
// sparse matrix serialize to bin https://gist.github.com/zishun/da277d30f4604108029d06db0e804773