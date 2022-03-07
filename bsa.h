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
        
        void bsa_worker(Eigen::Map<Eigen::MatrixXd> &b, 
            Eigen::Map<Eigen::MatrixXd> &x,
            Eigen::Map<Eigen::MatrixXd> &P,
            Eigen::Map<Eigen::MatrixXd> &Q, 
            Eigen::Map<Eigen::MatrixXi> &all_batches, 
            Eigen::Map<Eigen::MatrixXi> &rows_id_seq,
            bool multithread
        );
        void bsa_loop(
            Eigen::Map<Eigen::MatrixXd> &b, 
            Eigen::Map<Eigen::MatrixXd> &x,
            Eigen::Map<Eigen::MatrixXd> &P,
            Eigen::Map<Eigen::MatrixXd> &Q, 
            Eigen::Map<Eigen::MatrixXi> &all_batches, 
            Eigen::Map<Eigen::MatrixXi> &rows_id_seq
        );
        static std::atomic<bool> worker_end;
        static std::atomic<bool> worker_begin;
        static std::atomic<size_t> waiting_workers;

        float gamma;
        uint niter;
        SpMat A;

        uint n_butches;
        std::vector<int> list_batches;
        int batch_i = 0;
        int rows_id = 1;
        int batch_id = 0;
        int threads_num;
    };
}
#endif // BSA_H

//TODO:
// sparse matrix serialize to bin https://gist.github.com/zishun/da277d30f4604108029d06db0e804773