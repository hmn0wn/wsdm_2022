#ifndef BSA_H
#define BSA_H

#include "string.h"
#include <Eigen/Dense>

typedef unsigned int uint;

namespace predictc{
    class Bsa{
        public:
        Bsa();
        double bsa_operation(std::string dataset_name,uint size_, uint n_, uint m_, 
        Eigen::Map<Eigen::MatrixXd> &b, 
        Eigen::Map<Eigen::MatrixXd> &x, uint niter, 
        Eigen::Map<Eigen::MatrixXd> &P,
        Eigen::Map<Eigen::MatrixXd> &Q, 
        Eigen::Map<Eigen::MatrixXi> &all_batches, 
        float epsilon, float gamma, uint seed, uint threads_num);
    };
}
#endif // BSA_H

//TODO:
// sparse matrix serialize to bin https://gist.github.com/zishun/da277d30f4604108029d06db0e804773