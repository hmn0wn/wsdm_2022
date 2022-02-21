#ifndef BSA_H
#define BSA_H

#include "string.h"
#include <Eigen/Dense>

typedef unsigned int uint;

namespace predictc{
    class Bsa{
        public:
        Bsa();
        double bsa_operation(std::string dataset_name, uint n_, uint m_, Eigen::Map<Eigen::MatrixXd> &b, Eigen::Map<Eigen::MatrixXd> &x, uint niter, Eigen::Map<Eigen::MatrixXd> &P, Eigen::Map<Eigen::MatrixXd> &all_batches, float epsilon, float gamma, uint seed);
    };
}
#endif // BSA_H