#include "bsa.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <sys/time.h>

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Core>

#include <igl/cotmatrix.h>
#include <igl/slice.h>
#include <igl/slice_into.h>


template <typename DerivedX, typename DerivedY, typename DerivedR, typename DerivedC>
IGL_INLINE void igl::slice_into(
  const Eigen::MatrixBase<DerivedX> & X,
  const Eigen::MatrixBase<DerivedR> & R,
  const Eigen::MatrixBase<DerivedC> & C,
  Eigen::MatrixBase<DerivedY> & Y)
{

  int xm = X.rows();
  int xn = X.cols();
#ifndef NDEBUG
  assert(R.size() == xm);
  assert(C.size() == xn);
  int ym = Y.rows();
  int yn = Y.cols();
  assert(R.minCoeff() >= 0);
  assert(R.maxCoeff() < ym);
  assert(C.minCoeff() >= 0);
  assert(C.maxCoeff() < yn);
#endif

  // Build reindexing maps for columns and rows, -1 means not in map
  Eigen::Matrix<typename DerivedR::Scalar,Eigen::Dynamic,1> RI;
  RI.resize(xm);
  for(int i = 0;i<xm;i++)
  {
    for(int j = 0;j<xn;j++)
    {
      Y(int(R(i)),int(C(j))) = X(i,j);
    }
  }
}

using SpMat = Eigen::SparseMatrix<double>;
using Trip = Eigen::Triplet<double>;

std::string br = "====================================================================================================";
std::string br2 = "==================================================";
std::string br3 = "=========================";

template <typename Derived>
void write_to_csvfile(std::string name, const Eigen::SparseMatrix<Derived>& matrix)
{
    std::ofstream file(name.c_str());
    for (int i = 0; i < matrix.rows(); ++i)
    {
        for(uint j = 0; j < matrix.cols(); ++j)
        {
            file << matrix.coeff(i, j) << ", ";
        }
        file << std::endl;
    }
    //file << matrix.format(CSVFormat);

}

template <typename Derived>
void write_to_bitmap(std::string name, const Eigen::SparseMatrix<Derived>& matrix)
{

    int n = matrix.rows();
    int m = matrix.cols();
    FILE *f;
    unsigned char *img = NULL;
    int h = n, w = m;
    int filesize = 54 + 3*n*m;  //w is your image width, h is image height, both int

    img = (unsigned char *)malloc(3*h*w);
    memset(img,0,3*h*w);

    for (int i = 0; i < w; ++i)
    {
        for(uint j = 0; j < h; ++j)
        {
            unsigned char r = 0, g = 0, b = 0;
            if(matrix.coeff(i, j) > 0)
            {
                r = 255;
                g = 255;
                b = 255;
            }

            int x = i, y=j;
            img[(x+y*m)*3+2] = (r);
            img[(x+y*m)*3+1] = (g);
            img[(x+y*m)*3+0] = (b);

        }
    }

    unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
    unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
    unsigned char bmppad[3] = {0,0,0};

    bmpfileheader[ 2] = (unsigned char)(filesize    );
    bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
    bmpfileheader[ 4] = (unsigned char)(filesize>>16);
    bmpfileheader[ 5] = (unsigned char)(filesize>>24);

    bmpinfoheader[ 4] = (unsigned char)(       w    );
    bmpinfoheader[ 5] = (unsigned char)(       w>> 8);
    bmpinfoheader[ 6] = (unsigned char)(       w>>16);
    bmpinfoheader[ 7] = (unsigned char)(       w>>24);
    bmpinfoheader[ 8] = (unsigned char)(       h    );
    bmpinfoheader[ 9] = (unsigned char)(       h>> 8);
    bmpinfoheader[10] = (unsigned char)(       h>>16);
    bmpinfoheader[11] = (unsigned char)(       h>>24);

    f = fopen(name.c_str(),"wb");
    fwrite(bmpfileheader,1,14,f);
    fwrite(bmpinfoheader,1,40,f);
    for(uint i=0; i<h; i++)
    {
        fwrite(img+(w*(h-i-1)*3),3,w,f);
        fwrite(bmppad,1,(4-(w*3)%4)%4,f);
    }

    free(img);
    fclose(f);
}

void read_sparse_matrix(uint n_, uint m_, std::string &dataset_name, SpMat &A)
{

    std::vector<uint> el=std::vector<uint>(m_);
    std::vector<uint> pl=std::vector<uint>(n_+1);
    std::vector<double> dl=std::vector<double>(m_);


    std::string dataset_el="bsa_appnp/data/"+dataset_name+"_adj_el.txt";
    const char *p1=dataset_el.c_str();
    if (FILE *f1 = fopen(p1, "rb"))
    {
        size_t rtn = fread(el.data(), sizeof el[0], el.size(), f1);
        if(rtn!=m_)
            std::cout<<"Error! "<<dataset_el<<" Incorrect read! " << rtn <<"\r"<<std::endl;
        fclose(f1);
    }
    else
    {
        std::cout<<dataset_el<<" Not Exists.\r"<<std::endl;
        exit(1);
    }

    std::string dataset_pl="bsa_appnp/data/"+dataset_name+"_adj_pl.txt";
    const char *p2=dataset_pl.c_str();

    if (FILE *f2 = fopen(p2, "rb"))
    {
        size_t rtn = fread(pl.data(), sizeof pl[0], pl.size(), f2);
        if(rtn!=n_+1)
            std::cout<<"Error! "<<dataset_pl<<" Incorrect read!" << rtn << " " << n_+1 <<"\r"<<std::endl;
        fclose(f2);
    }
    else
    {
        std::cout<<dataset_pl<<" Not Exists."<<"\r"<<std::endl;
        exit(1);
    }

    std::string dataset_dl="bsa_appnp/data/"+dataset_name+"_adj_dl.txt";
    const char *p3=dataset_dl.c_str();

    if (FILE *f3 = fopen(p3, "rb"))
    {
        size_t rtn = fread(dl.data(), sizeof dl[0], dl.size(), f3);
        if(rtn!=n_+1)
            std::cout<<"Error! "<<dataset_dl<<" Incorrect read!" << rtn << " " << n_+1 <<"\r"<<std::endl;
        fclose(f3);
    }
    else
    {
        std::cout<<dataset_pl<<" Not Exists."<<"\r"<<std::endl;
        exit(1);
    }

    std::cout << "Read finished\r"<<std::endl;

    std::cout << "el: ";
    for (int i = 0; i < 10 && i < el.size(); ++i)
    {
        std::cout << el[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "pl: ";
    for (int i = 0; i < 10 && i < pl.size(); ++i)
    {
        std::cout << pl[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "dl: ";
    for (int i = 0; i < 10 && i < dl.size(); ++i)
    {
        std::cout << dl[i] << " ";
    }
    std::cout << std::endl;

    std::vector<Trip> triplets;
    triplets.reserve(el.size());
    for(uint i = 0; i < n_; ++i)
    {
        for(uint jptr = pl[i]; jptr < pl[i+1]; ++jptr)
        {
            int j = el[jptr];
            double d = dl[jptr];
            triplets.push_back(Trip(i, j, d));
            //std::cout <<  i << "\t\t" << j << ": " << d << std::endl;
        }
    }

    A.setFromTriplets(triplets.begin(), triplets.end());
}

const static Eigen::IOFormat MatFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", "\n");

template <typename Derived>
void print_mat(std::string dir_name, std::string mat_name, const Eigen::MatrixBase<Derived>& matrix, bool to_print = false)
{
    std::string path = dir_name + std::string("/") + mat_name + std::string("_mat.cpp.log");
    std::ofstream fcpplog(path);
    //fcpplog << matrix.format(MatFormat);
    uint maxn = 10;
    std::cout.precision(7);
    if (to_print) std::cout << mat_name << std::endl;
    for (uint i = 0; i < maxn && i < matrix.rows(); ++i)
    {
        for (uint j = 0; j < maxn && j < matrix.cols(); ++j)
        {
            fcpplog << matrix(i, j) << " ";
            if (to_print) std::cout << matrix(i, j) << " ";
        }
        fcpplog << std::endl;
        if (to_print) std::cout << std::endl;
    }
}

template <typename Derived>
void print_mat(std::string dir_name, std::string mat_name, const Eigen::SparseMatrix<Derived>& matrix, bool to_print = false)
{
    std::string path = dir_name + std::string("/") + mat_name + std::string("_mat.cpp.log");
    std::ofstream fcpplog(path);
    //fcpplog << matrix.format(MatFormat);
    uint maxn = 100;
    std::cout.precision(7);
    if (to_print) std::cout << mat_name << std::endl;
    for (uint i = 0; i < maxn && i < matrix.rows(); ++i)
    {
        for (uint j = 0; j < maxn && j < matrix.cols(); ++j)
        {
            fcpplog << matrix.coeff(i, j) << " ";
            if (to_print) std::cout << matrix.coeff(i, j) << " ";
        }
        fcpplog << std::endl;
        if (to_print) std::cout << std::endl;
    }
}

template <typename Derived>
void print_mat_i(std::string dir_name, std::string mat_name, const Eigen::SparseMatrix<Derived>& matrix)
{
    std::string path = dir_name + std::string("/") + mat_name + std::string("_mat.cpp.log");
    std::ofstream fcpplog(path);
    //fcpplog << matrix.format(MatFormat);
    uint maxn = 100;
    std::cout.precision(7);
    for (uint i = 0; i < maxn && i < matrix.rows(); ++i)
    {
        for (uint j = 0; j < maxn && j < matrix.cols(); ++j)
        {
            double el = matrix.coeff(i, j);
            double eps = 0.001;
            if(el - eps > 0.0)
            {
                fcpplog << i << "\t" << j << "\t\t: " << el << std::endl;
            }
        }
    }
}


template<typename XprType, typename RowIndices, typename ColIndices>
void print_mat(std::string dir_name, std::string mat_name, const Eigen::IndexedView<XprType, RowIndices, ColIndices>& matrix, bool to_print = false)
{
    std::string path = dir_name + std::string("/") + mat_name + std::string("_mat.cpp.log");
    std::ofstream fcpplog(path);
    uint maxn = 10;
    std::cout.precision(7);
    if (to_print) std::cout << mat_name << std::endl;
    for (uint i = 0; i < maxn && i < matrix.rows(); ++i)
    {
        for (uint j = 0; j < maxn && j < matrix.cols(); ++j)
        {
            fcpplog << matrix(i, j) << " ";
            if (to_print) std::cout << matrix(i, j) << " ";
        }
        fcpplog << std::endl;
        if (to_print) std::cout << std::endl;
    }
}



namespace predictc{
    Bsa::Bsa()
    {}
    
    double Bsa::bsa_operation(std::string dataset_name, uint size_, uint n_, uint m_, 
        Eigen::Map<Eigen::MatrixXd> &b, 
        Eigen::Map<Eigen::MatrixXd> &x, uint niter, 
        Eigen::Map<Eigen::MatrixXd> &P,
        Eigen::Map<Eigen::MatrixXd> &Q, 
        Eigen::Map<Eigen::MatrixXi> &all_batches, 
        float epsilon, float gamma, uint seed)
    {
        std::cout << "dataset name: " << dataset_name << std::endl;
        std::cout << "size_: " << size_ << "n: " << n_ << std::endl << "m: " << m_ << std::endl;
        std::cout << "b:" << b.rows() << " " << b.cols() << std::endl;
        std::cout << "x:" << x.rows() << " " << x.cols() << std::endl;
        std::cout << "niter" << niter << std::endl;
        std::cout << "P:" << P.rows() << " " << P.cols() << std::endl;
        std::cout << "Q:" << Q.rows() << " " << Q.cols() << std::endl;
        std::cout << "all_batches:" << all_batches.rows() << " " << all_batches.cols() << std::endl;
        std::cout << "epsilon" << epsilon << std::endl;
        std::cout << "gamma" << gamma << std::endl;
        std::cout << "seed" << seed << std::endl;

        //print_mat("./logs", "b", b);
        //print_mat("./logs", "x", x);
        //print_mat("./logs", "P", P);
        //print_mat("./logs", "Q", Q);
        //print_mat("./logs", "all_batches", all_batches);
        
        double prep_t, cclock_t;
        std::cout << br << std::endl;
        std::cout << "BSA cpp" << std::endl;
        SpMat A(size_, size_);
        read_sparse_matrix(n_, m_, dataset_name, A);
        //print_mat_i("./logs", std::string("A"), A);

        uint n_butches = all_batches.cols();

        std::vector<int> list_batches(n_butches);
        for(uint i = 0; i < n_butches; ++i)
        {
            list_batches[i] = i;
        }

        bool random_jump = false;
        int batch_i = 0;

        int rows_id = 1;
        int batch_id = 0;

            
        struct timeval t_start,t_end;
        clock_t start_t, end_t;
        gettimeofday(&t_start,NULL);
        start_t = clock();
        for(uint iter=0; iter < niter; ++iter)
        {
            //std::cout << br3 << std::endl;
            auto rows_ = all_batches.col(rows_id);
            auto cols_ = all_batches.col(batch_id);
            auto jump = P(rows_id, batch_id);
            auto qjump = Q(rows_id, batch_id);
            jump *= 1; qjump*= 1;

            //std::cout << "batch_id: " << batch_id << std::endl;
            //std::cout << "jump: " << jump << std::endl;
            //std::cout << "qjump: " << qjump << std::endl;
            if(false)
            {
                std::cout << "rows_: ";
                for (int i = 0; i < 10 && i < rows_.size(); ++i)
                {
                    std::cout << rows_[i] << " ";
                }
                std::cout << std::endl;

                std::cout << "cols_: ";
                for (int i = 0; i < 10 && i < cols_.size(); ++i)
                {
                    std::cout << cols_[i] << " ";
                }
                std::cout << std::endl;
            }

            //auto x_rows = x(Eigen::all, rows_);
            Eigen::MatrixXd x_rows;
            Eigen::VectorXi x_cols_all = igl::LinSpaced<Eigen::VectorXi >(x.cols(),0,x.cols()-1);
            igl::slice(x,rows_, x_cols_all, x_rows);
            //print_mat("./logs", std::string("x_rows") + std::to_string(iter), x_rows);

            SpMat A_rows_cols;
            igl::slice(A,rows_,cols_,A_rows_cols);
            //print_mat("./logs", std::string("A_") + std::to_string(iter), A_rows_cols);

            //auto x_cols = x(Eigen::all, cols_);
            Eigen::MatrixXd x_cols;
            igl::slice(x,cols_, x_cols_all, x_cols);
            //print_mat("./logs", std::string("x_cols") + std::to_string(iter), x_cols);

            //auto b_rows = b(Eigen::all, rows_);
            Eigen::MatrixXd b_rows;
            Eigen::VectorXi b_cols_all = igl::LinSpaced<Eigen::VectorXi >(b.cols(),0,b.cols()-1);
            igl::slice(b,rows_, b_cols_all, b_rows);
            //print_mat("./logs", std::string("b_rows") + std::to_string(iter), b_rows);

            //auto cur_A = A(rows_, cols_);
            //x(rows_, Eigen::all) = x(rows_, Eigen::all) * 2;
            double q = 1.0/pow((1+iter),gamma) * jump/qjump;
            Eigen::MatrixXd res = x_rows + q*((1/jump) * A_rows_cols * x_cols - x_rows + b_rows);
            //auto res = A_rows_cols*x_cols;
            //print_mat("./logs", std::string("res") + std::to_string(iter), res, true);
            
            igl::slice_into(res, rows_, x_cols_all, x);
            
            rows_id = batch_id;
            if(random_jump)
            {
                //batch_id = random
            }
            else
            {
                //
                batch_i = (batch_i + 1)%n_butches;
                batch_id = list_batches[batch_i];
            }
            //rows_ = copy cols_
            //cols_ =  all_batches.col(batch_id);
        }
        print_mat("./logs", std::string("x_res"), x, true);
        //writeToCSVfile("test.csv", A);
        //writeToBitmap("test.bmp", A);
        //saveMarket(A, "test.save");
        end_t = clock();
        cclock_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
        gettimeofday(&t_end, NULL);
        prep_t = t_end.tv_sec - t_start.tv_sec + (t_end.tv_usec - t_start.tv_usec)/1000000.0;
        std::cout << br3 << std::endl;
        std::cout<<"cpp BSA time: "<<prep_t<<" s"<<"\r"<<std::endl;
        std::cout<<"cpp BSA clock time : "<<cclock_t<<" s"<<"\r"<<std::endl;
        return 0;
    }
}