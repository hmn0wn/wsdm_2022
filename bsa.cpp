#include "bsa.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <sys/time.h>
#include <random>
#include <map>

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

std::string br100 = "====================================================================================================";
std::string br50 = "==================================================";
std::string br25 = "=========================";

template <typename Derived>
void write_to_csvfile(std::string name, const Eigen::SparseMatrix<Derived>& matrix)
{
    std::ofstream file(name.c_str());
    for (uint i = 0; i < matrix.rows(); ++i)
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

    for (uint i = 0; i < w; ++i)
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
    std::vector<uint> pl=std::vector<uint>(n_);
    std::vector<double> dl=std::vector<double>(m_);


    std::string dataset_el="bsa_appnp/data/"+dataset_name+"_adj_el.txt";
    const char *p1=dataset_el.c_str();
    if (FILE *f1 = fopen(p1, "rb"))
    {
        size_t rtn = fread(el.data(), sizeof el[0], el.size(), f1);
        if(rtn!=m_)
        {
            std::cout<<"Error! "<<dataset_el<<" Incorrect read! " << rtn <<"\r"<<std::endl;
        }
        fclose(f1);
    }
    else
    {
        std::cout<<dataset_el<<" Not Exists.\r"<<std::endl;
        assert(false);
        exit(1);
    }

    std::string dataset_pl="bsa_appnp/data/"+dataset_name+"_adj_pl.txt";
    const char *p2=dataset_pl.c_str();

    if (FILE *f2 = fopen(p2, "rb"))
    {
        uint rtn = fread(pl.data(), sizeof pl[0], pl.size(), f2);
        if(rtn!=n_)
        {
            std::cout<<"Error! "<<dataset_pl<<" Incorrect read!" << rtn << " " << n_ <<"\r"<<std::endl;
            assert(false);
        }
        fclose(f2);
    }
    else
    {
        std::cout<<dataset_pl<<" Not Exists."<<"\r"<<std::endl;
        assert(false);
        exit(1);
    }

    std::string dataset_dl="bsa_appnp/data/"+dataset_name+"_adj_dl.txt";
    const char *p3=dataset_dl.c_str();

    if (FILE *f3 = fopen(p3, "rb"))
    {
        size_t rtn = fread(dl.data(), sizeof dl[0], dl.size(), f3);
        if(rtn!=m_)
        {
            std::cout<<"Error! "<<dataset_dl<<" Incorrect read!" << rtn << " " << m_ <<"\r"<<std::endl;
            assert(false);
        }
        fclose(f3);
    }
    else
    {
        std::cout<<dataset_pl<<" Not Exists."<<"\r"<<std::endl;
        assert(false);
        exit(1);
    }

    std::cout << "Read finished\r"<<std::endl;

    std::cout << "el: ";
    for (uint i = 0; i < 20 && i < el.size(); ++i)
    {
        std::cout << el[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "pl: ";
    for (uint i = 0; i < 20 && i < pl.size(); ++i)
    {
        std::cout << pl[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "dl: ";
    for (uint i = 0; i < 20 && i < dl.size(); ++i)
    {
        std::cout << dl[i] << " ";
    }
    std::cout << std::endl;

    std::vector<Trip> triplets;
    triplets.reserve(el.size());
    for(uint i = 0; i < n_-1; ++i)
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

uint maxn = 100;

template <typename Derived>
void print_mat(std::string path,
const Eigen::MatrixBase<Derived>& matrix, bool to_print = false)
{
    std::ofstream fcpplog(path);
    //const static Eigen::IOFormat MatFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", "\n");
    //fcpplog << matrix.format(MatFormat);
    fcpplog << matrix.rows() << " " << matrix.cols() << std::endl;
    std::cout.precision(7);
    if (to_print) std::cout << path << std::endl;
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
void read_mat(std::string path, Eigen::PlainObjectBase<Derived>& matrix)
{
    typedef typename Derived::Scalar Scalar;
    std::ifstream fcpplog(path);
    assert(!fcpplog.fail());
    uint n, m;
    fcpplog >> n >> m;
    std::cout << std::endl << path  << std::endl << "shape: " << n << ' ' << m << std::endl;
    matrix.resize(n,m);
    for (uint i = 0; i < n; ++i)
    {
        for (uint j = 0; j < m; ++j)
        {
            double tmp;
            fcpplog >> tmp; 
            matrix(i, j) = (Scalar)tmp;
        }
    }
}

template <typename Derived>
void print_mat(std::string path,
const Eigen::SparseMatrix<Derived>& matrix, bool to_print = false)
{
    std::ofstream fcpplog(path);
    fcpplog << matrix.rows() << " " << matrix.cols() << std::endl;
    //fcpplog << matrix.format(MatFormat);
    std::cout.precision(7);
    if (to_print) std::cout << path << std::endl;
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
void print_mat_i(std::string path,
const Eigen::SparseMatrix<Derived>& matrix)
{
    std::ofstream fcpplog(path);
    //fcpplog << matrix.format(MatFormat);
    fcpplog << matrix.rows() << " " << matrix.cols() << std::endl;
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
void print_mat(std::string path, const Eigen::IndexedView<XprType, RowIndices, ColIndices>& matrix, bool to_print = false)
{
    std::ofstream fcpplog(path);
    uint maxn = 10;
    std::cout.precision(7);
    if (to_print) std::cout << path << std::endl;
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
        float epsilon, float gamma, uint seed, uint threads_num)
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

        if(true)
        {
            print_mat("./logs/b_mat.cpp.log", b);
            print_mat("./logs/x_mat.cpp.log", x);
            print_mat("./logs/P_mat.cpp.log", P);
            print_mat("./logs/Q_mat.cpp.log", Q);
            print_mat("./logs/all_batches_mat.cpp.log", all_batches);

            std::ofstream bsa_serialized("./logs/bsa_serialized.cpp.log");
            bsa_serialized << dataset_name << " " << size_ 
            << " " << n_ << " " << m_ 
            << " " << niter 
            << " " << epsilon << " " << gamma
            << " " << seed << " " << threads_num;
            bsa_serialized.close();
        }
        assert(dataset_name.size() > 0);
        assert(n_ > 0 && n_ < 20000 && m_ > 0 && m_ < 20000);

        double prep_t, cclock_t;
        std::cout << br100 << std::endl;
        std::cout << "BSA cpp" << std::endl;
        SpMat A(size_, size_);
        read_sparse_matrix(n_, m_, dataset_name, A);
        //print_mat_i("./logs", std::string("A"), A);

        uint n_butches = all_batches.rows();
        std::vector<int> list_batches(n_butches);
        for(uint i = 0; i < n_butches; ++i)
        {
            list_batches[i] = i;
        }
        std::default_random_engine generator;
        std::vector<std::discrete_distribution<int> > distributions(Q.rows());
        for(int i=0; i<Q.rows(); ++i)
        {
            //print_mat(std::string("./logs") + std::string("Q_") + std::to_string(i) + std::string("_mat.cpp.log"));
            std::discrete_distribution<int>(Q.row(i).data(), Q.row(i).data()+Q.row(i).size());
        }

        std::map<int, int> tmp;
        for(int k=0; k<10000; ++k)
        {
            ++tmp[distributions[0](generator)];
        }

        for(auto p:tmp)
        {
            std::cout << p.first << " generated " << p.second << " times\n";
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
            std::cout << br50 << std::endl;
            auto rows_ = all_batches.row(rows_id);
            auto cols_ = all_batches.row(batch_id);
            auto jump = P(rows_id, batch_id);
            auto qjump = Q(rows_id, batch_id);
            jump *= 1; qjump*= 1;

            std::cout << "batch_id: " << batch_id << std::endl;
            //std::cout << "jump: " << jump << std::endl;
            //std::cout << "qjump: " << qjump << std::endl;
            print_mat("./logs/rows_mat.cpp.log", rows_, true);
            print_mat("./logs/cols_mat.cpp.log", cols_, true);
           
            //auto x_rows = x(Eigen::all, rows_);
            Eigen::MatrixXd x_rows;
            Eigen::VectorXi x_cols_all = igl::LinSpaced<Eigen::VectorXi >(x.cols(),0,x.cols()-1);
            igl::slice(x,rows_, x_cols_all, x_rows);

            SpMat A_rows_cols;
            igl::slice(A,rows_,cols_,A_rows_cols);

            //auto x_cols = x(Eigen::all, cols_);
            Eigen::MatrixXd x_cols;
            igl::slice(x,cols_, x_cols_all, x_cols);

            //auto b_rows = b(Eigen::all, rows_);
            Eigen::MatrixXd b_rows;
            Eigen::VectorXi b_cols_all = igl::LinSpaced<Eigen::VectorXi >(b.cols(),0,b.cols()-1);
            igl::slice(b,rows_, b_cols_all, b_rows);

            //auto cur_A = A(rows_, cols_);
            //x(rows_, Eigen::all) = x(rows_, Eigen::all) * 2;
            double q = 1.0/pow((1+iter),gamma) * jump/qjump;
            Eigen::MatrixXd res = x_rows + q*((1/jump) * A_rows_cols * x_cols - x_rows + b_rows);
            //auto res = A_rows_cols*x_cols;
            
            if (true)
            {
                print_mat(std::string("./logs") + std::string("x_rows") + std::to_string(iter) + std::string("_mat.cpp.log"), x_rows);
                print_mat(std::string("./logs") + std::string("A_") + std::to_string(iter) + std::string("_mat.cpp.log"), A_rows_cols);
                print_mat(std::string("./logs") + std::string("x_cols") + std::to_string(iter) + std::string("_mat.cpp.log"), x_cols);
                print_mat(std::string("./logs") + std::string("b_rows") + std::to_string(iter) + std::string("_mat.cpp.log"), b_rows);
                print_mat(std::string("./logs") + std::string("res") + std::to_string(iter) + std::string("_mat.cpp.log"), res, true);
            }

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
            //cols_ =  all_batches.row(batch_id);
        }
        print_mat(std::string("./logs") + std::string("x_res_mat.cpp.log"), x, true);
        //writeToCSVfile("test.csv", A);
        //writeToBitmap("test.bmp", A);
        //saveMarket(A, "test.save");
        end_t = clock();
        cclock_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
        gettimeofday(&t_end, NULL);
        prep_t = t_end.tv_sec - t_start.tv_sec + (t_end.tv_usec - t_start.tv_usec)/1000000.0;
        std::cout << br50 << std::endl;
        std::cout<<"cpp BSA time: "<<prep_t<<" s"<<"\r"<<std::endl;
        std::cout<<"cpp BSA clock time : "<<cclock_t<<" s"<<"\r"<<std::endl;
        return 0;
    }
}

#ifndef CYTHON_COMPILE
int main()
{

    if (true)
    {
        std::string filename = "./run_bsa_appnp.py ";
        std::string command = "python3 ";
        command += filename;
        system(command.c_str());
    }
    std::cout << br50 << std::endl;

    std::string dataset_name;
    uint size_;
    uint n_;
    uint m_;
    Eigen::MatrixXd b;
    Eigen::MatrixXd x;
    uint niter;
    Eigen::MatrixXd P;
    Eigen::MatrixXd Q;
    Eigen::MatrixXi all_batches;
    float epsilon, gamma;
    uint seed, threads_num;

    Eigen::MatrixXd res_py;
    Eigen::MatrixXd res_cpp;

    std::ifstream bsa_serialized("./logs/bsa_serialized.py.log");
    assert(!bsa_serialized.fail());

    bsa_serialized >> dataset_name >> size_ 
    >> n_ >> m_ 
    >> niter 
    >> epsilon >> gamma
    >> seed >> threads_num;
    std::string log_dir("");
    bsa_serialized.close();


    std::cout << dataset_name << " " << size_ 
            << " " << n_ << " " << m_ 
            << " " << niter 
            << " " << epsilon << " " << gamma
            << " " << seed << " " << threads_num;

    read_mat("./logs/b_mat.py.log", b);
    read_mat("./logs/x_mat.py.log", x);
    read_mat("./logs/P_mat.py.log", P);
    read_mat("./logs/Q_mat.py.log", Q);
    read_mat("./logs/all_batches_mat.py.log", all_batches);
    read_mat("./logs/x_res_mat.py.log", res_py);
    
    auto b_ = Eigen::Map<Eigen::MatrixXd>(b.data(), b.rows(), b.cols());
    auto x_ = Eigen::Map<Eigen::MatrixXd>(x.data(), x.rows(), x.cols());
    auto P_ = Eigen::Map<Eigen::MatrixXd>(P.data(), P.rows(), P.cols());
    auto Q_ = Eigen::Map<Eigen::MatrixXd>(Q.data(), Q.rows(), Q.cols());
    auto all_batches_ = Eigen::Map<Eigen::MatrixXi>(all_batches.data(), all_batches.rows(), all_batches.cols());


    predictc::Bsa bsa;
    bsa.bsa_operation(dataset_name, size_, n_, m_, 
    b_,
    x_,
    niter, 
    P_,
    Q_,
    all_batches_, 
    epsilon, gamma, seed, threads_num);
    
    read_mat("./logs/x_res_mat.cpp.log", res_cpp);

    std::cout << br50 << std::endl;
    std::cout << "sum cpp: " << res_cpp.sum() << std::endl;
    std::cout << "sum py: " << res_py.sum() << std::endl;
    std::cout << "sum: " << (res_cpp - res_py).sum() << std::endl;

    return 0;
}
#endif