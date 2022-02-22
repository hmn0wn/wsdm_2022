#include "bsa.h"

#include <iostream>
#include <fstream>
#include <vector>

#include <Eigen/Sparse>

using SpMat = Eigen::SparseMatrix<double>;
using Trip = Eigen::Triplet<double>;


const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

template <typename Derived>
void writeToCSVfile(std::string name, const Eigen::SparseMatrix<Derived>& matrix)
{
    std::ofstream file(name.c_str());
    for (int i = 0; i < matrix.rows(); ++i)
    {
        for(int j = 0; j < matrix.cols(); ++j)
        {
            file << matrix.coeff(i, j) << ", ";
        }
        file << std::endl;
    }
    //file << matrix.format(CSVFormat);

}

template <typename Derived>
void writeToBitmap(std::string name, const Eigen::SparseMatrix<Derived>& matrix)
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
        for(int j = 0; j < h; ++j)
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
    for(int i=0; i<h; i++)
    {
        fwrite(img+(w*(h-i-1)*3),3,w,f);
        fwrite(bmppad,1,(4-(w*3)%4)%4,f);
    }

    free(img);
    fclose(f);
}


namespace predictc{
    Bsa::Bsa()
    {}
    
    double Bsa::bsa_operation(std::string dataset_name, uint n_, uint m_, 
        Eigen::Map<Eigen::MatrixXd> &b, 
        Eigen::Map<Eigen::MatrixXd> &x, uint niter, 
        Eigen::Map<Eigen::MatrixXd> &P,
        Eigen::Map<Eigen::MatrixXd> &Q, 
        Eigen::Map<Eigen::MatrixXi> &all_batches, 
        float epsilon, float gamma, uint seed)
    {
        std::cout << "dataset name: " << dataset_name << std::endl;
        std::cout << "n: " << n_ << std::endl << "m: " << m_ << std::endl;
        std::cout << "b:" << b.rows() << " " << b.cols() << std::endl;
        std::cout << "x:" << x.rows() << " " << x.cols() << std::endl;
        std::cout << "niter" << niter << std::endl;
        std::cout << "P:" << P.rows() << " " << P.cols() << std::endl;
        std::cout << "Q:" << Q.rows() << " " << Q.cols() << std::endl;
        std::cout << "all_batches:" << all_batches.rows() << " " << all_batches.cols() << std::endl;
        std::cout << "epsilon" << epsilon << std::endl;
        std::cout << "gamma" << gamma << std::endl;
        std::cout << "seed" << seed << std::endl;

        std::vector<uint> el=std::vector<uint>(m_);
        std::vector<uint> pl=std::vector<uint>(n_+1);
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

        std::cout << "Read finished\r"<<std::endl;

        std::cout << "el: ";
        for (int i = 0; i < 10; ++i)
        {
            std::cout << el[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "pl: ";
        for (int i = 0; i < 10; ++i)
        {
            std::cout << pl[i] << " ";
        }
        std::cout << std::endl;

        std::vector<Trip> triplets;
        triplets.reserve(el.size());
        for(int i = 0; i < n_; ++i)
        {
            for(int jptr = pl[i]; jptr < pl[i+1]; ++jptr)
            {
                int j = el[jptr];
                triplets.push_back(Trip(i, j, 1.0));
            }
        }
        SpMat A(n_, n_);
        A.setFromTriplets(triplets.begin(), triplets.end());

        int n_butches = all_batches.cols();
        
        //Q = epsilon / n_butches + (1 - epsilon) * P


        std::vector<int> list_batches(n_butches);
        for(int i = 0; i < n_butches; ++i)
        {
            list_batches[i] = i;
        }

        bool random_jump = false;
        int batch_i = 0;

        int rows_id = 1;
        int batch_id = 0;

        auto rows_ = all_batches.col(rows_id);
        auto cols_ = all_batches.col(batch_id);

        std::ofstream fcpplog("cpp.log");
        fcpplog << "all_batches: ";
        for (int j = 0; j < all_batches.cols(); ++j)
        {
            for (int i = 0; i < all_batches.rows(); ++i)
            {
                fcpplog << all_batches(i, j) << " ";
            }
            fcpplog << std::endl;
        }
        

        std::cout << "rows_: ";
        for (int i = 0; i < 10; ++i)
        {
            std::cout << rows_[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "cols_: ";
        for (int i = 0; i < 10; ++i)
        {
            std::cout << cols_[i] << " ";
        }
        std::cout << std::endl;

        writeToCSVfile("test.csv", A);
        writeToBitmap("test.bmp", A);
        //saveMarket(A, "test.save");
        return 0;
    }
}