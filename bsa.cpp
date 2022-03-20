#include "bsa.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip> 
#include <thread>

#include <sys/time.h>
#include <random>
#include <map>

#define BR100 "\n====================================================================================================\n"
#define BR50 "\n==================================================\n"

#define MAX_PRINT_NUM 999999

void slice(
    const fSpMat &X,
    const Eigen::Map<Eigen::VectorXi> &R,
    const Eigen::Map<Eigen::VectorXi> &C,
    fSpMat &Y)
{
  int xm = X.rows();
  int xn = X.cols();
  int ym = R.size();
  int yn = C.size();

  // special case when R or C is empty
  if (ym == 0 || yn == 0)
  {
    Y.resize(ym, yn);
    return;
  }

  //assert(R.minCoeff() >= 0);
  assert(R.maxCoeff() < xm);
  //assert(C.minCoeff() >= 0);
  assert(C.maxCoeff() < xn);

    std::vector<std::vector<int>> RI;
    std::vector<std::vector<int>> CI;
    std::vector<fTrip> entries;

  // Build reindexing maps for columns and rows
    RI.clear();
    CI.clear();
    entries.clear();

  RI.resize(xm);
  for (int i = 0; i < ym; i++)
  {
    if(R[i] < 0) continue;
    RI[R[i]].push_back(i);
  }
  CI.resize(xn);
  for (int i = 0; i < yn; i++)
  {
    if(C[i] < 0) continue;
    CI[C[i]].push_back(i);
  }

  // Take a guess at the number of nonzeros (this assumes uniform distribution
  // not banded or heavily diagonal)
  entries.reserve((X.nonZeros()/(X.rows()*X.cols())) * (ym*yn));

  // Iterate over outside
  for (int k = 0; k < X.outerSize(); ++k)
  {
    // Iterate over inside
    for (typename Eigen::SparseMatrix<float>::InnerIterator it(X, k); it; ++it)
    {
      for (auto rit = RI[it.row()].begin(); rit != RI[it.row()].end(); rit++)
      {
        for (auto cit = CI[it.col()].begin(); cit != CI[it.col()].end(); cit++)
        {
          entries.emplace_back(*rit, *cit, it.value());
        }
      }
    }
  }
  Y.resize(ym, yn);
  Y.setFromTriplets(entries.begin(), entries.end());
}

template <typename DerivedX, typename DerivedY, typename DerivedR, typename DerivedC>
void slice_into(
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
    //assert(R.minCoeff() >= 0);
    assert(R.maxCoeff() < ym);
    //assert(C.minCoeff() >= 0); 
    assert(C.maxCoeff() < yn);
#endif

    // Build reindexing maps for columns and rows, -1 means not in map
    Eigen::Matrix<typename DerivedR::Scalar,Eigen::Dynamic,1> RI;
    RI.resize(xm);
    for(int i = 0;i<xm;i++)
    {
        for(int j = 0;j<xn;j++)
        {
            auto R_ = R[i];
            auto C_ = C[j];
            if (R_ < 0 || C_ < 0) continue;
            Y(int(R_),int(C_)) = X(i,j);
        }
    }
}

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

template <typename Derived>
void print(std::string path,
const Eigen::MatrixBase<Derived>& matrix, bool to_print = false)
{
    std::ofstream fcpplog(path);
    //const static Eigen::IOFormat MatFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", "\n");
    //fcpplog << matrix.format(MatFormat);
    fcpplog << matrix.rows() << " " << matrix.cols() << std::endl;
    std::cout.precision(7);
    if (to_print) std::cout << path << std::endl;
    for (uint i = 0; i < MAX_PRINT_NUM && i < matrix.rows(); ++i)
    {
        for (uint j = 0; j < MAX_PRINT_NUM && j < matrix.cols(); ++j)
        {
            fcpplog << matrix(i, j) << " ";
            if (to_print) std::cout << matrix(i, j) << " ";
        }
        fcpplog << std::endl;
        if (to_print) std::cout << std::endl;
    }
}

template <typename Derived>
void print(std::string path,
const Eigen::ArrayBase<Derived>& matrix, bool to_print = false)
{
    std::ofstream fcpplog(path);
    fcpplog << matrix.matrix().rows() << " " << matrix.matrix().cols() << std::endl;
    std::cout.precision(7);
    if (to_print) std::cout << path << std::endl;
    for (uint i = 0; i < MAX_PRINT_NUM && i < matrix.matrix().rows(); ++i)
    {
        for (uint j = 0; j < MAX_PRINT_NUM && j < matrix.matrix().cols(); ++j)
        {
            fcpplog << matrix.matrix()(i, j) << " ";
            if (to_print) std::cout << matrix.matrix()(i, j) << " ";
        }
        fcpplog << std::endl;
        if (to_print) std::cout << std::endl;
    }
}

template <typename Derived>
void print(std::string path,
const Eigen::SparseMatrix<Derived>& matrix, bool to_print = false)
{
    std::ofstream fcpplog(path);
    fcpplog << matrix.rows() << " " << matrix.cols() << std::endl;
    //fcpplog << matrix.format(MatFormat);
    std::cout.precision(7);
    if (to_print) std::cout << path << std::endl;
    for (uint i = 0; i < MAX_PRINT_NUM && i < matrix.rows(); ++i)
    {
        for (uint j = 0; j < MAX_PRINT_NUM && j < matrix.cols(); ++j)
        {
            fcpplog << std::setw(5);
            fcpplog << matrix.coeff(i, j) << " ";
            std::cout << std::setw(5);
            if (to_print) std::cout << matrix.coeff(i, j) << " ";
        }
        fcpplog << std::endl;
        if (to_print) std::cout << std::endl;
    }
}

template <typename Derived>
void print_i(std::string path,
const Eigen::SparseMatrix<Derived>& matrix)
{
    std::ofstream fcpplog(path);
    //fcpplog << matrix.format(MatFormat);
    fcpplog << matrix.rows() << " " << matrix.cols() << std::endl;
    std::cout.precision(7);
    for (uint i = 0; i < MAX_PRINT_NUM && i < matrix.rows(); ++i)
    {
        for (uint j = 0; j < MAX_PRINT_NUM && j < matrix.cols(); ++j)
        {
            float el = matrix.coeff(i, j);
            float eps = 0.001;
            if(el - eps > 0.0)
            {
                fcpplog << i << "\t" << j << "\t\t: " << el << std::endl;
            }
        }
    }
}

template<typename XprType, typename RowIndices, typename ColIndices>
void print(std::string path, const Eigen::IndexedView<XprType, RowIndices, ColIndices>& matrix, bool to_print = false)
{
    std::ofstream fcpplog(path);
    std::cout.precision(7);
    if (to_print) std::cout << path << std::endl;
    for (uint i = 0; i < MAX_PRINT_NUM && i < matrix.rows(); ++i)
    {
        for (uint j = 0; j < MAX_PRINT_NUM && j < matrix.cols(); ++j)
        {
            fcpplog << matrix(i, j) << " ";
            if (to_print) std::cout << matrix(i, j) << " ";
        }
        fcpplog << std::endl;
        if (to_print) std::cout << std::endl;
    }
}

template<typename T>
void print(std::string path, const std::vector<T>& vec, bool to_print = false)
{
    std::ofstream fcpplog(path);
    std::cout.precision(7);
    if (to_print) std::cout << path << std::endl;
    for (uint i = 0; i < MAX_PRINT_NUM && i < vec.size(); ++i)
    {
        fcpplog << vec[i] << " ";
        if (to_print) std::cout << vec[i] << " ";
    }
    if (to_print) std::cout << std::endl;
    fcpplog << std::endl;
}

void print(std::string path, const std::vector<Eigen::Map<Eigen::VectorXi>>& vec, bool to_print = false)
{
    std::ofstream fcpplog(path);
    std::cout.precision(7);
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, " ");
    if (to_print) std::cout << path << std::endl;
    for (uint i = 0; i < MAX_PRINT_NUM && i < vec.size(); ++i)
    {
        for (uint j = 0; j < MAX_PRINT_NUM && j < vec[i].size(); ++j)
        {
            fcpplog << vec[i][j] << " ";
            if (to_print) std::cout << vec[i].format(CommaInitFmt) << " ";
        }
        if (to_print) std::cout << std::endl;
        fcpplog << std::endl;
    }
}

template <typename Derived>
void read_mat(std::string path, Eigen::PlainObjectBase<Derived>& matrix, bool to_print=false)
{
    typedef typename Derived::Scalar Scalar;
    std::ifstream fcpplog(path);
    assert(!fcpplog.fail());
    uint n, m;
    fcpplog >> n >> m;
    if(to_print) std::cout << std::endl << path  << std::endl << "shape: " << n << ' ' << m << std::endl;
    matrix.resize(n,m);
    for (uint i = 0; i < n; ++i)
    {
        for (uint j = 0; j < m; ++j)
        {
            float tmp;
            fcpplog >> tmp; 
            matrix(i, j) = (Scalar)tmp;
        }
    }
}

std::vector<Eigen::Map<Eigen::VectorXi>> remove_negative(Eigen::Map<RowMajorArray> &all_batches)
{
    std::cout << BR50;
    std::vector<Eigen::Map<Eigen::VectorXi>> all_batches_;
    
    for(uint i = 0; i < all_batches.rows(); ++i)
    {
        std::vector<int> v;
        int length = 0;
        for(uint j = 0; j < all_batches.matrix().cols(); ++j)
        {
            auto el = all_batches.matrix()(i, j);
            if(el >= 0)
            {
                ++length;
                v.push_back(el);
                //std::cout << el << " ";
            }
        }
        //std::cout << std::endl;
        all_batches_.push_back(Eigen::Map<Eigen::VectorXi>(all_batches.matrix().row(i).data(), length));
    }
    std::cout << BR50;
    return all_batches_;
} 

void read_sparse_matrixf(FILE *file, fSpMat &mat)
{
    uint32_t is_csr = 0;
    uint32_t indices_size = 0;
    uint32_t indptr_size = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;

    
    fread(&is_csr, sizeof(uint32_t), 1, file);
    fread(&rows, sizeof(uint32_t), 1, file);
    fread(&cols, sizeof(uint32_t), 1, file);
    fread(&indices_size, sizeof(uint32_t), 1, file);
    fread(&indptr_size, sizeof(uint32_t), 1, file);


    std::vector<uint32_t> indices(indices_size);
    std::vector<uint32_t> indptr(indptr_size);
    std::vector<float> data(indices_size);

    fread(indices.data(), sizeof(uint32_t), indices_size, file);
    fread(indptr.data(), sizeof(uint32_t), indptr_size, file);
    fread(data.data(), sizeof(float), indices_size, file);

    if(false)
    {
        std::cout << "is_csr      : " << is_csr << std::endl;
        std::cout << "rows        : " << rows << std::endl;
        std::cout << "cols        : " << cols << std::endl;
        std::cout << "indices_size: " << indices_size << std::endl;
        std::cout << "indptr_size : " << indptr_size << std::endl;

        std::cout << "indices: ";
        for (uint i = 0; i < 20 && i < indices.size(); ++i)
        {
            std::cout << indices[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "indptr: ";
        for (uint i = 0; i < 20 && i < indptr.size(); ++i)
        {
            std::cout << indptr[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "data: ";
        for (uint i = 0; i < 20 && i < data.size(); ++i)
        {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }
    else
    {
        //std::cout << "Read finished"<<std::endl;
    }


    std::vector<Eigen::Triplet<float>> triplets;
    triplets.reserve(indices_size);
    for(uint32_t i = 0; i < indptr_size-1; ++i)
    {
        for(uint32_t jptr = indptr[i]; jptr < indptr[i+1]; ++jptr)
        {
            uint32_t j = indices[jptr];
            float d = data[jptr];
            if(is_csr == 1)
            {
                triplets.push_back(Eigen::Triplet<float>(i, j, d));
            }
            else
            {
                triplets.push_back(Eigen::Triplet<float>(j, i, d));
            }
        }
    }
    mat.resize(rows,cols);
    mat.setFromTriplets(triplets.begin(), triplets.end());
    
}

void read_sparse_matrixf_fast(FILE *file, fSpMat &mat)
{
    uint32_t is_csr = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t indices_size = 0; //nnz
    uint32_t indptr_size = 0;  //outSz

    fread(&is_csr, sizeof(uint32_t), 1, file);
    fread(&rows, sizeof(uint32_t), 1, file);
    fread(&cols, sizeof(uint32_t), 1, file);
    fread(&indices_size, sizeof(uint32_t), 1, file);
    fread(&indptr_size, sizeof(uint32_t), 1, file);

    mat.resize(rows, cols);
    mat.makeCompressed();
    mat.resizeNonZeros(indices_size);

    fread(mat.innerIndexPtr(), sizeof(uint32_t), indices_size, file);       //indices
    fread(mat.outerIndexPtr(), sizeof(uint32_t), indptr_size, file);  //indptr
    fread(mat.valuePtr(), sizeof(float), indices_size, file);          //data

    mat.finalize();
    
    if(false)
    {
        std::cout << "is_csr      : " << is_csr << std::endl;
        std::cout << "rows        : " << rows << std::endl;
        std::cout << "cols        : " << cols << std::endl;
        std::cout << "indices_size: " << indices_size << std::endl;
        std::cout << "indptr_size : " << indptr_size << std::endl;

        std::cout << "indices: ";
        for (uint i = 0; i < 20 && i < indices_size; ++i)
        {
            std::cout << mat.innerIndexPtr()[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "indptr: ";
        for (uint i = 0; i < 20 && i < indptr_size; ++i)
        {
            std::cout << mat.outerIndexPtr()[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "data: ";
        for (uint i = 0; i < 20 && i < indices_size; ++i)
        {
            std::cout << mat.valuePtr()[i] << " ";
        }
        std::cout << std::endl;
    }
    else
    {
        //std::cout << "Read finished"<<std::endl;
    }
}

void parse_spmatrixf(std::string mat_path, fSpMat &mat)
{
    //std::cout << BR50;
    //std::cout << mat_path << std::endl;
    FILE *file = fopen(mat_path.c_str(), "rb");
    
    if(!file)
    {
        std::cout << "Read failed"<<std::endl;
        return;
    }
    //read_sparse_matrixf(file, mat);
    read_sparse_matrixf_fast(file, mat);
    fclose(file);
}

void parse_spmatrixf(std::string mat_path, fSpMatMap &mat_map)
{
     float prep_t, cclock_t;
    struct timeval t_start,t_end;
    clock_t start_t, end_t;
    gettimeofday(&t_start,NULL);
    start_t = clock();

    //std::cout << BR50;
    //std::cout << mat_path << std::endl;
    FILE *file = fopen(mat_path.c_str(), "rb");
    
    if(!file)
    {
        std::cout << "Read failed"<<std::endl;
        return;
    }

    uint32_t num = 0, row_id = 0, batch_id = 0;
    fread(&num, sizeof(uint32_t), 1, file);
    //std::cout << "num: " << num << std::endl;
    for(uint32_t i = 0; i < num; ++i)
    {
        fread(&row_id, sizeof(uint32_t), 1, file);
        fread(&batch_id, sizeof(uint32_t), 1, file);
        //std::cout << BR50;
        //std::cout << "row_id: " << row_id << std::endl;
        //std::cout << "batch_id: " << batch_id << std::endl;
        mat_map[row_id][batch_id] = fSpMat();
        //read_sparse_matrixf(file, mat_map[row_id][batch_id]);
        read_sparse_matrixf_fast(file, mat_map[row_id][batch_id]);
    }
    fclose(file);

    end_t = clock();
    cclock_t = (float)(end_t - start_t) / CLOCKS_PER_SEC;
    gettimeofday(&t_end, NULL);
    prep_t = t_end.tv_sec - t_start.tv_sec + (t_end.tv_usec - t_start.tv_usec)/1000000.0;

    std::cout << BR50;
    std::cout << "cpp parse_spmatrixf time: " << prep_t << " s" << std::endl;
    std::cout << "cpp parse_spmatrixf clock time : " << cclock_t <<" s" << std::endl;
}
namespace predictc{

    void Bsa::construct_sparse_blocks_vec()
    {
        float prep_t, cclock_t;
        struct timeval t_start,t_end;
        clock_t start_t, end_t;
        gettimeofday(&t_start,NULL);
        start_t = clock();

        A_blocks_vec.resize(all_batches.size());
        for(uint i = 0; i < all_batches.size(); ++i)
        {
            A_blocks_vec[i].resize(all_batches.size());
            auto rows_ = all_batches[i];
            for (uint j = 0; j < all_batches.size(); ++j)
            {
                std::cout << "construct: " << i << j << std::endl;
                auto cols_ = all_batches[j];
                A_blocks_vec[i][j].resize(rows_.size(), cols_.size());
                slice(A,rows_,cols_,A_blocks_vec[i][j]);
                
                //print(std::string("./logs/loops/") + std::to_string(i) + "_" +
                //std::to_string(j) + std::string("_A_block")  +
                //std::string("_mat.cpp.log"), A_blocks[i][j], false);
            }
        }

        end_t = clock();
        cclock_t = (float)(end_t - start_t) / CLOCKS_PER_SEC;
        gettimeofday(&t_end, NULL);
        prep_t = t_end.tv_sec - t_start.tv_sec + (t_end.tv_usec - t_start.tv_usec)/1000000.0;

        std::cout << BR50;
        std::cout << "cpp construct_sparse_blocks_vec time: " << prep_t << " s" << std::endl;
        std::cout << "cpp construct_sparse_blocks_vec clock time : " << cclock_t << " s" << std::endl;
    }

    void Bsa::construct_sparse_blocks_mat()
    {
        float prep_t, cclock_t;
        struct timeval t_start,t_end;
        clock_t start_t, end_t;
        gettimeofday(&t_start,NULL);
        start_t = clock();
        

        A_blocksf_map.reserve(tau);
        for (int i = 0; i < rows_id_seq.rows(); ++i)
        {
            //A_blocksf_map[i] = std::unordered_map<int32_t, std::shared_ptr<SpMat>>();
            for(uint work_index=0; work_index < tau; ++work_index)
            {   
                //uint batch_id = rows_id_seq(i, work_index);
                //A_blocksf_map[i][batch_id] = std::make_shared<SpMat>();
            }
        }
        uint thread_num_ = 1;
        std::vector<std::thread> workers;
        for (int i = 0; i < rows_id_seq.rows(); ++i)
        {
            auto rows_ = all_batches[i];
            for(uint work_index=0; work_index < tau; ++work_index)
            {   
                uint batch_id = rows_id_seq(i, work_index);
                workers.push_back(std::thread([=]() 
                {
                    auto cols_ = all_batches[batch_id];
                    //std::cout << "construct: " << i << " <- " << batch_id << std::endl;
                    if (A_blocksf_map[i].find(batch_id) == A_blocksf_map[i].end())
                    {
                        A_blocksf_map[i][batch_id] = fSpMat();
                        A_blocksf_map[i][batch_id].resize(rows_.size(), cols_.size());
                        slice(A,rows_,cols_, A_blocksf_map[i][batch_id]);
                    }
                }));
                if(workers.size() >= thread_num_)
                {
                    for(uint j=0; j < workers.size(); ++j)
                    {
                        workers[j].join();
                    }
                    workers.clear();
                }
            }
        }
        for(uint j=0; j < workers.size(); ++j)
        {
            workers[j].join();
        }
        workers.clear();

        end_t = clock();
        cclock_t = (float)(end_t - start_t) / CLOCKS_PER_SEC;
        gettimeofday(&t_end, NULL);
        prep_t = t_end.tv_sec - t_start.tv_sec + (t_end.tv_usec - t_start.tv_usec)/1000000.0;

        std::cout << BR50;
        std::cout << "cpp construct_sparse_blocks_mat time: " << prep_t << " s" << std::endl;
        std::cout << "cpp construct_sparse_blocks_mat clock time : " << cclock_t <<" s" << std::endl;
    }

    std::atomic<bool> Bsa::worker_func_end_wall;
    std::atomic<bool> Bsa::worker_func_begin_wall;
    std::atomic<bool> Bsa::dispatcher_must_exit;
    std::atomic<int> Bsa::waiting_workers;
    std::atomic<int> Bsa::global_time;
    std::atomic<int> Bsa::done_workers;

        
    Bsa::Bsa(
        fMMat &b_,
        fMMat &x_prev_,
        fMMat &x_,
        fMMat &P_,
        fMMat &Q_,
        Eigen::Map<Eigen::MatrixXi> &rows_id_seq_,
        Eigen::Map<RowMajorArray> &all_batches_,
        std::string dataset_name_,
        float epsilon_, float gamma_, 
        uint niter_, uint threads_num_,
        uint extra_logs_, uint tau_, uint optimal_batch_size_
    ) : 
        b(b_), x_prev(x_prev_), x(x_), P(P_), Q(Q_), rows_id_seq(rows_id_seq_),
        all_batches(remove_negative(all_batches_)),
        dataset_name(dataset_name_),
        epsilon(epsilon_), gamma(gamma_),
        niter(niter_), threads_num(threads_num_),
        extra_logs(extra_logs_), tau(tau_), optimal_batch_size(optimal_batch_size_)
    {
        worker_func_begin_wall = true;
		worker_func_end_wall = true;
        waiting_workers = 0;
        global_time = 0;
        dispatcher_must_exit = false;
        done_workers = 0;

        std::cout << "BSA cpp" << std::endl;
        if(extra_logs)
        {
            std::cout << "dataset name: " << dataset_name << std::endl;
            std::cout << "b:" << b.rows() << " " << b.cols() << std::endl;
            std::cout << "x:" << x.rows() << " " << x.cols() << std::endl;
            std::cout << "niter" << niter << std::endl;
            std::cout << "P:" << P.rows() << " " << P.cols() << std::endl;
            std::cout << "Q:" << Q.rows() << " " << Q.cols() << std::endl;
            std::cout << "all_batches:" << all_batches.size() << " " << std::endl;
            std::cout << "epsilon" << epsilon << std::endl;
            std::cout << "gamma" << gamma << std::endl;
            std::cout << "tau" << tau << std::endl;
        }

        assert(dataset_name.size() > 0);

        std::string sp_name = std::string("bsa_appnp/data/") + dataset_name + std::string("_A_sp.pack");
        std::string spmap_name = std::string("bsa_appnp/data/") + dataset_name + std::string("_")
        + std::to_string(optimal_batch_size) + std::string("_A_map_sp.pack");
        //parse_spmatrixf(sp_name, A);
        parse_spmatrixf(spmap_name, A_blocksf_map);
        
        //construct_sparse_blocks_vec();
        //construct_sparse_blocks_mat();

        if(extra_logs)
        {
            print("./logs/b_mat.cpp.log", b);
            print("./logs/x_mat.cpp.log", x);
            print("./logs/P_mat.cpp.log", P);
            print("./logs/Q_mat.cpp.log", Q);
            print("./logs/A_mat.cpp.log", A);
            //print("./logs/all_batches_mat.cpp.log", all_batches);
            print("./logs/all_batches_vec.cpp.log", all_batches);
            print("./logs/rows_id_seq_mat.cpp.log", rows_id_seq);
            

            std::ofstream bsa_serialized("./logs/bsa_serialized.cpp.log");
            bsa_serialized << dataset_name << " "
            << " " << niter 
            << " " << epsilon << " " << gamma
            << " " << threads_num << " " << tau;
            bsa_serialized.close();
        }
    }
    
    float Bsa::bsa_operation()
    {
        float prep_t, cclock_t;
        struct timeval t_start,t_end;
        clock_t start_t, end_t;
        gettimeofday(&t_start,NULL);
        start_t = clock();

        std::cout << BR50;
        std::cout << "update: ";
        
        //bsa();
        bsa_multithread_all();
        //bsa_multithread();
        //bsa_multithread1();


        end_t = clock();
        cclock_t = (float)(end_t - start_t) / CLOCKS_PER_SEC;
        gettimeofday(&t_end, NULL);
        prep_t = t_end.tv_sec - t_start.tv_sec + (t_end.tv_usec - t_start.tv_usec)/1000000.0;
        std::cout << BR50;
        std::cout<<"cpp BSA time: "<<prep_t<<" s"<<"\r"<<std::endl;
        std::cout<<"cpp BSA clock time : "<<cclock_t<<" s"<<"\r"<<std::endl;

        //writeToCSVfile("test.csv", A);
        //writeToBitmap("test.bmp", A);
        //saveMarket(A, "test.save");
        print(std::string("./logs/x_res_mat.cpp.log"), x, false);
        return 0;
    }

    void Bsa::bsa()
    {
        Eigen::MatrixXf x_rows;
        Eigen::MatrixXf x_cols;
        Eigen::MatrixXf b_rows;

        std::cout << "bsa tau: " << tau << std::endl;
        for(uint iter=0; iter < tau; ++iter)
        {
            for (int worker_index = 0; worker_index < rows_id_seq.rows(); ++worker_index)
            {
                //std::cout << "bsa update: " worker_index << " " << iter << std::endl;
                uint rows_id = worker_index;
                uint batch_id = rows_id_seq(worker_index, iter);

                auto rows_ = all_batches[rows_id];
                auto cols_ = all_batches[batch_id];
                auto jump = P(rows_id, batch_id);
                auto qjump = Q(rows_id, batch_id);
                jump *= 1; qjump*= 1;

               
                if(true)
                {
                    auto &A_rows_cols_map = A_blocksf_map[rows_id][batch_id];
                    if (extra_logs)
                    {
                        fSpMat A_rows_cols;
                        slice(A, rows_, cols_, A_rows_cols);
                        auto &A_rows_cols_vec = A_blocks_vec[rows_id][batch_id];
                        std::string findex = std::to_string(iter) + std::string("_") + std::to_string(worker_index) +
                            std::string("_") + std::to_string(batch_id) + std::string("->") + std::to_string(rows_id);
                        print(std::string("./logs/loops/") + findex + std::string("_A")  + 
                        std::string("_mat.cpp.log"), A_rows_cols, true);
                        print(std::string("./logs/loops/") + findex + std::string("_A_map")  + 
                        std::string("_mat.cpp.log"), A_rows_cols_map, true);
                        print(std::string("./logs/loops/") + findex + std::string("_A_vec")  + 
                        std::string("_mat.cpp.log"), A_rows_cols_vec, true);
                    }

                    float q = 1.0/pow((1+iter),this->gamma) * jump/qjump;
                    if(iter%2)
                    {
                        x_prev(rows_, Eigen::all) = x(rows_, Eigen::all) + q*((1/jump) * A_rows_cols_map* x(cols_,Eigen::all) - 
                        x(rows_, Eigen::all) + b(rows_,Eigen::all));
                    }
                    else
                    {
                        x(rows_, Eigen::all) = x_prev(rows_, Eigen::all) + q*((1/jump) * A_rows_cols_map * x_prev(cols_,Eigen::all) - 
                        x_prev(rows_, Eigen::all) + b(rows_,Eigen::all));                
                    }
                }
                
                if(false)
                {
                    float q = 1.0/pow((1+iter),this->gamma) * jump/qjump;
                    
                    for (int ir = 0; ir < rows_.size(); ++ir)
                    {
                        auto row_index = rows_[ir];
                        auto x_row = iter%2 ? x_prev.row(row_index) : x.row(row_index);
                        auto x_prev_row = iter%2 ? x.row(row_index) : x_prev.row(row_index);
                        auto b_row = b.row(row_index);
                        auto &Af_map_row = Af_map[row_index];
                        for(int jr = 0; jr < x_row.size(); ++jr)
                        {
                            float A_xcol = 0;
                            for(int ic = 0; ic < cols_.size(); ++ic)
                            {
                                auto col_index =cols_[ic];
                                auto x_prev_col = iter%2 ? x.row(col_index) : x_prev.row(col_index);
                                //int64_t A_index = (int64_t)row_index << 32 | (int64_t)col_index;
                                //auto A_ir_ic = A_map[A_index];
                                auto A_ir_ic = Af_map_row[col_index];
                                //std::cout << "A_" << row_index << "_" << col_index << " :" << A_ir_ic << std::endl;
                                auto x_prev_col_jr = x_prev_col[jr];
                                A_xcol += A_ir_ic * x_prev_col_jr;
                            }
                            auto x_ir_jr = x_prev_row[jr] + q*((1/jump) * A_xcol - x_prev_row[jr] + b_row[jr]);
                            //std::cout << "bsa: " << x_row[jr] << " vs " << x_ir_jr << std::endl;
                            x_row[jr] = x_ir_jr;
                        }
                    }
                }

                if(false)
                {
                    float q = 1.0/pow((1+iter),this->gamma) * jump/qjump;
                    
                    for (int ir = 0; ir < rows_.size(); ++ir)
                    {
                        auto row_index = rows_[ir];
                        float* x_ = iter%2 ? x_prev.data() : x.data();
                        float* x_prev_ = iter%2 ? x.data() : x_prev.data();
                        int n = x.rows();
                        float* b_ = b.data();
                        //auto &Af_map_row = Af_map[row_index];

                        for(int jr = 0; jr < x.cols(); ++jr)
                        {
                            float A_xcol = 0;
                            for(int ic = 0; ic < cols_.size(); ++ic)
                            {
                                auto col_index =cols_[ic];
                                auto A_ir_ic = 0;//Af_map_row[col_index];
                                //std::cout << "A_" << row_index << "_" << col_index << " :" << A_ir_ic << std::endl;

                                auto x_prev_col_jr = x_prev_[jr*n + col_index];
                                A_xcol += A_ir_ic * x_prev_col_jr;
                            }
                            auto x_ir_jr = x_prev_[jr*n + row_index] + q*((1/jump) * A_xcol - x_prev_[jr*n + row_index] + b_[jr*n + row_index]);
                            //std::cout << "bsa: " << x_[jr*n + row_index] << " vs " << x_ir_jr << std::endl;
                            x_[jr*n + row_index] = x_ir_jr;
                        }
                    }
                }

                if(false)
                {
                    float q = 1.0/pow((1+iter),this->gamma) * jump/qjump;
                    
                    for (int ir = 0; ir < rows_.size(); ++ir)
                    {
                        auto row_index = rows_[ir];
                        float* x_ = iter%2 ? x_prev.data() : x.data();
                        float* x_prev_ = iter%2 ? x.data() : x_prev.data();
                        int n = x.rows();
                        float* b_ = b.data();
                        //auto &Af_map_row = Af_map[row_index];

                        for(int jr = 0; jr < x.cols(); ++jr)
                        {
                            float A_xcol = 0;
                            for(int ic = 0; ic < cols_.size(); ++ic)
                            {
                                auto col_index =cols_[ic];
                                auto A_ir_ic = 0;//Af_map_row[col_index];
                                //std::cout << "A_" << row_index << "_" << col_index << " :" << A_ir_ic << std::endl;

                                auto x_prev_col_jr = x_prev_[jr*n + col_index];
                                A_xcol += A_ir_ic * x_prev_col_jr;
                            }
                            auto x_ir_jr = x_prev_[jr*n + row_index] + q*((1/jump) * A_xcol - x_prev_[jr*n + row_index] + b_[jr*n + row_index]);
                            //std::cout << "bsa: " << x_[jr*n + row_index] << " vs " << x_ir_jr << std::endl;
                            x_[jr*n + row_index] = x_ir_jr;
                        }
                    }
                }

                if (extra_logs)
                {
                    std::cout << worker_index << " " << iter << std::endl;

                    std::cout << "jump: " << jump << std::endl;
                    std::cout << "qjump: " << qjump << std::endl;

                    std::string findex = std::to_string(iter) + std::string("_") + std::to_string(worker_index) +
                        std::string("_") + std::to_string(batch_id) + std::string("->") + std::to_string(rows_id);
                        
                    //print(std::string("./logs/loops/") +findex + std::string("_A")  + std::string("_mat.cpp.log"), A_rows_cols, true);
                                
                    print(std::string("./logs/loops/") +findex + std::string("_rows")   + std::string("_mat.cpp.log"), rows_);
                    print(std::string("./logs/loops/") +findex + std::string("_cols")   + std::string("_mat.cpp.log"), cols_);

                    print(std::string("./logs/loops/") +findex + std::string("_x_rows") + std::string("_mat.cpp.log"), x(rows_, Eigen::all));
                    print(std::string("./logs/loops/") +findex + std::string("_x_cols") + std::string("_mat.cpp.log"), x(cols_, Eigen::all));
                    print(std::string("./logs/loops/") +findex + std::string("_b_rows") + std::string("_mat.cpp.log"), b(rows_,Eigen::all));
                    
                    print(std::string("./logs/loops/") +findex + std::string("_x")      + std::string("_mat.cpp.log"), x, false);
                    print(std::string("./logs/loops/") +findex + std::string("_x_prev") + std::string("_mat.cpp.log"), x_prev, false);
                    std::cout << BR50;
                }
            }
        }
    }

    void Bsa::bsa_multithread1()
    {
        for(uint work_index=0; work_index < tau; ++work_index)
        {
            std::vector<std::thread> ths;
        
            for (int worker_index = 0; worker_index < rows_id_seq.rows(); ++worker_index)
            {
                ths.push_back(std::thread(&Bsa::bsa_worker1, this, worker_index, work_index));
            
            }

            for (int i = 0; i < rows_id_seq.rows(); ++i)
            {
                ths[i].join();
            }
            std::vector<std::thread>().swap(ths);
        }
    }

    void Bsa::bsa_worker1(uint worker_index, uint work_index)
    {
        fSpMat A_rows_cols;
        uint rows_id = rows_id_seq(worker_index, work_index);
        uint batch_id = rows_id_seq(worker_index, work_index+1);

        auto rows_ = all_batches[rows_id];
        auto cols_ = all_batches[batch_id];
        auto jump = P(rows_id, batch_id);
        auto qjump = Q(rows_id, batch_id);
        jump *= 1; qjump*= 1;

        slice(A,rows_,cols_,A_rows_cols);
        float q = 1.0/pow((1+work_index),this->gamma) * jump/qjump;
        x(rows_, Eigen::all) = x(rows_, Eigen::all) + q*((1/jump) * A_rows_cols * x(cols_,Eigen::all) - x(rows_, Eigen::all) + b(rows_,Eigen::all));
    }

    void Bsa::bsa_multithread_all()
    {
        for(uint work_index=0; work_index < tau; ++work_index)
        {
            std::vector<std::thread> ths;
        
            for (int worker_index = 0; worker_index < rows_id_seq.rows(); ++worker_index)
            {
               
                //std::cout << "x --> x_prev" << std::endl;
                ths.push_back(std::thread(&Bsa::bsa_worker_all, this, worker_index, work_index));

                //ths[worker_index].join();
                if(worker_index % threads_num == 0)
                {
                    for (uint i = 0; i < ths.size(); ++i)
                    {
                        ths[i].join();
                    }
                    ths.clear();
                }
            }

            for (uint i = 0; i < ths.size(); ++i)
            {
                ths[i].join();
            }
            std::vector<std::thread>().swap(ths);
        }
    }

    void Bsa::bsa_worker_all(uint worker_index, uint work_index)
    {
        uint rows_id = worker_index;
        uint batch_id = rows_id_seq(worker_index, work_index);
        //std::cout << "id: " << batch_id << " --> " << rows_id << std::endl;
        auto rows_ = all_batches[rows_id];
        auto cols_ = all_batches[batch_id];
        auto jump = P(rows_id, batch_id);
        auto qjump = Q(rows_id, batch_id);
        jump *= 1; qjump*= 1;


        //SpMat A_rows_cols;
        //slice(A,rows_,cols_,A_rows_cols);
        auto &A_rows_cols_map = A_blocksf_map[rows_id][batch_id];
        float q = 1.0/pow((1+work_index),this->gamma) * jump/qjump;
        if(work_index % 2)
        {
            x_prev(rows_, Eigen::all) = x(rows_, Eigen::all) + q*((1/jump) * A_rows_cols_map* x(cols_,Eigen::all) - 
            x(rows_, Eigen::all) + b(rows_,Eigen::all));
        }
        else
        {
            x(rows_, Eigen::all) = x_prev(rows_, Eigen::all) + q*((1/jump) * A_rows_cols_map * x_prev(cols_,Eigen::all) - 
            x_prev(rows_, Eigen::all) + b(rows_,Eigen::all));                
        }

        if (extra_logs)
        {
            std::string findex = std::to_string(work_index) + std::string("_") + std::to_string(worker_index) +
                std::string("_") + std::to_string(batch_id) + std::string("->") + std::to_string(rows_id);
                
            print(std::string("./logs/loops/") +findex + std::string("_A")  + std::string("_mat.cpp.log"), A_rows_cols_map);
                        
            print(std::string("./logs/loops/") +findex + std::string("_rows")   + std::string("_mat.cpp.log"), rows_);
            print(std::string("./logs/loops/") +findex + std::string("_cols")   + std::string("_mat.cpp.log"), cols_);

            print(std::string("./logs/loops/") +findex + std::string("_x_rows") + std::string("_mat.cpp.log"), x(rows_, Eigen::all));
            print(std::string("./logs/loops/") +findex + std::string("_x_cols") + std::string("_mat.cpp.log"), x(cols_, Eigen::all));
            print(std::string("./logs/loops/") +findex + std::string("_b_rows") + std::string("_mat.cpp.log"), b(rows_,Eigen::all));
            
            print(std::string("./logs/loops/") +findex + std::string("_x")      + std::string("_mat.cpp.log"), x, false);
            print(std::string("./logs/loops/") +findex + std::string("_x_prev") + std::string("_mat.cpp.log"), x_prev, false);
        }

    }
}

#ifndef CYTHON_COMPILE

void accuracy_check()
{
    std::string filename = "./run_bsa_appnp.py -e";
    std::string command = "python3 ";
    command += filename;
    system(command.c_str());

}

int main()
{
    if (false)
    {
        std::string filename = "./run_bsa_appnp.py -p";
        std::string command = "python3 ";
        command += filename;
        system(command.c_str());
    }
    
    std::cout << BR50;

    std::string dataset_name;
    Eigen::MatrixXf b;
    Eigen::MatrixXf x_prev;
    Eigen::MatrixXf x;
    uint niter;
    Eigen::MatrixXf P;
    Eigen::MatrixXf Q;
    uint tau;
    uint optimal_batch_size;

    MatrixXiRowMajor all_batches;

    Eigen::MatrixXi rows_id_seq;
    float epsilon, gamma;
    uint threads_num;

    Eigen::MatrixXf res_py;
    Eigen::MatrixXf res_cpp;

    std::ifstream bsa_serialized("./logs/bsa_serialized.py.log");
    assert(!bsa_serialized.fail());

    bsa_serialized >> dataset_name
    >> niter 
    >> epsilon >> gamma
    >> threads_num >> tau >> optimal_batch_size;
    std::string log_dir("");
    bsa_serialized.close();



    std::cout << dataset_name << " "
            << " " << niter 
            << " " << epsilon << " " << gamma
            << " " << threads_num << " " << tau << optimal_batch_size;

    read_mat("./logs/b_mat.py.log", b);
    read_mat("./logs/x_mat.py.log", x_prev);
    read_mat("./logs/x_mat.py.log", x);
    read_mat("./logs/P_mat.py.log", P);
    read_mat("./logs/Q_mat.py.log", Q);
    read_mat("./logs/all_batches_squared_mat.py.log", all_batches);
    read_mat("./logs/rows_id_seq_mat.py.log", rows_id_seq);
    
    
    auto b_ = fMMat(b.data(), b.rows(), b.cols());
    auto x_prev_ = fMMat(x_prev.data(), x_prev.rows(), x_prev.cols());
    auto x_ = fMMat(x.data(), x.rows(), x.cols());
    auto P_ = fMMat(P.data(), P.rows(), P.cols());
    auto Q_ = fMMat(Q.data(), Q.rows(), Q.cols());
    
    Eigen::Map<RowMajorArray> all_batches_ = Eigen::Map<RowMajorArray>(all_batches.data(), all_batches.rows(), all_batches.cols());
    auto rows_id_seq_ = Eigen::Map<Eigen::MatrixXi>(rows_id_seq.data(), rows_id_seq.rows(), rows_id_seq.cols());

    uint extra_logs = 0;

    predictc::Bsa bsa(
    b_,
    x_prev_,
    x_,
    P_,
    Q_,
    rows_id_seq_,
    all_batches_,
    dataset_name,
    epsilon,
    gamma,
    niter, 
    threads_num,
    extra_logs,
    tau,
    optimal_batch_size
    );
    
    
    bsa.bsa_operation();
    
    read_mat("./logs/x_res_mat.py.log", res_py);
    read_mat("./logs/x_res_mat.cpp.log", res_cpp);

    std::cout << BR50;
    std::cout << "sum cpp : " << res_cpp.sum() << std::endl;
    std::cout << "sum py  : " << res_py.sum() << std::endl;
    std::cout << "diff sum_: " << (res_cpp - res_py).sum() << std::endl;

    accuracy_check();

    return 0;
}

int main1()
{
    if (true)
    {
        std::string filename = "./sparse_serialize.py";
        std::string command = "python3 ";
        command += filename;
        system(command.c_str());
    }

    fSpMat fA_row;
    parse_spmatrixf("./logs/test/Ar_sp.pack", fA_row);
    print("./logs/test/fA_row_spmat.cpp.log", fA_row);

    fSpMat fA_col;
    parse_spmatrixf("./logs/test/Ac_sp.pack", fA_col);
    print("./logs/test/fA_col_spmat.cpp.log", fA_col);

    fSpMatMap mat_map;
    parse_spmatrixf("./logs/test/Acr_map_sp.pack", mat_map);
    print("./logs/test/fA12_col_spmat.cpp.log", mat_map[1][2]);
    print("./logs/test/fA15_col_spmat.cpp.log", mat_map[1][5]);
    print("./logs/test/fA54_col_spmat.cpp.log", mat_map[5][4]);
    print("./logs/test/fA57_col_spmat.cpp.log", mat_map[5][7]);

    return 0;
}

#endif
