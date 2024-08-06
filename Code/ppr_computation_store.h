#pragma once

extern "C"
{
#include "matrix_vector_functions_intel_mkl.h"
#include "matrix_vector_functions_intel_mkl_ext.h"
#include "string.h"
}

#undef max
#undef min



#include "malloc.h"


#include <cmath>
#include <cstdint>
#include <random>



#include <algorithm>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <deque>
#include <vector>
#include <unordered_map>
// #include "Graph.h"
#include <fstream>
#include <cstring>
#include <thread>
#include <mutex>
#include "Eigen/Sparse"
#include "Eigen/Dense"
#include <chrono>
#include <climits>


#include <boost/archive/basic_archive.hpp>
#include <boost/serialization/nvp.hpp>
#include <iomanip>
#include <cassert>
#include<functional>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include <map>
#include <boost/serialization/map.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include<numeric>


#include<queue>
#include "my_queue.h"



// #include "unordered_map_serialization.h"



#include<assert.h>
#include <unordered_map>
#include<cmath>

#include<list>

#include<memory.h>

#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>


#include "../emhash/hash_table7.hpp"

//#include "metis_test_csdn.h"

using namespace Eigen;

using namespace std;





using third_float_map = emhash7::HashMap<int, float>;
using third_bool_map = emhash7::HashMap<int, bool>;
using third_int_map = emhash7::HashMap<int, int>;















std::mt19937 rand_uint{(std::random_device())()};

double rand_uniformf() {
  return 0x1.0p-32 * rand_uint();
}

uint32_t rand_uniform(uint32_t n) {
  uint64_t m = (uint64_t)rand_uint() * n;
  if ((uint32_t)m < n) {
    uint32_t t = -n;
    if (t >= n) t -= n;
    if (t >= n) t %= n;
    while ((uint32_t)m < t)
      m = (uint64_t)rand_uint() * n;
  }
  return m >> 32;
}







class d_row_tree_mkl{
  public:
  int nParts;

  int level_p;
  int total_nodes;
  vector<mat*> matrix_vec;
  int largest_level_start_index;
  int largest_level_end_index;
  vector<mat*> hierarchy_matrix_vec;

  Eigen::VectorXd vS_cur_iter;

  Eigen::MatrixXd U_cur_iter;

  vector<mat*> near_n_matrix_vec;
  vector<mat*> less_near_n;

  // unordered_map<int, SparseMatrix<double, 0, long long int>> mat_mapping;

  // unordered_map<int, MatrixXd> dense_mat_mapping;

//   int start_row;
//   int end_row;
  int hierarchy_n;
  long long int row_dim;

//   vector<double> norm_B_Bid_difference_vec;

  vector<int> block_svd_indicator;
    

  d_row_tree_mkl(long long int row_dim, int d, int nParts, int hierarchy_n, unordered_map<int, vector<long long int>> &vec_mapping
  ){

    this->nParts = nParts;

    this->row_dim = row_dim;

    // this->start_row = start_row;
    // this->end_row = end_row;
    this->hierarchy_n = hierarchy_n;


    // norm_B_Bid_difference_vec.resize(vec_mapping.size());

    // for(int i = 0; i < vec_mapping.size(); i++){
    //   vector<long long int> &v = vec_mapping[i];
    //   int current_group_size = vec_mapping[i].size();
    //   // mat_mapping[i].resize(row_dim, current_group_size);
    //   // dense_mat_mapping[i].resize(row_dim, current_group_size);
    // //   norm_B_Bid_difference_vec[i] = 0;
    // }


    level_p = log(nParts) / log(hierarchy_n) + 1;

    // assert(pow(hierarchy_n, level_p - 1) == nParts);

    total_nodes = (pow(hierarchy_n, level_p) - 1) / (hierarchy_n - 1);

    largest_level_start_index = total_nodes - nParts;

    largest_level_end_index = total_nodes - 1;

    hierarchy_matrix_vec.resize(total_nodes);

    // for(int i = 0; i < total_nodes; i++){
    //   hierarchy_matrix_vec[i] = matrix_new(row_dim, d);
    // }

    // near_n_matrix_vec.resize(total_nodes - nParts);

    // less_near_n.resize(total_nodes - nParts);
  }
  
};






















void parallel_store_matrix_nParts_NUMTHREAD(
// vector<map<long long int, double>>& all_csr_entries,
vector<vector<map<long long int, double>>>& Partition_thread_all_csr_entries,
mat_coo *ppr_matrix_coo,
long long int start,
long long int end,
// long long int start_column,
long long int start_nnz,
int threads_number,
int stored_interval
){

  long long int nnz_iter=start_nnz;

  // cout<<"start_nnz = "<<start_nnz<<endl;




  for (long long int k = start; k < end; ++k){
    int interval_index = (k - start) / stored_interval;

    if(interval_index == Partition_thread_all_csr_entries.size()){
      interval_index--;
    }

    int inner_index = k - start - interval_index * stored_interval;



    map<long long int, double> all_csr_entries = Partition_thread_all_csr_entries[interval_index][inner_index];

    // cout<<"k = "<<k<<endl;
    // cout<<"interval_index = "<<interval_index<<endl;
    // cout<<"inner_index = "<<inner_index<<endl;

    // cout<<"all_csr_entries.size() = "<<all_csr_entries.size()<<endl;

    // cout<<"all_csr_entries[k - start].size() = "<<all_csr_entries[k - start].size()<<endl;
    // cout<<"k = "<<k<<endl;
    // cout<<"nnz_iter = "<<nnz_iter<<endl;
    // for (auto & key_value: all_csr_entries[k]){
    // for (auto & key_value: all_csr_entries[k - start]){
    for (auto & key_value: all_csr_entries){
      // double value1 = log10(key_value.second);
      double value1 = log10(1 + key_value.second);
      // cout<<"value1 = "<<value1<<endl;
      
      // ppr_matrix_coo->rows[nnz_iter] = k - start_column + 1;
      ppr_matrix_coo->rows[nnz_iter] = k + 1;
      
      ppr_matrix_coo->cols[nnz_iter] = key_value.first + 1;

      // if(nnz_iter <= 148){
      //   cout<<"nnz_iter <= 148: ppr_matrix_coo->rows["<<nnz_iter<<"] = "<<ppr_matrix_coo->rows[nnz_iter]<<endl;
      //   cout<<"nnz_iter <= 148: ppr_matrix_coo->cols["<<nnz_iter<<"] = "<<ppr_matrix_coo->cols[nnz_iter]<<endl;
      // }

      ppr_matrix_coo->values[nnz_iter] = value1;
      nnz_iter++;
      // if(threads_number == 64){
      //   cout<<"64: nnz_iter = "<<nnz_iter<<endl;
      //   cout<<"13:ppr_matrix_coo->rows["<<0<<"] = "<<ppr_matrix_coo->rows[0]<<endl;
      //   cout<<"13:ppr_matrix_coo->cols["<<0<<"] = "<<ppr_matrix_coo->cols[0]<<endl;
      //   cout<<endl;
      // }
    }

    // all_csr_entries[k].clear();    
    // all_csr_entries[k - start].clear();    
    all_csr_entries.clear();    
  }

  // cout<<threads_number<<": "<<"nnz_iter = "<<nnz_iter<<endl;

}





void parallel_store_matrix(
vector<map<long long int, double>>& all_csr_entries,
mat_coo *ppr_matrix_coo,
long long int start,
long long int end,
long long int start_column,
long long int start_nnz
){

  long long int nnz_iter=start_nnz;


  for (long long int k = start; k < end; ++k){
    // cout<<"k = "<<k<<endl;
    // cout<<"nnz_iter = "<<nnz_iter<<endl;
    for (auto & key_value: all_csr_entries[k]){
      double value1 = log10(1 + key_value.second);
      // double value1 = 1 + key_value.second;
      // cout<<"value1 = "<<value1<<endl;
      
      ppr_matrix_coo->rows[nnz_iter] = k - start_column + 1;
      ppr_matrix_coo->cols[nnz_iter] = key_value.first + 1;
      ppr_matrix_coo->values[nnz_iter] = value1;
      nnz_iter ++;
    }

    all_csr_entries[k].clear();    
  }

  malloc_trim(0);

}




void parallel_store_matrix_less_space(
vector<map<long long int, double>>& all_csr_entries,
mat_coo *ppr_matrix_coo,
long long int start,
long long int end,
long long int start_column,
long long int start_nnz,
int** ppr_col = NULL,
double** ppr_nnz = NULL
){

  long long int nnz_iter=start_nnz;

  if(ppr_nnz != NULL)
  {
    for (long long int k = start; k < end; ++k)
    {
      // int ssz = (ppr_nnz[k] == NULL) ? 0 : (ppr_nnz[k][0].first);
      int ssz = ppr_col[k][0];

      for(int j = 1; j <= ssz; ++j){
        // double value1 = log10(1 + ppr_nnz[k].second);
        
        ppr_matrix_coo->rows[nnz_iter] = k - start_column + 1;
        ppr_matrix_coo->cols[nnz_iter] = ppr_col[k][j] + 1;
        ppr_matrix_coo->values[nnz_iter] = log10(1 + ppr_nnz[k][j]);
        nnz_iter ++;
      }
      if(ssz > 0) 
      {
        delete[] ppr_nnz[k]; 
        delete[] ppr_col[k];
      }
    }
  }
  else 
  for (long long int k = start; k < end; ++k){
    // cout<<"k = "<<k<<endl;
    // cout<<"nnz_iter = "<<nnz_iter<<endl;
    for (auto & key_value: all_csr_entries[k]){
      double value1 = log10(1 + key_value.second);
      // double value1 = 1 + key_value.second;
      // cout<<"value1 = "<<value1<<endl;
      
      ppr_matrix_coo->rows[nnz_iter] = k - start_column + 1;
      ppr_matrix_coo->cols[nnz_iter] = key_value.first + 1;
      ppr_matrix_coo->values[nnz_iter] = value1;
      nnz_iter ++;
    }

    all_csr_entries[k].clear();    
  }

  malloc_trim(0);

}



void parallel_store_matrix_less_space_rows_cols(
vector<map<long long int, double>>& all_csr_entries,
mat_coo_2 *ppr_matrix_coo,
long long int start,
long long int end,
long long int start_column,
long long int start_nnz,
int** ppr_col = NULL
){

  long long int nnz_iter = start_nnz;

  if(ppr_col != NULL)
  {
    for (long long int k = start; k < end; ++k)
    {
      // int ssz = (ppr_nnz[k] == NULL) ? 0 : (ppr_nnz[k][0].first);
      int ssz = ppr_col[k][0];

      for(int j = 1; j <= ssz; ++j){
        // double value1 = log10(1 + ppr_nnz[k].second);
        
        ppr_matrix_coo->rows[nnz_iter] = k - start_column + 1;
        ppr_matrix_coo->cols[nnz_iter] = ppr_col[k][j] + 1;
        nnz_iter ++;
      }
      if(ppr_col[k] != NULL) 
      {
        delete[] ppr_col[k];
        ppr_col[k] = NULL;
      }
    }
  }
  else 
  for (long long int k = start; k < end; ++k){
    // cout<<"k = "<<k<<endl;
    // cout<<"nnz_iter = "<<nnz_iter<<endl;
    for (auto & key_value: all_csr_entries[k]){
      double value1 = log10(1 + key_value.second);
      // double value1 = 1 + key_value.second;
      // cout<<"value1 = "<<value1<<endl;
      
      ppr_matrix_coo->rows[nnz_iter] = k - start_column + 1;
      ppr_matrix_coo->cols[nnz_iter] = key_value.first + 1;
      ppr_matrix_coo->values[nnz_iter] = value1;
      nnz_iter ++;
    }

    all_csr_entries[k].clear();    
  }

  malloc_trim(0);

}



void parallel_store_matrix_less_space_nnz(
vector<map<long long int, double>>& all_csr_entries,
mat_coo_2 *ppr_matrix_coo,
long long int start,
long long int end,
long long int start_column,
long long int start_nnz,
int** ppr_col = NULL,
double** ppr_nnz = NULL
){

  long long int nnz_iter = start_nnz;

  if(ppr_nnz != NULL)
  {
    for (long long int k = start; k < end; ++k)
    {
      // int ssz = (ppr_nnz[k] == NULL) ? 0 : (ppr_nnz[k][0].first);
      int ssz = ppr_col[k][0];

      for(int j = 1; j <= ssz; ++j){
        // double value1 = log10(1 + ppr_nnz[k].second);
        
        ppr_matrix_coo->values[nnz_iter] = log10(1 + ppr_nnz[k][j]);
        nnz_iter ++;
      }
      if(ppr_nnz[k] != NULL) 
      {
        delete[] ppr_nnz[k]; 
        ppr_nnz[k] = NULL;
      }
    }
  }
  else 
  for (long long int k = start; k < end; ++k){
    // cout<<"k = "<<k<<endl;
    // cout<<"nnz_iter = "<<nnz_iter<<endl;
    for (auto & key_value: all_csr_entries[k]){
      double value1 = log10(1 + key_value.second);
      // double value1 = 1 + key_value.second;
      // cout<<"value1 = "<<value1<<endl;
      
      ppr_matrix_coo->rows[nnz_iter] = k - start_column + 1;
      ppr_matrix_coo->cols[nnz_iter] = key_value.first + 1;
      ppr_matrix_coo->values[nnz_iter] = value1;
      nnz_iter ++;
    }

    all_csr_entries[k].clear();    
  }

  malloc_trim(0);

}







void parallel_get_matrix_from_mkl(mat* mkl, MatrixXd &Eig, 
long long int row_start, long long int row_end, 
long long int col_start, long long int col_end,
long long int eigen_row_more_than_mkl,
long long int eigen_col_more_than_mkl){
  // Eig.resize(mkl->nrows, mkl->ncols);
  for(long long int i = row_start; i < row_end; i++){
    for(long long int j = col_start; j < col_end; j++){
      Eig(eigen_row_more_than_mkl + i, eigen_col_more_than_mkl + j) = matrix_get_element(mkl, i, j);
    }
  }
}

















void parallel_rescale_degree_directed(MatrixXd &U_V, int* out_degree, int* in_degree,
long long int row_start, long long int row_end, int d
){
  
  for(long long int i = row_start; i < row_end; i++){
    for(int j = 0; j < d; j++){
      U_V(i, j) = out_degree[i] * U_V(i, j);
    }
    for(int j = d; j < 2*d; j++){
      U_V(i, j) = in_degree[i] * U_V(i, j);
    }
  }

}





void parallel_rescale_degree_undirected(
MatrixXd &U_V, 
int* degree,
long long int row_start, long long int row_end, int d
){
  
  for(long long int i = row_start; i < row_end; i++){
    for(int j = 0; j < d; j++){
      U_V(i, j) = degree[i] * U_V(i, j);
    }
    for(int j = d; j < 2*d; j++){
      U_V(i, j) = degree[i] * U_V(i, j);
    }
  }

}



void parallel_rescale_degree_undirected_U_and_V(
MatrixXd &U, 
MatrixXd &V, 
int* degree,
long long int row_start, long long int row_end, int d
){
  
  for(long long int i = row_start; i < row_end; i++){
    for(int j = 0; j < d; j++){
      U(i, j) = degree[i] * U(i, j);
    }
    for(int j = 0; j < d; j++){
      V(i, j) = degree[i] * V(i, j);
    }
  }

}




void parallel_rescale_degree_directed_U_and_V(
MatrixXd &U, 
MatrixXd &V, 
// int* degree,
int* out_degree, int* in_degree,
long long int row_start, long long int row_end, int d
){
  
  for(long long int i = row_start; i < row_end; i++){
    for(int j = 0; j < d; j++){
      U(i, j) = out_degree[i] * U(i, j);
    }
    for(int j = 0; j < d; j++){
      V(i, j) = in_degree[i] * V(i, j);
    }
  }

}




void single_part_parallel_rescale_degree_undirected(
// MatrixXd &U_V, 
MatrixXd &U, 
MatrixXd &V, 
int* degree,
long long int U_row_start, long long int U_row_end,
long long int V_row_start, long long int V_row_end,
int d
){
  
  for(long long int i = U_row_start; i < U_row_end; i++){
    for(int j = 0; j < d; j++){
      U(i, j) = degree[i] * U(i, j);
    }
  }
  
  for(long long int i = V_row_start; i < V_row_start; i++){
    // for(int j = d; j < 2*d; j++){
    for(int j = 0; j < d; j++){
      V(i, j) = degree[i] * V(i, j);
    }
  }

}



void single_part_parallel_rescale_degree_directed(
// MatrixXd &U_V, 
MatrixXd &U, 
MatrixXd &V, 
// int* degree,
int* out_degree, int* in_degree,
long long int U_row_start, long long int U_row_end,
long long int V_row_start, long long int V_row_end,
int d
){
  
  for(long long int i = U_row_start; i < U_row_end; i++){
    for(int j = 0; j < d; j++){
      U(i, j) = out_degree[i] * U(i, j);
    }
  }
  
  for(long long int i = V_row_start; i < V_row_start; i++){
    // for(int j = d; j < 2*d; j++){
    for(int j = 0; j < d; j++){
      V(i, j) = in_degree[i] * V(i, j);
    }
  }

}






void PPR_retrieve(
// int thread_number, 
long long int start, long long int end, 
// long long int& all_count, 
vector<map<long long int, double>>& all_csr_entries,
std::ifstream& ifs,
std::vector<long long int>& offsets,
long long int& read_in_io_all_count
){
    auto start_io_read_time = chrono::system_clock::now();

    boost::archive::binary_iarchive ia(ifs,
        boost::archive::no_header | boost::archive::no_tracking);

    for(int it = start; it < end; it++){

        ifs.seekg(offsets[it]);

        ia >> all_csr_entries[it];
        // std::cout<<"in_map_vec[i].size() = "<<in_map_vec[i].size()<<std::endl;
    
        // all_count += all_csr_entries[it].size();
       
        ifs.clear();
    
    }

    auto end_io_read_time = chrono::system_clock::now();
    auto elapsed_read_time = chrono::duration_cast<std::chrono::seconds>(end_io_read_time - start_io_read_time);
    read_in_io_all_count += elapsed_read_time.count();

}





void PPR_retrieve_less_space(
// int thread_number, 
long long int start, long long int end, 
// long long int& all_count, 
vector<map<long long int, double>>& all_csr_entries,
std::ifstream& ifs,
std::vector<long long int>& offsets,
long long int& read_in_io_all_count,
long long* all_count = NULL,
int** ppr_col = NULL,
double** ppr_nnz = NULL
){
    auto start_io_read_time = chrono::system_clock::now();

    // boost::archive::binary_iarchive ia(ifs,
    //     boost::archive::no_header | boost::archive::no_tracking);
    int ttt = 100;

    long long tot = 0;

    pair<int, double> header_num;

    for(int it = start; it < end; it++){

        ifs.seekg(offsets[it]);

        // ia >> all_csr_entries[it];
        // std::cout<<"in_map_vec[i].size() = "<<in_map_vec[i].size()<<std::endl;
        
        int ssz = 0; //all_csr_entries[it].size();

        ifs.read(reinterpret_cast<char*>(&ssz), sizeof(int));

        tot += ssz;
        
        // all_count += all_csr_entries[it].size();
        if(ppr_nnz != NULL)
        {
          // if(ssz != 0)
          // {
            // ppr_nnz[it] = new pair<int, double>[ssz + 1];
            ppr_col[it] = new int[ssz + 1];
            ppr_nnz[it] = new double[ssz + 1];
            ppr_col[it][0] = ssz;

            if(ssz > 0)
              ifs.read(reinterpret_cast<char*>(&ppr_col[it][1]), sizeof(int) * (long long)ssz);
            
            ifs.read(reinterpret_cast<char*>(ppr_nnz[it]), sizeof(double) * (long long)(ssz+1));
            
            // int nnz_iter = 1;
            // for (auto & key_value: all_csr_entries[it])
            // {
            //   // double value1 = 1 + key_value.second;
            //   // cout<<"value1 = "<<value1<<endl;
            //   ppr_nnz[it][nnz_iter].first = key_value.first;
            //   ppr_nnz[it][nnz_iter].second = key_value.second;
            //   nnz_iter ++;
            // }
          // } else ppr_nnz[it] = NULL;

          all_csr_entries[it].clear();
          --ttt;
          if(ttt == 0)
          {
            malloc_trim(0);
            ttt = 100;
          }
        }

        ifs.clear();
    
    }

    if(all_count != NULL)
    {
      (*all_count) = tot;
    }

    auto end_io_read_time = chrono::system_clock::now();
    auto elapsed_read_time = chrono::duration_cast<std::chrono::seconds>(end_io_read_time - start_io_read_time);
    read_in_io_all_count += elapsed_read_time.count();

}








void PPR_retrieve_New(
// int thread_number, 
long long int start, long long int end, 
// long long int& all_count, 
vector<map<long long int, double>>& all_csr_entries,
std::ifstream& ifs,
std::vector<long long int>& offsets,
long long int& read_in_io_all_count,
long long* all_count = NULL
){
    auto start_io_read_time = chrono::system_clock::now();

    boost::archive::binary_iarchive ia(ifs,
        boost::archive::no_header | boost::archive::no_tracking);

    for(int it = start; it < end; it++){

        ifs.seekg(offsets[it]);

        ia >> all_csr_entries[it];
        // std::cout<<"in_map_vec[i].size() = "<<in_map_vec[i].size()<<std::endl;
    
        // all_count += all_csr_entries[it].size();
    
        ifs.clear();
    
    }

    if(all_count != NULL)
    {
      // *all_count = 0;
      for(int it = start; it < end; it++)
        (*all_count) += all_csr_entries[it].size();
    }

    auto end_io_read_time = chrono::system_clock::now();
    auto elapsed_read_time = chrono::duration_cast<std::chrono::seconds>(end_io_read_time - start_io_read_time);
    read_in_io_all_count += elapsed_read_time.count();

}








void PPR_retrieve_nParts_NUMTHREAD(
// int thread_number, 
// long long int start, long long int end, 
// long long int& all_count, 
// vector<map<long long int, double>>& all_csr_entries,
vector< vector<map<long long int, double>> >& Partition_all_csr_entries,
std::ifstream& ifs,
// long long int& offsets,
vector< long long int >& Partition_offsets,
long long int& read_in_io_all_count
){
    auto start_io_read_time = chrono::system_clock::now();

    boost::archive::binary_iarchive ia(ifs,
        boost::archive::no_header | boost::archive::no_tracking);

    // for(int it = start; it < end; it++){

    //     ifs.seekg(offsets[it]);

    //     ia >> all_csr_entries[it];
    //     // std::cout<<"in_map_vec[i].size() = "<<in_map_vec[i].size()<<std::endl;
    
    //     // all_count += all_csr_entries[it].size();
    
    //     ifs.clear();
    
    // }


    // ifs.seekg(offsets[it]);

    // ia >> all_csr_entries[it];

    // ifs.seekg(offsets);

    // ia >> all_csr_entries;


    // // std::cout<<"in_map_vec[i].size() = "<<in_map_vec[i].size()<<std::endl;

    // // all_count += all_csr_entries[it].size();

    // ifs.clear();

    cout<<"Partition_all_csr_entries.size() = "<<Partition_all_csr_entries.size()<<endl;

    for(int i = 0; i < Partition_all_csr_entries.size(); i++){      
      cout<<"i = "<<i<<endl;
      cout<<"Partition_all_csr_entries["<<i<<"].size() = "<<Partition_all_csr_entries[i].size()<<endl;
      cout<<"Partition_offsets["<<i<<"] = "<<Partition_offsets[i]<<endl;
      ifs.seekg(Partition_offsets[i]);

      ia >> Partition_all_csr_entries[i];

      ifs.clear();
    }

    auto end_io_read_time = chrono::system_clock::now();
    auto elapsed_read_time = chrono::duration_cast<std::chrono::seconds>(end_io_read_time - start_io_read_time);
    read_in_io_all_count += elapsed_read_time.count();

}

















void AnytimeDirectedRowbasedMixedPush(int thread_number, long long int start, long long int end, 
Graph* g, double residuemax, double reservemin, 
long long int& all_count, double alpha,
vector<map<long long int, double>>& all_csr_entries,
std::ofstream& ofs,
std::vector<long long int>& offsets,
long long int& write_out_io_all_count,
int IO_Flag = 1,
int** ppr_col = NULL,
double** ppr_nnz = NULL
){

  // cout<<"Start AnytimeDirectedRowbasedMixedPush!"<<endl;
  // cout<<"start = "<<start<<endl;
  // cout<<"end = "<<end<<endl;

  long long int vertices_number = g->n;

  MKL_INT LEN_MAX = vertices_number + (MKL_INT)5;

  //Create Queue
	Queue forback_Q =
	{
		//.arr = 
		(MKL_INT*)malloc( sizeof(MKL_INT) * (LEN_MAX) ),
		//.capacity = 
		// INT_MAX,
    LEN_MAX,
  	//.front =
		0,
		//.rear = 
		0
	};


  //The queue to record used elements in this node
	Queue record_Q =
	{
		//.arr = 
		(MKL_INT*)malloc(sizeof(MKL_INT) * (LEN_MAX) ),
		//.capacity = 
		// INT_MAX,
		LEN_MAX,
		//.front =
		0,
		//.rear = 
		0
	};

//Forward Initialization
//Backward Initialization
  // float * transposed_forward_pi = new float[vertices_number];
  // float *backward_pi = new float[vertices_number];
  float *pi = new float[vertices_number];
  float *residue = new float[vertices_number];
  bool *flags = new bool[vertices_number];
  int *Record_Q_times = new int[vertices_number]();
  
  // memset(transposed_forward_pi, 0, sizeof(float) * vertices_number);
  // memset(backward_pi, 0, sizeof(float) * vertices_number);
  memset(pi, 0.0, sizeof(float) * vertices_number);
  memset(residue, 0.0, sizeof(float) * vertices_number);
  memset(flags, false, sizeof(bool) * vertices_number);
  memset(Record_Q_times, 0, sizeof(int) * vertices_number);


  // string stored_filename = to_string(thread_order) + ".bin";


  int ttt = 100;
  
  for(int it = start; it < end; it++){

//Forward Computation
    forback_Q.front = 0;
    forback_Q.rear = 0;

    int src = it;


    while(!isEmpty(&record_Q)){
      int current = get_front(&record_Q);
      residue[current] = 0;
      flags[current] = false;

      // transposed_forward_pi[current] = 0;
      pi[current] = 0;
      dequeue(&record_Q);
    }

    record_Q.front = 0;
    record_Q.rear = 0;


    residue[src] = 1;

    flags[src] = true;

    enqueue(&forback_Q, src);
    enqueue(&record_Q, src);
    Record_Q_times[it] = 1;





    while(!isEmpty(&forback_Q)){
      int transposed_forward_v = get_front(&forback_Q);
      
      //是这里的问题，如果一个节点符合条件，但是degree = 0，这就意味着它没有损失(没有alpha * rmax)，这样的话我们要用双倍空间，
      //给每个节点的queue多提供一份

      // // already add self loop, this can be omitted
      // if(g->indegree[transposed_forward_v] == 0){
      //   //注释掉 flags[v] = false; 这样最多推进来一次
      //   transposed_forward_flags[transposed_forward_v] = false;
        
      //   //处理dead end
      //   transposed_forward_pi[transposed_forward_v] = alpha * transposed_forward_residue[transposed_forward_v];
      //   cout<<"Forward indegree[v] == 0"<<endl;
        
      //   transposed_forward_residue[transposed_forward_v] = 0;
      //   transposed_forward_Q.front++;
      //   // dequeue(&Q);
      //   continue;
      // }

      if(residue[transposed_forward_v] / g->indegree[transposed_forward_v] > residuemax){

        for(int j = 0; j < g->indegree[transposed_forward_v]; j++){
          int transposed_forward_u = g->inAdjList[transposed_forward_v][j];

          // // already add self loop, this can be omitted
          // if(g->indegree[transposed_forward_u] == 0){
          //   //处理dead end
          //   cout<<"Forward indegree[forward_u] == 0"<<endl;
          //   transposed_forward_pi[transposed_forward_u] += alpha * transposed_forward_residue[transposed_forward_v];
          //   transposed_forward_residue[src] += (1-alpha) * transposed_forward_residue[transposed_forward_v];
          //   continue;
          // }

          residue[transposed_forward_u] += (1-alpha) * residue[transposed_forward_v] / g->indegree[transposed_forward_v];
          ++Record_Q_times[transposed_forward_u];
          if(Record_Q_times[transposed_forward_u] == 1)
            enqueue(&record_Q, transposed_forward_u);
          
          
          if(residue[transposed_forward_u] / g->indegree[transposed_forward_u] > residuemax && !flags[transposed_forward_u]){
            enqueue(&forback_Q, transposed_forward_u);
            ++Record_Q_times[transposed_forward_u];
            if(Record_Q_times[transposed_forward_u] == 1)
              enqueue(&record_Q, transposed_forward_u);
            flags[transposed_forward_u] = true;
          }

        }
        pi[transposed_forward_v] += alpha * residue[transposed_forward_v];
        residue[transposed_forward_v] = 0;
      }
      
      flags[transposed_forward_v] = false;

      // forback_Q.front++;
      dequeue(&forback_Q);
    }



    // for(int i = 0; i < forback_Q.rear; i++){
    //   int transposed_forward_index = forback_Q.arr[i];
    //   if(pi[transposed_forward_index] != 0 && !flags[transposed_forward_index]){ 
    //     all_csr_entries[it][transposed_forward_index] += pi[transposed_forward_index] / reservemin;
    //     flags[transposed_forward_index] = true;
    //   }
    // }

    // for(int i = 0; i < record_Q.rear; i++){
    //   int transposed_forward_index = record_Q.arr[i];
    //   if(pi[transposed_forward_index] != 0 && !flags[transposed_forward_index]){ 
    //     all_csr_entries[it][transposed_forward_index] += pi[transposed_forward_index] / reservemin;
    //     flags[transposed_forward_index] = true;
    //   }
    // }

    for(int i = 0; i < record_Q.rear; i++){
      int transposed_forward_index = record_Q.arr[i];
      //times number entered into the queue
      if(pi[transposed_forward_index] != 0){ 
        all_csr_entries[it][transposed_forward_index] = pi[transposed_forward_index] / reservemin * Record_Q_times[transposed_forward_index];
        // flags[transposed_forward_index] = true;
      }
      Record_Q_times[transposed_forward_index] = 0;
    }


    // Maybe duplicate nnz, so we need to accumulate duplicate, cannot directly sum the nnz of forward and the nnz of backward.
    // single_source_ppr_nnz_number += transposed_forward_Q.rear - 0;

//End of Forward PPR computation






// we should refresh all the entries, except forward_pi, at the same time use a set to record the indices we used in the forward.


// //Backward Computation
    // backward_Q.front = 0;
    // backward_Q.rear = 0;

//     int backward_src = it;


//     while(!isEmpty(&backward_record_Q)){
//       int current = get_front(&backward_record_Q);
//       backward_residue[current] = 0;
//       backward_flags[current] = false;

//       backward_pi[current] = 0;
//       dequeue(&backward_record_Q);
//     }

//     backward_record_Q.front = 0;
//     backward_record_Q.rear = 0;


//     backward_residue[backward_src] = 1;
//     backward_flags[backward_src] = true;

//     enqueue(&backward_Q, backward_src);

//Backward Computation
    forback_Q.front = 0;
    forback_Q.rear = 0;

    src = it;


    while(!isEmpty(&record_Q)){
      int current = get_front(&record_Q);
      residue[current] = 0;
      flags[current] = false;
      pi[current] = 0;
      dequeue(&record_Q);
    }

    record_Q.front = 0;
    record_Q.rear = 0;


    residue[src] = 1;

    flags[src] = true;

    enqueue(&forback_Q, src);
    enqueue(&record_Q, src);
    Record_Q_times[src] = 1;

    while(!isEmpty(&forback_Q)){
        
      int backward_v = get_front(&forback_Q);

      // already add self loop, this can be omitted
      // if(g->indegree[backward_v] == 0){
      //   flags[backward_v] = false;
      //   backward_Q.front++;
      //   cout<<"Backward Indegree[v] == 0"<<endl;      
      //   // dequeue(&Q);
      //   continue;
      // }


      if(residue[backward_v] > residuemax){
        for(int j = 0; j < g->indegree[backward_v]; j++){

          int backward_u = g->inAdjList[backward_v][j];

          // already add self loop, this can be omitted
          // if(g->outdegree[backward_u] == 0){
          //   cout<<"Backward Outdegree[backward_u] == 0"<<endl;      
          //   continue;
          // }

          residue[backward_u] += (1 - alpha) * residue[backward_v] / g->outdegree[backward_u];

          ++Record_Q_times[backward_u];
          if(Record_Q_times[backward_u] == 1)
            enqueue(&record_Q, backward_u);

          if(residue[backward_u] > residuemax && !flags[backward_u]){

            enqueue(&forback_Q, backward_u);
            ++Record_Q_times[backward_u];
            if(Record_Q_times[backward_u] == 1)
              enqueue(&record_Q, backward_u);

            flags[backward_u] = true;
          }
        }
        pi[backward_v] += alpha * residue[backward_v];
        residue[backward_v] = 0;
      }
      
      flags[backward_v] = false;

      // forback_Q.front++;
      dequeue(&forback_Q);
    }




    // for(int i = 0; i < forback_Q.rear; i++){
    //   int backward_index = forback_Q.arr[i];
    //   if(pi[backward_index] != 0 && !flags[backward_index]){ 
    //     all_csr_entries[it][backward_index] += pi[backward_index] / reservemin;
    //     flags[backward_index] = true;
    //   }
    // }

    // for(int i = 0; i < record_Q.rear; i++){
    //   int backward_index = record_Q.arr[i];
    //   if(pi[backward_index] != 0 && !flags[backward_index]){ 
    //     all_csr_entries[it][backward_index] += pi[backward_index] / reservemin;
    //     flags[backward_index] = true;
    //   }
    // }



    for(int i = 0; i < record_Q.rear; i++){
      int backward_index = record_Q.arr[i];
      //times number entered into the queue
      if(pi[backward_index] != 0){ 
        all_csr_entries[it][backward_index] += pi[backward_index] / reservemin * Record_Q_times[backward_index];
        // flags[backward_index] = true;
      }
      Record_Q_times[backward_index] = 0;
    }




    // Maybe duplicate nnz, so we need to accumulate duplicate, cannot directly sum the nnz of forward and the nnz of backward.
    // single_source_ppr_nnz_number += backward_Q.rear - 0;




    // offsets[it] = ofs.tellp();

    // oa << all_csr_entries[it];

    // all_count += all_csr_entries[it].size();


    if(ppr_nnz != NULL)
    {
      offsets[it] = ofs.tellp();

      int ssz = all_csr_entries[it].size();

      all_count += ssz;
      // #ttag
      // if(ssz != 0)
      // {
        // ppr_nnz[it] = new pair<int, double>[ssz + 1];
        ppr_col[it] = new int[ssz + 1];
        ppr_nnz[it] = new double[ssz + 1];
        ppr_col[it][0] = ssz;

        int nnz_iter = 1;
        for (auto & key_value: all_csr_entries[it])
        {
          // double value1 = 1 + key_value.second;
          // cout<<"value1 = "<<value1<<endl;
          ppr_col[it][nnz_iter] = key_value.first;
          ppr_nnz[it][nnz_iter] = key_value.second;
          nnz_iter ++;
        }
      // } else ppr_nnz[it] = NULL;

      all_csr_entries[it].clear();
      map<long long, double>().swap(all_csr_entries[it]);

      --ttt;
      if(ttt == 0)
      {
        malloc_trim(0);
        ttt = 100;
      }
      // cout << it << endl;
      
      if(IO_Flag == 1) 
      {
        ofs.write(reinterpret_cast<char*>(ppr_col[it]), sizeof(int) * (long long)(ssz+1));//oa << all_csr_entries[it];
        ofs.write(reinterpret_cast<char*>(ppr_nnz[it]), sizeof(double) * (long long)(ssz+1));
      }
        
    }


  }

// //End of Backward PPR computation

  
  // {
  malloc_trim(0);
  auto start_io_write_time = chrono::system_clock::now();


  if(ppr_nnz == NULL)
  {
    boost::archive::binary_oarchive oa(ofs, 
      boost::archive::no_header | boost::archive::no_tracking);

    for(int it = start; it < end; it++){
      offsets[it] = ofs.tellp();

      if(IO_Flag == 1) oa << all_csr_entries[it];

      all_count += all_csr_entries[it].size();
    }
  }

  auto end_io_write_time = chrono::system_clock::now();
  auto elapsed_write_time = chrono::duration_cast<std::chrono::seconds>(end_io_write_time - start_io_write_time);
  write_out_io_all_count = elapsed_write_time.count();

  // }



//Deletion of Forward Allocation
  delete[] residue;
  delete[] pi;
  // delete[] transposed_forward_pi;
  delete[] flags;
  delete[] Record_Q_times;
//Deletion of Backward Allocation
  // delete[] backward_pi;

  // if (NULL != forward_Q.arr)
  // {
  //     free(forward_Q.arr);
  //     forward_Q.arr = NULL;
  // }

  if (NULL != forback_Q.arr)
  {
      free(forback_Q.arr);
      forback_Q.arr = NULL;
  }

  if (NULL != record_Q.arr)
  {
      free(record_Q.arr);
      record_Q.arr = NULL;
  }






  return;

}



































































void AnytimeDirectedRowbasedMixedKatz(int thread_number, long long int start, long long int end, Graph* g, double residuemax, double reservemin, 
long long int& all_count, double alpha,
vector<map<long long int, double>>& all_csr_entries,
std::ofstream& ofs,
std::vector<long long int>& offsets,
long long int& write_out_io_all_count
){

  cout<<"Start AnytimeDirectedRowbasedMixedPush!"<<endl;
  cout<<"start = "<<start<<endl;
  cout<<"end = "<<end<<endl;

  long long int vertices_number = g->n;


  //Create Queue
	Queue forback_Q =
	{
		//.arr = 
		(MKL_INT*)malloc( sizeof(MKL_INT) * (INT_MAX-1) ),
		//.capacity = 
		// INT_MAX,
    INT_MAX - 1,
  	//.front =
		0,
		//.rear = 
		0
	};


  //The queue to record used elements in this node
	Queue record_Q =
	{
		//.arr = 
		(MKL_INT*)malloc(sizeof(MKL_INT) * (INT_MAX-1) ),
		//.capacity = 
		// INT_MAX,
		INT_MAX - 1,
		//.front =
		0,
		//.rear = 
		0
	};

//Forward Initialization
//Backward Initialization
  // float * transposed_forward_pi = new float[vertices_number];
  // float *backward_pi = new float[vertices_number];
  float *pi = new float[vertices_number];
  float *residue = new float[vertices_number];
  bool *flags = new bool[vertices_number];
  
  // memset(transposed_forward_pi, 0, sizeof(float) * vertices_number);
  // memset(backward_pi, 0, sizeof(float) * vertices_number);
  memset(pi, 0.0, sizeof(float) * vertices_number);
  memset(residue, 0.0, sizeof(float) * vertices_number);
  memset(flags, false, sizeof(bool) * vertices_number);


  // string stored_filename = to_string(thread_order) + ".bin";



  
  for(int it = start; it < end; it++){

//Forward Computation
    forback_Q.front = 0;
    forback_Q.rear = 0;

    int src = it;


    while(!isEmpty(&record_Q)){
      int current = get_front(&record_Q);
      residue[current] = 0;
      flags[current] = false;

      // transposed_forward_pi[current] = 0;
      pi[current] = 0;
      dequeue(&record_Q);
    }

    record_Q.front = 0;
    record_Q.rear = 0;


    residue[src] = 1;

    flags[src] = true;

    enqueue(&forback_Q, src);
    enqueue(&record_Q, src);



    int iter_forward = 0;
    int max_iter = 100000000;
    // while(!isEmpty(&forback_Q)){
    while( iter_forward <= max_iter ){
      iter_forward++;
      int transposed_forward_v = get_front(&forback_Q);
      
      //是这里的问题，如果一个节点符合条件，但是degree = 0，这就意味着它没有损失(没有alpha * rmax)，这样的话我们要用双倍空间，
      //给每个节点的queue多提供一份

      // // already add self loop, this can be omitted
      // if(g->indegree[transposed_forward_v] == 0){
      //   //注释掉 flags[v] = false; 这样最多推进来一次
      //   transposed_forward_flags[transposed_forward_v] = false;
        
      //   //处理dead end
      //   transposed_forward_pi[transposed_forward_v] = alpha * transposed_forward_residue[transposed_forward_v];
      //   cout<<"Forward indegree[v] == 0"<<endl;
        
      //   transposed_forward_residue[transposed_forward_v] = 0;
      //   transposed_forward_Q.front++;
      //   // dequeue(&Q);
      //   continue;
      // }

      // if(residue[transposed_forward_v] / g->indegree[transposed_forward_v] > residuemax){
      if(residue[transposed_forward_v] > residuemax){

        for(int j = 0; j < g->indegree[transposed_forward_v]; j++){
          int transposed_forward_u = g->inAdjList[transposed_forward_v][j];

          // // already add self loop, this can be omitted
          // if(g->indegree[transposed_forward_u] == 0){
          //   //处理dead end
          //   cout<<"Forward indegree[forward_u] == 0"<<endl;
          //   transposed_forward_pi[transposed_forward_u] += alpha * transposed_forward_residue[transposed_forward_v];
          //   transposed_forward_residue[src] += (1-alpha) * transposed_forward_residue[transposed_forward_v];
          //   continue;
          // }

          // residue[transposed_forward_u] += (1-alpha) * residue[transposed_forward_v] / g->indegree[transposed_forward_v];
          residue[transposed_forward_u] += (1-alpha) * residue[transposed_forward_v];
          enqueue(&record_Q, transposed_forward_u);
          
          
          // if(residue[transposed_forward_u] / g->indegree[transposed_forward_u] > residuemax && !flags[transposed_forward_u]){
          if(residue[transposed_forward_u] > residuemax && !flags[transposed_forward_u]){
            enqueue(&forback_Q, transposed_forward_u);
            enqueue(&record_Q, transposed_forward_u);
            flags[transposed_forward_u] = true;
          }

        }
        pi[transposed_forward_v] += alpha * residue[transposed_forward_v];
        residue[transposed_forward_v] = 0;
      }
      
      flags[transposed_forward_v] = false;

      forback_Q.front++;
    }



    // for(int i = 0; i < forback_Q.rear; i++){
    //   int transposed_forward_index = forback_Q.arr[i];
    //   if(pi[transposed_forward_index] != 0 && !flags[transposed_forward_index]){ 
    //     all_csr_entries[it][transposed_forward_index] += pi[transposed_forward_index] / reservemin;
    //     flags[transposed_forward_index] = true;
    //   }
    // }

    for(int i = 0; i < record_Q.rear; i++){
      int transposed_forward_index = record_Q.arr[i];
      if(pi[transposed_forward_index] != 0 && !flags[transposed_forward_index]){ 
        all_csr_entries[it][transposed_forward_index] += pi[transposed_forward_index] / reservemin;
        flags[transposed_forward_index] = true;
      }
    }

    // for(int i = 0; i < record_Q.rear; i++){
    //   int transposed_forward_index = record_Q.arr[i];
    //   //times number entered into the queue
    //   if(pi[transposed_forward_index] != 0){ 
    //     all_csr_entries[it][transposed_forward_index] += pi[transposed_forward_index] / reservemin;
    //     // flags[transposed_forward_index] = true;
    //   }
    // }


    // Maybe duplicate nnz, so we need to accumulate duplicate, cannot directly sum the nnz of forward and the nnz of backward.
    // single_source_ppr_nnz_number += transposed_forward_Q.rear - 0;

//End of Forward PPR computation






// we should refresh all the entries, except forward_pi, at the same time use a set to record the indices we used in the forward.


// //Backward Computation
    // backward_Q.front = 0;
    // backward_Q.rear = 0;

//     int backward_src = it;


//     while(!isEmpty(&backward_record_Q)){
//       int current = get_front(&backward_record_Q);
//       backward_residue[current] = 0;
//       backward_flags[current] = false;

//       backward_pi[current] = 0;
//       dequeue(&backward_record_Q);
//     }

//     backward_record_Q.front = 0;
//     backward_record_Q.rear = 0;


//     backward_residue[backward_src] = 1;
//     backward_flags[backward_src] = true;

//     enqueue(&backward_Q, backward_src);

//Backward Computation
    forback_Q.front = 0;
    forback_Q.rear = 0;

    src = it;


    while(!isEmpty(&record_Q)){
      int current = get_front(&record_Q);
      residue[current] = 0;
      flags[current] = false;
      pi[current] = 0;
      dequeue(&record_Q);
    }

    record_Q.front = 0;
    record_Q.rear = 0;


    residue[src] = 1;

    flags[src] = true;

    enqueue(&forback_Q, src);
    enqueue(&record_Q, src);


    int iter_backward = 0;
    // while(!isEmpty(&forback_Q)){
    while( iter_backward <= max_iter ){
      iter_backward++;
      
    int backward_v = get_front(&forback_Q);

      // already add self loop, this can be omitted
      // if(g->indegree[backward_v] == 0){
      //   flags[backward_v] = false;
      //   backward_Q.front++;
      //   cout<<"Backward Indegree[v] == 0"<<endl;      
      //   // dequeue(&Q);
      //   continue;
      // }


      if(residue[backward_v] > residuemax){
        for(int j = 0; j < g->indegree[backward_v]; j++){

          int backward_u = g->inAdjList[backward_v][j];

          // already add self loop, this can be omitted
          // if(g->outdegree[backward_u] == 0){
          //   cout<<"Backward Outdegree[backward_u] == 0"<<endl;      
          //   continue;
          // }

          // residue[backward_u] += (1-alpha) * residue[backward_v] / g->outdegree[backward_u];
          residue[backward_u] += (1-alpha) * residue[backward_v];

          enqueue(&record_Q, backward_u);

          if(residue[backward_u] > residuemax && !flags[backward_u]){

            enqueue(&forback_Q, backward_u);
            enqueue(&record_Q, backward_u);

            flags[backward_u] = true;
          }
        }
        pi[backward_v] += alpha * residue[backward_v];
        residue[backward_v] = 0;
      }
      
      flags[backward_v] = false;

      forback_Q.front++;
    }




    // for(int i = 0; i < forback_Q.rear; i++){
    //   int backward_index = forback_Q.arr[i];
    //   if(pi[backward_index] != 0 && !flags[backward_index]){ 
    //     all_csr_entries[it][backward_index] += pi[backward_index] / reservemin;
    //     flags[backward_index] = true;
    //   }
    // }

    for(int i = 0; i < record_Q.rear; i++){
      int backward_index = record_Q.arr[i];
      if(pi[backward_index] != 0 && !flags[backward_index]){ 
        all_csr_entries[it][backward_index] += pi[backward_index] / reservemin;
        flags[backward_index] = true;
      }
    }



    // for(int i = 0; i < record_Q.rear; i++){
    //   int backward_index = record_Q.arr[i];
    //   //times number entered into the queue
    //   if(pi[backward_index] != 0){ 
    //     all_csr_entries[it][backward_index] += pi[backward_index] / reservemin;
    //     // flags[backward_index] = true;
    //   }
    // }




    // Maybe duplicate nnz, so we need to accumulate duplicate, cannot directly sum the nnz of forward and the nnz of backward.
    // single_source_ppr_nnz_number += backward_Q.rear - 0;




    // offsets[it] = ofs.tellp();

    // oa << all_csr_entries[it];

    // all_count += all_csr_entries[it].size();


  }

// //End of Backward PPR computation



  auto start_io_write_time = chrono::system_clock::now();

  boost::archive::binary_oarchive oa(ofs, 
    boost::archive::no_header | boost::archive::no_tracking);

  for(int it = start; it < end; it++){
    offsets[it] = ofs.tellp();

    oa << all_csr_entries[it];

    all_count += all_csr_entries[it].size();
  }


  auto end_io_write_time = chrono::system_clock::now();
  auto elapsed_write_time = chrono::duration_cast<std::chrono::seconds>(end_io_write_time - start_io_write_time);
  write_out_io_all_count = elapsed_write_time.count();





//Deletion of Forward Allocation
  delete[] residue;
  delete[] pi;
  // delete[] transposed_forward_pi;
  delete[] flags;
//Deletion of Backward Allocation
  // delete[] backward_pi;

  // if (NULL != forward_Q.arr)
  // {
  //     free(forward_Q.arr);
  //     forward_Q.arr = NULL;
  // }

  if (NULL != forback_Q.arr)
  {
      free(forback_Q.arr);
      forback_Q.arr = NULL;
  }

  if (NULL != record_Q.arr)
  {
      free(record_Q.arr);
      record_Q.arr = NULL;
  }






  return;

}






























void AnytimeUndirectedRowbasedMixedPush(int thread_number, long long int start, long long int end, 
UGraph* g, double residuemax, double reservemin, 
long long int& all_count, double alpha,
vector<map<long long int, double>>& all_csr_entries,
std::ofstream& ofs,
std::vector<long long int>& offsets,
long long int& write_out_io_all_count,
int IO_Flag = 1,
int** ppr_col = NULL,
double** ppr_nnz = NULL
){


  // cout<<"Start AnytimeUndirectedRowbasedMixedPush!"<<endl;
  // cout<<"start = "<<start<<endl;
  // cout<<"end = "<<end<<endl;

  long long int vertices_number = g->n;

  MKL_INT LEN_MAX = vertices_number + (MKL_INT)5;

  //Create Queue
	Queue forback_Q =
	{
		//.arr = 
		(MKL_INT*)malloc( sizeof(MKL_INT) * (LEN_MAX) ),
		//.capacity = 
		// INT_MAX,
    LEN_MAX,
  	//.front =
		0,
		//.rear = 
		0
	};


  //The queue to record used elements in this node
	Queue record_Q =
	{
		//.arr = 
		(MKL_INT*)malloc(sizeof(MKL_INT) * (LEN_MAX) ),
		//.capacity = 
		// INT_MAX,
		LEN_MAX,
		//.front =
		0,
		//.rear = 
		0
	};

//Forward Initialization
//Backward Initialization

  float *pi = new float[vertices_number];
  float *residue = new float[vertices_number];
  bool *flags = new bool[vertices_number];
  int *Record_Q_times = new int[vertices_number]();
  
  // memset(transposed_forward_pi, 0, sizeof(float) * vertices_number);
  // memset(backward_pi, 0, sizeof(float) * vertices_number);
  memset(pi, 0.0, sizeof(float) * vertices_number);
  memset(residue, 0.0, sizeof(float) * vertices_number);
  memset(flags, false, sizeof(bool) * vertices_number);
  memset(Record_Q_times, 0, sizeof(int) * vertices_number);





  int ttt = 100;

  for(int it = start; it < end; it++){

//Forward Computation
    forback_Q.front = 0;
    forback_Q.rear = 0;

    int src = it;


    while(!isEmpty(&record_Q)){
      int current = get_front(&record_Q);
      residue[current] = 0;
      flags[current] = false;

      // transposed_forward_pi[current] = 0;
      pi[current] = 0;
      dequeue(&record_Q);
    }

    record_Q.front = 0;
    record_Q.rear = 0;


    residue[src] = 1;

    flags[src] = true;

    enqueue(&forback_Q, src);
    enqueue(&record_Q, src);
    Record_Q_times[src] = 1;


    while(!isEmpty(&forback_Q)){
      int transposed_forward_v = get_front(&forback_Q);
      
      //是这里的问题，如果一个节点符合条件，但是degree = 0，这就意味着它没有损失(没有alpha * rmax)，这样的话我们要用双倍空间，
      //给每个节点的queue多提供一份

      // // already add self loop, this can be omitted
      // if(g->indegree[transposed_forward_v] == 0){
      //   //注释掉 flags[v] = false; 这样最多推进来一次
      //   transposed_forward_flags[transposed_forward_v] = false;
        
      //   //处理dead end
      //   transposed_forward_pi[transposed_forward_v] = alpha * transposed_forward_residue[transposed_forward_v];
      //   cout<<"Forward indegree[v] == 0"<<endl;
        
      //   transposed_forward_residue[transposed_forward_v] = 0;
      //   transposed_forward_Q.front++;
      //   // dequeue(&Q);
      //   continue;
      // }

      if(residue[transposed_forward_v] / g->degree[transposed_forward_v] > residuemax){

        for(int j = 0; j < g->degree[transposed_forward_v]; j++){
          int transposed_forward_u = g->AdjList[transposed_forward_v][j];

          // // already add self loop, this can be omitted
          // if(g->indegree[transposed_forward_u] == 0){
          //   //处理dead end
          //   cout<<"Forward indegree[forward_u] == 0"<<endl;
          //   transposed_forward_pi[transposed_forward_u] += alpha * transposed_forward_residue[transposed_forward_v];
          //   transposed_forward_residue[src] += (1-alpha) * transposed_forward_residue[transposed_forward_v];
          //   continue;
          // }

          residue[transposed_forward_u] += (1-alpha) * residue[transposed_forward_v] / g->degree[transposed_forward_v];
          ++Record_Q_times[transposed_forward_u];
          if(Record_Q_times[transposed_forward_u] == 1)
            enqueue(&record_Q, transposed_forward_u);
          
          
          if(residue[transposed_forward_u] / g->degree[transposed_forward_u] > residuemax && !flags[transposed_forward_u]){
            enqueue(&forback_Q, transposed_forward_u);
            ++Record_Q_times[transposed_forward_u];
            if(Record_Q_times[transposed_forward_u] == 1)
              enqueue(&record_Q, transposed_forward_u);
            flags[transposed_forward_u] = true;
          }

        }
        pi[transposed_forward_v] += alpha * residue[transposed_forward_v];
        residue[transposed_forward_v] = 0;
      }
      
      flags[transposed_forward_v] = false;

      // forback_Q.front++;
      dequeue(&forback_Q);
    }



    // for(int i = 0; i < forback_Q.rear; i++){
    //   int transposed_forward_index = forback_Q.arr[i];
    //   if(pi[transposed_forward_index] != 0 && !flags[transposed_forward_index]){ 
    //     // forward_single_source_all_ppr_triplet->push_back(Triplet<double>(transposed_forward_index, src, transposed_forward_pi[transposed_forward_index]));
    //     all_csr_entries[it][transposed_forward_index] += pi[transposed_forward_index] / reservemin;
    //     flags[transposed_forward_index] = true;
    //   }
    // }

    // for(int i = 0; i < record_Q.rear; i++){
    //   int transposed_forward_index = record_Q.arr[i];
    //   if(pi[transposed_forward_index] != 0 && !flags[transposed_forward_index]){ 
    //     // forward_single_source_all_ppr_triplet->push_back(Triplet<double>(transposed_forward_index, src, transposed_forward_pi[transposed_forward_index]));
    //     all_csr_entries[it][transposed_forward_index] += pi[transposed_forward_index] / reservemin;
    //     flags[transposed_forward_index] = true;
    //   }
    // }

    for(int i = 0; i < record_Q.rear; i++){
      int transposed_forward_index = record_Q.arr[i];
      if(pi[transposed_forward_index] != 0){ 
        all_csr_entries[it][transposed_forward_index] += pi[transposed_forward_index] / reservemin * Record_Q_times[transposed_forward_index];
      }
      Record_Q_times[transposed_forward_index] = 0;
    }


    // Maybe duplicate nnz, so we need to accumulate duplicate, cannot directly sum the nnz of forward and the nnz of backward.
    // single_source_ppr_nnz_number += transposed_forward_Q.rear - 0;

//End of Forward PPR computation






// we should refresh all the entries, except forward_pi, at the same time use a set to record the indices we used in the forward.


// //Backward Computation
    // backward_Q.front = 0;
    // backward_Q.rear = 0;

//     int backward_src = it;


//     while(!isEmpty(&backward_record_Q)){
//       int current = get_front(&backward_record_Q);
//       backward_residue[current] = 0;
//       backward_flags[current] = false;

//       backward_pi[current] = 0;
//       dequeue(&backward_record_Q);
//     }

//     backward_record_Q.front = 0;
//     backward_record_Q.rear = 0;


//     backward_residue[backward_src] = 1;
//     backward_flags[backward_src] = true;

//     enqueue(&backward_Q, backward_src);

//Backward Computation
    forback_Q.front = 0;
    forback_Q.rear = 0;

    src = it;


    while(!isEmpty(&record_Q)){
      int current = get_front(&record_Q);
      residue[current] = 0;
      flags[current] = false;
      pi[current] = 0;
      dequeue(&record_Q);
    }

    record_Q.front = 0;
    record_Q.rear = 0;


    residue[src] = 1;

    flags[src] = true;

    enqueue(&forback_Q, src);
    enqueue(&record_Q, src);
    Record_Q_times[src] = 1;



    while(!isEmpty(&forback_Q)){
      
    int backward_v = get_front(&forback_Q);

      // already add self loop, this can be omitted
      // if(g->indegree[backward_v] == 0){
      //   flags[backward_v] = false;
      //   backward_Q.front++;
      //   cout<<"Backward Indegree[v] == 0"<<endl;      
      //   // dequeue(&Q);
      //   continue;
      // }


      if(residue[backward_v] > residuemax){
        for(int j = 0; j < g->degree[backward_v]; j++){

          int backward_u = g->AdjList[backward_v][j];

          // already add self loop, this can be omitted
          // if(g->outdegree[backward_u] == 0){
          //   cout<<"Backward Outdegree[backward_u] == 0"<<endl;      
          //   continue;
          // }

          residue[backward_u] += (1-alpha) * residue[backward_v] / g->degree[backward_u];

          ++Record_Q_times[backward_u];
          if(Record_Q_times[backward_u] == 1)
            enqueue(&record_Q, backward_u);

          if(residue[backward_u] > residuemax && !flags[backward_u]){
            enqueue(&forback_Q, backward_u);
            ++Record_Q_times[backward_u];
            if(Record_Q_times[backward_u] == 1)
              enqueue(&record_Q, backward_u);
            flags[backward_u] = true;
          }
        }
        pi[backward_v] += alpha * residue[backward_v];
        residue[backward_v] = 0;
      }
      
      flags[backward_v] = false;

      dequeue(&forback_Q);
    }




    // for(int i = 0; i < forback_Q.rear; i++){
    //   int backward_index = forback_Q.arr[i];
    //   if(pi[backward_index] != 0 && !flags[backward_index]){ 
    //     // backward_single_source_all_ppr_triplet->push_back(Triplet<double>(backward_index, src, backward_pi[backward_index]));
    //     all_csr_entries[it][backward_index] += pi[backward_index] / reservemin;
    //     flags[backward_index] = true;
    //   }
    // }

    // for(int i = 0; i < record_Q.rear; i++){
    //   int backward_index = record_Q.arr[i];
    //   if(pi[backward_index] != 0 && !flags[backward_index]){ 
    //     // backward_single_source_all_ppr_triplet->push_back(Triplet<double>(backward_index, src, backward_pi[backward_index]));
    //     all_csr_entries[it][backward_index] += pi[backward_index] / reservemin;
    //     flags[backward_index] = true;
    //   }
    // }

    for(int i = 0; i < record_Q.rear; i++){
      int backward_index = record_Q.arr[i];
      if(pi[backward_index] != 0){ 
        all_csr_entries[it][backward_index] += pi[backward_index] / reservemin * Record_Q_times[backward_index];       
      }
      Record_Q_times[backward_index] = 0;
    }

    // Maybe duplicate nnz, so we need to accumulate duplicate, cannot directly sum the nnz of forward and the nnz of backward.
    // single_source_ppr_nnz_number += backward_Q.rear - 0;


   if(ppr_nnz != NULL)
    {
      offsets[it] = ofs.tellp();

      int ssz = all_csr_entries[it].size();

      all_count += ssz;
      // #ttag
      // if(ssz != 0)
      // {
        // ppr_nnz[it] = new pair<int, double>[ssz + 1];
        ppr_nnz[it] = new double[ssz + 1];
        ppr_col[it] = new int[ssz + 1];
        ppr_col[it][0] = ssz;
        int nnz_iter = 1;

        for(auto & key_value: all_csr_entries[it])
        {
          // double value1 = 1 + key_value.second;
          // cout<<"value1 = "<<value1<<endl;
          ppr_col[it][nnz_iter] = key_value.first;
          ppr_nnz[it][nnz_iter] = key_value.second;
          nnz_iter ++;
        }
      // } else ppr_nnz[it] = NULL;

      // cout << it << " " << ssz << endl;
      if(IO_Flag == 1) 
      {
        ofs.write(reinterpret_cast<char*>(ppr_col[it]), sizeof(int) * (long long) (ssz+1));
        ofs.write(reinterpret_cast<char*>(ppr_nnz[it]), sizeof(double) * (long long) (ssz+1));
      }
       //oa << all_csr_entries[it];

      all_csr_entries[it].clear();
      map<long long, double>().swap(all_csr_entries[it]);

      --ttt;
      if(ttt == 0)
      {
        malloc_trim(0);
        ttt = 100;
      }
    }

  }

// //End of Backward PPR computation

  // {
    malloc_trim(0);

  auto start_io_write_time = chrono::system_clock::now();

  if(ppr_nnz == NULL)
  {
    boost::archive::binary_oarchive oa(ofs, 
      boost::archive::no_header | boost::archive::no_tracking);

    for(int it = start; it < end; it++){

      offsets[it] = ofs.tellp();

      if(IO_Flag == 1) oa << all_csr_entries[it];

      all_count += all_csr_entries[it].size();

    }
  }

  auto end_io_write_time = chrono::system_clock::now();
  auto elapsed_write_time = chrono::duration_cast<std::chrono::seconds>(end_io_write_time - start_io_write_time);
  write_out_io_all_count = elapsed_write_time.count();

  // }

//Deletion of Forward Allocation
  delete[] residue;
  delete[] pi;
  // delete[] transposed_forward_pi;
  delete[] flags;
  delete[] Record_Q_times;
//Deletion of Backward Allocation
  // delete[] backward_pi;

  // if (NULL != forward_Q.arr)
  // {
  //     free(forward_Q.arr);
  //     forward_Q.arr = NULL;
  // }

  if (NULL != forback_Q.arr)
  {
      free(forback_Q.arr);
      forback_Q.arr = NULL;
  }

  if (NULL != record_Q.arr)
  {
      free(record_Q.arr);
      record_Q.arr = NULL;
  }






  return;

}






































void AnytimeUndirectedRowbasedMixedPush_PrepareDynamic(int thread_number, long long int start, long long int end, UGraph* g, double residuemax, double reservemin, 
long long int& all_count, double alpha,

// vector<std::map<long long int, double>> & thread_all_csr_entries,

// vector<unordered_map<long long int, double>> & thread_forward_pi_forward_residue_backward_pi_backward_residue_entries,
vector< vector<std::map<long long int, double>> > & Partition_thread_all_csr_entries,
// vector< vector<unordered_map<long long int, double>> > & Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries,

vector< vector< third_float_map > > & Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries,


std::ofstream& ofs_all_csr,
std::ofstream& ofs_dynamic_ppr,


// long long int & all_csr_offsets,

// long long int & dynamic_ppr_offsets,
vector< long long int > & Partition_all_csr_offsets,

vector< long long int > & Partition_dynamic_ppr_offsets,

long long int& write_out_io_all_count,

int stored_interval,

std::ofstream& ofs_debug
){


  cout<<"Start AnytimeUndirectedRowbasedMixedPush!"<<endl;
  cout<<"start = "<<start<<endl;
  cout<<"end = "<<end<<endl;

  long long int vertices_number = g->n;


  //Create Queue
	Queue forback_Q =
	{
		//.arr = 
		(MKL_INT*)malloc( sizeof(MKL_INT) * (INT_MAX-1) ),
		//.capacity = 
		// INT_MAX,
    INT_MAX - 1,
  	//.front =
		0,
		//.rear = 
		0
	};


  //The queue to record used elements in this node
	Queue record_Q =
	{
		//.arr = 
		(MKL_INT*)malloc(sizeof(MKL_INT) * (INT_MAX-1) ),
		//.capacity = 
		// INT_MAX,
		INT_MAX - 1,
		//.front =
		0,
		//.rear = 
		0
	};

//Forward Initialization
//Backward Initialization

  float *pi = new float[vertices_number];
  float *residue = new float[vertices_number];
  bool *flags = new bool[vertices_number];
  
  // memset(transposed_forward_pi, 0, sizeof(float) * vertices_number);
  // memset(backward_pi, 0, sizeof(float) * vertices_number);
  memset(pi, 0.0, sizeof(float) * vertices_number);
  memset(residue, 0.0, sizeof(float) * vertices_number);
  memset(flags, false, sizeof(bool) * vertices_number);


  boost::archive::binary_oarchive oa_all_csr(ofs_all_csr, 
    boost::archive::no_header | boost::archive::no_tracking);
    

  boost::archive::binary_oarchive oa_dynamic_ppr(ofs_dynamic_ppr, 
    boost::archive::no_header | boost::archive::no_tracking);


  // ofs_debug<<"start = "<<start<<endl;
  // ofs_debug<<"end = "<<end<<endl;

  // cout<<"start = "<<start<<endl;
  // cout<<"end = "<<end<<endl;


  int reserve_size = 1 / residuemax / alpha;


  int interval_index = 0;

  int inner_index_all_csr = 0;

  int inner_index_dynamic_ppr = 0;


  for(int it = start; it < end; it++){

    // int interval_index = (it - start) / stored_interval;

    // if(interval_index == Partition_thread_all_csr_entries.size()){
    //   interval_index--;
    // }

    // int inner_index = it - start - interval_index * stored_interval;


    // ofs_debug<<"it = "<<it<<endl;
    // ofs_debug<<"interval_index = "<<interval_index<<endl;
    // ofs_debug<<"inner_index = "<<inner_index<<endl;
    // ofs_debug<<"Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[interval_index].size() = "<<Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[interval_index].size()<<endl;
    // ofs_debug<<"Partition_thread_all_csr_entries[interval_index].size() = "<<Partition_thread_all_csr_entries[interval_index].size()<<endl;
    // ofs_debug<<"4 * inner_index + 3 = "<<4 * inner_index + 3<<endl;


    // cout<<"it = "<<it<<endl;
    // cout<<"interval_index = "<<interval_index<<endl;
    // cout<<"inner_index_dynamic_ppr = "<<inner_index_dynamic_ppr<<endl;
    // cout<<"Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[interval_index].size() = "<<Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[interval_index].size()<<endl;
    // cout<<"Partition_thread_all_csr_entries[interval_index].size() = "<<Partition_thread_all_csr_entries[interval_index].size()<<endl;
    // cout<<"4 * inner_index_dynamic_ppr + 3 = "<<4 * inner_index_dynamic_ppr + 3<<endl;


    // unordered_map<long long int, double>& forward_pi_entries = Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[interval_index][4 * inner_index + 0];
    // unordered_map<long long int, double>& forward_residue_entries = Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[interval_index][4 * inner_index + 1];    
    // unordered_map<long long int, double>& backward_pi_entries = Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[interval_index][4 * inner_index + 2];    
    // unordered_map<long long int, double>& backward_residue_entries = Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[interval_index][4 * inner_index + 3];

    // cout<<"Static Before"<<endl;
    third_float_map& forward_pi_entries = Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[interval_index][inner_index_dynamic_ppr++];
    third_float_map& forward_residue_entries = Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[interval_index][inner_index_dynamic_ppr++];    
    third_float_map& backward_pi_entries = Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[interval_index][inner_index_dynamic_ppr++];    
    third_float_map& backward_residue_entries = Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[interval_index][inner_index_dynamic_ppr++];

    forward_pi_entries.reserve(reserve_size);
    forward_residue_entries.reserve(reserve_size);
    backward_pi_entries.reserve(reserve_size);
    backward_residue_entries.reserve(reserve_size);

    map<long long int, double>& all_csr_entries = Partition_thread_all_csr_entries[interval_index][inner_index_all_csr++];


    // cout<<"Static After"<<endl;

    // unordered_map<long long int, double>& forward_pi_entries = thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[4 * (it-start) + 0];
    // unordered_map<long long int, double>& forward_residue_entries = thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[4 * (it-start) + 1];    
    // unordered_map<long long int, double>& backward_pi_entries = thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[4 * (it-start) + 2];    
    // unordered_map<long long int, double>& backward_residue_entries = thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[4 * (it-start) + 3];
    
//Forward Computation
    forback_Q.front = 0;
    forback_Q.rear = 0;

    int src = it;

    // cout<<0<<endl;
    // ofs_debug<<0<<endl;

    while(!isEmpty(&record_Q)){
      int current = get_front(&record_Q);
      residue[current] = 0;
      flags[current] = false;

      // transposed_forward_pi[current] = 0;
      pi[current] = 0;
      dequeue(&record_Q);
    }

    record_Q.front = 0;
    record_Q.rear = 0;

    // cout<<1<<endl;
    // ofs_debug<<1<<endl;

    residue[src] = 1;

    flags[src] = true;

    enqueue(&forback_Q, src);
    enqueue(&record_Q, src);


    // cout<<2<<endl;
    // ofs_debug<<2<<endl;

    while(!isEmpty(&forback_Q)){
      int transposed_forward_v = get_front(&forback_Q);
      
      //是这里的问题，如果一个节点符合条件，但是degree = 0，这就意味着它没有损失(没有alpha * rmax)，这样的话我们要用双倍空间，
      //给每个节点的queue多提供一份

      // // already add self loop, this can be omitted
      // if(g->indegree[transposed_forward_v] == 0){
      //   //注释掉 flags[v] = false; 这样最多推进来一次
      //   transposed_forward_flags[transposed_forward_v] = false;
        
      //   //处理dead end
      //   transposed_forward_pi[transposed_forward_v] = alpha * transposed_forward_residue[transposed_forward_v];
      //   cout<<"Forward indegree[v] == 0"<<endl;
        
      //   transposed_forward_residue[transposed_forward_v] = 0;
      //   transposed_forward_Q.front++;
      //   // dequeue(&Q);
      //   continue;
      // }

      if(residue[transposed_forward_v] / g->degree[transposed_forward_v] > residuemax){

        for(int j = 0; j < g->degree[transposed_forward_v]; j++){
          int transposed_forward_u = g->AdjList[transposed_forward_v][j];

          // // already add self loop, this can be omitted
          // if(g->indegree[transposed_forward_u] == 0){
          //   //处理dead end
          //   cout<<"Forward indegree[forward_u] == 0"<<endl;
          //   transposed_forward_pi[transposed_forward_u] += alpha * transposed_forward_residue[transposed_forward_v];
          //   transposed_forward_residue[src] += (1-alpha) * transposed_forward_residue[transposed_forward_v];
          //   continue;
          // }

          residue[transposed_forward_u] += (1-alpha) * residue[transposed_forward_v] / g->degree[transposed_forward_v];
          enqueue(&record_Q, transposed_forward_u);
          
          
          if(residue[transposed_forward_u] / g->degree[transposed_forward_u] > residuemax && !flags[transposed_forward_u]){
            enqueue(&forback_Q, transposed_forward_u);
            enqueue(&record_Q, transposed_forward_u);
            flags[transposed_forward_u] = true;
          }

        }
        pi[transposed_forward_v] += alpha * residue[transposed_forward_v];
        residue[transposed_forward_v] = 0;
      }
      
      flags[transposed_forward_v] = false;

      forback_Q.front++;
    }


    // cout<<3<<endl;
    // ofs_debug<<3<<endl;



    // for(int i = 0; i < forback_Q.rear; i++){
    //   int transposed_forward_index = forback_Q.arr[i];
    //   if(pi[transposed_forward_index] != 0 && !flags[transposed_forward_index]){ 
    //     // forward_single_source_all_ppr_triplet->push_back(Triplet<double>(transposed_forward_index, src, transposed_forward_pi[transposed_forward_index]));
    //     all_csr_entries[it][transposed_forward_index] += pi[transposed_forward_index] / reservemin;
    //     flags[transposed_forward_index] = true;
    //   }
    // }

    // for(int i = 0; i < record_Q.rear; i++){
    //   int transposed_forward_index = record_Q.arr[i];
    //   if(pi[transposed_forward_index] != 0 && !flags[transposed_forward_index]){ 
    //     // forward_single_source_all_ppr_triplet->push_back(Triplet<double>(transposed_forward_index, src, transposed_forward_pi[transposed_forward_index]));
    //     all_csr_entries[it][transposed_forward_index] += pi[transposed_forward_index] / reservemin;
    //     flags[transposed_forward_index] = true;
    //   }
    // }

    // for(int i = 0; i < record_Q.rear; i++){
    //   int transposed_forward_index = record_Q.arr[i];
    //   if(pi[transposed_forward_index] != 0){
    //     all_csr_entries[it][transposed_forward_index] += pi[transposed_forward_index] / reservemin;
    //   }
    // }

    for(int i = 0; i < record_Q.rear; i++){
      int transposed_forward_index = record_Q.arr[i];
      // if(pi[transposed_forward_index] != 0){
      //   all_csr_entries[it][transposed_forward_index] += pi[transposed_forward_index] / reservemin;
      // }
      
      // cout<<"3:1"<<endl;
      // ofs_debug<<"3:1"<<endl;

      if(forward_residue_entries.find(transposed_forward_index) == forward_residue_entries.end()){
        forward_residue_entries[transposed_forward_index] = residue[transposed_forward_index];
      }

      // cout<<"3:2"<<endl;
      // ofs_debug<<"3:2"<<endl;

      if(pi[transposed_forward_index] != 0){
        // thread_all_csr_entries[it - start][transposed_forward_index] += pi[transposed_forward_index] / reservemin;
        all_csr_entries[transposed_forward_index] += pi[transposed_forward_index] / reservemin;
      }
      else{
        continue;
      }

      // cout<<"3:3"<<endl;
      // ofs_debug<<"3:3"<<endl;

      if(forward_pi_entries.find(transposed_forward_index) == forward_pi_entries.end()){
        forward_pi_entries[transposed_forward_index] = pi[transposed_forward_index];
      }

    }

    // cout<<4<<endl;
    // ofs_debug<<4<<endl;

    // Maybe duplicate nnz, so we need to accumulate duplicate, cannot directly sum the nnz of forward and the nnz of backward.
    // single_source_ppr_nnz_number += transposed_forward_Q.rear - 0;

//End of Forward PPR computation






// we should refresh all the entries, except forward_pi, at the same time use a set to record the indices we used in the forward.


// //Backward Computation
    // backward_Q.front = 0;
    // backward_Q.rear = 0;

//     int backward_src = it;


//     while(!isEmpty(&backward_record_Q)){
//       int current = get_front(&backward_record_Q);
//       backward_residue[current] = 0;
//       backward_flags[current] = false;

//       backward_pi[current] = 0;
//       dequeue(&backward_record_Q);
//     }

//     backward_record_Q.front = 0;
//     backward_record_Q.rear = 0;


//     backward_residue[backward_src] = 1;
//     backward_flags[backward_src] = true;

//     enqueue(&backward_Q, backward_src);

//Backward Computation
    forback_Q.front = 0;
    forback_Q.rear = 0;

    src = it;


    while(!isEmpty(&record_Q)){
      int current = get_front(&record_Q);
      residue[current] = 0;
      flags[current] = false;
      pi[current] = 0;
      dequeue(&record_Q);
    }

    // cout<<5<<endl;
    // ofs_debug<<5<<endl;

    record_Q.front = 0;
    record_Q.rear = 0;


    residue[src] = 1;

    flags[src] = true;

    enqueue(&forback_Q, src);
    enqueue(&record_Q, src);


    // cout<<6<<endl;
    // ofs_debug<<6<<endl;


    while(!isEmpty(&forback_Q)){
      
    int backward_v = get_front(&forback_Q);

      // already add self loop, this can be omitted
      // if(g->indegree[backward_v] == 0){
      //   flags[backward_v] = false;
      //   backward_Q.front++;
      //   cout<<"Backward Indegree[v] == 0"<<endl;      
      //   // dequeue(&Q);
      //   continue;
      // }


      if(residue[backward_v] > residuemax){
        for(int j = 0; j < g->degree[backward_v]; j++){

          int backward_u = g->AdjList[backward_v][j];

          // already add self loop, this can be omitted
          // if(g->outdegree[backward_u] == 0){
          //   cout<<"Backward Outdegree[backward_u] == 0"<<endl;      
          //   continue;
          // }

          residue[backward_u] += (1-alpha) * residue[backward_v] / g->degree[backward_u];

          enqueue(&record_Q, backward_u);

          if(residue[backward_u] > residuemax && !flags[backward_u]){
            enqueue(&forback_Q, backward_u);
            enqueue(&record_Q, backward_u);
            flags[backward_u] = true;
          }
        }
        pi[backward_v] += alpha * residue[backward_v];
        residue[backward_v] = 0;
      }
      
      flags[backward_v] = false;

      forback_Q.front++;
    }



    // cout<<7<<endl;
    // ofs_debug<<7<<endl;



    // for(int i = 0; i < forback_Q.rear; i++){
    //   int backward_index = forback_Q.arr[i];
    //   if(pi[backward_index] != 0 && !flags[backward_index]){ 
    //     // backward_single_source_all_ppr_triplet->push_back(Triplet<double>(backward_index, src, backward_pi[backward_index]));
    //     all_csr_entries[it][backward_index] += pi[backward_index] / reservemin;
    //     flags[backward_index] = true;
    //   }
    // }

    // for(int i = 0; i < record_Q.rear; i++){
    //   int backward_index = record_Q.arr[i];
    //   if(pi[backward_index] != 0 && !flags[backward_index]){ 
    //     // backward_single_source_all_ppr_triplet->push_back(Triplet<double>(backward_index, src, backward_pi[backward_index]));
    //     all_csr_entries[it][backward_index] += pi[backward_index] / reservemin;
    //     flags[backward_index] = true;
    //   }
    // }


    // for(int i = 0; i < record_Q.rear; i++){
    //   int backward_index = record_Q.arr[i];
    //   if(pi[backward_index] != 0){ 
    //     all_csr_entries[it][backward_index] += pi[backward_index] / reservemin;
    //   }
    // }



    for(int i = 0; i < record_Q.rear; i++){
      int backward_index = record_Q.arr[i];
      // if(pi[backward_index] != 0){
      //   all_csr_entries[it][backward_index] += pi[backward_index] / reservemin;
      // }
      if(backward_residue_entries.find(backward_index) == backward_residue_entries.end()){
        backward_residue_entries[backward_index] = residue[backward_index];
      }

      if(pi[backward_index] != 0){
        // thread_all_csr_entries[it - start][backward_index] += pi[backward_index] / reservemin;
        all_csr_entries[backward_index] += pi[backward_index] / reservemin;
      }
      else{
        continue;
      }

      if(backward_pi_entries.find(backward_index) == backward_pi_entries.end()){
        backward_pi_entries[backward_index] = pi[backward_index];
      }

    }


    // cout<<8<<endl;
    // ofs_debug<<8<<endl;


    // Maybe duplicate nnz, so we need to accumulate duplicate, cannot directly sum the nnz of forward and the nnz of backward.
    // single_source_ppr_nnz_number += backward_Q.rear - 0;



    // all_count += thread_all_csr_entries[it - start].size();
    all_count += all_csr_entries.size();

    //write out
    if(Partition_thread_all_csr_entries[interval_index].size() == inner_index_all_csr){
      // ofs_debug<<"start: write_out interval_index = "<<interval_index<<endl;
      // cout<<"start: write_out interval_index = "<<interval_index<<endl;
      // cout<<"inner_index_all_csr = "<<inner_index_all_csr<<endl;
      // cout<<"inner_index_dynamic_ppr = "<<inner_index_dynamic_ppr<<endl;   
      // cout<<"Partition_thread_all_csr_entries[interval_index].size() - 1 = "<<Partition_thread_all_csr_entries[interval_index].size() - 1<<endl;

      
      Partition_all_csr_offsets[interval_index] = ofs_all_csr.tellp();
      oa_all_csr << Partition_thread_all_csr_entries[interval_index];

      Partition_dynamic_ppr_offsets[interval_index] = ofs_dynamic_ppr.tellp();
      // oa_dynamic_ppr << Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[interval_index];


      //write_out dynamic ppr
      for(int i = 0; i < Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[interval_index].size(); i++){
        third_float_map& dynamic_ppr_map = Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[interval_index][i];
        for(auto& key_value: dynamic_ppr_map){
          int index = key_value.first;
          float value = key_value.second;
          if(value <= 0){
            continue;
          }

          
        }

        // Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[interval_index][i].clear();
        third_float_map().swap(Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[interval_index][i]);
      }

      interval_index++;
      inner_index_all_csr = 0;
      inner_index_dynamic_ppr = 0;
      // ofs_debug<<"finish: write_out interval_index = "<<interval_index<<endl;
      // cout<<"finish: write_out interval_index = "<<interval_index<<endl;
    }


  }


  // cout<<9<<endl;
  // ofs_debug<<9<<endl;


  // cout<<"finish writing"<<endl;


// // //End of Backward PPR computation


//   auto start_io_write_time = chrono::system_clock::now();



//     // all_csr_offsets = ofs_all_csr.tellp();
//     // oa_all_csr << thread_all_csr_entries;



//     // dynamic_ppr_offsets = ofs_dynamic_ppr.tellp();
//     // oa_dynamic_ppr << thread_forward_pi_forward_residue_backward_pi_backward_residue_entries;

    
//   // for(int it = start; it < end; it++){

//   //   unordered_map<long long int, double>& forward_pi_entries = thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[4 * (it-start) + 0];
//   //   unordered_map<long long int, double>& forward_residue_entries = thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[4 * (it-start) + 1];    
//   //   unordered_map<long long int, double>& backward_pi_entries = thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[4 * (it-start) + 2];    
//   //   unordered_map<long long int, double>& backward_residue_entries = thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[4 * (it-start) + 3];

//   //   forward_pi_entries.clear();
//   //   forward_residue_entries.clear();
//   //   backward_pi_entries.clear();
//   //   backward_residue_entries.clear();

//   // }




//   auto end_io_write_time = chrono::system_clock::now();
//   auto elapsed_write_time = chrono::duration_cast<std::chrono::seconds>(end_io_write_time - start_io_write_time);
//   write_out_io_all_count = elapsed_write_time.count();





//Deletion of Forward Allocation
  delete[] residue;
  delete[] pi;
  // delete[] transposed_forward_pi;
  delete[] flags;
//Deletion of Backward Allocation
  // delete[] backward_pi;

  // if (NULL != forward_Q.arr)
  // {
  //     free(forward_Q.arr);
  //     forward_Q.arr = NULL;
  // }

  if (NULL != forback_Q.arr)
  {
      free(forback_Q.arr);
      forback_Q.arr = NULL;
  }

  if (NULL != record_Q.arr)
  {
      free(record_Q.arr);
      record_Q.arr = NULL;
  }






  return;

}































































































































































// void ForwardPush(int* random_w, int start, int end, UGraph* g, double residuemax, double reservemin, vector<Triplet<double>>* answer, 
// long long int& all_count, double alpha){

void ForwardPush(int start, int end, UGraph* g, double residuemax, double reservemin, vector<Triplet<double>>* answer, 
long long int& all_count, double alpha, vector<int>& labeled_node_vec){

  cout<<"Start ForwardPush!"<<endl;
  cout<<"start = "<<start<<endl;
  cout<<"end = "<<end<<endl;

  int vertices_number = g->n;

  // cout<<"g->n = "<<vertices_number<<endl;
  double *residue = new double[vertices_number];

  double *pi = new double[vertices_number];

  bool *flags = new bool[vertices_number];

  int upper_nnz = ceil(1 / residuemax / alpha);


  // cout<<"upper_nnz = "<<upper_nnz<<endl;

  //Create Queue
	Queue Q =
	{
		//.arr = 
		(MKL_INT*)malloc( sizeof(MKL_INT) * (upper_nnz + 2) * 2 ),
		//.capacity = 
		(upper_nnz + 2) * 2,
		//.front =
		0,
		//.rear = 
		0
	};

  //The queue to record used elements in this node
	Queue record_Q =
	{
		//.arr = 
		(MKL_INT*)malloc(sizeof(MKL_INT) * (upper_nnz + 2) * 2 ),
		//.capacity = 
		(upper_nnz + 2) * 2,
		//.front =
		0,
		//.rear = 
		0
	};



  // //这个要改成memset
  // for(int i = 0; i < vertices_number; i++){
  //   residue[i] = 0;
  //   flags[i] = false;
  //   //顺带初始化pi
  //   pi[i] = 0;
  // }

  memset(residue, 0, sizeof(double) * vertices_number);

  memset(pi, 0, sizeof(double) * vertices_number);

  //bool变量，1个字节
  memset(flags, false, sizeof(bool) * vertices_number);

  int record_max;
  
  for(int it = start; it < end; it++){
    // cout<<"******"<<endl;

    Q.front = 0;
    Q.rear = 0;

    // int src = labeled_node_vec[it];
    int src = it;

    // if(record_Q.rear-record_Q.front > record_max){
    //   record_max = record_Q.rear-record_Q.front;
    // }

    // cout<<"record_Q.rear - record_Q.front = "<<record_max<<endl;

    while(!isEmpty(&record_Q)){
      int current = get_front(&record_Q);
      residue[current] = 0;
      flags[current] = false;
      //顺带初始化pi
      pi[current] = 0;
      dequeue(&record_Q);
    }

    record_Q.front = 0;
    record_Q.rear = 0;


    residue[src] = 1;
    flags[src] = true;
    // Q.push(src);
    // cout<<"Enqueue Q1!"<<endl;
    enqueue(&Q, src);

  //问题是到底需要多少容量，是1/residuemax/alpha吗？
  //对于Q，record_Q和pi,residue的记录，到底内循环走到了多少个节点

    // int front_pointer = 0;

    while(!isEmpty(&Q)){
      // int v = Q.front();
      int v = get_front(&Q);
      
      // int v = Q.arr[front_pointer];

      //是这里的问题，如果一个节点符合条件，但是degree = 0，这就意味着它没有损失(没有alpha * rmax)，这样的话我们要用双倍空间，
      //给每个节点的queue多提供一份
      if(g->degree[v] == 0){
        flags[v] = false;
        Q.front++;
        // dequeue(&Q);
        continue;
      }

      if(residue[v] / g->degree[v] > residuemax){
        for(int j = 0; j < g->degree[v]; j++){
          int u = g->AdjList[v][j];
          residue[u] += (1-alpha) * residue[v] / g->degree[v];
          
          // cout<<"Enqueue record_Q!"<<endl;
          enqueue(&record_Q, u);

          if(g->degree[u] == 0){
            continue;
          }
          
          if(residue[u] / g->degree[u] > residuemax && !flags[u]){
            // Q.push(u);
            // cout<<"Enqueue Q2!"<<endl;
            enqueue(&Q, u);
            flags[u] = true;
          }
        }
        pi[v] += alpha * residue[v];
        residue[v] = 0;
      }
      
      flags[v] = false;
      // Q.pop();

      // dequeue(&Q);
      Q.front++;
    }



    //在这里记录对于当前节点的pi和residue
    
    //1. check一下有多少节点是需要推入的，同时记录index
    //2. 新开一个这么多节点的空间
    //3. 把需要推入的节点推入
    //4. 把之前的长度为g.n的空间销毁
    //5. 把留下的节点记录到矩阵中去（这样方便后面的有序优化查询，不用开一个g.n的空间对应每个节点对应的cluster，
    //同时应该可以用insertBackUncompressed，这样省时间很多）
    // for (int i = 0; i < g->n; ++i){
    //   if(pi[i] > reservemin){
    //     answer->push_back(Triplet<double>(i, src, pi[i]));
    //     answer->push_back(Triplet<double>(src, i, pi[i]));
    //     all_count += 1;
    //   }
    // }

    // for (int i = 0; i < g->n; ++i){
    //   if(pi[i] > reservemin){
    //     answer->push_back(Triplet<double>(i, src, pi[i]));
    //     answer->push_back(Triplet<double>(src, i, pi[i]));
    //     all_count += 1;
    //   }
    // }

    for(int i = 0; i < Q.rear; i++){
      int index = Q.arr[i];
      if(pi[index] != 0){ 
        // answer->push_back(Triplet<double>(index, src, pi[index]));
        answer->push_back(Triplet<double>(src, index, pi[index]));
        all_count += 1;
      }
    }

    // double temp_sum = 0;
    // for(int i = 0; i < g->n; i++){
    //   temp_sum += residue[i];
    //   temp_sum += pi[i];
    // }



    // cout<<"temp_sum = "<<temp_sum<<endl;

    // cout<<"g->degree[src] = "<<g->degree[src]<<endl;

    // auto once_end_time = chrono::system_clock::now();


    // auto elapsed_once_time = chrono::duration_cast<std::chrono::seconds>(once_end_time - once_start_time);
    // cout << "Once used time: "<< elapsed_once_time.count() << endl;

  }

  // cout<<"record_max = "<<record_max<<endl;

  delete[] residue;
  delete[] pi;
  delete[] flags;

  if (NULL != Q.arr)
    {
        free(Q.arr);
        Q.arr = NULL;
    }

  if (NULL != record_Q.arr)
    {
        free(record_Q.arr);
        record_Q.arr = NULL;
    }

  cout<<"End ForwardPush!"<<endl;
  return;

}













void DirectedForwardPush(int start, int end, Graph* g, double residuemax, double reservemin, vector<Triplet<double>>* answer, 
long long int& all_count, double alpha, vector<int>& labeled_node_vec){

  cout<<"Start DirectedForwardPush!"<<endl;
  cout<<"start = "<<start<<endl;
  cout<<"end = "<<end<<endl;

  int vertices_number = g->n;

  // cout<<"g->n = "<<vertices_number<<endl;
  double *residue = new double[vertices_number];

  double *pi = new double[vertices_number];

  bool *flags = new bool[vertices_number];

  int upper_nnz = ceil(1 / residuemax / alpha);


  // cout<<"upper_nnz = "<<upper_nnz<<endl;

  //Create Queue
	Queue Q =
	{
		//.arr = 
		(MKL_INT*)malloc( sizeof(MKL_INT) * (upper_nnz + 2) * 2 ),
		//.capacity = 
		(upper_nnz + 2) * 2,
		//.front =
		0,
		//.rear = 
		0
	};

  //The queue to record used elements in this node
	Queue record_Q =
	{
		//.arr = 
		(MKL_INT*)malloc(sizeof(MKL_INT) * (upper_nnz + 2) * 2 ),
		//.capacity = 
		(upper_nnz + 2) * 2,
		//.front =
		0,
		//.rear = 
		0
	};



  // //这个要改成memset
  // for(int i = 0; i < vertices_number; i++){
  //   residue[i] = 0;
  //   flags[i] = false;
  //   //顺带初始化pi
  //   pi[i] = 0;
  // }

  memset(residue, 0, sizeof(double) * vertices_number);

  memset(pi, 0, sizeof(double) * vertices_number);

  //bool变量，1个字节
  memset(flags, false, sizeof(bool) * vertices_number);

  int record_max;
  
  for(int it = start; it < end; it++){
    // cout<<"******"<<endl;

    Q.front = 0;
    Q.rear = 0;

    int src = labeled_node_vec[it];

    // if(record_Q.rear-record_Q.front > record_max){
    //   record_max = record_Q.rear-record_Q.front;
    // }

    // cout<<"record_Q.rear - record_Q.front = "<<record_max<<endl;

    while(!isEmpty(&record_Q)){
      int current = get_front(&record_Q);
      residue[current] = 0;
      flags[current] = false;
      //顺带初始化pi
      pi[current] = 0;
      dequeue(&record_Q);
    }

    record_Q.front = 0;
    record_Q.rear = 0;


    residue[src] = 1;
    flags[src] = true;
    // Q.push(src);
    // cout<<"Enqueue Q1!"<<endl;
    enqueue(&Q, src);

  //问题是到底需要多少容量，是1/residuemax/alpha吗？
  //对于Q，record_Q和pi,residue的记录，到底内循环走到了多少个节点

    // int front_pointer = 0;

    while(!isEmpty(&Q)){
      // int v = Q.front();
      int v = get_front(&Q);
      
      // int v = Q.arr[front_pointer];

      //是这里的问题，如果一个节点符合条件，但是degree = 0，这就意味着它没有损失(没有alpha * rmax)，这样的话我们要用双倍空间，
      //给每个节点的queue多提供一份
      if(g->outdegree[v] == 0){
        flags[v] = false;
        Q.front++;
        // dequeue(&Q);
        continue;
      }

      if(residue[v] / g->outdegree[v] > residuemax){
        for(int j = 0; j < g->outdegree[v]; j++){
          int u = g->outAdjList[v][j];
          residue[u] += (1-alpha) * residue[v] / g->outdegree[v];
          
          // cout<<"Enqueue record_Q!"<<endl;
          enqueue(&record_Q, u);

          if(g->outdegree[u] == 0){
            continue;
          }
          
          if(residue[u] / g->outdegree[u] > residuemax && !flags[u]){
            // Q.push(u);
            // cout<<"Enqueue Q2!"<<endl;
            enqueue(&Q, u);
            flags[u] = true;
          }
        }
        pi[v] += alpha * residue[v];
        residue[v] = 0;
      }
      
      flags[v] = false;
      // Q.pop();

      // dequeue(&Q);
      Q.front++;
    }



    //在这里记录对于当前节点的pi和residue
    
    //1. check一下有多少节点是需要推入的，同时记录index
    //2. 新开一个这么多节点的空间
    //3. 把需要推入的节点推入
    //4. 把之前的长度为g.n的空间销毁
    //5. 把留下的节点记录到矩阵中去（这样方便后面的有序优化查询，不用开一个g.n的空间对应每个节点对应的cluster，
    //同时应该可以用insertBackUncompressed，这样省时间很多）
    // for (int i = 0; i < g->n; ++i){
    //   if(pi[i] > reservemin){
    //     answer->push_back(Triplet<double>(i, src, pi[i]));
    //     answer->push_back(Triplet<double>(src, i, pi[i]));
    //     all_count += 1;
    //   }
    // }

    // for (int i = 0; i < g->n; ++i){
    //   if(pi[i] > reservemin){
    //     answer->push_back(Triplet<double>(i, src, pi[i]));
    //     answer->push_back(Triplet<double>(src, i, pi[i]));
    //     all_count += 1;
    //   }
    // }

    for(int i = 0; i < Q.rear; i++){
      int index = Q.arr[i];
      // answer->push_back(Triplet<double>(index, src, pi[index]));
      if(pi[index] != 0){ 
        answer->push_back(Triplet<double>(src, index, pi[index]));
        all_count += 1;
      }
    }

    // double temp_sum = 0;
    // for(int i = 0; i < g->n; i++){
    //   temp_sum += residue[i];
    //   temp_sum += pi[i];
    // }



    // cout<<"temp_sum = "<<temp_sum<<endl;

    // cout<<"g->degree[src] = "<<g->degree[src]<<endl;

    // auto once_end_time = chrono::system_clock::now();


    // auto elapsed_once_time = chrono::duration_cast<std::chrono::seconds>(once_end_time - once_start_time);
    // cout << "Once used time: "<< elapsed_once_time.count() << endl;

  }

  // cout<<"record_max = "<<record_max<<endl;

  delete[] residue;
  delete[] pi;
  delete[] flags;

  if (NULL != Q.arr)
    {
        free(Q.arr);
        Q.arr = NULL;
    }

  if (NULL != record_Q.arr)
    {
        free(record_Q.arr);
        record_Q.arr = NULL;
    }

  cout<<"End DirectedForwardPush!"<<endl;
  return;

}







void DirectedForwardPushTranspose(int start, int end, Graph* g, double residuemax, double reservemin, vector<Triplet<double>>* answer, 
long long int& all_count, double alpha, vector<int>& labeled_node_vec){

  cout<<"Start DirectedForwardPushTranspose!"<<endl;
  cout<<"start = "<<start<<endl;
  cout<<"end = "<<end<<endl;

  int vertices_number = g->n;

  // cout<<"g->n = "<<vertices_number<<endl;
  double *residue = new double[vertices_number];

  double *pi = new double[vertices_number];

  bool *flags = new bool[vertices_number];

  int upper_nnz = ceil(1 / residuemax / alpha);


  // cout<<"upper_nnz = "<<upper_nnz<<endl;

  //Create Queue
	Queue Q =
	{
		//.arr = 
		(MKL_INT*)malloc( sizeof(MKL_INT) * (upper_nnz + 2) * 2 ),
		//.capacity = 
		(upper_nnz + 2) * 2,
		//.front =
		0,
		//.rear = 
		0
	};

  //The queue to record used elements in this node
	Queue record_Q =
	{
		//.arr = 
		(MKL_INT*)malloc(sizeof(MKL_INT) * (upper_nnz + 2) * 2 ),
		//.capacity = 
		(upper_nnz + 2) * 2,
		//.front =
		0,
		//.rear = 
		0
	};



  // //这个要改成memset
  // for(int i = 0; i < vertices_number; i++){
  //   residue[i] = 0;
  //   flags[i] = false;
  //   //顺带初始化pi
  //   pi[i] = 0;
  // }

  memset(residue, 0, sizeof(double) * vertices_number);

  memset(pi, 0, sizeof(double) * vertices_number);

  //bool变量，1个字节
  memset(flags, false, sizeof(bool) * vertices_number);

  int record_max;
  
  for(int it = start; it < end; it++){
    // cout<<"******"<<endl;

    Q.front = 0;
    Q.rear = 0;

    int src = labeled_node_vec[it];

    // if(record_Q.rear-record_Q.front > record_max){
    //   record_max = record_Q.rear-record_Q.front;
    // }

    // cout<<"record_Q.rear - record_Q.front = "<<record_max<<endl;

    while(!isEmpty(&record_Q)){
      int current = get_front(&record_Q);
      residue[current] = 0;
      flags[current] = false;
      //顺带初始化pi
      pi[current] = 0;
      dequeue(&record_Q);
    }

    record_Q.front = 0;
    record_Q.rear = 0;


    residue[src] = 1;
    flags[src] = true;
    // Q.push(src);
    // cout<<"Enqueue Q1!"<<endl;
    enqueue(&Q, src);

  //问题是到底需要多少容量，是1/residuemax/alpha吗？
  //对于Q，record_Q和pi,residue的记录，到底内循环走到了多少个节点

    // int front_pointer = 0;

    while(!isEmpty(&Q)){
      // int v = Q.front();
      int v = get_front(&Q);
      
      // int v = Q.arr[front_pointer];

      //是这里的问题，如果一个节点符合条件，但是degree = 0，这就意味着它没有损失(没有alpha * rmax)，这样的话我们要用双倍空间，
      //给每个节点的queue多提供一份
      if(g->indegree[v] == 0){
        flags[v] = false;
        Q.front++;
        // dequeue(&Q);
        continue;
      }

      if(residue[v] / g->indegree[v] > residuemax){
        for(int j = 0; j < g->indegree[v]; j++){
          int u = g->inAdjList[v][j];
          residue[u] += (1-alpha) * residue[v] / g->indegree[v];
          
          // cout<<"Enqueue record_Q!"<<endl;
          enqueue(&record_Q, u);

          if(g->indegree[u] == 0){
            continue;
          }
          
          if(residue[u] / g->indegree[u] > residuemax && !flags[u]){
            // Q.push(u);
            // cout<<"Enqueue Q2!"<<endl;
            enqueue(&Q, u);
            flags[u] = true;
          }
        }
        pi[v] += alpha * residue[v];
        residue[v] = 0;
      }
      
      flags[v] = false;
      // Q.pop();

      // dequeue(&Q);
      Q.front++;
    }



    //在这里记录对于当前节点的pi和residue
    
    //1. check一下有多少节点是需要推入的，同时记录index
    //2. 新开一个这么多节点的空间
    //3. 把需要推入的节点推入
    //4. 把之前的长度为g.n的空间销毁
    //5. 把留下的节点记录到矩阵中去（这样方便后面的有序优化查询，不用开一个g.n的空间对应每个节点对应的cluster，
    //同时应该可以用insertBackUncompressed，这样省时间很多）
    // for (int i = 0; i < g->n; ++i){
    //   if(pi[i] > reservemin){
    //     answer->push_back(Triplet<double>(i, src, pi[i]));
    //     answer->push_back(Triplet<double>(src, i, pi[i]));
    //     all_count += 1;
    //   }
    // }

    // for (int i = 0; i < g->n; ++i){
    //   if(pi[i] > reservemin){
    //     answer->push_back(Triplet<double>(i, src, pi[i]));
    //     answer->push_back(Triplet<double>(src, i, pi[i]));
    //     all_count += 1;
    //   }
    // }

    for(int i = 0; i < Q.rear; i++){
      int index = Q.arr[i];
      if(pi[index] != 0){ 
      // answer->push_back(Triplet<double>(index, src, pi[index]));
        answer->push_back(Triplet<double>(src, index, pi[index]));
        all_count += 1;
      }
    }

    // double temp_sum = 0;
    // for(int i = 0; i < g->n; i++){
    //   temp_sum += residue[i];
    //   temp_sum += pi[i];
    // }



    // cout<<"temp_sum = "<<temp_sum<<endl;

    // cout<<"g->degree[src] = "<<g->degree[src]<<endl;

    // auto once_end_time = chrono::system_clock::now();


    // auto elapsed_once_time = chrono::duration_cast<std::chrono::seconds>(once_end_time - once_start_time);
    // cout << "Once used time: "<< elapsed_once_time.count() << endl;

  }

  // cout<<"record_max = "<<record_max<<endl;

  delete[] residue;
  delete[] pi;
  delete[] flags;

  if (NULL != Q.arr)
    {
        free(Q.arr);
        Q.arr = NULL;
    }

  if (NULL != record_Q.arr)
    {
        free(record_Q.arr);
        record_Q.arr = NULL;
    }

  cout<<"End DirectedForwardPush!"<<endl;
  return;

}








void DirectedBackwardPushTranspose(int start, int end, Graph* g, double residuemax, double reservemin, vector<Triplet<double>>* answer, 
long long int& all_count, double alpha, vector<int>& labeled_node_vec){

  cout<<"Start DirectedTransposeBackwardPush!"<<endl;
  cout<<"start = "<<start<<endl;
  cout<<"end = "<<end<<endl;

  int vertices_number = g->n;

  // cout<<"g->n = "<<vertices_number<<endl;
  double *residue = new double[vertices_number];

  double *pi = new double[vertices_number];

  bool *flags = new bool[vertices_number];

  int upper_nnz = ceil(1 / residuemax / alpha * 100);


  // cout<<"upper_nnz = "<<upper_nnz<<endl;

  //Create Queue
	Queue Q =
	{
		//.arr = 
		(MKL_INT*)malloc( sizeof(MKL_INT) * (upper_nnz + 2) * 2 ),
		//.capacity = 
		(upper_nnz + 2) * 2,
		//.front =
		0,
		//.rear = 
		0
	};

  //The queue to record used elements in this node
	Queue record_Q =
	{
		//.arr = 
		(MKL_INT*)malloc(sizeof(MKL_INT) * (upper_nnz + 2) * 2 ),
		//.capacity = 
		(upper_nnz + 2) * 2,
		//.front =
		0,
		//.rear = 
		0
	};



  // //这个要改成memset
  // for(int i = 0; i < vertices_number; i++){
  //   residue[i] = 0;
  //   flags[i] = false;
  //   //顺带初始化pi
  //   pi[i] = 0;
  // }

  memset(residue, 0, sizeof(double) * vertices_number);

  memset(pi, 0, sizeof(double) * vertices_number);

  //bool变量，1个字节
  memset(flags, false, sizeof(bool) * vertices_number);

  int record_max;
  
  for(int it = start; it < end; it++){
    // cout<<"******"<<endl;

    Q.front = 0;
    Q.rear = 0;

    int src = labeled_node_vec[it];

    // if(record_Q.rear-record_Q.front > record_max){
    //   record_max = record_Q.rear-record_Q.front;
    // }

    // cout<<"record_Q.rear - record_Q.front = "<<record_max<<endl;

    while(!isEmpty(&record_Q)){
      int current = get_front(&record_Q);
      residue[current] = 0;
      flags[current] = false;
      //顺带初始化pi
      pi[current] = 0;
      dequeue(&record_Q);
    }

    record_Q.front = 0;
    record_Q.rear = 0;


    residue[src] = 1;
    flags[src] = true;
    // Q.push(src);
    // cout<<"Enqueue Q1!"<<endl;
    enqueue(&Q, src);

  //问题是到底需要多少容量，是1/residuemax/alpha吗？
  //对于Q，record_Q和pi,residue的记录，到底内循环走到了多少个节点

    // int front_pointer = 0;

    while(!isEmpty(&Q)){
      // int v = Q.front();
      int v = get_front(&Q);
      
      // int v = Q.arr[front_pointer];

      //是这里的问题，如果一个节点符合条件，但是degree = 0，这就意味着它没有损失(没有alpha * rmax)，这样的话我们要用双倍空间，
      //给每个节点的queue多提供一份
      if(g->indegree[v] == 0){
        flags[v] = false;
        Q.front++;
        // dequeue(&Q);
        continue;
      }

      if(residue[v] > residuemax){
        for(int j = 0; j < g->outdegree[v]; j++){
          int u = g->outAdjList[v][j];
          residue[u] += (1-alpha) * residue[v] / g->indegree[u];
          
          // cout<<"Enqueue record_Q!"<<endl;
          enqueue(&record_Q, u);

          // if(g->indegree[u] == 0){
          //   continue;
          // }
          
          if(residue[u] > residuemax && !flags[u]){
            // Q.push(u);
            // cout<<"Enqueue Q2!"<<endl;
            enqueue(&Q, u);
            flags[u] = true;
          }
        }
        pi[v] += alpha * residue[v];
        residue[v] = 0;
      }
      
      flags[v] = false;
      // Q.pop();

      // dequeue(&Q);
      Q.front++;
    }



    //在这里记录对于当前节点的pi和residue
    
    //1. check一下有多少节点是需要推入的，同时记录index
    //2. 新开一个这么多节点的空间
    //3. 把需要推入的节点推入
    //4. 把之前的长度为g.n的空间销毁
    //5. 把留下的节点记录到矩阵中去（这样方便后面的有序优化查询，不用开一个g.n的空间对应每个节点对应的cluster，
    //同时应该可以用insertBackUncompressed，这样省时间很多）
    // for (int i = 0; i < g->n; ++i){
    //   if(pi[i] > reservemin){
    //     answer->push_back(Triplet<double>(i, src, pi[i]));
    //     answer->push_back(Triplet<double>(src, i, pi[i]));
    //     all_count += 1;
    //   }
    // }

    // for (int i = 0; i < g->n; ++i){
    //   if(pi[i] > reservemin){
    //     answer->push_back(Triplet<double>(i, src, pi[i]));
    //     answer->push_back(Triplet<double>(src, i, pi[i]));
    //     all_count += 1;
    //   }
    // }

    for(int i = 0; i < Q.rear; i++){
      int index = Q.arr[i];
      if(pi[index] != 0){ 
      // answer->push_back(Triplet<double>(index, src, pi[index]));
        answer->push_back(Triplet<double>(src, index, pi[index]));
        all_count += 1;
      }
    }

    // double temp_sum = 0;
    // for(int i = 0; i < g->n; i++){
    //   temp_sum += residue[i];
    //   temp_sum += pi[i];
    // }



    // cout<<"temp_sum = "<<temp_sum<<endl;

    // cout<<"g->degree[src] = "<<g->degree[src]<<endl;

    // auto once_end_time = chrono::system_clock::now();


    // auto elapsed_once_time = chrono::duration_cast<std::chrono::seconds>(once_end_time - once_start_time);
    // cout << "Once used time: "<< elapsed_once_time.count() << endl;

  }

  // cout<<"record_max = "<<record_max<<endl;

  delete[] residue;
  delete[] pi;
  delete[] flags;

  if (NULL != Q.arr)
    {
        free(Q.arr);
        Q.arr = NULL;
    }

  if (NULL != record_Q.arr)
    {
        free(record_Q.arr);
        record_Q.arr = NULL;
    }

  cout<<"End DirectedTransposeBackwardPush!"<<endl;
  return;

}




























































































































void AnytimeUndirectedBackwardPush(int start, int end, 
UGraph* g, 
double residuemax, double reservemin, vector<Triplet<double>>* answer, 
long long int& all_count, double alpha
// , vector<int>& labeled_node_vec
){

  // cout<<"Start UndirectedBackwardPush!"<<endl;
  // cout<<"start = "<<start<<endl;
  // cout<<"end = "<<end<<endl;

  int vertices_number = g->n;

  // cout<<"point 1"<<endl;

  double *residue = new double[vertices_number];

  double *pi = new double[vertices_number];

  bool *flags = new bool[vertices_number];

  // int upper_nnz = ceil(1 / residuemax / alpha * 100);
  int upper_nnz = INT_MAX;


  // cout<<"point 2"<<endl;

  //Create Queue
	Queue Q =
	{
		//.arr = 
		(MKL_INT*)malloc( sizeof(MKL_INT) * (INT_MAX-1) ),
		//.capacity = 
		// INT_MAX,
		INT_MAX - 1,
		//.front =
		0,
		//.rear = 
		0
	};

  //The queue to record used elements in this node
	Queue record_Q =
	{
		//.arr = 
		(MKL_INT*)malloc(sizeof(MKL_INT) * (INT_MAX-1) ),
		//.capacity = 
		// INT_MAX,
		INT_MAX - 1,
		//.front =
		0,
		//.rear = 
		0
	};




  memset(residue, 0, sizeof(double) * vertices_number);

  memset(pi, 0, sizeof(double) * vertices_number);

  //bool变量，1个字节
  memset(flags, false, sizeof(bool) * vertices_number);

  
  // cout<<"point 3"<<endl;

  for(int it = start; it < end; it++){

    Q.front = 0;
    Q.rear = 0;

    int src = it;

    // cout<<"point 4"<<endl;

    while(!isEmpty(&record_Q)){
      int current = get_front(&record_Q);
      residue[current] = 0;
      flags[current] = false;

      pi[current] = 0;
      dequeue(&record_Q);
    }

    record_Q.front = 0;
    record_Q.rear = 0;

    // cout<<"point 5"<<endl;

    residue[src] = 1;
    flags[src] = true;

    enqueue(&Q, src);

    // cout<<"point 6"<<endl;

    while(!isEmpty(&Q)){
      
    int v = get_front(&Q);
      
    // cout<<"point 7"<<endl;

      if(g->degree[v] == 0){
        flags[v] = false;
        Q.front++;
        cout<<"Backward degree[v] == 0"<<endl;
        // dequeue(&Q);
        continue;
      }

      // cout<<"point 8"<<endl;      

      if(residue[v] > residuemax){
        for(int j = 0; j < g->degree[v]; j++){
      // cout<<"point 8:1"<<endl;
      // cout<<"g->degree[v] = "<<g->degree[v]<<endl;  
      // cout<<"j = "<<j<<endl;
      // cout<<"v = "<<v<<endl;
          int u = g->AdjList[v][j];
      // cout<<"g->degree[u] = "<<g->degree[u]<<endl;      
      // cout<<"point 8:2"<<endl;      
          if(g->degree[u] == 0){
      // cout<<"point 8:3"<<endl;      
            cout<<"Forward degree[u] == 0"<<endl;
            continue;
          }
      // cout<<"point 8:4"<<endl;      
          residue[u] += (1-alpha) * residue[v] / g->degree[u];
          
      // cout<<"point 8:5"<<endl;      
          // cout<<"Enqueue record_Q!"<<endl;
          enqueue(&record_Q, u);

      // cout<<"point 8:6"<<endl;      

          // cout<<"point 9"<<endl;

          // if(g->indegree[u] == 0){
          //   continue;
          // }
          
          if(residue[u] > residuemax && !flags[u]){

            enqueue(&Q, u);
            flags[u] = true;
          }
        }
        pi[v] += alpha * residue[v];
        residue[v] = 0;
      }
      
      flags[v] = false;

      Q.front++;
    }

    // cout<<"point 10"<<endl;

    for(int i = 0; i < Q.rear; i++){
      int index = Q.arr[i];
      if(pi[index] != 0){ 
        answer->push_back(Triplet<double>(index, src, pi[index]));
        // answer->push_back(Triplet<double>(src, index, pi[index]));
        all_count += 1;
      }
    }

    // cout<<"point 11"<<endl;
  }

  // cout<<"point 12"<<endl;

  delete[] residue;
  delete[] pi;
  delete[] flags;

  if (NULL != Q.arr)
    {
        free(Q.arr);
        Q.arr = NULL;
    }

  if (NULL != record_Q.arr)
    {
        free(record_Q.arr);
        record_Q.arr = NULL;
    }

  // cout<<"End UndirectedBackwardPush!"<<endl;
  return;

}






















void AnytimeUndirectedForwardPush(int start, int end, UGraph* g, double residuemax, double reservemin, vector<Triplet<double>>* answer, 
long long int& all_count, double alpha
){

  cout<<"Start ForwardPush!"<<endl;
  cout<<"start = "<<start<<endl;
  cout<<"end = "<<end<<endl;

  int vertices_number = g->n;

  // cout<<"g->n = "<<vertices_number<<endl;
  double *residue = new double[vertices_number];

  double *pi = new double[vertices_number];

  bool *flags = new bool[vertices_number];

  int upper_nnz = ceil(1 / residuemax / alpha);


  cout<<"upper_nnz = "<<upper_nnz<<endl;

  //Create Queue
	Queue Q =
	{
		//.arr = 
		(MKL_INT*)malloc( sizeof(MKL_INT) * (upper_nnz + 2) * 2 ),
		//.capacity = 
		(upper_nnz + 2) * 2,
		//.front =
		0,
		//.rear = 
		0
	};

  // //The queue to record used elements in this node
	// Queue record_Q =
	// {
	// 	//.arr = 
	// 	(int*)malloc(sizeof(int) * (upper_nnz + 2) * 2 ),
	// 	//.capacity = 
	// 	(upper_nnz + 2) * 2,
	// 	//.front =
	// 	0,
	// 	//.rear = 
	// 	0
	// };
  //The queue to record used elements in this node
	Queue record_Q =
	{
		//.arr = 
		(MKL_INT*)malloc(sizeof(MKL_INT) * (INT_MAX - 1) ),
		//.capacity = 
		(INT_MAX - 1),
		//.front =
		0,
		//.rear = 
		0
	};



  // //这个要改成memset
  // for(int i = 0; i < vertices_number; i++){
  //   residue[i] = 0;
  //   flags[i] = false;
  //   //顺带初始化pi
  //   pi[i] = 0;
  // }

  memset(residue, 0, sizeof(double) * vertices_number);

  memset(pi, 0, sizeof(double) * vertices_number);

  //bool变量，1个字节
  memset(flags, false, sizeof(bool) * vertices_number);

  int record_max;
  
  for(int it = start; it < end; it++){
    // cout<<"it = "<<it<<endl;

    Q.front = 0;
    Q.rear = 0;

    // int src = labeled_node_vec[it];
    int src = it;

    // if(record_Q.rear-record_Q.front > record_max){
    //   record_max = record_Q.rear-record_Q.front;
    // }

    // cout<<"record_Q.rear - record_Q.front = "<<record_max<<endl;

    while(!isEmpty(&record_Q)){
      int current = get_front(&record_Q);
      residue[current] = 0;
      flags[current] = false;
      //顺带初始化pi
      pi[current] = 0;
      dequeue(&record_Q);
    }

    record_Q.front = 0;
    record_Q.rear = 0;


    residue[src] = 1;
    flags[src] = true;
    // Q.push(src);
    // cout<<"Enqueue Q1!"<<endl;
    enqueue(&Q, src);

  //问题是到底需要多少容量，是1/residuemax/alpha吗？
  //对于Q，record_Q和pi,residue的记录，到底内循环走到了多少个节点

    // int front_pointer = 0;

    while(!isEmpty(&Q)){
      // int v = Q.front();
      int v = get_front(&Q);
      
      // int v = Q.arr[front_pointer];

      //是这里的问题，如果一个节点符合条件，但是degree = 0，这就意味着它没有损失(没有alpha * rmax)，这样的话我们要用双倍空间，
      //给每个节点的queue多提供一份
      if(g->degree[v] == 0){
        flags[v] = false;

        //处理dead end
        pi[v] = alpha * residue[v];

        Q.front++;

        cout<<"Forward degree[v] == 0"<<endl;
        // dequeue(&Q);
        continue;
      }

      if(residue[v] / g->degree[v] > residuemax){
      // if(residue[v] > residuemax){
        for(int j = 0; j < g->degree[v]; j++){
          int u = g->AdjList[v][j];
          residue[u] += (1-alpha) * residue[v] / g->degree[v];
          
          // cout<<"Enqueue record_Q!"<<endl;
          enqueue(&record_Q, u);

          if(g->degree[u] == 0){
            pi[u] += alpha * residue[v];
            residue[src] += (1-alpha) * residue[v];
            cout<<"Forward degree[u] == 0"<<endl;
            continue;
          }
          
          if(residue[u] / g->degree[u] > residuemax && !flags[u]){
          // if(residue[u] > residuemax && !flags[u]){
            // Q.push(u);
            // cout<<"Enqueue Q2!"<<endl;
            enqueue(&Q, u);
            flags[u] = true;
          }
        }
        pi[v] += alpha * residue[v];
        residue[v] = 0;
      }
      
      flags[v] = false;
      // Q.pop();

      // dequeue(&Q);
      Q.front++;
    }



    //在这里记录对于当前节点的pi和residue
    
    //1. check一下有多少节点是需要推入的，同时记录index
    //2. 新开一个这么多节点的空间
    //3. 把需要推入的节点推入
    //4. 把之前的长度为g.n的空间销毁
    //5. 把留下的节点记录到矩阵中去（这样方便后面的有序优化查询，不用开一个g.n的空间对应每个节点对应的cluster，
    //同时应该可以用insertBackUncompressed，这样省时间很多）
    // for (int i = 0; i < g->n; ++i){
    //   if(pi[i] > reservemin){
    //     answer->push_back(Triplet<double>(i, src, pi[i]));
    //     answer->push_back(Triplet<double>(src, i, pi[i]));
    //     all_count += 1;
    //   }
    // }

    // for (int i = 0; i < g->n; ++i){
    //   if(pi[i] > reservemin){
    //     answer->push_back(Triplet<double>(i, src, pi[i]));
    //     answer->push_back(Triplet<double>(src, i, pi[i]));
    //     all_count += 1;
    //   }
    // }

    for(int i = 0; i < Q.rear; i++){
      int index = Q.arr[i];
      if(pi[index] != 0){ 
        // answer->push_back(Triplet<double>(index, src, pi[index]));
        answer->push_back(Triplet<double>(index, src, pi[index]));
        all_count += 1;
      }
    }

    // double temp_sum = 0;
    // for(int i = 0; i < g->n; i++){
    //   temp_sum += residue[i];
    //   temp_sum += pi[i];
    // }



    // cout<<"temp_sum = "<<temp_sum<<endl;

    // cout<<"g->degree[src] = "<<g->degree[src]<<endl;

    // auto once_end_time = chrono::system_clock::now();


    // auto elapsed_once_time = chrono::duration_cast<std::chrono::seconds>(once_end_time - once_start_time);
    // cout << "Once used time: "<< elapsed_once_time.count() << endl;

  }

  // cout<<"record_max = "<<record_max<<endl;

  delete[] residue;
  delete[] pi;
  delete[] flags;

  if (NULL != Q.arr)
    {
        free(Q.arr);
        Q.arr = NULL;
    }

  if (NULL != record_Q.arr)
    {
        free(record_Q.arr);
        record_Q.arr = NULL;
    }

  return;

}




























void AnytimeDirectedBackwardPush(int start, int end, 
Graph* g, 
double residuemax, double reservemin, vector<Triplet<double>>* answer, 
long long int& all_count, double alpha
// , vector<int>& labeled_node_vec
){

  // cout<<"Start UndirectedBackwardPush!"<<endl;
  // cout<<"start = "<<start<<endl;
  // cout<<"end = "<<end<<endl;

  int vertices_number = g->n;

  // cout<<"point 1"<<endl;

  double *residue = new double[vertices_number];

  double *pi = new double[vertices_number];

  bool *flags = new bool[vertices_number];

  // int upper_nnz = ceil(1 / residuemax / alpha * 100);
  int upper_nnz = INT_MAX;


  // cout<<"point 2"<<endl;

  //Create Queue
	Queue Q =
	{
		//.arr = 
		(MKL_INT*)malloc( sizeof(MKL_INT) * (INT_MAX-1) ),
		//.capacity = 
		// INT_MAX,
    INT_MAX - 1,
  	//.front =
		0,
		//.rear = 
		0
	};

  //The queue to record used elements in this node
	Queue record_Q =
	{
		//.arr = 
		(MKL_INT*)malloc(sizeof(MKL_INT) * (INT_MAX-1) ),
		//.capacity = 
		// INT_MAX,
		INT_MAX - 1,
		//.front =
		0,
		//.rear = 
		0
	};




  memset(residue, 0, sizeof(double) * vertices_number);

  memset(pi, 0, sizeof(double) * vertices_number);

  //bool变量，1个字节
  memset(flags, false, sizeof(bool) * vertices_number);

  
  // cout<<"point 3"<<endl;

  for(int it = start; it < end; it++){

    Q.front = 0;
    Q.rear = 0;

    int src = it;

    // cout<<"point 4"<<endl;

    while(!isEmpty(&record_Q)){
      int current = get_front(&record_Q);
      residue[current] = 0;
      flags[current] = false;

      pi[current] = 0;
      dequeue(&record_Q);
    }

    record_Q.front = 0;
    record_Q.rear = 0;

    // cout<<"point 5"<<endl;

    residue[src] = 1;
    flags[src] = true;

    enqueue(&Q, src);

    // cout<<"point 6"<<endl;

    while(!isEmpty(&Q)){
      
    int v = get_front(&Q);
      
    // cout<<"point 7"<<endl;

      if(g->indegree[v] == 0){
        flags[v] = false;
        Q.front++;
        cout<<"Backward Indegree[v] == 0"<<endl;      
        // dequeue(&Q);
        continue;
      }

      // cout<<"point 8"<<endl;      

      if(residue[v] > residuemax){
        for(int j = 0; j < g->indegree[v]; j++){
      // cout<<"point 8:1"<<endl;
      // cout<<"g->degree[v] = "<<g->degree[v]<<endl;  
      // cout<<"j = "<<j<<endl;
      // cout<<"v = "<<v<<endl;
          int u = g->inAdjList[v][j];
      // cout<<"g->degree[u] = "<<g->degree[u]<<endl;      
      // cout<<"point 8:2"<<endl;      
          if(g->outdegree[u] == 0){
      // cout<<"point 8:3"<<endl;
            cout<<"Backward Outdegree[u] == 0"<<endl;      
            continue;
          }
      // cout<<"point 8:4"<<endl;      
          residue[u] += (1-alpha) * residue[v] / g->outdegree[u];
          
      // cout<<"point 8:5"<<endl;      
          // cout<<"Enqueue record_Q!"<<endl;
          enqueue(&record_Q, u);

      // cout<<"point 8:6"<<endl;      

          // cout<<"point 9"<<endl;

          // if(g->indegree[u] == 0){
          //   continue;
          // }

                  
          // if(residue[u] > residuemax && !flags[u]){
          if(residue[u] > residuemax && !flags[u]){

            enqueue(&Q, u);
            flags[u] = true;
          }
        }
        pi[v] += alpha * residue[v];
        residue[v] = 0;
      }
      
      flags[v] = false;

      Q.front++;
    }

    // cout<<"point 10"<<endl;

    for(int i = 0; i < Q.rear; i++){
      int index = Q.arr[i];
      if(pi[index] != 0){ 
        answer->push_back(Triplet<double>(index, src, pi[index]));
        // answer->push_back(Triplet<double>(src, index, pi[index]));
        all_count += 1;
      }
    }

    // cout<<"point 11"<<endl;
  }

  // cout<<"point 12"<<endl;

  delete[] residue;
  delete[] pi;
  delete[] flags;

  if (NULL != Q.arr)
    {
        free(Q.arr);
        Q.arr = NULL;
    }

  if (NULL != record_Q.arr)
    {
        free(record_Q.arr);
        record_Q.arr = NULL;
    }

  // cout<<"End UndirectedBackwardPush!"<<endl;
  return;

}





















































void AnytimeDirectedTransposedForwardPush(int start, int end, Graph* g, double residuemax, double reservemin, vector<Triplet<double>>* answer, 
long long int& all_count, double alpha
){

  cout<<"Start AnytimeDirectedForwardPush!"<<endl;
  cout<<"start = "<<start<<endl;
  cout<<"end = "<<end<<endl;

  int vertices_number = g->n;

  // cout<<"g->n = "<<vertices_number<<endl;
  double *residue = new double[vertices_number];

  double *pi = new double[vertices_number];

  bool *flags = new bool[vertices_number];

  int upper_nnz = ceil(1 / residuemax / alpha);


  cout<<"upper_nnz = "<<upper_nnz<<endl;

  //Create Queue
	Queue Q =
	{
		//.arr = 
		(MKL_INT*)malloc( sizeof(MKL_INT) * (upper_nnz + 2) * 2 ),
		//.capacity = 
		(upper_nnz + 2) * 2,
		//.front =
		0,
		//.rear = 
		0
	};

  //The queue to record used elements in this node
	// Queue record_Q =
	// {
	// 	//.arr = 
	// 	(int*)malloc(sizeof(int) * (upper_nnz + 2) * 2 ),
	// 	//.capacity = 
	// 	(upper_nnz + 2) * 2,
	// 	//.front =
	// 	0,
	// 	//.rear = 
	// 	0
	// };
	Queue record_Q =
	{
		//.arr = 
		(MKL_INT*)malloc(sizeof(MKL_INT) * (INT_MAX - 1) ),
		//.capacity = 
		INT_MAX - 1,
		//.front =
		0,
		//.rear = 
		0
	};



  // //这个要改成memset
  // for(int i = 0; i < vertices_number; i++){
  //   residue[i] = 0;
  //   flags[i] = false;
  //   //顺带初始化pi
  //   pi[i] = 0;
  // }

  memset(residue, 0, sizeof(double) * vertices_number);

  memset(pi, 0, sizeof(double) * vertices_number);

  //bool变量，1个字节
  memset(flags, false, sizeof(bool) * vertices_number);

  int record_max;
  
  for(int it = start; it < end; it++){
    // cout<<"it = "<<it<<endl;

    Q.front = 0;
    Q.rear = 0;

    // int src = labeled_node_vec[it];
    int src = it;

    // if(record_Q.rear-record_Q.front > record_max){
    //   record_max = record_Q.rear-record_Q.front;
    // }

    // cout<<"record_Q.rear - record_Q.front = "<<record_max<<endl;

    while(!isEmpty(&record_Q)){
      int current = get_front(&record_Q);
      residue[current] = 0;
      flags[current] = false;
      //顺带初始化pi
      pi[current] = 0;
      dequeue(&record_Q);
    }

    record_Q.front = 0;
    record_Q.rear = 0;


    residue[src] = 1;
    // for(int i = 0; i < vertices_number; i++){
    //   residue[i] = 1 /vertices_number;
    // }

    flags[src] = true;
    // Q.push(src);
    // cout<<"Enqueue Q1!"<<endl;
    enqueue(&Q, src);

  //问题是到底需要多少容量，是1/residuemax/alpha吗？
  //对于Q，record_Q和pi,residue的记录，到底内循环走到了多少个节点

    // int front_pointer = 0;

    while(!isEmpty(&Q)){
      // int v = Q.front();
      int v = get_front(&Q);
      
      // int v = Q.arr[front_pointer];

      //是这里的问题，如果一个节点符合条件，但是degree = 0，这就意味着它没有损失(没有alpha * rmax)，这样的话我们要用双倍空间，
      //给每个节点的queue多提供一份
      if(g->indegree[v] == 0){
        //注释掉 flags[v] = false; 这样最多推进来一次
        flags[v] = false;
        
        //处理dead end
        pi[v] = alpha * residue[v];
        cout<<"Forward indegree[v] == 0"<<endl;
        
        residue[v] = 0;
        Q.front++;
        // dequeue(&Q);
        continue;
      }

      if(residue[v] / g->indegree[v] > residuemax){
      // if(residue[v] > residuemax){
        for(int j = 0; j < g->indegree[v]; j++){
          int u = g->inAdjList[v][j];

          if(g->indegree[u] == 0){
            //处理dead end
            cout<<"Forward indegree[u] == 0"<<endl;
            pi[u] += alpha * residue[v];
            residue[src] += (1-alpha) * residue[v];
            continue;
          }

          // residue[u] += (1-alpha) * residue[v] / g->indegree[v];
          residue[u] += (1-alpha) * residue[v] / g->indegree[v];
          enqueue(&record_Q, u);
          
          // cout<<"Enqueue record_Q!"<<endl;
          
          if(residue[u] / g->indegree[u] > residuemax && !flags[u]){
          // if(residue[u] > residuemax && !flags[u]){
            // Q.push(u);
            // cout<<"Enqueue Q2!"<<endl;
            enqueue(&Q, u);
            flags[u] = true;
          }

        }
        pi[v] += alpha * residue[v];
        residue[v] = 0;
      }
      
      flags[v] = false;
      // Q.pop();

      // dequeue(&Q);
      Q.front++;
    }



    //在这里记录对于当前节点的pi和residue
    
    //1. check一下有多少节点是需要推入的，同时记录index
    //2. 新开一个这么多节点的空间
    //3. 把需要推入的节点推入
    //4. 把之前的长度为g.n的空间销毁
    //5. 把留下的节点记录到矩阵中去（这样方便后面的有序优化查询，不用开一个g.n的空间对应每个节点对应的cluster，
    //同时应该可以用insertBackUncompressed，这样省时间很多）
    // for (int i = 0; i < g->n; ++i){
    //   if(pi[i] > reservemin){
    //     answer->push_back(Triplet<double>(i, src, pi[i]));
    //     answer->push_back(Triplet<double>(src, i, pi[i]));
    //     all_count += 1;
    //   }
    // }

    // for (int i = 0; i < g->n; ++i){
    //   if(pi[i] > reservemin){
    //     answer->push_back(Triplet<double>(i, src, pi[i]));
    //     answer->push_back(Triplet<double>(src, i, pi[i]));
    //     all_count += 1;
    //   }
    // }

    for(int i = 0; i < Q.rear; i++){
      int index = Q.arr[i];
      if(pi[index] != 0){ 
        answer->push_back(Triplet<double>(index, src, pi[index]));
        all_count += 1;
      }
    }

    // double temp_sum = 0;
    // for(int i = 0; i < g->n; i++){
    //   temp_sum += residue[i];
    //   temp_sum += pi[i];
    // }



    // cout<<"temp_sum = "<<temp_sum<<endl;

    // cout<<"g->degree[src] = "<<g->degree[src]<<endl;

    // auto once_end_time = chrono::system_clock::now();


    // auto elapsed_once_time = chrono::duration_cast<std::chrono::seconds>(once_end_time - once_start_time);
    // cout << "Once used time: "<< elapsed_once_time.count() << endl;

  }

  // cout<<"record_max = "<<record_max<<endl;

  delete[] residue;
  delete[] pi;
  delete[] flags;

  if (NULL != Q.arr)
    {
        free(Q.arr);
        Q.arr = NULL;
    }

  if (NULL != record_Q.arr)
    {
        free(record_Q.arr);
        record_Q.arr = NULL;
    }

  return;

}



























































































































































































































































void ForwardPush_IndexVersion(int start, int end, UGraph* g, double residuemax, double reservemin, vector<Triplet<double>>* answer, 
long long int& all_count, double alpha){

  cout<<"Start ForwardPush!"<<endl;
  cout<<"start = "<<start<<endl;
  cout<<"end = "<<end<<endl;

  int vertices_number = g->n;

  // cout<<"g->n = "<<vertices_number<<endl;
  double *residue = new double[vertices_number];

  double *pi = new double[vertices_number];

  bool *flags = new bool[vertices_number];

  int upper_nnz = ceil(1 / residuemax / alpha);


  // cout<<"upper_nnz = "<<upper_nnz<<endl;

  //Create Queue
	Queue Q =
	{
		//.arr = 
		(MKL_INT*)malloc( sizeof(MKL_INT) * (upper_nnz + 2) * 2 ),
		//.capacity = 
		(upper_nnz + 2) * 2,
		//.front =
		0,
		//.rear = 
		0
	};

  //The queue to record used elements in this node
	Queue record_Q =
	{
		//.arr = 
		(MKL_INT*)malloc(sizeof(MKL_INT) * (upper_nnz + 2) * 2 ),
		//.capacity = 
		(upper_nnz + 2) * 2,
		//.front =
		0,
		//.rear = 
		0
	};



  // //这个要改成memset
  // for(int i = 0; i < vertices_number; i++){
  //   residue[i] = 0;
  //   flags[i] = false;
  //   //顺带初始化pi
  //   pi[i] = 0;
  // }

  memset(residue, 0, sizeof(double) * vertices_number);

  memset(pi, 0, sizeof(double) * vertices_number);

  //bool变量，1个字节
  memset(flags, false, sizeof(bool) * vertices_number);

  int record_max;
  
  for(int it = start; it < end; it++){
    // cout<<"******"<<endl;

    Q.front = 0;
    Q.rear = 0;

    int src = it;

    // if(record_Q.rear-record_Q.front > record_max){
    //   record_max = record_Q.rear-record_Q.front;
    // }

    // cout<<"record_Q.rear - record_Q.front = "<<record_max<<endl;

    while(!isEmpty(&record_Q)){
      int current = get_front(&record_Q);
      residue[current] = 0;
      flags[current] = false;
      //顺带初始化pi
      pi[current] = 0;
      dequeue(&record_Q);
    }

    record_Q.front = 0;
    record_Q.rear = 0;


    residue[src] = 1;
    flags[src] = true;
    // Q.push(src);
    // cout<<"Enqueue Q1!"<<endl;
    enqueue(&Q, src);

  //问题是到底需要多少容量，是1/residuemax/alpha吗？
  //对于Q，record_Q和pi,residue的记录，到底内循环走到了多少个节点

    // int front_pointer = 0;

    while(!isEmpty(&Q)){
      // int v = Q.front();
      int v = get_front(&Q);
      
      // int v = Q.arr[front_pointer];

      //是这里的问题，如果一个节点符合条件，但是degree = 0，这就意味着它没有损失(没有alpha * rmax)，这样的话我们要用双倍空间，
      //给每个节点的queue多提供一份
      if(g->degree[v] == 0){
        flags[v] = false;
        Q.front++;
        // dequeue(&Q);
        continue;
      }

      if(residue[v] / g->degree[v] > residuemax){
        for(int j = 0; j < g->degree[v]; j++){
          int u = g->AdjList[v][j];
          residue[u] += (1-alpha) * residue[v] / g->degree[v];
          
          // cout<<"Enqueue record_Q!"<<endl;
          enqueue(&record_Q, u);

          if(g->degree[u] == 0){
            continue;
          }
          
          if(residue[u] / g->degree[u] > residuemax && !flags[u]){
            // Q.push(u);
            // cout<<"Enqueue Q2!"<<endl;
            enqueue(&Q, u);
            flags[u] = true;
          }
        }
        pi[v] += alpha * residue[v];
        residue[v] = 0;
      }
      
      flags[v] = false;
      // Q.pop();

      // dequeue(&Q);
      Q.front++;
    }



    //在这里记录对于当前节点的pi和residue
    
    //1. check一下有多少节点是需要推入的，同时记录index
    //2. 新开一个这么多节点的空间
    //3. 把需要推入的节点推入
    //4. 把之前的长度为g.n的空间销毁
    //5. 把留下的节点记录到矩阵中去（这样方便后面的有序优化查询，不用开一个g.n的空间对应每个节点对应的cluster，
    //同时应该可以用insertBackUncompressed，这样省时间很多）
    // for (int i = 0; i < g->n; ++i){
    //   if(pi[i] > reservemin){
    //     answer->push_back(Triplet<double>(i, src, pi[i]));
    //     answer->push_back(Triplet<double>(src, i, pi[i]));
    //     all_count += 1;
    //   }
    // }

    // for (int i = 0; i < g->n; ++i){
    //   if(pi[i] > reservemin){
    //     answer->push_back(Triplet<double>(i, src, pi[i]));
    //     answer->push_back(Triplet<double>(src, i, pi[i]));
    //     all_count += 1;
    //   }
    // }

    for(int i = 0; i < Q.rear; i++){
      int index = Q.arr[i];
      answer->push_back(Triplet<double>(index, src, pi[index]));
      answer->push_back(Triplet<double>(src, index, pi[index]));
      all_count += 1;
    }

    // double temp_sum = 0;
    // for(int i = 0; i < g->n; i++){
    //   temp_sum += residue[i];
    //   temp_sum += pi[i];
    // }



    // cout<<"temp_sum = "<<temp_sum<<endl;

    // cout<<"g->degree[src] = "<<g->degree[src]<<endl;

    // auto once_end_time = chrono::system_clock::now();


    // auto elapsed_once_time = chrono::duration_cast<std::chrono::seconds>(once_end_time - once_start_time);
    // cout << "Once used time: "<< elapsed_once_time.count() << endl;

  }

  // cout<<"record_max = "<<record_max<<endl;

  delete[] residue;
  delete[] pi;
  delete[] flags;

  if (NULL != Q.arr)
    {
        free(Q.arr);
        Q.arr = NULL;
    }

  if (NULL != record_Q.arr)
    {
        free(record_Q.arr);
        record_Q.arr = NULL;
    }

  cout<<"End ForwardPush!"<<endl;
  return;

}



















void Random_Walk_Undirected(
int thread_number, 
long long int start, long long int end, UGraph* g, double residuemax, double reservemin, 
long long int& all_count, double alpha,
vector<map<long long int, double>>& all_csr_entries,
std::ofstream& ofs,
std::vector<long long int>& offsets,
long long int& write_out_io_all_count
){

  long long int vertices_number = g->n;

  // vector<int>queue_vec;

  // double *pi = new double[vertices_number];

  boost::archive::binary_oarchive oa(ofs, 
    boost::archive::no_header | boost::archive::no_tracking);



  double total_number = 1.0 / reservemin;

  // int STEP_SIZE_LESS_THAN = 10;
  int STEP_SIZE_LESS_THAN = 0;
  int exponential_2 = 1;
  while(exponential_2 < total_number){
    exponential_2 *= 2;
    STEP_SIZE_LESS_THAN++;
  }

  cout<<"STEP_SIZE_LESS_THAN = "<<STEP_SIZE_LESS_THAN<<endl;
  
  vector<int> steps_to_samples_number(STEP_SIZE_LESS_THAN, 0);
  vector<double> alpha_1_minus_alpha_vec(STEP_SIZE_LESS_THAN, 0);

  // double total_number = 1.0 / reservemin;
  alpha_1_minus_alpha_vec[0] = alpha;

  // cout<<0<<endl;

  for(int step_length = 0; step_length < STEP_SIZE_LESS_THAN; step_length++){
    double cur_number = total_number * alpha;
    double alpha_1_minus_alpha = alpha;
    // cout<<"alpha_1_minus_alpha = "<<alpha_1_minus_alpha<<endl;
    for(int expo = 0; expo < step_length; expo++){
      cur_number *= (1.0 - alpha);
      alpha_1_minus_alpha *= (1.0 - alpha);
    }
    steps_to_samples_number[step_length] = ceil(cur_number);
    alpha_1_minus_alpha_vec[step_length] = alpha_1_minus_alpha;
    // cout<<"steps_to_samples_number["<<step_length<<"] = "<<steps_to_samples_number[step_length]<<endl;
    // cout<<"alpha_1_minus_alpha_vec["<<step_length<<"] = "<<alpha_1_minus_alpha_vec[step_length]<<endl;
  }

  // cout<<1<<endl;

  //undirected forward( = backward?) random walk
  for(int source = start; source < end; source++){

    map<long long int, double> &pi = all_csr_entries[source];

    // cout<<"1:1"<<endl;

    if(g->degree[source] <= 0){
      continue;
    }


    // pi[source] += alpha; //step_length == 0;  
    pi[source] += alpha / reservemin; //step_length == 0;  
    
    //要求1: revisit了
    //要求2: revisit的时候residue是大于residuemax的（或者除以degree是大于residuemax的）
    for(int step_length = 1; step_length < STEP_SIZE_LESS_THAN; step_length++){
    // cout<<"1:2"<<endl;
      if(g->degree[source] <= 0){
        // pi[source] += alpha_1_minus_alpha_vec[step_length];
        pi[source] += alpha_1_minus_alpha_vec[step_length] / reservemin;
        continue;
      }

    // cout<<"1:3"<<endl;
      // double changed_value = alpha_1_minus_alpha_vec[step_length] * 1.0 / steps_to_samples_number[step_length];
      double changed_value = alpha_1_minus_alpha_vec[step_length] * 1.0 / steps_to_samples_number[step_length] / reservemin;
      // cout<<"changed_value = "<<changed_value<<endl;

    // cout<<"1:4"<<endl;
      for(int sample_index = 0; sample_index <= steps_to_samples_number[step_length]; sample_index++){
        int left_step = step_length;

        int v = source;

        while(left_step > 0){
          
          //无向图，除非是起始点，否则不会degree == 0的
          // if(g->degree[v] <= 0){
          //   continue;
          // }
    // cout<<"v = "<<v<<endl;
          // cout<<"g->degree[v] = "<<g->degree[v]<<endl;
    // cout<<"1:4:0"<<endl;
          int edge_number = rand_uniform(g->degree[v]);
          // cout<<"edge_number = "<<edge_number<<endl;
    // cout<<"1:4:1"<<endl;
          int neighbor = g->AdjList[v][edge_number];
          // cout<<"neighbor = "<<neighbor<<endl;
    // cout<<"1:4:2"<<endl;
          // if (rand_uniformf() < alpha){
          //   // v = v;
          //   pi[v] += alpha_1_minus_alpha_vec[step_length] * 1 / steps_to_samples_number[step_length];
          // }
          // else{
            v = neighbor;
            pi[v] += changed_value;
          // }
          left_step--;

        }

      }
    // cout<<"1:5"<<endl;

    }


    // for(auto& key_value: pi){
    //   // cout<<"key_value.f
      
    // }

  }


  for(int it = start; it < end; it++){
    offsets[it] = ofs.tellp();

    oa << all_csr_entries[it];

    all_count += all_csr_entries[it].size();
  }


}






void Random_Walk_Directed(
int thread_number, 
long long int start, long long int end, Graph* g, double residuemax, double reservemin, 
long long int& all_count, double alpha,
vector<map<long long int, double>>& all_csr_entries,
std::ofstream& ofs,
std::vector<long long int>& offsets,
long long int& write_out_io_all_count
){

  long long int vertices_number = g->n;

  // vector<int>queue_vec;

  // double *pi = new double[vertices_number];

  boost::archive::binary_oarchive oa(ofs, 
    boost::archive::no_header | boost::archive::no_tracking);



  double total_number = 1.0 / reservemin;

  // int STEP_SIZE_LESS_THAN = 10;
  int STEP_SIZE_LESS_THAN = 0;
  int exponential_2 = 1;
  while(exponential_2 < total_number){
    exponential_2 *= 2;
    STEP_SIZE_LESS_THAN++;
  }

  cout<<"STEP_SIZE_LESS_THAN = "<<STEP_SIZE_LESS_THAN<<endl;
  
  vector<int> steps_to_samples_number(STEP_SIZE_LESS_THAN, 0);
  vector<double> alpha_1_minus_alpha_vec(STEP_SIZE_LESS_THAN, 0);

  // double total_number = 1.0 / reservemin;
  alpha_1_minus_alpha_vec[0] = alpha;

  // cout<<0<<endl;

  for(int step_length = 0; step_length < STEP_SIZE_LESS_THAN; step_length++){
    double cur_number = total_number * alpha;
    double alpha_1_minus_alpha = alpha;
    // cout<<"alpha_1_minus_alpha = "<<alpha_1_minus_alpha<<endl;
    for(int expo = 0; expo < step_length; expo++){
      cur_number *= (1.0 - alpha);
      alpha_1_minus_alpha *= (1.0 - alpha);
    }
    steps_to_samples_number[step_length] = ceil(cur_number);
    alpha_1_minus_alpha_vec[step_length] = alpha_1_minus_alpha;
    // cout<<"steps_to_samples_number["<<step_length<<"] = "<<steps_to_samples_number[step_length]<<endl;
    // cout<<"alpha_1_minus_alpha_vec["<<step_length<<"] = "<<alpha_1_minus_alpha_vec[step_length]<<endl;
  }

  // cout<<1<<endl;

  //undirected forward( = backward?) random walk
  for(int source = start; source < end; source++){

    map<long long int, double> &pi = all_csr_entries[source];

    // cout<<"1:1"<<endl;

    if(g->indegree[source] <= 0){
      continue;
    }


    // pi[source] += alpha; //step_length == 0;  
    pi[source] += alpha / reservemin; //step_length == 0;  
    
    //要求1: revisit了
    //要求2: revisit的时候residue是大于residuemax的（或者除以degree是大于residuemax的）
    for(int step_length = 1; step_length < STEP_SIZE_LESS_THAN; step_length++){
    // cout<<"1:2"<<endl;
      if(g->indegree[source] <= 0){
        // pi[source] += alpha_1_minus_alpha_vec[step_length];
        pi[source] += alpha_1_minus_alpha_vec[step_length] / reservemin;
        continue;
      }

    // cout<<"1:3"<<endl;
      // double changed_value = alpha_1_minus_alpha_vec[step_length] * 1.0 / steps_to_samples_number[step_length];
      double changed_value = alpha_1_minus_alpha_vec[step_length] * 1.0 / steps_to_samples_number[step_length] / reservemin;
      // cout<<"changed_value = "<<changed_value<<endl;

    // cout<<"1:4"<<endl;
      for(int sample_index = 0; sample_index <= steps_to_samples_number[step_length]; sample_index++){
        int left_step = step_length;

        int v = source;

        while(left_step > 0){
          
          //无向图，除非是起始点，否则不会degree == 0的
          // if(g->degree[v] <= 0){
          //   continue;
          // }
    // cout<<"v = "<<v<<endl;
          // cout<<"g->degree[v] = "<<g->degree[v]<<endl;
    // cout<<"1:4:0"<<endl;
          int edge_number = rand_uniform(g->indegree[v]);
          // cout<<"edge_number = "<<edge_number<<endl;
    // cout<<"1:4:1"<<endl;
          int neighbor = g->inAdjList[v][edge_number];
          // cout<<"neighbor = "<<neighbor<<endl;
    // cout<<"1:4:2"<<endl;
          // if (rand_uniformf() < alpha){
          //   // v = v;
          //   pi[v] += alpha_1_minus_alpha_vec[step_length] * 1 / steps_to_samples_number[step_length];
          // }
          // else{
            v = neighbor;
            pi[v] += changed_value;
          // }
          left_step--;

        }

      }
    // cout<<"1:5"<<endl;

    }


    // for(auto& key_value: pi){
    //   // cout<<"key_value.f
      
    // }

  }


  for(int it = start; it < end; it++){
    offsets[it] = ofs.tellp();

    oa << all_csr_entries[it];

    all_count += all_csr_entries[it].size();
  }


}