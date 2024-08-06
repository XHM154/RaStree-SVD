extern "C"
{
#include "matrix_vector_functions_intel_mkl.h"
#include "matrix_vector_functions_intel_mkl_ext.h"
#include "string.h"
}

#undef max
#undef min

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
// #include "my_queue.h"


#include<assert.h>
#include <unordered_map>
#include<cmath>

#include<list>

#include<memory.h>

#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>



//#include "metis_test_csdn.h"

using namespace Eigen;

using namespace std;



double sqr(double x)
{return x * x;}


// int find_top_k_index(vector<int>& nums1, vector<int>& nums2, int k, 
int find_top_k_index(vector<double>& nums1, vector<double>& nums2, int k, 
                int nums1_begin, int nums1_end, int nums2_begin, int nums2_end){
    if(nums1_end < nums1_begin){
        return (-1) * (nums2_begin + k - 1 + 1);
    }
    if(nums2_end < nums2_begin){
        return nums1_begin + k - 1 + 1;
    }
    if(k == 1){
        return nums1[nums1_begin] > nums2[nums2_begin]? 
            (-1 * (nums2_begin + 1) ) : (nums1_begin + 1);
    }
    int half_k = k / 2;
    int next_k = 0;
    if( half_k > min((nums1_end - nums1_begin + 1), (nums2_end - nums2_begin + 1)) ){
        half_k = min((nums1_end - nums1_begin + 1), (nums2_end - nums2_begin + 1));
    }
    
    int nums1_num = nums1[nums1_begin + half_k - 1];
    int nums2_num = nums2[nums2_begin + half_k - 1];
    
    if(nums1_num < nums2_num){
        nums1_begin = nums1_begin + half_k - 1 + 1;
        nums1_end = nums1.size() - 1;
        next_k = k - half_k;
    }
    else{            
        nums2_begin = nums2_begin + half_k - 1 + 1;
        nums2_end = nums2.size() - 1;
        next_k = k - half_k;
    }
    
    return find_top_k_index(nums1, nums2, next_k, 
                        nums1_begin, nums1_end, nums2_begin, nums2_end);        
}
    

// void take_out_index12(vector<int>& nums1, vector<int>& nums2, int top_d, vector<int>& result){
void take_out_index12(vector<double>& nums1, vector<double>& nums2, int top_d, vector<int>& result){

    if(top_d == 0)
    {
      result.clear();
      result.push_back(0);
      result.push_back(0);
      return;
    }

    int nums1_size = nums1.size();
    int nums2_size = nums2.size();

    double n1_sum = 0;
    double n2_sum = 0;
    for(int i = 0; i < nums1_size; ++i)
      n1_sum += nums1[i];
    
    for(int i = 0; i < nums2_size; ++i)
      n2_sum += nums2[i];

    // printf("n1[0] = %.12lf  n1[size] = %.12lf\n", nums1[0], nums1[nums1_size-1]);
    // printf("n2[0] = %.12lf  n2[size] = %.12lf\n", nums2[0], nums2[nums2_size-1]);

    // printf("n1_sum = %.12lf\nn2_sum = %.12lf\n", n1_sum, n2_sum);

    int nums_all_size = nums1_size + nums2_size;

    int index = find_top_k_index(nums1, nums2, top_d, 
                        0, nums1.size() - 1, 0, nums2.size() - 1);
    
    // cout<<"index = "<<index<<endl;

    int index1, index2;

    if(index >= 0){
        index1 = index;
        index2 = top_d - index;
    }
    else{
        index1 = top_d + index;
        index2 = -1 * index;
    }

    result.push_back(index1);
    result.push_back(index2);
}




void take_out_index_OfSons
(vector<vector<double>>& low_approx_value_vec, vector<int>& son_node_list, 
vector<int>& factorization_d_vec, vector<int>& result, int number_of_son, int top_d
){
  for(int i = 0; i < number_of_son; ++i)
    result[i] = 0;
  for(int i = 0; i < top_d; ++i)
  {
    double val = -1;
    int po = -1;
    for(int j = 0; j < number_of_son; ++j)
      {
        int x = son_node_list[j];
        if(result[j] < factorization_d_vec[x] && (po == -1 || low_approx_value_vec[x][result[j]] < val))
        {
          val = low_approx_value_vec[x][result[j]];
          po = j;
        }
      }
    ++result[po];
  }
}




void parallel_resize_vectors(
long long int start,
long long int end,
vector< vector<std::map<long long int, double>> > & Partition_thread_all_csr_entries,
// vector< vector<unordered_map<long long int, double>> > & Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries,

vector< vector<third_float_map> > & Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries,

vector< long long int > & Partition_all_csr_offsets,
vector< long long int > & Partition_dynamic_ppr_offsets,
long long int group_size,
int stored_interval,
std::ofstream& ofs_debug
){
  

  cout<<"start = "<<start<<endl;
  cout<<"end = "<<end<<endl;

  // ofs_debug<<"start = "<<start<<endl;
  // ofs_debug<<"end = "<<end<<endl;

  if((end - start) > stored_interval){

    // ofs_debug<<"(end - start) > stored_interval"<<endl;
    cout<<"(end - start) > stored_interval"<<endl;

    // ofs_debug<<"(end - start) / stored_interval = "<<(end - start) / stored_interval<<endl;
    cout<<"(end - start) / stored_interval = "<<(end - start) / stored_interval <<endl;


    Partition_thread_all_csr_entries.resize( (end - start) / stored_interval );
    Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries.resize( (end - start) / stored_interval );

    Partition_all_csr_offsets.resize( (end - start) / stored_interval );
    Partition_dynamic_ppr_offsets.resize( (end - start) / stored_interval );

    for(int j = 0; j < (end - start) / stored_interval ; j++){

      if(j != ( (end - start) / stored_interval - 1 ) ){
        Partition_thread_all_csr_entries[j].resize( stored_interval );
        Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[j].resize( 4 * stored_interval );

        // ofs_debug<<"j = "<<j<<" , stored_interval = "<<stored_interval<<endl;
        // cout<<"j = "<<j<<" , stored_interval = "<<stored_interval<<endl;
      }
      else{
        Partition_thread_all_csr_entries[j].resize( end - start - (end - start) / stored_interval * stored_interval + stored_interval );
        Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[j].resize( 4 * (end - start - (end - start) / stored_interval * stored_interval + stored_interval) );
        // ofs_debug<<"j = "<<j<<" ,end - start - (end - start) / stored_interval * stored_interval + stored_interval = "<<end - start - (end - start) / stored_interval * stored_interval + stored_interval<<endl;
        cout<<"j = "<<j<<" ,end - start - (end - start) / stored_interval * stored_interval + stored_interval = "
          <<end - start - (end - start) / stored_interval * stored_interval + stored_interval<<endl;
      }
      
    }

  }
  else{

      Partition_thread_all_csr_entries.resize( 1 );
      Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries.resize( 1 );

      Partition_all_csr_offsets.resize( 1 );
      Partition_dynamic_ppr_offsets.resize( 1 );

      Partition_thread_all_csr_entries[0].resize( end - start );
      Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[0].resize( 4 * (end - start) );

      cout<<"end - start = "<<end - start<<endl;
  }


  // cout<<"start = "<<start<<endl;
  // cout<<"end = "<<end<<endl;

  cout<<"end resize function"<<endl;
  
  // ofs_debug<<"finish: start = "<<start<<endl;
  // ofs_debug<<"finish: end = "<<end<<endl;
  // ofs_debug<<endl;
  
}






























//first memory optimized
void sparse_sub_svd_function_take_out_singular_values(
vector<double>& low_approx_value_vec,  
int d, int pass, 
mat* matrix_vec_t,
long long int vertex_number,
vector<map<long long int, double>>& all_csr_entries,
long long int start_column,
long long int group_size,
long long int nnz,
vector<long long int>& all_count,
int NUMTHREAD,
double *ppr_retain_time = NULL
){

cout<<"sparse matrix: 1"<<endl;


long long int submatrix_rows = vertex_number;
long long int submatrix_cols = group_size;


// cout<<"submatrix_cols = "<<submatrix_cols<<endl;
// cout<<"submatrix_rows = "<<submatrix_rows<<endl;

//   cout<<"After nnz = "<<nnz<<endl;

  auto hash_coo_time = chrono::system_clock::now();

cout<<"sparse matrix: 4"<<endl;


// cout<<"submatrix_cols = "<<submatrix_cols<<endl;
// cout<<"submatrix_rows = "<<submatrix_rows<<endl;
// cout<<"nnz = "<<nnz<<endl;

  mat_coo *ppr_matrix_coo = coo_matrix_new(submatrix_cols, submatrix_rows, nnz);

  ppr_matrix_coo->nnz = nnz;

cout<<"sparse matrix: 5"<<endl;




    vector<thread> threads;

    for (int t = 1; t <= NUMTHREAD; t++){

      long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
      long long int end = 0;
      if (t == NUMTHREAD){
        end = start_column + group_size;
      } else{
        end = start_column + t*(group_size/NUMTHREAD);
      }

      threads.push_back(thread(parallel_store_matrix, 
      std::ref(all_csr_entries),
      ppr_matrix_coo, 
      start,
      end,
      start_column,
      all_count[t-1] ));

    }

    for (int t = 0; t < NUMTHREAD ; t++){
      threads[t].join();
    }
    vector<thread>().swap(threads);













cout<<"sparse matrix: 8"<<endl;

  mat_csr* ppr_matrix = csr_matrix_new();
  csr_init_from_coo(ppr_matrix, ppr_matrix_coo);

cout<<"sparse matrix: 9"<<endl;

  auto coo_csr_time = chrono::system_clock::now();
  auto elapsed_sparse_coo_time = chrono::duration<double>(coo_csr_time- hash_coo_time);

  if(ppr_retain_time != NULL)
    *ppr_retain_time += elapsed_sparse_coo_time.count();

  mat *U = matrix_new(submatrix_cols, d);
  mat *S = matrix_new(d, 1);
  mat *V = matrix_new(submatrix_rows, d);

cout<<"sparse matrix: 10"<<endl;

  cout<<"out:U->nrows = "<<U->nrows<<endl;

  // frPCAt(ppr_matrix, &matrix_vec_t, &S, &V, d, pass);
  frPCA(ppr_matrix, &U, &S, &V, d, pass);

  // cout<<"out:V->nrows = "<<V->nrows<<endl;
  // cout<<"out:V->ncols = "<<V->ncols<<endl;

  // double temp_err = get_matrix_frobenius_A_minus_AVVT(ppr_matrix_coo, ppr_matrix, V);
  // cout << fixed << setprecision(12) << temp_err << endl;
  
  coo_matrix_delete(ppr_matrix_coo);
  ppr_matrix_coo = NULL;

cout<<"sparse matrix: 10:2"<<endl;
  csr_matrix_delete(ppr_matrix);
  ppr_matrix = NULL;

cout<<"sparse matrix: 11"<<endl;

  mat * S_full = matrix_new(d, d);

  for(int i = 0; i < d; i++){
    // cout<<"matrix_get_element(S, "<<i<<", 0) = "<<matrix_get_element(S, i, 0)<<endl;
    matrix_set_element(S_full, i, i, matrix_get_element(S, i, 0));
  }

  MKL_INT Nrows = matrix_vec_t->nrows;

  for(int i = 0; i < d; i++){
    // cout<<"matrix_get_element(S, "<<i<<", 0) = "<<matrix_get_element(S, i, 0)<<endl;
    double cur_element = matrix_get_element(S, i, 0);
    low_approx_value_vec[i] = cur_element;
    // matrix_set_element(S_full, i, i, cur_element);
    // MKL_INT st_pos = Nrows * i;
    // for(int j = 0; j < Nrows; ++j)
    // matrix_vec_t->d[st_pos + j] = V->d[st_pos + j] * cur_element;
  }


  cout<<"V->nrows = "<<V->nrows<<endl;
  cout<<"V->ncols = "<<V->ncols<<endl;

  // cout<<"S_full->nrows = "<<S_full->nrows<<endl;
  // cout<<"S_full->ncols = "<<S_full->ncols<<endl;

  cout<<"matrix_vec_t->nrows = "<<matrix_vec_t->nrows<<endl;
  cout<<"matrix_vec_t->ncols = "<<matrix_vec_t->ncols<<endl;


  cout<<"sparse matrix: 11:1"<<endl;

  matrix_matrix_mult(V, S_full, matrix_vec_t);

  
  cout<<"sparse matrix: 11:2"<<endl;

  // matrix_matrix_mult(U, S_full, matrix_vec_t);
  // second_pointer_matrix_matrix_mult(&U, &S_full, &matrix_vec_t);

  // for(int i = 0; i < d; i++){
  //   cout<<"matrix_get_element(US, "<<i<<", 0) = "<<matrix_get_element(matrix_vec_t, i, i)<<endl;
  // }

  matrix_delete(V);
  V = NULL;
    
  matrix_delete(U);
  U = NULL;
  // matrix_matrix_mult(matrix_vec_t, S_full, matrix_vec_t);

cout<<"sparse matrix: 12"<<endl;  

  matrix_delete(S);

  matrix_delete(S_full);
  S = NULL;
  S_full = NULL;
  
cout<<"sparse matrix: 13"<<endl;


cout<<"sparse matrix: 14"<<endl;


}






//first memory optimized Only sparse matrix got (For Claim)
void sparse_sub_svd_function_take_out_sparse_matrix(
mat_csr ** sparse_mat,
long long int vertex_number,
vector<map<long long int, double>>& all_csr_entries,
long long int start_column,
long long int group_size,
long long int nnz,
vector<long long int>& all_count,
int NUMTHREAD,
double *ppr_retain_time = NULL
){

cout<<"sparse matrix: 1"<<endl;


long long int submatrix_rows = vertex_number;
long long int submatrix_cols = group_size;


// cout<<"submatrix_cols = "<<submatrix_cols<<endl;
// cout<<"submatrix_rows = "<<submatrix_rows<<endl;

//   cout<<"After nnz = "<<nnz<<endl;

  auto hash_coo_time = chrono::system_clock::now();

cout<<"sparse matrix: 4"<<endl;


// cout<<"submatrix_cols = "<<submatrix_cols<<endl;
// cout<<"submatrix_rows = "<<submatrix_rows<<endl;
// cout<<"nnz = "<<nnz<<endl;

  mat_coo *ppr_matrix_coo = coo_matrix_new(submatrix_cols, submatrix_rows, nnz);

  ppr_matrix_coo->nnz = nnz;

cout<<"sparse matrix: 5"<<endl;




    vector<thread> threads;

    for (int t = 1; t <= NUMTHREAD; t++){

      long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
      long long int end = 0;
      if (t == NUMTHREAD){
        end = start_column + group_size;
      } else{
        end = start_column + t*(group_size/NUMTHREAD);
      }

      threads.push_back(thread(parallel_store_matrix, 
      std::ref(all_csr_entries),
      ppr_matrix_coo, 
      start,
      end,
      start_column,
      all_count[t-1] ));

    }

    for (int t = 0; t < NUMTHREAD ; t++){
      threads[t].join();
    }
    vector<thread>().swap(threads);

    malloc_trim(0);


cout<<"sparse matrix: 8"<<endl;

  mat_csr* ppr_matrix = csr_matrix_new();
  csr_init_from_coo(ppr_matrix, ppr_matrix_coo);

  *sparse_mat = ppr_matrix;

cout<<"sparse matrix: 9"<<endl;

  auto coo_csr_time = chrono::system_clock::now();
  auto elapsed_sparse_coo_time = chrono::duration<double>(coo_csr_time- hash_coo_time);

  if(ppr_retain_time != NULL)
    *ppr_retain_time += elapsed_sparse_coo_time.count();

  coo_matrix_delete(ppr_matrix_coo);
  ppr_matrix_coo = NULL;


}




void Erase_row_col_nonzeros(mat_coo* A, pair<double,pair<double, double>> remain_para, double* snapnum_col, double* snapnum_row)
{
  long long now_nnz = 0;
  for(long long i = 0; i < A->nnz; ++i)
  if(max(snapnum_col[A->cols[i] - 1], snapnum_row[A->rows[i] - 1]) < remain_para.first)
  {
    A->values[now_nnz] = A->values[i];
    A->rows[now_nnz] = A->rows[i];
    A->cols[now_nnz] = A->cols[i];
    ++now_nnz;
  }

  A->nnz = now_nnz;
  cout << now_nnz << endl;

}



void Erase_row_col_nonzeros(mat_coo_2* A, pair<double,pair<double, double>> remain_para, double* snapnum_col, double* snapnum_row)
{
  long long now_nnz = 0;
  for(long long i = 0; i < A->nnz; ++i)
  if(max(snapnum_col[A->cols[i] - 1], snapnum_row[A->rows[i] - 1]) < remain_para.first)
  {
    // if(i>A->nnz - 10) cout <<i << " " << A->nnz << endl; 
    A->values[now_nnz] = A->values[i];
    A->rows[now_nnz] = A->rows[i];
    A->cols[now_nnz] = A->cols[i];
    ++now_nnz;
  }

  A->nnz = now_nnz;
  cout << now_nnz << endl;

}




void sparse_sub_svd_function_take_out_sparse_matrix_preserve_someofelements(
mat_csr ** sparse_mat,
long long int vertex_number,
vector<map<long long int, double>>& all_csr_entries,
long long int start_column,
long long int group_size,
long long int nnz,
vector<long long int>& all_count,
int NUMTHREAD,
pair<double, pair<double, double>> remain_para,
int random_seed,
int compute_fnorm_sig, int erase_way,
vector<double>& Frobeniusnorm_per_snapshots,
int** ppr_col,
double** ppr_nnz,
// double* Frobenius_norm_per_col = NULL,
double* snapnum_columns = NULL,
double* snapnum_rows = NULL,
// double* Frobenius_norm_per_row = NULL,
double *ppr_retain_time = NULL
){

cout<<"sparse matrix: 1"<<endl;


long long int submatrix_rows = vertex_number;
long long int submatrix_cols = group_size;


// cout<<"submatrix_cols = "<<submatrix_cols<<endl;
// cout<<"submatrix_rows = "<<submatrix_rows<<endl;

//   cout<<"After nnz = "<<nnz<<endl;

  auto hash_coo_time = chrono::system_clock::now();

cout<<"sparse matrix: 4"<<endl;


// cout<<"submatrix_cols = "<<submatrix_cols<<endl;
// cout<<"submatrix_rows = "<<submatrix_rows<<endl;
cout<<"nnz = "<<nnz<<endl;

  mat_coo_2 *ppr_matrix_coo = coo_2_matrix_new(submatrix_cols, submatrix_rows, nnz);

  ppr_matrix_coo->nnz = nnz;

cout<<"sparse matrix: 5"<<endl;




    vector<thread> threads;

    for(int uuu = 0; uuu < 2; ++uuu)
    {
      if(uuu == 0) ppr_matrix_coo->values = new double[nnz];
      else {
        ppr_matrix_coo->rows = new int[nnz];
        ppr_matrix_coo->cols = new long long[nnz];
      }

      for (int t = 1; t <= NUMTHREAD; t++)
      {
        long long int start = start_column + (t - 1) * (group_size / NUMTHREAD);
        long long int end = 0;
        if (t == NUMTHREAD){
          end = start_column + group_size;
        } else{
          end = start_column + t * (group_size / NUMTHREAD);
        }
        if(uuu == 0)
          threads.push_back(thread(parallel_store_matrix_less_space_nnz, 
            std::ref(all_csr_entries),
            ppr_matrix_coo, 
            start,
            end,
            start_column,
            all_count[t-1],
            ppr_col, ppr_nnz ));
        else
          threads.push_back(thread(parallel_store_matrix_less_space_rows_cols, 
            std::ref(all_csr_entries),
            ppr_matrix_coo, 
            start,
            end,
            start_column,
            all_count[t-1],
            ppr_col));

      }

      for (int t = 0; t < NUMTHREAD ; t++){
        threads[t].join();
      }

      vector<thread>().swap(threads);
      malloc_trim(0);
    }


    if(compute_fnorm_sig > 0)
    {
      int* cnt = new int[16]{0};
      double LowerBound = remain_para.second.first;
      double UpperBound = remain_para.second.second - LowerBound;
      // double bb = UpperBound - sqrt((UpperBound - remain_para.first) * (UpperBound - LowerBound));
      double bb = sqrt(remain_para.first * UpperBound);
      cout << "bb = === " << bb << endl;
      for(long long i = 0; i < ppr_matrix_coo->nnz; ++i)
      {
        // double tsqr = sqr(ppr_matrix_coo->values[i]);
        int col_num = ppr_matrix_coo->cols[i] - 1;
        double temp = max(snapnum_columns[ppr_matrix_coo->cols[i] - 1], snapnum_rows[ppr_matrix_coo->rows[i] - 1]);
        // double temp = min(snapnum_columns[ppr_matrix_coo->cols[i] - 1], snapnum_rows[ppr_matrix_coo->rows[i] - 1]);
        // Frobenius_norm_per_col[col_num] += tsqr;

        // int snap_num = (temp < remain_para.first) ? 0 : (1 + temp - remain_para.first);
        int snap_num = 0;
        if(temp >= bb) snap_num = sqr(temp - LowerBound) / UpperBound + LowerBound 
                        - remain_para.first + 1;
                        // cout <<temp << ' '<< snap_num << endl;
        // if(temp >= bb) snap_num = UpperBound - remain_para.first 
        //                 - sqr(UpperBound - temp) / (UpperBound - LowerBound) + 1;

        Frobeniusnorm_per_snapshots[snap_num] += sqr(ppr_matrix_coo->values[i]);
        ++cnt[snap_num];
      }

      remain_para.first = bb;

      delete[] cnt;
    }


cout<<"sparse matrix: 6"<<endl;

  if(erase_way > 0)
    Erase_row_col_nonzeros(ppr_matrix_coo, remain_para, snapnum_columns, snapnum_rows);

cout << "erase_way number = " << erase_way << endl;
cout<<"sparse matrix: 8"<<endl;

  mat_csr* ppr_matrix = csr_matrix_new();
  csr_init_from_coo_2(ppr_matrix, ppr_matrix_coo);

  *sparse_mat = ppr_matrix;

cout<<"sparse matrix: 9"<<endl;

  auto coo_csr_time = chrono::system_clock::now();
  auto elapsed_sparse_coo_time = chrono::duration<double>(coo_csr_time - hash_coo_time);

  if(ppr_retain_time != NULL)
    *ppr_retain_time += elapsed_sparse_coo_time.count();

  coo_2_matrix_delete(ppr_matrix_coo);
  ppr_matrix_coo = NULL;


}








void Read_image_Matrix(ifstream& infile, mat_coo* A)
{
  // int a,b,c;
  for(long long i = 0; i < A->nnz; ++i)
    infile >> A->rows[i] >> A->cols[i] >> A->values[i];
}








void sparse_sub_svd_function_take_out_sparse_image_matrix_preserve_someofelements(
mat_csr ** sparse_mat,
long long int item_number,
ifstream& ifs_vec,
vector<map<long long int, double>>& all_csr_entries,
long long int start_column,
long long int group_size,
long long int nnz,
vector<long long int>& all_count,
int NUMTHREAD,
pair<double, pair<double, double>> remain_para,
int random_seed,
int compute_fnorm_sig, int erase_way,
vector<double>& Frobeniusnorm_per_snapshots,
mat_coo** cache_matrix,
double* snapnum_columns = NULL,
double* snapnum_rows = NULL,
double *ppr_retain_time = NULL
){

cout<<"sparse matrix: 1"<<endl;


long long int submatrix_rows = item_number;
long long int submatrix_cols = group_size;



  auto hash_coo_time = chrono::system_clock::now();

cout<<"sparse matrix: 4"<<endl;



  mat_coo *ppr_matrix_coo = coo_matrix_new(submatrix_cols, submatrix_rows, nnz);

  ppr_matrix_coo->nnz = nnz;

cout<<"sparse matrix: 5"<<endl;
    


    if(cache_matrix != NULL) 
    {
      if(*cache_matrix == NULL)
      {
      // cout << "2\n";
        (*cache_matrix) = ppr_matrix_coo;
        Read_image_Matrix(std::ref(ifs_vec), (*cache_matrix));
        ppr_matrix_coo = coo_matrix_new(submatrix_cols, submatrix_rows, nnz);
        ppr_matrix_coo->nnz = nnz;
      } // else cout << "sparse matrix initilaized from the cache matrix in memory.\n";

      assert(ppr_matrix_coo->nnz == (*cache_matrix)->nnz);
      assert(ppr_matrix_coo->nrows == (*cache_matrix)->nrows);
      assert(ppr_matrix_coo->ncols == (*cache_matrix)->ncols);

      for(long long u = 0; u < nnz; ++u)
        ppr_matrix_coo->rows[u] = (*cache_matrix)->rows[u];
      for(long long u = 0; u < nnz; ++u)
        ppr_matrix_coo->cols[u] = (*cache_matrix)->cols[u];
      for(long long u = 0; u < nnz; ++u)
        ppr_matrix_coo->values[u] = (*cache_matrix)->values[u];

    } else 
      Read_image_Matrix(std::ref(ifs_vec), ppr_matrix_coo);
    
    for(long long ii = 0; ii < ppr_matrix_coo->nnz; ++ii)
      ppr_matrix_coo->rows[ii] -= (start_column);

    malloc_trim(0);



    double keep_tot_fnorm = -1;

    if(compute_fnorm_sig != 0)
    {
      int* cnt = new int[16]{0};
      double LowerBound = remain_para.second.first;
      double UpperBound = remain_para.second.second - LowerBound;
      // double bb = UpperBound - sqrt((UpperBound - remain_para.first) * (UpperBound - LowerBound));
      double bb = sqrt(remain_para.first * UpperBound);
      // cout << "bb = === " << bb << endl;
      for(long long i = 0; i < ppr_matrix_coo->nnz; ++i)
      {
        // double tsqr = sqr(ppr_matrix_coo->values[i]);
        int col_num = ppr_matrix_coo->cols[i] - 1;
        double temp = max(snapnum_columns[ppr_matrix_coo->cols[i] - 1], snapnum_rows[ppr_matrix_coo->rows[i] - 1]);
        // double temp = min(snapnum_columns[ppr_matrix_coo->cols[i] - 1], snapnum_rows[ppr_matrix_coo->rows[i] - 1]);
        // Frobenius_norm_per_col[col_num] += tsqr;

        // int snap_num = (temp < remain_para.first) ? 0 : (1 + temp - remain_para.first);
        int snap_num = 0;
        if(temp >= bb) snap_num = sqr(temp - LowerBound) / UpperBound + LowerBound 
                        - remain_para.first + 1;
                        // cout <<temp << ' '<< snap_num << endl;
        // if(temp >= bb) snap_num = UpperBound - remain_para.first 
        //                 - sqr(UpperBound - temp) / (UpperBound - LowerBound) + 1;

        Frobeniusnorm_per_snapshots[snap_num] += sqr(ppr_matrix_coo->values[i]);
        ++cnt[snap_num];
      }

      remain_para.first = bb;


      delete[] cnt;
    }

cout<<"sparse matrix: 6"<<endl;


  if(erase_way > 0)
    Erase_row_col_nonzeros(ppr_matrix_coo, remain_para, snapnum_columns, snapnum_rows);

cout << "erase_way number = " << erase_way << endl;
cout<<"sparse matrix: 8"<<endl;

  mat_csr* ppr_matrix = csr_matrix_new();
  csr_init_from_coo(ppr_matrix, ppr_matrix_coo);

  *sparse_mat = ppr_matrix;

cout<<"sparse matrix: 9"<<endl;

  auto coo_csr_time = chrono::system_clock::now();
  auto elapsed_sparse_coo_time = chrono::duration<double>(coo_csr_time - hash_coo_time);

  if(ppr_retain_time != NULL)
    *ppr_retain_time += elapsed_sparse_coo_time.count();

  coo_matrix_delete(ppr_matrix_coo);
  ppr_matrix_coo = NULL;


}













//first memory optimized only frpca on sparse matrix (For Claim)
void sparse_matrix_take_out_singular_values(
vector<double>& low_approx_value_vec,  
int d, int pass, 
mat* matrix_vec_t,
mat_csr* ppr_matrix,
long long int vertex_number,
long long int nnz
){

long long int submatrix_rows = ppr_matrix->ncols;
long long int submatrix_cols = ppr_matrix->nrows;


  mat *U = matrix_new(submatrix_cols, d);
  mat *S = matrix_new(d, 1);
  mat *V = matrix_new(submatrix_rows, d);

cout<<"sparse matrix: 10"<<endl;

  cout<<"out:U->nrows = "<<U->nrows<<endl;

  cout << "nrows:" << ppr_matrix->nrows << endl;
  cout << "ncols:" << ppr_matrix->ncols << endl;
  cout << "nnz:" << ppr_matrix->nnz << endl;


  // frPCAt(ppr_matrix, &matrix_vec_t, &S, &V, d, pass);
  frPCA(ppr_matrix, &U, &S, &V, d, pass);


cout<<"sparse matrix: 10:2"<<endl;
  csr_matrix_delete(ppr_matrix);
  ppr_matrix = NULL;

cout<<"sparse matrix: 11"<<endl;

  mat * S_full = matrix_new(d, d);

  for(int i = 0; i < d; i++){
    // cout<<"matrix_get_element(S, "<<i<<", 0) = "<<matrix_get_element(S, i, 0)<<endl;
    matrix_set_element(S_full, i, i, matrix_get_element(S, i, 0));
  }

  MKL_INT Nrows = matrix_vec_t->nrows;

  for(int i = 0; i < d; i++){
    // cout<<"matrix_get_element(S, "<<i<<", 0) = "<<matrix_get_element(S, i, 0)<<endl;
    double cur_element = matrix_get_element(S, i, 0);
    low_approx_value_vec[i] = cur_element;
    // matrix_set_element(S_full, i, i, cur_element);
    // MKL_INT st_pos = Nrows * i;
    // for(int j = 0; j < Nrows; ++j)
    // matrix_vec_t->d[st_pos + j] = V->d[st_pos + j] * cur_element;
  }


  cout<<"V->nrows = "<<V->nrows<<endl;
  cout<<"V->ncols = "<<V->ncols<<endl;

  // cout<<"S_full->nrows = "<<S_full->nrows<<endl;
  // cout<<"S_full->ncols = "<<S_full->ncols<<endl;

  cout<<"matrix_vec_t->nrows = "<<matrix_vec_t->nrows<<endl;
  cout<<"matrix_vec_t->ncols = "<<matrix_vec_t->ncols<<endl;


  cout<<"sparse matrix: 11:1"<<endl;

  matrix_matrix_mult(V, S_full, matrix_vec_t);

  
  cout<<"sparse matrix: 11:2"<<endl;

  // matrix_matrix_mult(U, S_full, matrix_vec_t);
  // second_pointer_matrix_matrix_mult(&U, &S_full, &matrix_vec_t);

  // for(int i = 0; i < d; i++){
  //   cout<<"matrix_get_element(US, "<<i<<", 0) = "<<matrix_get_element(matrix_vec_t, i, i)<<endl;
  // }

  matrix_delete(V);
  V = NULL;
    
  matrix_delete(U);
  U = NULL;
  // matrix_matrix_mult(matrix_vec_t, S_full, matrix_vec_t);

cout<<"sparse matrix: 12"<<endl;  

  matrix_delete(S);

  matrix_delete(S_full);
  S = NULL;
  S_full = NULL;
  
cout<<"sparse matrix: 13"<<endl;


cout<<"sparse matrix: 14"<<endl;


}






































































//first memory optimized
void sparse_sub_svd_function(int d, int pass, 
mat* matrix_vec_t,
long long int vertex_number,
vector<map<long long int, double>>& all_csr_entries,
long long int start_column,
long long int group_size,
long long int nnz,
vector<long long int>& all_count,
int NUMTHREAD
){

cout<<"sparse matrix: 1"<<endl;


long long int submatrix_rows = vertex_number;
long long int submatrix_cols = group_size;


  cout<<"After nnz = "<<nnz<<endl;

  auto hash_coo_time = chrono::system_clock::now();

cout<<"sparse matrix: 4"<<endl;

  mat_coo *ppr_matrix_coo = coo_matrix_new(submatrix_cols, submatrix_rows, nnz);

  ppr_matrix_coo->nnz = nnz;

cout<<"sparse matrix: 5"<<endl;




    vector<thread> threads;

    for (int t = 1; t <= NUMTHREAD; t++){

      long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
      long long int end = 0;
      if (t == NUMTHREAD){
        end = start_column + group_size;
      } else{
        end = start_column + t*(group_size/NUMTHREAD);
      }

      threads.push_back(thread(parallel_store_matrix, 
      std::ref(all_csr_entries),
      ppr_matrix_coo, 
      start,
      end,
      start_column,
      all_count[t-1] ));

    }

    for (int t = 0; t < NUMTHREAD ; t++){
      threads[t].join();
    }
    vector<thread>().swap(threads);













cout<<"sparse matrix: 8"<<endl;
  auto coo_csr_time = chrono::system_clock::now();
  auto elapsed_sparse_coo_time = chrono::duration_cast<std::chrono::seconds>(coo_csr_time- hash_coo_time);

  mat_csr* ppr_matrix = csr_matrix_new();
  csr_init_from_coo(ppr_matrix, ppr_matrix_coo);

cout<<"sparse matrix: 9"<<endl;

  coo_matrix_delete(ppr_matrix_coo);
  ppr_matrix_coo = NULL;

  mat *U = matrix_new(submatrix_cols, d);
  mat *S = matrix_new(d, 1);
  mat *V = matrix_new(submatrix_rows, d);

cout<<"sparse matrix: 10"<<endl;

  cout<<"out:U->nrows = "<<U->nrows<<endl;

  // frPCAt(ppr_matrix, &matrix_vec_t, &S, &V, d, pass);
  frPCA(ppr_matrix, &U, &S, &V, d, pass);

cout<<"sparse matrix: 10:2"<<endl;
  csr_matrix_delete(ppr_matrix);
  ppr_matrix = NULL;

cout<<"sparse matrix: 11"<<endl;

  mat * S_full = matrix_new(d, d);
  for(int i = 0; i < d; i++){
    // cout<<"matrix_get_element(S, "<<i<<", 0) = "<<matrix_get_element(S, i, 0)<<endl;
    matrix_set_element(S_full, i, i, matrix_get_element(S, i, 0));
  }

  cout<<"V->nrows = "<<V->nrows<<endl;
  cout<<"V->ncols = "<<V->ncols<<endl;

  cout<<"S_full->nrows = "<<S_full->nrows<<endl;
  cout<<"S_full->ncols = "<<S_full->ncols<<endl;

  cout<<"matrix_vec_t->nrows = "<<matrix_vec_t->nrows<<endl;
  cout<<"matrix_vec_t->ncols = "<<matrix_vec_t->ncols<<endl;


  cout<<"sparse matrix: 11:1"<<endl;

  matrix_matrix_mult(V, S_full, matrix_vec_t);

  
  cout<<"sparse matrix: 11:2"<<endl;

  // matrix_matrix_mult(U, S_full, matrix_vec_t);
  // second_pointer_matrix_matrix_mult(&U, &S_full, &matrix_vec_t);

  // for(int i = 0; i < d; i++){
  //   cout<<"matrix_get_element(US, "<<i<<", 0) = "<<matrix_get_element(matrix_vec_t, i, i)<<endl;
  // }

  matrix_delete(V);
  V = NULL;
    
  matrix_delete(U);
  U = NULL;
  // matrix_matrix_mult(matrix_vec_t, S_full, matrix_vec_t);

cout<<"sparse matrix: 12"<<endl;  

  matrix_delete(S);

  matrix_delete(S_full);
  S = NULL;
  S_full = NULL;
  
cout<<"sparse matrix: 13"<<endl;


cout<<"sparse matrix: 14"<<endl;


}



















//first memory optimized
void sparse_sub_svd_function_nparts_NUMTHREAD(
// double & approx_low_value,  
vector<double>& low_approx_value_vec,
int d, 
int pass, 
mat* matrix_vec_t,
long long int vertex_number,
// vector<map<long long int, double>>& all_csr_entries,

// vector< vector<map<long long int, double>> >& NUMTHREAD_all_csr_entries,
vector< vector< vector<map<long long int, double>> > > & NUMTHREAD_Partition_all_csr_entries,

long long int start_column,
long long int group_size,
long long int nnz,
vector<long long int>& all_count,
int NUMTHREAD,
int stored_interval
){

// cout<<"sparse matrix: 1"<<endl;


// cout<<"NUMTHREAD_Partition_all_csr_entries[0][0].size() = "<<NUMTHREAD_Partition_all_csr_entries[0][0][0].size()<<endl;


long long int submatrix_rows = vertex_number;
long long int submatrix_cols = group_size;


  cout<<"After nnz = "<<nnz<<endl;

  auto hash_coo_time = chrono::system_clock::now();

// cout<<"sparse matrix: 4"<<endl;

  mat_coo *ppr_matrix_coo = coo_matrix_new(submatrix_cols, submatrix_rows, nnz);

  cout<<"ppr_matrix_coo->nrows = "<<std::dec<<ppr_matrix_coo->nrows<<endl;
  cout<<"ppr_matrix_coo->ncols = "<<std::dec<<ppr_matrix_coo->ncols<<endl;

  ppr_matrix_coo->nnz = nnz;


  // for(long long int i = 0; i < 148; i++){
  //     cout<<"1:ppr_matrix_coo->rows["<<i<<"] = "<<ppr_matrix_coo->rows[i]<<endl;
  //     cout<<"1:ppr_matrix_coo->cols["<<i<<"] = "<<ppr_matrix_coo->cols[i]<<endl;
  //     cout<<endl;
  // }

cout<<"sparse matrix: 5"<<endl;

    // for(int t = 1 ; t <= NUMTHREAD; t++){
    //   cout<<"sparse_sub_svd_function_nparts_NUMTHREAD: all_count["<<t-1<<"] = "<<all_count[t-1]<<endl;
    // }


    vector<thread> threads;

    for (int t = 1; t <= NUMTHREAD; t++){

      // long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
      // long long int end = 0;
      // if (t == NUMTHREAD){
      //   end = start_column + group_size;
      // } else{
      //   end = start_column + t*(group_size/NUMTHREAD);
      // }

      // threads.push_back(thread(parallel_store_matrix, 
      // std::ref(all_csr_entries),
      // ppr_matrix_coo, 
      // start,
      // end,
      // start_column,
      // all_count[t-1] ));

      long long int start = (t-1)*(group_size/NUMTHREAD);
      long long int end = 0;
      if (t == NUMTHREAD){
        end = group_size;
      } else{
        end = t*(group_size/NUMTHREAD);
      }

      // cout<<"start = "<<start<<endl;
      // cout<<"end = "<<end<<endl;
      // cout<<endl;

      threads.push_back(thread(parallel_store_matrix_nParts_NUMTHREAD, 
      // std::ref(all_csr_entries),
      // std::ref(NUMTHREAD_all_csr_entries[t-1]),
      std::ref(NUMTHREAD_Partition_all_csr_entries[t-1]),
      ppr_matrix_coo, 
      start,
      end,
      all_count[t-1],
      t,
      stored_interval      
      ));

      // cout<<"t = "<<t<<endl;
      // for(long long int i = 0; i < 148; i++){
      //   cout<<"11:ppr_matrix_coo->rows["<<i<<"] = "<<ppr_matrix_coo->rows[i]<<endl;
      //   cout<<"11:ppr_matrix_coo->cols["<<i<<"] = "<<ppr_matrix_coo->cols[i]<<endl;
      //   cout<<endl;
      // }


      // parallel_store_matrix_nParts_NUMTHREAD(
      // // std::ref(all_csr_entries),
      // NUMTHREAD_all_csr_entries[t-1],
      // ppr_matrix_coo, 
      // start,
      // end,
      // // start_column,
      // all_count[t-1],
      // t);

      // cout<<"t = "<<t<<endl;
      // for(long long int i = 0; i < 148; i++){
      //   cout<<"12:ppr_matrix_coo->rows["<<i<<"] = "<<ppr_matrix_coo->rows[i]<<endl;
      //   cout<<"12:ppr_matrix_coo->cols["<<i<<"] = "<<ppr_matrix_coo->cols[i]<<endl;
      //   cout<<endl;
      // }

    }

    for (int t = 0; t < NUMTHREAD ; t++){
      threads[t].join();
    }
    vector<thread>().swap(threads);








  // for(long long int i = 0; i < nnz; i++){
  //     cout<<"after:ppr_matrix_coo->rows["<<i<<"] = "<<ppr_matrix_coo->rows[i]<<endl;
  //     cout<<"after:ppr_matrix_coo->cols["<<i<<"] = "<<ppr_matrix_coo->cols[i]<<endl;
  //     cout<<"after:ppr_matrix_coo->values["<<i<<"] = "<<ppr_matrix_coo->values[i]<<endl;
  //     cout<<endl;
  // }






cout<<"sparse matrix: 8"<<endl;
  auto coo_csr_time = chrono::system_clock::now();
  auto elapsed_sparse_coo_time = chrono::duration_cast<std::chrono::seconds>(coo_csr_time- hash_coo_time);

  mat_csr* ppr_matrix = csr_matrix_new();
cout<<"sparse matrix: 8:1"<<endl;


  // for(long long int i = 0; i < 148; i++){
  //   ppr_matrix_coo->rows[i] = 0;
  //   ppr_matrix_coo->cols[i] = 0;
  // }

  csr_init_from_coo(ppr_matrix, ppr_matrix_coo);
  

  // for(int i = 0; i < ppr_matrix->nnz; i++){
  //   cout<<"i = "<<i<<endl;
  //   cout<<"ppr_matrix->cols[i] = "<<ppr_matrix->cols[i]<<endl;
  //   cout<<"ppr_matrix->values[i] = "<<ppr_matrix->values[i]<<endl;
  //   cout<<endl;
  // }

  // for(int i = 0; i < ppr_matrix->nrows; i++){
  //   cout<<"ppr_matrix->pointerB[i] = "<<ppr_matrix->pointerB[i]<<endl;
  //   cout<<"ppr_matrix->pointerE[i] = "<<ppr_matrix->pointerE[i]<<endl;
  //   cout<<endl;
  // }



cout<<"sparse matrix: 9"<<endl;

  coo_matrix_delete(ppr_matrix_coo);
  ppr_matrix_coo = NULL;

  mat *U = matrix_new(submatrix_cols, d);
  mat *S = matrix_new(d, 1);
  mat *V = matrix_new(submatrix_rows, d);

cout<<"sparse matrix: 10"<<endl;



  // cout<<"out:U->nrows = "<<U->nrows<<endl;

  cout<<"U->nrows = "<<U->nrows<<endl;
  cout<<"U->ncols = "<<U->ncols<<endl;

  cout<<"S->nrows = "<<S->nrows<<endl;
  cout<<"S->ncols = "<<S->ncols<<endl;

  cout<<"V->nrows = "<<V->nrows<<endl;
  cout<<"V->ncols = "<<V->ncols<<endl;

  // frPCAt(ppr_matrix, &matrix_vec_t, &S, &V, d, pass);
  frPCA(ppr_matrix, &U, &S, &V, d, pass);

cout<<"sparse matrix: 10:2"<<endl;
  csr_matrix_delete(ppr_matrix);
  ppr_matrix = NULL;

cout<<"sparse matrix: 11"<<endl;

  mat * S_full = matrix_new(d, d);
  

cout<<"sparse matrix: 11:0"<<endl;

  for(int i = 0; i < d; i++){
    // cout<<"matrix_get_element(S, "<<i<<", 0) = "<<matrix_get_element(S, i, 0)<<endl;
    double cur_element = matrix_get_element(S, i, 0);
    low_approx_value_vec[i] = cur_element;
    matrix_set_element(S_full, i, i, cur_element);
  }


  // cout<<"matrix_get_element(S, d-1, 0) = "<<matrix_get_element(S, d-1, 0)<<endl;
  // cout<<"matrix_get_element(S, 0, 0) = "<<matrix_get_element(S, 0, 0)<<endl;
  // approx_low_value = matrix_get_element(S, d-1, 0);






  cout<<"V->nrows = "<<V->nrows<<endl;
  cout<<"V->ncols = "<<V->ncols<<endl;

  cout<<"S_full->nrows = "<<S_full->nrows<<endl;
  cout<<"S_full->ncols = "<<S_full->ncols<<endl;

  // cout<<"matrix_vec_t->nrows = "<<matrix_vec_t->nrows<<endl;
  // cout<<"matrix_vec_t->ncols = "<<matrix_vec_t->ncols<<endl;


  // cout<<"sparse matrix: 11:1"<<endl;

  matrix_matrix_mult(V, S_full, matrix_vec_t);

  
  cout<<"sparse matrix: 11:2"<<endl;

  // matrix_matrix_mult(U, S_full, matrix_vec_t);
  // second_pointer_matrix_matrix_mult(&U, &S_full, &matrix_vec_t);

  // for(int i = 0; i < d; i++){
  //   cout<<"matrix_get_element(US, "<<i<<", 0) = "<<matrix_get_element(matrix_vec_t, i, i)<<endl;
  // }

  matrix_delete(V);
  V = NULL;

  cout<<"sparse matrix: 11:3"<<endl;

  matrix_delete(U);
  U = NULL;
  // matrix_matrix_mult(matrix_vec_t, S_full, matrix_vec_t);

  cout<<"sparse matrix: 12"<<endl;  

  matrix_delete(S);
  S = NULL;
  
  cout<<"sparse matrix: 13"<<endl;

  matrix_delete(S_full);
  S_full = NULL;
  


  // cout<<"sparse matrix: 14"<<endl;


}








//second memory optimized
void dense_sub_svd_function(int d, int pass, 
// int update_i, 
mat* submatrix, 
mat* matrix_vec_t
// ,long long int vertex_number 
){

    // mat *U = matrix_new(submatrix->nrows, d);

    mat * S_full = matrix_new(d, d);

    mat *Vt = matrix_new(d, submatrix->ncols);

    truncated_singular_value_decomposition(submatrix, matrix_vec_t, S_full, Vt, d);
    
    matrix_matrix_mult(matrix_vec_t, S_full, matrix_vec_t);
    // second_pointer_matrix_matrix_mult(&matrix_vec_t, &S_full, &matrix_vec_t);

    
    // matrix_delete(U);

    matrix_delete(S_full);

    matrix_delete(Vt);

    // U = NULL;
    S_full = NULL;
    Vt = NULL;

}

void dense_sub_svd_store_function(int d, int pass, 
// int update_i, 
mat* submatrix, 
mat* matrix_vec_t,
vector<double>& low_approx_value_vec
// ,long long int vertex_number 
){

    // mat *U = matrix_new(submatrix->nrows, d);

    mat * S_full = matrix_new(d, d);

    mat *Vt = matrix_new(d, submatrix->ncols);

    truncated_singular_value_decomposition(submatrix, matrix_vec_t, S_full, Vt, d);
    
    matrix_matrix_mult(matrix_vec_t, S_full, matrix_vec_t);
    // second_pointer_matrix_matrix_mult(&matrix_vec_t, &S_full, &matrix_vec_t);

    MKL_INT Nrows = matrix_vec_t->nrows;
    for(int i = 0; i < d; ++i)
    {
        low_approx_value_vec[d - 1 - i] = matrix_get_element(S_full, i, i);
        // MKL_INT st_pos = Nrows * i;
        // for(int j = 0; j < Nrows; ++j)
        // matrix_vec_t->d[st_pos + j] *= low_approx_value_vec[d - 1 - i];
        // printf("i-th singular value: %.12lf\n", low_approx_value_vec[d - 1 - i]);
    }
      
    
    // MKL_INT Nrows = matrix_vec_t->nrows;
    MKL_INT Ncols = matrix_vec_t->ncols;
    for(MKL_INT i = 0; i < Ncols; ++i)
    if(i < Ncols - 1 - i)
    {
      MKL_INT st_i = i * Nrows;
      MKL_INT st_rev_i = (Ncols - 1 - i) * Nrows;
      for(int j = 0; j < Nrows; ++j)
      swap(matrix_vec_t->d[st_i + j], matrix_vec_t->d[st_rev_i + j]);
    } else break;

    // matrix_delete(U);

    matrix_delete(S_full);

    matrix_delete(Vt);

    // U = NULL;
    S_full = NULL;
    Vt = NULL;

}











void mkl_right_matrix_multiplication(
d_row_tree_mkl* d_row_tree,
mat* mkl_left_matrix, Eigen::MatrixXd &U_V,
long long int vertex_number,
int d,
mat* SS,
int NUMTHREAD,
int largest_level_start_index,
long long int common_group_size,
long long int final_group_size,
vector<map<long long int, double>>& all_csr_entries,
vector<long long int>& offsets,
vector<ifstream>& ifs_vec,
vector<vector<long long int>>& nparts_all_count,
vector<long long int> nnz_nparts_vec,
vector<vector<long long int>>& nparts_read_in_io_all_count
)
{
    int unique_update_times = 0;
    
    auto total_right_matrix_start_time = chrono::system_clock::now();
    
    int nParts = d_row_tree->nParts;

    double FF1 = 0, FF2 = 0;

    for(int i = 0; i < mkl_left_matrix->ncols; i++)
    {
      double cor = get_matrix_column_norm_squared(mkl_left_matrix, i);
      cor = sqrt(cor);
      for(int j = 0; j < mkl_left_matrix->nrows; j++)
        matrix_set_element(mkl_left_matrix, j ,i, matrix_get_element(mkl_left_matrix, j, i) / cor);
    }

    for(int iter = 0; iter < d_row_tree->nParts; iter++){
      int index = largest_level_start_index + iter;

      vector<long long int> &all_count = nparts_all_count[iter];

      vector<long long int> &read_in_io_all_count = nparts_read_in_io_all_count[iter];

      cout << "ppr computation " << endl;
      auto ppr_start_time = std::chrono::system_clock::now();
      vector<thread> threads;

      int order = index - largest_level_start_index;

      long long int start_column = order * common_group_size;

      long long int group_size;

      if(order == nParts - 1){
        group_size = final_group_size;
      }
      else{
        group_size = common_group_size;
      }

cout<<0<<endl;

      cout<<"group_size = "<<group_size<<endl;


      long long int nnz = nnz_nparts_vec[iter];

      long long int submatrix_rows = vertex_number;
      long long int submatrix_cols = group_size;


      cout<<"After nnz = "<<nnz<<endl;

      auto hash_coo_time = chrono::system_clock::now();

      cout<<"sparse matrix: 4"<<endl;

      // mat_coo *ppr_matrix_coo = coo_matrix_new(group_size, vertex_number, nnz);
      mat_coo *ppr_matrix_coo = coo_matrix_new(group_size, vertex_number, nnz);

      ppr_matrix_coo->nnz = nnz;





      for (int t = 1; t <= NUMTHREAD; t++){
        long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
        long long int end = 0;
        if (t == NUMTHREAD){
          end = start_column + group_size;
        } 
        else{
          end = start_column + t*(group_size/NUMTHREAD);
        }

        threads.push_back(thread(PPR_retrieve, 
        // t-1, 
        start, end, 
        // std::ref(all_count[t-1]), 
        std::ref(all_csr_entries),
        std::ref(ifs_vec[t-1]),
        std::ref(offsets),
        std::ref(read_in_io_all_count[t-1])
        ));

      }

      for (int t = 0; t < NUMTHREAD ; t++){
        threads[t].join();
      }
      vector<thread>().swap(threads);




    for (int t = 1; t <= NUMTHREAD; t++){

      long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
      long long int end = 0;
      if (t == NUMTHREAD){
        end = start_column + group_size;
      } else{
        end = start_column + t*(group_size/NUMTHREAD);
      }

      threads.push_back(thread(parallel_store_matrix, 
      std::ref(all_csr_entries),
      ppr_matrix_coo, 
      start,
      end,
      start_column,
      all_count[t-1] ));

    }

    for (int t = 0; t < NUMTHREAD ; t++){
      threads[t].join();
    }
    vector<thread>().swap(threads);




      mat_csr* ppr_matrix = csr_matrix_new();

      csr_init_from_coo(ppr_matrix, ppr_matrix_coo);

      for(MKL_INT i = 0; i < ppr_matrix_coo->nnz; ++i)
        FF1 += ppr_matrix_coo->values[i] * ppr_matrix_coo->values[i];


      cout<<"mkl_left_matrix->ncols = "<<mkl_left_matrix->ncols<<endl;

      //mkl_left_matrix->ncols == d
      mat *mkl_result_mat = matrix_new(group_size, mkl_left_matrix->ncols);

      auto right_matrix_start_time = chrono::system_clock::now();

      // csr_matrix_matrix_mult(ppr_matrix, mkl_left_matrix, mkl_result_mat);
      csr_matrix_matrix_mult(ppr_matrix, mkl_left_matrix, mkl_result_mat);

      FF2 += get_matrix_frobenius_A_minus_AVVT(ppr_matrix, mkl_left_matrix);

      coo_matrix_delete(ppr_matrix_coo);

      ppr_matrix_coo = NULL;

      csr_matrix_delete(ppr_matrix);
      ppr_matrix = NULL;

      mat *temp_mkl_result_mat = matrix_new(group_size, mkl_left_matrix->ncols);


      //This SS is already reversed
      matrix_matrix_mult(mkl_result_mat, SS, temp_mkl_result_mat);
      // second_pointer_matrix_matrix_mult(&mkl_result_mat, &SS, &temp_mkl_result_mat);

      matrix_copy(mkl_result_mat, temp_mkl_result_mat);

      cout<<"mkl_result_mat->nrows = "<<mkl_result_mat->nrows<<endl;
      cout<<"mkl_result_mat->ncols = "<<mkl_result_mat->ncols<<endl;
      

      matrix_delete(temp_mkl_result_mat);
      temp_mkl_result_mat = NULL;

      auto right_matrix_end_time = chrono::system_clock::now();
      auto elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(right_matrix_end_time - right_matrix_start_time);


      cout<<"U_V.rows() = "<<U_V.rows()<<endl;
      cout<<"U_V.cols() = "<<U_V.cols()<<endl;

      // if(iter != d_row_tree->nParts-1){
      //   for(long long int i = 0; i < group_size; i++){
      //       for(long long int j = 0; j < mkl_left_matrix->ncols; j++){
      //           U_V(iter * group_size + i, d + j) = matrix_get_element(mkl_result_mat, i, j);
      //           if(isnan(U_V(iter * group_size + i, d + j)) || isinf(U_V(iter * group_size + i, d + j))){
      //               cout<<"U_V("<<i<<", "<<j<<") = "<<U_V(i, j)<<endl;
      //           }
      //       }
      //   }
      // }
      // else{
      //   for(long long int i = 0; i < group_size; i++){
      //       for(long long int j = 0; j < mkl_left_matrix->ncols; j++){
      //           U_V(vertex_number - group_size + i, d + j) = matrix_get_element(mkl_result_mat, i, j);
      //           if(isnan(U_V(vertex_number - group_size + i, d + j)) || isinf(U_V(vertex_number - group_size + i, d + j))){
      //               cout<<"U_V("<<i<<", "<<j<<") = "<<U_V(i, j)<<endl;
      //           }
      //       }
      //   }
      // }
      for (int t = 1; t <= NUMTHREAD; t++){
        long long int row_start = (t-1)*(group_size/NUMTHREAD);
        long long int row_end = 0;
        if (t == NUMTHREAD){
          row_end = group_size;
        } else{
          row_end = t * (group_size/NUMTHREAD);
        }

        threads.push_back(thread(parallel_get_matrix_from_mkl, 
        mkl_result_mat, std::ref(U_V), row_start, row_end, 0, d, start_column, d
        ));

      }

      for (int t = 0; t < NUMTHREAD ; t++){
        threads[t].join();
      }
      vector<thread>().swap(threads);




      matrix_delete(mkl_result_mat);
      mkl_result_mat = NULL;

    }
    cout<<"right_unique_update_times = "<<unique_update_times<<endl;

    auto total_right_matrix_end_time = chrono::system_clock::now();
    auto total_elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(total_right_matrix_end_time - total_right_matrix_start_time);
    cout << "Total right matrix cost time: "<< total_elapsed_right_matrix_time.count() << endl;
    cout << "Total F norm of the whole ppr matrix = " << fixed << setprecision(12) << FF1 << endl;
    cout << "Total F norm of A - A * V * V^T = " << fixed << setprecision(12) << FF2 << endl;
    // cout << "Total norm cost time = "<< total_norm_time << endl;

}






















void mkl_right_matrix_multiplication_U_and_V(
d_row_tree_mkl* d_row_tree,
// mat* mkl_left_matrix, 
// Eigen::MatrixXd &U_V,
mat* mkl_left_matrix, 
Eigen::MatrixXd &V,

long long int vertex_number,
int d,
mat* SS,
int NUMTHREAD,
int largest_level_start_index,
long long int common_group_size,
long long int final_group_size,
vector<map<long long int, double>>& all_csr_entries,
vector<long long int>& offsets,
vector<ifstream>& ifs_vec,
vector<vector<long long int>>& nparts_all_count,
vector<long long int> nnz_nparts_vec,
vector<vector<long long int>>& nparts_read_in_io_all_count
)
{
    int unique_update_times = 0;
    
    auto total_right_matrix_start_time = chrono::system_clock::now();
    
    int nParts = d_row_tree->nParts;

    for(int iter = 0; iter < d_row_tree->nParts; iter++){
      int index = largest_level_start_index + iter;

      vector<long long int> &all_count = nparts_all_count[iter];

      vector<long long int> &read_in_io_all_count = nparts_read_in_io_all_count[iter];

      cout << "ppr computation " << endl;
      auto ppr_start_time = std::chrono::system_clock::now();
      vector<thread> threads;

      int order = index - largest_level_start_index;

      long long int start_column = order * common_group_size;

      long long int group_size;

      if(order == nParts - 1){
        group_size = final_group_size;
      }
      else{
        group_size = common_group_size;
      }

cout<<0<<endl;

      cout<<"group_size = "<<group_size<<endl;


      long long int nnz = nnz_nparts_vec[iter];

      long long int submatrix_rows = vertex_number;
      long long int submatrix_cols = group_size;



      cout<<"submatrix_cols = "<<submatrix_cols<<endl;
      cout<<"submatrix_rows = "<<submatrix_rows<<endl;
      cout<<"After nnz = "<<nnz<<endl;



      auto hash_coo_time = chrono::system_clock::now();

      cout<<"right matrix: 4"<<endl;

      // mat_coo *ppr_matrix_coo = coo_matrix_new(group_size, vertex_number, nnz);
      mat_coo *ppr_matrix_coo = coo_matrix_new(group_size, vertex_number, nnz);

      ppr_matrix_coo->nnz = nnz;





      for (int t = 1; t <= NUMTHREAD; t++){
        long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
        long long int end = 0;
        if (t == NUMTHREAD){
          end = start_column + group_size;
        } 
        else{
          end = start_column + t*(group_size/NUMTHREAD);
        }

        threads.push_back(thread(PPR_retrieve, 
        // t-1, 
        start, end, 
        // std::ref(all_count[t-1]), 
        std::ref(all_csr_entries),
        std::ref(ifs_vec[t-1]),
        std::ref(offsets),
        std::ref(read_in_io_all_count[t-1])
        ));

      }

      for (int t = 0; t < NUMTHREAD ; t++){
        threads[t].join();
      }
      vector<thread>().swap(threads);




    for (int t = 1; t <= NUMTHREAD; t++){

      long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
      long long int end = 0;
      if (t == NUMTHREAD){
        end = start_column + group_size;
      } else{
        end = start_column + t*(group_size/NUMTHREAD);
      }

      threads.push_back(thread(parallel_store_matrix, 
      std::ref(all_csr_entries),
      ppr_matrix_coo, 
      start,
      end,
      start_column,
      all_count[t-1] ));

    }

    for (int t = 0; t < NUMTHREAD ; t++){
      threads[t].join();
    }
    vector<thread>().swap(threads);




      mat_csr* ppr_matrix = csr_matrix_new();

      csr_init_from_coo(ppr_matrix, ppr_matrix_coo);

      coo_matrix_delete(ppr_matrix_coo);

      ppr_matrix_coo = NULL;


      cout<<"mkl_left_matrix->ncols = "<<mkl_left_matrix->ncols<<endl;

      //mkl_left_matrix->ncols == d
      mat *mkl_result_mat = matrix_new(group_size, mkl_left_matrix->ncols);

      auto right_matrix_start_time = chrono::system_clock::now();

      // csr_matrix_matrix_mult(ppr_matrix, mkl_left_matrix, mkl_result_mat);
      csr_matrix_matrix_mult(ppr_matrix, mkl_left_matrix, mkl_result_mat);

      csr_matrix_delete(ppr_matrix);
      ppr_matrix = NULL;

      mat *temp_mkl_result_mat = matrix_new(group_size, mkl_left_matrix->ncols);


      //This SS is already reversed
      matrix_matrix_mult(mkl_result_mat, SS, temp_mkl_result_mat);
      // second_pointer_matrix_matrix_mult(&mkl_result_mat, &SS, &temp_mkl_result_mat);

      matrix_copy(mkl_result_mat, temp_mkl_result_mat);






      








      cout<<"mkl_result_mat->nrows = "<<mkl_result_mat->nrows<<endl;
      cout<<"mkl_result_mat->ncols = "<<mkl_result_mat->ncols<<endl;
      

      matrix_delete(temp_mkl_result_mat);
      temp_mkl_result_mat = NULL;

      auto right_matrix_end_time = chrono::system_clock::now();
      auto elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(right_matrix_end_time - right_matrix_start_time);


      cout<<"V.rows() = "<<V.rows()<<endl;
      cout<<"V.cols() = "<<V.cols()<<endl;

      // if(iter != d_row_tree->nParts-1){
      //   for(long long int i = 0; i < group_size; i++){
      //       for(long long int j = 0; j < mkl_left_matrix->ncols; j++){
      //           U_V(iter * group_size + i, d + j) = matrix_get_element(mkl_result_mat, i, j);
      //           if(isnan(U_V(iter * group_size + i, d + j)) || isinf(U_V(iter * group_size + i, d + j))){
      //               cout<<"U_V("<<i<<", "<<j<<") = "<<U_V(i, j)<<endl;
      //           }
      //       }
      //   }
      // }
      // else{
      //   for(long long int i = 0; i < group_size; i++){
      //       for(long long int j = 0; j < mkl_left_matrix->ncols; j++){
      //           U_V(vertex_number - group_size + i, d + j) = matrix_get_element(mkl_result_mat, i, j);
      //           if(isnan(U_V(vertex_number - group_size + i, d + j)) || isinf(U_V(vertex_number - group_size + i, d + j))){
      //               cout<<"U_V("<<i<<", "<<j<<") = "<<U_V(i, j)<<endl;
      //           }
      //       }
      //   }
      // }
      for (int t = 1; t <= NUMTHREAD; t++){
        long long int row_start = (t-1)*(group_size/NUMTHREAD);
        long long int row_end = 0;
        if (t == NUMTHREAD){
          row_end = group_size;
        } else{
          row_end = t * (group_size/NUMTHREAD);
        }

        // threads.push_back(thread(parallel_get_matrix_from_mkl, 
        // mkl_result_mat, std::ref(U_V), row_start, row_end, 0, d, start_column, d
        // ));

        threads.push_back(thread(parallel_get_matrix_from_mkl, 
        mkl_result_mat, 
        std::ref(V), row_start, row_end, 0, d, start_column, 0
        ));
      }

      for (int t = 0; t < NUMTHREAD ; t++){
        threads[t].join();
      }
      vector<thread>().swap(threads);




      matrix_delete(mkl_result_mat);
      mkl_result_mat = NULL;

    }
    cout<<"right_unique_update_times = "<<unique_update_times<<endl;

    auto total_right_matrix_end_time = chrono::system_clock::now();
    auto total_elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(total_right_matrix_end_time - total_right_matrix_start_time);
    cout << "Total right matrix cost time: "<< total_elapsed_right_matrix_time.count() << endl;

    // cout << "Total norm cost time = "<< total_norm_time << endl;

}







































void mkl_right_matrix_multiplication_U_and_V_MAE(
d_row_tree_mkl* d_row_tree,
// mat* mkl_left_matrix, 
// Eigen::MatrixXd &U_V,
mat* mkl_left_matrix, 
// Eigen::MatrixXd &V,
vector<double>& MAE_vec,
long long int vertex_number,
int d,
mat* SS,
int NUMTHREAD,
int largest_level_start_index,
long long int common_group_size,
long long int final_group_size,
vector<map<long long int, double>>& all_csr_entries,
vector<long long int>& offsets,
vector<ifstream>& ifs_vec,
vector<vector<long long int>>& nparts_all_count,
vector<long long int> nnz_nparts_vec,
vector<vector<long long int>>& nparts_read_in_io_all_count
)
{
    int unique_update_times = 0;
    
    auto total_right_matrix_start_time = chrono::system_clock::now();
    
    int nParts = d_row_tree->nParts;

    long long int total_nnz = 0;

    for(int iter = 0; iter < d_row_tree->nParts; iter++){
      int index = largest_level_start_index + iter;

      vector<long long int> &all_count = nparts_all_count[iter];

      vector<long long int> &read_in_io_all_count = nparts_read_in_io_all_count[iter];

      cout << "ppr computation " << endl;
      auto ppr_start_time = std::chrono::system_clock::now();
      vector<thread> threads;

      int order = index - largest_level_start_index;

      long long int start_column = order * common_group_size;

      long long int group_size;

      if(order == nParts - 1){
        group_size = final_group_size;
      }
      else{
        group_size = common_group_size;
      }

cout<<0<<endl;

      cout<<"group_size = "<<group_size<<endl;


      long long int nnz = nnz_nparts_vec[iter];

      long long int submatrix_rows = vertex_number;
      long long int submatrix_cols = group_size;



      cout<<"submatrix_cols = "<<submatrix_cols<<endl;
      cout<<"submatrix_rows = "<<submatrix_rows<<endl;
      cout<<"After nnz = "<<nnz<<endl;



      auto hash_coo_time = chrono::system_clock::now();

      cout<<"right matrix: 4"<<endl;

      // mat_coo *ppr_matrix_coo = coo_matrix_new(group_size, vertex_number, nnz);
      mat_coo *ppr_matrix_coo = coo_matrix_new(group_size, vertex_number, nnz);

      ppr_matrix_coo->nnz = nnz;





      for (int t = 1; t <= NUMTHREAD; t++){
        long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
        long long int end = 0;
        if (t == NUMTHREAD){
          end = start_column + group_size;
        } 
        else{
          end = start_column + t*(group_size/NUMTHREAD);
        }

        threads.push_back(thread(PPR_retrieve, 
        // t-1, 
        start, end, 
        // std::ref(all_count[t-1]), 
        std::ref(all_csr_entries),
        std::ref(ifs_vec[t-1]),
        std::ref(offsets),
        std::ref(read_in_io_all_count[t-1])
        ));

      }

      for (int t = 0; t < NUMTHREAD ; t++){
        threads[t].join();
      }
      vector<thread>().swap(threads);




      for (int t = 1; t <= NUMTHREAD; t++){

        long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
        long long int end = 0;
        if (t == NUMTHREAD){
          end = start_column + group_size;
        } else{
          end = start_column + t*(group_size/NUMTHREAD);
        }

        threads.push_back(thread(parallel_store_matrix, 
        std::ref(all_csr_entries),
        ppr_matrix_coo, 
        start,
        end,
        start_column,
        all_count[t-1] ));

      }

      for (int t = 0; t < NUMTHREAD ; t++){
        threads[t].join();
      }
      vector<thread>().swap(threads);




      mat_csr* ppr_matrix = csr_matrix_new();

      csr_init_from_coo(ppr_matrix, ppr_matrix_coo);


      cout<<"mkl_left_matrix->ncols = "<<mkl_left_matrix->ncols<<endl;

      // mkl_left_matrix->ncols == d
      mat *mkl_result_mat = matrix_new(group_size, mkl_left_matrix->ncols);

      csr_matrix_matrix_mult(ppr_matrix, mkl_left_matrix, mkl_result_mat);


      auto right_matrix_start_time = chrono::system_clock::now();



      // get_matrix_frobenius_A_minus_AVVT(ppr_matrix_coo, ppr_matrix, mkl_left_matrix);








      csr_matrix_delete(ppr_matrix);
      ppr_matrix = NULL;


      matrix_get_mean_absolute_error(mkl_left_matrix, mkl_result_mat, 
                                        ppr_matrix_coo, &MAE_vec[iter]);

      
      cout<<"MAE_vec["<<iter<<"] = "<<MAE_vec[iter]<<endl;


      cout<<"mkl_result_mat->nrows = "<<mkl_result_mat->nrows<<endl;
      cout<<"mkl_result_mat->ncols = "<<mkl_result_mat->ncols<<endl;
      


      coo_matrix_delete(ppr_matrix_coo);
      ppr_matrix_coo = NULL;



      matrix_delete(mkl_result_mat);
      mkl_result_mat = NULL;


      auto right_matrix_end_time = chrono::system_clock::now();
      auto elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(right_matrix_end_time - right_matrix_start_time);



    }
      
    double total_MAE = 0.0;

    for(int iter = 0; iter < d_row_tree->nParts; iter++){
      total_MAE += MAE_vec[iter];
    }


    for(int iter = 0; iter < d_row_tree->nParts; iter++){
      total_nnz += nnz_nparts_vec[iter];
    }

    cout<<"total_MAE = "<<total_MAE<<endl;

    total_MAE /= total_nnz;

    cout<<" total_MAE /= total_nnz = "<<total_MAE<<endl;
    
    cout<<"right_unique_update_times = "<<unique_update_times<<endl;

    auto total_right_matrix_end_time = chrono::system_clock::now();
    auto total_elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(total_right_matrix_end_time - total_right_matrix_start_time);
    cout << "Total right matrix cost time: "<< total_elapsed_right_matrix_time.count() << endl;

    // cout << "Total norm cost time = "<< total_norm_time << endl;

}


























void mkl_right_matrix_multiplication_U_and_V_frobenius(
d_row_tree_mkl* d_row_tree,
// mat* mkl_left_matrix, 
// Eigen::MatrixXd &U_V,
mat* mkl_left_matrix, 
// Eigen::MatrixXd &V,
vector<double>& frobenius_vec,
long long int vertex_number,
int d,
mat* SS,
int NUMTHREAD,
int largest_level_start_index,
long long int common_group_size,
long long int final_group_size,
vector<map<long long int, double>>& all_csr_entries,
vector<long long int>& offsets,
vector<ifstream>& ifs_vec,
vector<vector<long long int>>& nparts_all_count,
vector<long long int> nnz_nparts_vec,
vector<vector<long long int>>& nparts_read_in_io_all_count
)
{
    int unique_update_times = 0;
    
    auto total_right_matrix_start_time = chrono::system_clock::now();
    
    int nParts = d_row_tree->nParts;

    long long int total_nnz = 0;

    double coo_frobenius_norm = 0;
    MKL_INT tot_nnz = 0;

    for(int iter = 0; iter < d_row_tree->nParts; iter++){
      int index = largest_level_start_index + iter;

      vector<long long int> &all_count = nparts_all_count[iter];

      vector<long long int> &read_in_io_all_count = nparts_read_in_io_all_count[iter];

      cout << "ppr computation " << endl;
      auto ppr_start_time = std::chrono::system_clock::now();
      vector<thread> threads;

      int order = index - largest_level_start_index;

      long long int start_column = order * common_group_size;

      long long int group_size;

      if(order == nParts - 1){
        group_size = final_group_size;
      }
      else{
        group_size = common_group_size;
      }

cout<<0<<endl;

      cout<<"group_size = "<<group_size<<endl;


      long long int nnz = nnz_nparts_vec[iter];

      long long int submatrix_rows = vertex_number;
      long long int submatrix_cols = group_size;



      cout<<"submatrix_cols = "<<submatrix_cols<<endl;
      cout<<"submatrix_rows = "<<submatrix_rows<<endl;
      cout<<"After nnz = "<<nnz<<endl;



      auto hash_coo_time = chrono::system_clock::now();

      cout<<"right matrix: 4"<<endl;

      // mat_coo *ppr_matrix_coo = coo_matrix_new(group_size, vertex_number, nnz);
      mat_coo *ppr_matrix_coo = coo_matrix_new(group_size, vertex_number, nnz);

      ppr_matrix_coo->nnz = nnz;





      for (int t = 1; t <= NUMTHREAD; t++){
        long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
        long long int end = 0;
        if (t == NUMTHREAD){
          end = start_column + group_size;
        } 
        else{
          end = start_column + t*(group_size/NUMTHREAD);
        }

        threads.push_back(thread(PPR_retrieve, 
        // t-1, 
        start, end, 
        // std::ref(all_count[t-1]), 
        std::ref(all_csr_entries),
        std::ref(ifs_vec[t-1]),
        std::ref(offsets),
        std::ref(read_in_io_all_count[t-1])
        ));

      }

      for (int t = 0; t < NUMTHREAD ; t++){
        threads[t].join();
      }
      vector<thread>().swap(threads);




      for (int t = 1; t <= NUMTHREAD; t++){

        long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
        long long int end = 0;
        if (t == NUMTHREAD){
          end = start_column + group_size;
        } else{
          end = start_column + t*(group_size/NUMTHREAD);
        }

        threads.push_back(thread(parallel_store_matrix, 
        std::ref(all_csr_entries),
        ppr_matrix_coo, 
        start,
        end,
        start_column,
        all_count[t-1] ));

      }

      for (int t = 0; t < NUMTHREAD ; t++){
        threads[t].join();
      }
      vector<thread>().swap(threads);




      mat_csr* ppr_matrix = csr_matrix_new();

      csr_init_from_coo(ppr_matrix, ppr_matrix_coo);


      cout<<"mkl_left_matrix->ncols = "<<mkl_left_matrix->ncols<<endl;

      // mkl_left_matrix->ncols == d
      // mat *mkl_result_mat = matrix_new(group_size, mkl_left_matrix->ncols);

      // csr_matrix_matrix_mult(ppr_matrix, mkl_left_matrix, mkl_result_mat);


      auto right_matrix_start_time = chrono::system_clock::now();



      // get_matrix_frobenius_A_minus_AVVT(ppr_matrix_coo, ppr_matrix, mkl_left_matrix);






      double* tempv = ppr_matrix_coo->values;
      double temp_fnorm = 0;
      for(MKL_INT temp_i = 0; temp_i < ppr_matrix_coo->nnz; ++temp_i)
        temp_fnorm+= tempv[temp_i] * tempv[temp_i];
      coo_frobenius_norm += temp_fnorm;

      frobenius_vec[iter] = get_matrix_frobenius_A_minus_AVVT(ppr_matrix, mkl_left_matrix);
      tot_nnz += ppr_matrix_coo->nnz;

      cout << "First row with " << ppr_matrix->pointerE[0] - ppr_matrix->pointerB[0] << "columns.\n";
      cout << "Last row with " << ppr_matrix->pointerE[ppr_matrix->nrows - 1] - ppr_matrix->pointerB[ppr_matrix->nrows - 1] << "columns.\n";
      cout<<"frobenius norm of part "<<iter<<" = "<< fixed << setprecision(12) << temp_fnorm <<endl;
      cout<<"frobenius_vec["<<iter<<"] = "<< fixed << setprecision(12) <<frobenius_vec[iter]<<endl;

      // matrix_get_mean_absolute_error(mkl_left_matrix, mkl_result_mat, 
      //                                   ppr_matrix_coo, &MAE_vec[iter]);

      
      // cout<<"MAE_vec["<<iter<<"] = "<<MAE_vec[iter]<<endl;


      // cout<<"mkl_result_mat->nrows = "<<mkl_result_mat->nrows<<endl;
      // cout<<"mkl_result_mat->ncols = "<<mkl_result_mat->ncols<<endl;
      

      csr_matrix_delete(ppr_matrix);
      ppr_matrix = NULL;


      coo_matrix_delete(ppr_matrix_coo);
      ppr_matrix_coo = NULL;



      // matrix_delete(mkl_result_mat);
      // mkl_result_mat = NULL;


      auto right_matrix_end_time = chrono::system_clock::now();
      auto elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(right_matrix_end_time - right_matrix_start_time);
      cout<<"frobenius norm time = "<<elapsed_right_matrix_time.count()<<endl;


    }
      
    double total_frobenius = 0.0;

    for(int iter = 0; iter < d_row_tree->nParts; iter++){
      total_frobenius += frobenius_vec[iter];
    }


    // for(int iter = 0; iter < d_row_tree->nParts; iter++){
    //   total_nnz += nnz_nparts_vec[iter];
    // }

    cout << "Total Frobenius Norm Square as A - A * V * V^T = " << fixed << setprecision(12) << total_frobenius << endl;
    cout << "Total Frobenius Norm Square of the whole PPR Matrix = " << fixed << setprecision(12) << coo_frobenius_norm << endl;
    cout << "Total NNZ of the whole PPR Matrix = " << tot_nnz << endl;
    // total_frobenius /= total_nnz;

    // cout<<" total_frobenius /= total_nnz = "<<total_frobenius<<endl;
    
    cout<<"right_unique_update_times = "<<unique_update_times<<endl;

    auto total_right_matrix_end_time = chrono::system_clock::now();
    auto total_elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(total_right_matrix_end_time - total_right_matrix_start_time);
    cout << "Total right matrix cost time: "<< total_elapsed_right_matrix_time.count() << endl;

    // cout << "Total norm cost time = "<< total_norm_time << endl;

}






void mkl_right_matrix_multiplication_U_and_V_frobenius_with_GroupSize(
d_row_tree_mkl* d_row_tree,
// mat* mkl_left_matrix, 
// Eigen::MatrixXd &U_V,
mat* mkl_left_matrix, 
// Eigen::MatrixXd &V,
vector<double>& frobenius_vec,
long long int vertex_number,
int d,
mat* SS,
int NUMTHREAD,
int largest_level_start_index,
int* GroupSize,
vector<map<long long int, double>>& all_csr_entries,
vector<long long int>& offsets,
vector<ifstream>& ifs_vec,
vector<vector<long long int>>& nparts_all_count,
vector<long long int> nnz_nparts_vec,
vector<vector<long long int>>& nparts_read_in_io_all_count,
vector<vector<std::pair<long long, pair<double, double>>>>& changed_nonzeros,
mat_csr** leaf_matrix = NULL,
double* leaf_fnorm = NULL,
long long* leaf_nnz = NULL,
double* Final_Error = NULL
)
{
    auto AVVT_st_time = chrono::system_clock::now();

    int unique_update_times = 0;
    
    // auto total_right_matrix_start_time = chrono::system_clock::now();
    
    int nParts = d_row_tree->nParts;

    long long int total_nnz = 0;

    double coo_frobenius_norm = 0;
    MKL_INT tot_nnz = 0;

    for(int iter = 0; iter < d_row_tree->nParts; iter++){
      int index = largest_level_start_index + iter;

      vector<long long int> &all_count = nparts_all_count[iter];

      vector<long long int> &read_in_io_all_count = nparts_read_in_io_all_count[iter];

      // cout << "ppr computation " << endl;//output
      auto ppr_start_time = std::chrono::system_clock::now();
      vector<thread> threads;

      int order = index - largest_level_start_index;

      long long int start_column = 0;

      long long int group_size = GroupSize[iter];

      for(int tti = 0; tti < iter; ++tti)
        start_column += GroupSize[tti];


      // cout<<"group_size = "<<group_size<<endl;//output


      long long int nnz = nnz_nparts_vec[iter];
      if(leaf_matrix != NULL && leaf_matrix[iter] !=NULL || nnz == 0) 
        nnz = leaf_matrix[iter]->nnz;

      long long int submatrix_rows = vertex_number;
      long long int submatrix_cols = group_size;



      // cout<<"submatrix_cols = "<<submatrix_cols<<endl;//output
      // cout<<"submatrix_rows = "<<submatrix_rows<<endl;//output
      // cout<<"After nnz = "<<nnz<<endl;//output



      auto hash_coo_time = chrono::system_clock::now();

      // cout<<"right matrix: 4"<<endl;//output

      // mat_coo *ppr_matrix_coo = coo_matrix_new(group_size, vertex_number, nnz);
      mat_coo *ppr_matrix_coo = coo_matrix_new(group_size, vertex_number, nnz);

      ppr_matrix_coo->nnz = nnz;


      mat_csr* ppr_matrix = csr_matrix_new();

      if(leaf_matrix == NULL){

        for (int t = 1; t <= NUMTHREAD; t++){
          long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
          long long int end = 0;
          if (t == NUMTHREAD){
            end = start_column + group_size;
          } 
          else{
            end = start_column + t*(group_size/NUMTHREAD);
          }

          threads.push_back(thread(PPR_retrieve, 
          // t-1, 
          start, end, 
          // std::ref(all_count[t-1]), 
          std::ref(all_csr_entries),
          std::ref(ifs_vec[t-1]),
          std::ref(offsets),
          std::ref(read_in_io_all_count[t-1])
          ));

        }

        for (int t = 0; t < NUMTHREAD ; t++){
          threads[t].join();
        }
        vector<thread>().swap(threads);

        // cout << "11" << endl;

        threads.clear();

        for (int t = 1; t <= NUMTHREAD; t++){

          long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
          long long int end = 0;
          if (t == NUMTHREAD){
            end = start_column + group_size;
          } else{
            end = start_column + t*(group_size/NUMTHREAD);
          }

          threads.push_back(thread(parallel_store_matrix, 
          std::ref(all_csr_entries),
          ppr_matrix_coo, 
          start,
          end,
          start_column,
          all_count[t-1] ));

        }

        malloc_trim(0);

        for (int t = 0; t < NUMTHREAD ; t++){
          threads[t].join();
        }
        vector<thread>().swap(threads);

        csr_init_from_coo(ppr_matrix, ppr_matrix_coo);
    
      } 

      double temp_ss = 0;
      // for(int xx = 0; xx < changed_nonzeros[iter].size(); ++xx)
      for(auto x : changed_nonzeros[iter])
      {
        // std::pair<long long, pair<double, double>> x = changed_nonzeros[iter][xx];
        ppr_matrix->values[x.first] += x.second.second; // sqrt(x.second);
        ppr_matrix_coo->values[x.first] += x.second.second; //sqrt(x.second);
        if(x.first < 0 || x.first >= ppr_matrix->nnz)
          cout << "Warning " << x.first << " " << ppr_matrix->nnz << endl;
        temp_ss += x.second.second;
      }

      cout << temp_ss << " ";

      // cout<<"mkl_left_matrix->ncols = "<<mkl_left_matrix->ncols<<endl; //output

      // mkl_left_matrix->ncols == d
      // mat *mkl_result_mat = matrix_new(group_size, mkl_left_matrix->ncols);

      // csr_matrix_matrix_mult(ppr_matrix, mkl_left_matrix, mkl_result_mat);


      auto right_matrix_start_time = chrono::system_clock::now();



      // get_matrix_frobenius_A_minus_AVVT(ppr_matrix_coo, ppr_matrix, mkl_left_matrix);






      double* tempv = ppr_matrix->values;
      double temp_fnorm = 0;

      for(MKL_INT temp_i = 0; temp_i < ppr_matrix->nnz; ++temp_i)
        temp_fnorm += tempv[temp_i] * tempv[temp_i];

      coo_frobenius_norm += temp_fnorm;

      frobenius_vec[iter] = get_matrix_frobenius_A_minus_AVVT(ppr_matrix, mkl_left_matrix);
      tot_nnz += ppr_matrix->nnz;

      leaf_nnz[iter] = ppr_matrix->nnz;
      leaf_fnorm[iter] = temp_fnorm;


      //output
      // cout << "First row with " << ppr_matrix->pointerE[0] - ppr_matrix->pointerB[0] << "columns.\n";
      // cout << "Last row with " << ppr_matrix->pointerE[ppr_matrix->nrows - 1] - ppr_matrix->pointerB[ppr_matrix->nrows - 1] << "columns.\n";
      cout<<"frobenius norm of part "<<iter<<" = "<< fixed << setprecision(12) << temp_fnorm <<endl;
      // cout<<"frobenius_vec["<<iter<<"] = "<< fixed << setprecision(12) <<frobenius_vec[iter]<<endl;
      //output

      // matrix_get_mean_absolute_error(mkl_left_matrix, mkl_result_mat, 
      //                                   ppr_matrix_coo, &MAE_vec[iter]);
      
      // cout<<"MAE_vec["<<iter<<"] = "<<MAE_vec[iter]<<endl;

      // cout<<"mkl_result_mat->nrows = "<<mkl_result_mat->nrows<<endl;
      // cout<<"mkl_result_mat->ncols = "<<mkl_result_mat->ncols<<endl;
      

      csr_matrix_delete(ppr_matrix);
      ppr_matrix = NULL;


      coo_matrix_delete(ppr_matrix_coo);
      ppr_matrix_coo = NULL;



      // matrix_delete(mkl_result_mat);
      // mkl_result_mat = NULL;


      auto right_matrix_end_time = chrono::system_clock::now();
      auto elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(right_matrix_end_time - right_matrix_start_time);
      // cout<<"frobenius norm time = "<<elapsed_right_matrix_time.count()<<endl;


    }
      
    double total_frobenius = 0.0;

    for(int iter = 0; iter < d_row_tree->nParts; iter++){
      total_frobenius += frobenius_vec[iter];
    }


    // for(int iter = 0; iter < d_row_tree->nParts; iter++){
    //   total_nnz += nnz_nparts_vec[iter];
    // }

    cout << "Total Frobenius Norm Square as A - A * V * V^T = " << fixed << setprecision(12) << total_frobenius << endl;
    cout << "Total Frobenius Norm Square of the whole PPR Matrix = " << fixed << setprecision(12) << coo_frobenius_norm << endl;
    cout << "Total Error Percentage is " << (total_frobenius / coo_frobenius_norm) << endl;
    cout << "Total NNZ of the whole PPR Matrix = " << tot_nnz << endl;

    if(Final_Error != NULL) (*Final_Error) = (total_frobenius / coo_frobenius_norm);
    // total_frobenius /= total_nnz;

    // cout<<" total_frobenius /= total_nnz = "<<total_frobenius<<endl;
    
    cout<<"right_unique_update_times = "<<unique_update_times<<endl;

    // auto total_right_matrix_end_time = chrono::system_clock::now();
    // auto total_elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(total_right_matrix_end_time - total_right_matrix_start_time);
    // cout << "Total right matrix cost time: "<< total_elapsed_right_matrix_time.count() << endl;

    // cout << "Total norm cost time = "<< total_norm_time << endl;
    auto AVVT_en_time = chrono::system_clock::now();
    auto AVVT_comp_time = chrono::duration<double>(AVVT_en_time - AVVT_st_time);
    cout << "Retrieve PPR and Compute the F norm of || A - A * V * V^T || Time Use : " << AVVT_comp_time.count() << endl;

}








void mkl_right_matrix_multiplication_V_frobenius_error_with_recover_update(
d_row_tree_mkl* d_row_tree,
mat* mkl_left_matrix, 
vector<double>& frobenius_vec,
long long int vertex_number,
int d,
mat* SS,
int NUMTHREAD,
int largest_level_start_index,
int* GroupSize,
vector<map<long long int, double>>& all_csr_entries,
vector<long long int>& offsets,
vector<ifstream>& ifs_vec,
vector<vector<long long int>>& nparts_all_count,
vector<long long int> nnz_nparts_vec,
vector<vector<long long int>>& nparts_read_in_io_all_count,
vector<int>& random_seed,
pair<double, pair<double, double>> remain_para,
int** ppr_col,
double** ppr_nnz,
int erase_way = 0,
// double** Frobenius_norm_each_row = NULL,
// double* Frobenius_norm_each_col = NULL,
double* snapnum_col = NULL,
double** snapnum_row = NULL,
double* leaf_fnorm = NULL,
long long* leaf_nnz = NULL,
double* L_1_1_norm = NULL,
double* Final_Error = NULL
)
{
    cout << "erase_way number = " << erase_way << endl;
    auto AVVT_st_time = chrono::system_clock::now();

    int unique_update_times = 0;
    
    // auto total_right_matrix_start_time = chrono::system_clock::now();
    cout << "begin sss" << endl;
    int nParts = d_row_tree->nParts;

    long long int total_nnz = 0;

    double coo_frobenius_norm = 0;
    MKL_INT tot_nnz = 0;

    for(int iter = 0; iter < d_row_tree->nParts; iter++){
      int index = largest_level_start_index + iter;

      vector<long long int> &all_count = nparts_all_count[iter];

      vector<long long int> &read_in_io_all_count = nparts_read_in_io_all_count[iter];

      cout << "ppr computation " << endl;
      auto ppr_start_time = std::chrono::system_clock::now();
      vector<thread> threads;

      int order = index - largest_level_start_index;

      long long int start_column = 0;

      long long int group_size = GroupSize[iter];

      for(int tti = 0; tti < iter; ++tti)
        start_column += GroupSize[tti];


      // cout<<"group_size = "<<group_size<<endl;//output


      long long int nnz = nnz_nparts_vec[iter];

      long long int submatrix_rows = vertex_number;
      long long int submatrix_cols = group_size;



      // cout<<"submatrix_cols = "<<submatrix_cols<<endl;//output
      // cout<<"submatrix_rows = "<<submatrix_rows<<endl;//output
      // cout<<"After nnz = "<<nnz<<endl;//output



      auto hash_coo_time = chrono::system_clock::now();

      // cout<<"right matrix: 4"<<endl;//output

      // mat_coo *ppr_matrix_coo = coo_matrix_new(group_size, vertex_number, nnz);
      mat_coo_2 *ppr_matrix_coo = coo_2_matrix_new(group_size, vertex_number, nnz);

      ppr_matrix_coo->nnz = nnz;

      cout << "nnz = " << nnz << endl;
      mat_csr* ppr_matrix = csr_matrix_new();

      // cout << "0\n"; 
      
        for (int t = 1; t <= NUMTHREAD; t++){
          long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
          long long int end = 0;
          if (t == NUMTHREAD){
            end = start_column + group_size;
          } 
          else{
            end = start_column + t*(group_size/NUMTHREAD);
          }

          long long temp_tt = 0;
          threads.push_back(thread(PPR_retrieve_less_space, 
          // t-1, 
          start, end, 
          // std::ref(all_count[t-1]), 
          std::ref(all_csr_entries),
          std::ref(ifs_vec[t-1]),
          std::ref(offsets),
          std::ref(read_in_io_all_count[t-1]),
          &temp_tt,
          ppr_col, ppr_nnz
          ));

        }

        for (int t = 0; t < NUMTHREAD ; t++){
          threads[t].join();
        }
        vector<thread>().swap(threads);

        // cout << "11" << endl;

        threads.clear();

        for(int uuu = 0; uuu < 2; ++uuu)
        {
          if(uuu == 0) ppr_matrix_coo->values = new double[nnz];
          else {
            ppr_matrix_coo->rows = new int[nnz];
            ppr_matrix_coo->cols = new long long[nnz];
          }

          for (int t = 1; t <= NUMTHREAD; t++){

            long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
            long long int end = 0;
            if (t == NUMTHREAD){
              end = start_column + group_size;
            } else{
              end = start_column + t*(group_size/NUMTHREAD);
            }

            if(uuu == 0)
              threads.push_back(thread(parallel_store_matrix_less_space_nnz, 
              std::ref(all_csr_entries),
              ppr_matrix_coo, 
              start,
              end,
              start_column,
              all_count[t-1],
              ppr_col, ppr_nnz ));
            else
              threads.push_back(thread(parallel_store_matrix_less_space_rows_cols, 
              std::ref(all_csr_entries),
              ppr_matrix_coo, 
              start,
              end,
              start_column,
              all_count[t-1],
              ppr_col));

          }

          malloc_trim(0);

          for (int t = 0; t < NUMTHREAD ; t++){
            threads[t].join();
          }
          vector<thread>().swap(threads);
        }


        if(erase_way > 0)
        {
          Erase_row_col_nonzeros(ppr_matrix_coo, remain_para, snapnum_col, snapnum_row[iter]);
        }

        csr_init_from_coo_2(ppr_matrix, ppr_matrix_coo);
    
      
      



      auto right_matrix_start_time = chrono::system_clock::now();



      // get_matrix_frobenius_A_minus_AVVT(ppr_matrix_coo, ppr_matrix, mkl_left_matrix);


      // cout << "2" << endl;



      double* tempv = ppr_matrix->values;
      double temp_fnorm = 0;

      if(L_1_1_norm != NULL)
      {
        L_1_1_norm[iter] = 0;
        for(MKL_INT temp_i = 0; temp_i < ppr_matrix->nnz; ++temp_i)
          L_1_1_norm[iter] += tempv[temp_i];
      }

      // cout << 3 << endl;
      for(MKL_INT temp_i = 0; temp_i < ppr_matrix->nnz; ++temp_i)
        temp_fnorm += tempv[temp_i] * tempv[temp_i];

      coo_frobenius_norm += temp_fnorm;

      // cout << 4 << endl;
      cout << ppr_matrix->nnz << endl;
      cout << ppr_matrix->nrows << endl;
      cout << ppr_matrix->ncols << endl;
      frobenius_vec[iter] = get_matrix_frobenius_A_minus_AVVT(ppr_matrix, mkl_left_matrix);
      tot_nnz += ppr_matrix->nnz;

      // cout << 5 << endl;
      if(leaf_nnz != NULL) leaf_nnz[iter] = ppr_matrix->nnz;
      if(leaf_fnorm != NULL) leaf_fnorm[iter] = temp_fnorm;


      //output
      // cout << "First row with " << ppr_matrix->pointerE[0] - ppr_matrix->pointerB[0] << "columns.\n";
      // cout << "Last row with " << ppr_matrix->pointerE[ppr_matrix->nrows - 1] - ppr_matrix->pointerB[ppr_matrix->nrows - 1] << "columns.\n";
      cout<<"frobenius norm of part "<<iter<<" = "<< fixed << setprecision(12) << temp_fnorm <<endl;
      // cout<<"frobenius_vec["<<iter<<"] = "<< fixed << setprecision(12) <<frobenius_vec[iter]<<endl;
      //output

      // matrix_get_mean_absolute_error(mkl_left_matrix, mkl_result_mat, 
      //                                   ppr_matrix_coo, &MAE_vec[iter]);
      
      // cout<<"MAE_vec["<<iter<<"] = "<<MAE_vec[iter]<<endl;

      // cout<<"mkl_result_mat->nrows = "<<mkl_result_mat->nrows<<endl;
      // cout<<"mkl_result_mat->ncols = "<<mkl_result_mat->ncols<<endl;
      

      csr_matrix_delete(ppr_matrix);
      ppr_matrix = NULL;


      coo_2_matrix_delete(ppr_matrix_coo);
      ppr_matrix_coo = NULL;



      // matrix_delete(mkl_result_mat);
      // mkl_result_mat = NULL;


      auto right_matrix_end_time = chrono::system_clock::now();
      auto elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(right_matrix_end_time - right_matrix_start_time);
      // cout<<"frobenius norm time = "<<elapsed_right_matrix_time.count()<<endl;


    }
      
    double total_frobenius = 0.0;

    for(int iter = 0; iter < d_row_tree->nParts; iter++){
      total_frobenius += frobenius_vec[iter];
    }


    // for(int iter = 0; iter < d_row_tree->nParts; iter++){
    //   total_nnz += nnz_nparts_vec[iter];
    // }

    cout << "Total Frobenius Norm Square as A - A * V * V^T = " << fixed << setprecision(12) << total_frobenius << endl;
    cout << "Total Frobenius Norm Square of the whole PPR Matrix = " << fixed << setprecision(12) << coo_frobenius_norm << endl;
    cout << "Total Error Percentage is " << (total_frobenius / coo_frobenius_norm) << endl;
    cout << "Total NNZ of the whole PPR Matrix = " << tot_nnz << endl;

    if(Final_Error != NULL) (*Final_Error) = (total_frobenius / coo_frobenius_norm);
    // total_frobenius /= total_nnz;

    // cout<<" total_frobenius /= total_nnz = "<<total_frobenius<<endl;
    
    cout<<"right_unique_update_times = "<<unique_update_times<<endl;

    // auto total_right_matrix_end_time = chrono::system_clock::now();
    // auto total_elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(total_right_matrix_end_time - total_right_matrix_start_time);
    // cout << "Total right matrix cost time: "<< total_elapsed_right_matrix_time.count() << endl;

    // cout << "Total norm cost time = "<< total_norm_time << endl;
    auto AVVT_en_time = chrono::system_clock::now();
    auto AVVT_comp_time = chrono::duration<double>(AVVT_en_time - AVVT_st_time);
    cout << "Retrieve PPR and Compute the F norm of || A - A * V * V^T || Time Use : " << AVVT_comp_time.count() << endl;

}






















void mkl_right_matrix_multiplication_V_frobenius_error_with_image_recover_update(
d_row_tree_mkl* d_row_tree,
mat* mkl_left_matrix, 
vector<double>& frobenius_vec,
long long int user_number,
long long int item_number,
int d,
mat* SS,
int NUMTHREAD,
int largest_level_start_index,
int* GroupSize,
vector<map<long long int, double>>& all_csr_entries,
vector<long long int>& offsets,
vector<ifstream>& ifs_vec,
vector<vector<long long int>>& nparts_all_count,
vector<long long int> nnz_nparts_vec,
vector<vector<long long int>>& nparts_read_in_io_all_count,
vector<int>& random_seed,
pair<double, pair<double, double>> remain_para,
mat_coo** cache_matrix,
int erase_way = 0,
double* snapnum_col = NULL,
double** snapnum_row = NULL,
double* leaf_fnorm = NULL,
long long* leaf_nnz = NULL,
double* L_1_1_norm = NULL,
double* Final_Error = NULL
)
{
    cout << "erase_way number = " << erase_way << endl;
    auto AVVT_st_time = chrono::system_clock::now();

    int unique_update_times = 0;
    
    // auto total_right_matrix_start_time = chrono::system_clock::now();
    cout << "begin!" << endl;
    int nParts = d_row_tree->nParts;

    long long int total_nnz = 0;

    double coo_frobenius_norm = 0;
    MKL_INT tot_nnz = 0;

    for(int iter = 0; iter < d_row_tree->nParts; iter++){
      int index = largest_level_start_index + iter;

      vector<long long int> &all_count = nparts_all_count[iter];

      vector<long long int> &read_in_io_all_count = nparts_read_in_io_all_count[iter];

      cout << "matrix computation" << endl;//output
      auto ppr_start_time = std::chrono::system_clock::now();
      vector<thread> threads;

      int order = index - largest_level_start_index;

      long long int start_column = 0;

      long long int group_size = GroupSize[iter];

      for(int tti = 0; tti < iter; ++tti)
        start_column += GroupSize[tti];


      // cout<<"group_size = "<<group_size<<endl;//output


      long long int nnz = nnz_nparts_vec[iter];

      long long int submatrix_rows = item_number;
      long long int submatrix_cols = group_size;



      // cout<<"submatrix_cols = "<<submatrix_cols<<endl;//output
      // cout<<"submatrix_rows = "<<submatrix_rows<<endl;//output
      // cout<<"After nnz = "<<nnz<<endl;//output



      auto hash_coo_time = chrono::system_clock::now();

      // cout<<"right matrix: 4"<<endl;//output

      // mat_coo *ppr_matrix_coo = coo_matrix_new(group_size, vertex_number, nnz);
      mat_coo *ppr_matrix_coo = coo_matrix_new(group_size, item_number, nnz);

      ppr_matrix_coo->nnz = nnz;


      mat_csr* ppr_matrix = csr_matrix_new();

      // cout << "0\n"; 
      
        // for (int t = 1; t <= NUMTHREAD; t++){
        //   long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
        //   long long int end = 0;
        //   if (t == NUMTHREAD){
        //     end = start_column + group_size;
        //   } 
        //   else{
        //     end = start_column + t*(group_size/NUMTHREAD);
        //   }

        //   threads.push_back(thread(PPR_retrieve, 
        //   // t-1, 
        //   start, end, 
        //   // std::ref(all_count[t-1]), 
        //   std::ref(all_csr_entries),
        //   std::ref(ifs_vec[t-1]),
        //   std::ref(offsets),
        //   std::ref(read_in_io_all_count[t-1])
        //   ));

        // }

        // for (int t = 0; t < NUMTHREAD ; t++){
        //   threads[t].join();
        // }
        // vector<thread>().swap(threads);

        // cout << "11" << endl;

        // threads.clear();

        // for (int t = 1; t <= NUMTHREAD; t++){

        //   long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
        //   long long int end = 0;
        //   if (t == NUMTHREAD){
        //     end = start_column + group_size;
        //   } else{
        //     end = start_column + t*(group_size/NUMTHREAD);
        //   }

        //   threads.push_back(thread(parallel_store_matrix, 
        //   std::ref(all_csr_entries),
        //   ppr_matrix_coo, 
        //   start,
        //   end,
        //   start_column,
        //   all_count[t-1] ));

        // }

        // malloc_trim(0);

        // for (int t = 0; t < NUMTHREAD ; t++){
        //   threads[t].join();
        // }
        // vector<thread>().swap(threads);

        if(cache_matrix != NULL && cache_matrix[iter] != NULL)
        {
          cout << "copy matrix " << iter << endl;
          assert(ppr_matrix_coo->nnz == cache_matrix[iter]->nnz);
          assert(ppr_matrix_coo->nrows == cache_matrix[iter]->nrows);
          assert(ppr_matrix_coo->ncols == cache_matrix[iter]->ncols);

          for(long long u = 0; u < nnz; ++u)
            ppr_matrix_coo->rows[u] = cache_matrix[iter]->rows[u];
          for(long long u = 0; u < nnz; ++u)
            ppr_matrix_coo->cols[u] = cache_matrix[iter]->cols[u];
          for(long long u = 0; u < nnz; ++u)
            ppr_matrix_coo->values[u] = cache_matrix[iter]->values[u];
        }
        else {
          Read_image_Matrix(std::ref(ifs_vec[iter]), ppr_matrix_coo);
          cout << "copy matrix " << iter << endl;
        }

        for(long long ii = 0; ii < ppr_matrix_coo->nnz; ++ii)
          ppr_matrix_coo->rows[ii] -= (start_column);


        if(erase_way > 0)
        {
          Erase_row_col_nonzeros(ppr_matrix_coo, remain_para, snapnum_col, snapnum_row[iter]);
        }

        csr_init_from_coo(ppr_matrix, ppr_matrix_coo);
    
      
      

      // cout << "PPR proximity matrix read from Disk.\n";//output

      // cout << "1" << endl;


      // cout<<"mkl_left_matrix->ncols = "<<mkl_left_matrix->ncols<<endl; //output

      // mkl_left_matrix->ncols == d
      // mat *mkl_result_mat = matrix_new(group_size, mkl_left_matrix->ncols);

      // csr_matrix_matrix_mult(ppr_matrix, mkl_left_matrix, mkl_result_mat);


      auto right_matrix_start_time = chrono::system_clock::now();



      // get_matrix_frobenius_A_minus_AVVT(ppr_matrix_coo, ppr_matrix, mkl_left_matrix);


      // cout << "2" << endl;



      double* tempv = ppr_matrix->values;
      double temp_fnorm = 0;

      if(L_1_1_norm != NULL)
      {
        L_1_1_norm[iter] = 0;
        for(MKL_INT temp_i = 0; temp_i < ppr_matrix->nnz; ++temp_i)
          L_1_1_norm[iter] += tempv[temp_i];
      }

      // cout << 3 << endl;
      for(MKL_INT temp_i = 0; temp_i < ppr_matrix->nnz; ++temp_i)
        temp_fnorm += tempv[temp_i] * tempv[temp_i];

      coo_frobenius_norm += temp_fnorm;

      // cout << 4 << endl;
      cout << ppr_matrix->nnz << endl;
      cout << ppr_matrix->nrows << endl;
      cout << ppr_matrix->ncols << endl;
      frobenius_vec[iter] = get_matrix_frobenius_A_minus_AVVT(ppr_matrix, mkl_left_matrix);
      tot_nnz += ppr_matrix->nnz;

      // cout << 5 << endl;
      if(leaf_nnz != NULL) leaf_nnz[iter] = ppr_matrix->nnz;
      if(leaf_fnorm != NULL) leaf_fnorm[iter] = temp_fnorm;


      //output
      // cout << "First row with " << ppr_matrix->pointerE[0] - ppr_matrix->pointerB[0] << "columns.\n";
      // cout << "Last row with " << ppr_matrix->pointerE[ppr_matrix->nrows - 1] - ppr_matrix->pointerB[ppr_matrix->nrows - 1] << "columns.\n";
      cout<<"frobenius norm of part "<<iter<<" = "<< fixed << setprecision(12) << temp_fnorm <<endl;
      // cout<<"frobenius_vec["<<iter<<"] = "<< fixed << setprecision(12) <<frobenius_vec[iter]<<endl;
      //output

      // matrix_get_mean_absolute_error(mkl_left_matrix, mkl_result_mat, 
      //                                   ppr_matrix_coo, &MAE_vec[iter]);
      
      // cout<<"MAE_vec["<<iter<<"] = "<<MAE_vec[iter]<<endl;

      // cout<<"mkl_result_mat->nrows = "<<mkl_result_mat->nrows<<endl;
      // cout<<"mkl_result_mat->ncols = "<<mkl_result_mat->ncols<<endl;
      

      csr_matrix_delete(ppr_matrix);
      ppr_matrix = NULL;


      coo_matrix_delete(ppr_matrix_coo);
      ppr_matrix_coo = NULL;



      // matrix_delete(mkl_result_mat);
      // mkl_result_mat = NULL;


      auto right_matrix_end_time = chrono::system_clock::now();
      auto elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(right_matrix_end_time - right_matrix_start_time);
      // cout<<"frobenius norm time = "<<elapsed_right_matrix_time.count()<<endl;


    }
      
    double total_frobenius = 0.0;

    for(int iter = 0; iter < d_row_tree->nParts; iter++){
      total_frobenius += frobenius_vec[iter];
    }


    // for(int iter = 0; iter < d_row_tree->nParts; iter++){
    //   total_nnz += nnz_nparts_vec[iter];
    // }

    cout << "Total Frobenius Norm Square as A - A * V * V^T = " << fixed << setprecision(12) << total_frobenius << endl;
    cout << "Total Frobenius Norm Square of the whole PPR Matrix = " << fixed << setprecision(12) << coo_frobenius_norm << endl;
    cout << "Total Error Percentage is " << (total_frobenius / coo_frobenius_norm) << endl;
    cout << "Total NNZ of the whole PPR Matrix = " << tot_nnz << endl;

    if(Final_Error != NULL) (*Final_Error) = (total_frobenius / coo_frobenius_norm);
    // total_frobenius /= total_nnz;

    // cout<<" total_frobenius /= total_nnz = "<<total_frobenius<<endl;
    
    cout<<"right_unique_update_times = "<<unique_update_times<<endl;

    // auto total_right_matrix_end_time = chrono::system_clock::now();
    // auto total_elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(total_right_matrix_end_time - total_right_matrix_start_time);
    // cout << "Total right matrix cost time: "<< total_elapsed_right_matrix_time.count() << endl;

    // cout << "Total norm cost time = "<< total_norm_time << endl;
    auto AVVT_en_time = chrono::system_clock::now();
    auto AVVT_comp_time = chrono::duration<double>(AVVT_en_time - AVVT_st_time);
    cout << "Retrieve PPR and Compute the F norm of || A - A * V * V^T || Time Use : " << AVVT_comp_time.count() << endl;

}
















































































void mkl_right_matrix_multiplication_nParts_NUMTHREAD(
d_row_tree_mkl* d_row_tree,
mat* mkl_left_matrix, 

// Eigen::MatrixXd &U_V,
// Eigen::MatrixXd &U,
Eigen::MatrixXd &V,

long long int vertex_number,
int d,
mat* SS,
int NUMTHREAD,
long long int largest_level_start_index,
long long int common_group_size,
long long int final_group_size,

// vector<map<long long int, double>>& all_csr_entries,
// vector<long long int>& offsets,


// vector< vector< vector<std::map<long long int, double>> > > & nParts_NUMTHREAD_all_csr_entries,

// vector<std::vector<long long int>>& nParts_NUMTHREAD_all_csr_offsets,

vector< vector< vector< vector<std::map<long long int, double>> > > >& nParts_NUMTHREAD_Partition_all_csr_entries,

vector< vector<std::vector<long long int>> >& nParts_NUMTHREAD_Partition_all_csr_offsets,

vector<ifstream>& ifs_vec,
vector<vector<long long int>>& nparts_all_count,
vector<long long int>& nnz_nparts_vec,
vector<vector<long long int>>& nparts_read_in_io_all_count,
int stored_interval

)
{
    // int unique_update_times = 0;
    
    auto total_right_matrix_start_time = chrono::system_clock::now();
    
    int nParts = d_row_tree->nParts;

    for(int iter = 0; iter < d_row_tree->nParts; iter++){

      // vector< vector<std::map<long long int, double>> > & NUMTHREAD_all_csr_entries = nParts_NUMTHREAD_all_csr_entries[iter];

      // std::vector<long long int> & NUMTHREAD_all_csr_offsets = nParts_NUMTHREAD_all_csr_offsets[iter];
      vector< vector< vector<std::map<long long int, double>> > > & NUMTHREAD_Partition_all_csr_entries = nParts_NUMTHREAD_Partition_all_csr_entries[iter];

      vector< std::vector<long long int> > & NUMTHREAD_Partition_all_csr_offsets = nParts_NUMTHREAD_Partition_all_csr_offsets[iter];

      int index = largest_level_start_index + iter;

      vector<long long int> &all_count = nparts_all_count[iter];
      vector<long long int> &read_in_io_all_count = nparts_read_in_io_all_count[iter];

      cout << "ppr retrieve " << endl;
      auto ppr_start_time = std::chrono::system_clock::now();
      vector<thread> threads;

      int order = index - largest_level_start_index;

      long long int start_column = order * common_group_size;

      long long int group_size;

      if(order == nParts - 1){
        group_size = final_group_size;
      }
      else{
        group_size = common_group_size;
      }

// cout<<0<<endl;

      // cout<<"group_size = "<<group_size<<endl;


      long long int nnz = nnz_nparts_vec[iter];

      long long int submatrix_rows = vertex_number;
      long long int submatrix_cols = group_size;


      cout<<"After nnz = "<<nnz<<endl;

      auto hash_coo_time = chrono::system_clock::now();

      cout<<"right matrix: 1"<<endl;

      // mat_coo *ppr_matrix_coo = coo_matrix_new(group_size, vertex_number, nnz);
      //等于transpose了
      mat_coo *ppr_matrix_coo = coo_matrix_new(group_size, vertex_number, nnz);

      ppr_matrix_coo->nnz = nnz;



      cout<<"right matrix: 2"<<endl;

      
      for (int t = 1; t <= NUMTHREAD; t++){
        // long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
        // long long int end = 0;
        // if (t == NUMTHREAD){
        //   end = start_column + group_size;
        // } else{
        //   end = start_column + t*(group_size/NUMTHREAD);
        // }

        // threads.push_back(thread(PPR_retrieve, 
        // // t-1, 
        // start, end, 
        // // std::ref(all_count[t-1]), 
        // std::ref(all_csr_entries),
        // std::ref(ifs_vec[t-1]),
        // std::ref(offsets),
        // std::ref(read_in_io_all_count[t-1])
        // ));

        // long long int start = (t-1)*(group_size/NUMTHREAD);
        // long long int end = 0;
        // if (t == NUMTHREAD){
        //   end = group_size;
        // } else{
        //   end = t*(group_size/NUMTHREAD);
        // }

        // threads.push_back(thread(parallel_store_matrix_nParts_NUMTHREAD, 
        // // std::ref(all_csr_entries),
        // std::ref(NUMTHREAD_all_csr_entries[t-1]),
        // ppr_matrix_coo, 
        // start,
        // end,
        // start_column,
        // all_count[t-1] ));
        

        // threads.push_back(thread(PPR_retrieve_nParts_NUMTHREAD, 
        // // t-1, 
        // // start, end, 
        // // std::ref(all_count[t-1]), 
        // // std::ref(all_csr_entries),
        // // std::ref(NUMTHREAD_all_csr_entries[t-1]),
        // std::ref(NUMTHREAD_Partition_all_csr_entries[t-1]),

        // std::ref(ifs_vec[t-1]),
        // // std::ref(offsets),
        // // std::ref(NUMTHREAD_all_csr_offsets[t-1]),
        // std::ref(NUMTHREAD_Partition_all_csr_offsets[t-1]),
        // std::ref(read_in_io_all_count[t-1])
        // ));

        PPR_retrieve_nParts_NUMTHREAD(
        NUMTHREAD_Partition_all_csr_entries[t-1],
        ifs_vec[t-1],
        NUMTHREAD_Partition_all_csr_offsets[t-1],
        read_in_io_all_count[t-1]
        );

      }

      // for (int t = 0; t < NUMTHREAD ; t++){
      //   threads[t].join();
      // }
      // vector<thread>().swap(threads);


      cout<<"right matrix: 3"<<endl;


    for (int t = 1; t <= NUMTHREAD; t++){

    //   long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
    //   long long int end = 0;
    //   if (t == NUMTHREAD){
    //     end = start_column + group_size;
    //   } else{
    //     end = start_column + t*(group_size/NUMTHREAD);
    //   }

    //   threads.push_back(thread(parallel_store_matrix, 
    //   std::ref(all_csr_entries),
    //   ppr_matrix_coo, 
    //   start,
    //   end,
    //   start_column,
    //   all_count[t-1] ));

    // }

      long long int start = (t-1)*(group_size/NUMTHREAD);
      long long int end = 0;
      if (t == NUMTHREAD){
        end = group_size;
      } else{
        end = t*(group_size/NUMTHREAD);
      }

      threads.push_back(thread(parallel_store_matrix_nParts_NUMTHREAD, 
      // std::ref(all_csr_entries),
      // std::ref(NUMTHREAD_all_csr_entries[t-1]),
      std::ref(NUMTHREAD_Partition_all_csr_entries[t-1]),

      ppr_matrix_coo, 
      start,
      end,
      // start_column,
      all_count[t-1],
      t,
      stored_interval
      ));
    }

    for (int t = 0; t < NUMTHREAD ; t++){
      threads[t].join();
    }
    vector<thread>().swap(threads);






      cout<<"right matrix: 4"<<endl;


      mat_csr* ppr_matrix = csr_matrix_new();


      cout<<"right matrix: 5"<<endl;

      csr_init_from_coo(ppr_matrix, ppr_matrix_coo);

      cout<<"right matrix: 6"<<endl;

      coo_matrix_delete(ppr_matrix_coo);


      cout<<"right matrix: 7"<<endl;


      ppr_matrix_coo = NULL;


      // cout<<"mkl_left_matrix->ncols = "<<mkl_left_matrix->ncols<<endl;

      //mkl_left_matrix->ncols == d
      mat *mkl_result_mat = matrix_new(group_size, mkl_left_matrix->ncols);

      auto right_matrix_start_time = chrono::system_clock::now();

      // csr_matrix_matrix_mult(ppr_matrix, mkl_left_matrix, mkl_result_mat);
      csr_matrix_matrix_mult(ppr_matrix, mkl_left_matrix, mkl_result_mat);

      csr_matrix_delete(ppr_matrix);
      ppr_matrix = NULL;

      mat *temp_mkl_result_mat = matrix_new(group_size, mkl_left_matrix->ncols);


      //This SS is already reversed
      matrix_matrix_mult(mkl_result_mat, SS, temp_mkl_result_mat);
      // second_pointer_matrix_matrix_mult(&mkl_result_mat, &SS, &temp_mkl_result_mat);

      matrix_copy(mkl_result_mat, temp_mkl_result_mat);

      // cout<<"mkl_result_mat->nrows = "<<mkl_result_mat->nrows<<endl;
      // cout<<"mkl_result_mat->ncols = "<<mkl_result_mat->ncols<<endl;
      

      matrix_delete(temp_mkl_result_mat);
      temp_mkl_result_mat = NULL;

      auto right_matrix_end_time = chrono::system_clock::now();
      auto elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(right_matrix_end_time - right_matrix_start_time);


      // cout<<"U_V.rows() = "<<U_V.rows()<<endl;
      // cout<<"U_V.cols() = "<<U_V.cols()<<endl;




      // for(long long int i = 0; i < group_size; i++){
      //     for(long long int j = 0; j < mkl_left_matrix->ncols; j++){
      //         U_V(iter * group_size + i, d + j) = matrix_get_element(mkl_result_mat, i, j);
      //         // if(isnan(U_V(iter * group_size + i, d + j)) || isinf(U_V(iter * group_size + i, d + j))){
      //         //     cout<<"U_V("<<i<<", "<<j<<") = "<<U_V(i, j)<<endl;
      //         // }
      //     }
      // }

      for (int t = 1; t <= NUMTHREAD; t++){
        long long int row_start = (t-1)*(group_size/NUMTHREAD);
        long long int row_end = 0;
        if (t == NUMTHREAD){
          row_end = group_size;
        } else{
          row_end = t * (group_size/NUMTHREAD);
        }

        // threads.push_back(thread(parallel_get_matrix_from_mkl, 
        // mkl_result_mat, 
        // std::ref(U_V), row_start, row_end, 0, d, start_column, d
        // ));

        threads.push_back(thread(parallel_get_matrix_from_mkl, 
        mkl_result_mat, 
        std::ref(V), row_start, row_end, 0, d, start_column, 0
        ));


      }

      for (int t = 0; t < NUMTHREAD ; t++){
        threads[t].join();
      }
      vector<thread>().swap(threads);

      // if(iter != d_row_tree->nParts-1){
      //   for(long long int i = 0; i < group_size; i++){
      //       for(long long int j = 0; j < mkl_left_matrix->ncols; j++){
      //           U_V(iter * group_size + i, d + j) = matrix_get_element(mkl_result_mat, i, j);
      //           // if(isnan(U_V(iter * group_size + i, d + j)) || isinf(U_V(iter * group_size + i, d + j))){
      //           //     cout<<"U_V("<<i<<", "<<j<<") = "<<U_V(i, j)<<endl;
      //           // }
      //       }
      //   }
      // }
      // else{
      //   for(long long int i = 0; i < group_size; i++){
      //       for(long long int j = 0; j < mkl_left_matrix->ncols; j++){
      //           U_V(vertex_number - group_size + i, d + j) = matrix_get_element(mkl_result_mat, i, j);
      //           // if(isnan(U_V(vertex_number - group_size + i, d + j)) || isinf(U_V(vertex_number - group_size + i, d + j))){
      //           //     cout<<"U_V("<<i<<", "<<j<<") = "<<U_V(i, j)<<endl;
      //           // }
      //       }
      //   }
      // }


      matrix_delete(mkl_result_mat);
      mkl_result_mat = NULL;

    // }
    // cout<<"right_unique_update_times = "<<unique_update_times<<endl;

    auto total_right_matrix_end_time = chrono::system_clock::now();
    auto total_elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(total_right_matrix_end_time - total_right_matrix_start_time);
    cout << "Total right matrix cost time: "<< total_elapsed_right_matrix_time.count() << endl;

    // cout << "Total norm cost time = "<< total_norm_time << endl;

  }
}

















void dynamic_single_mkl_right_matrix_multiplication(
d_row_tree_mkl* d_row_tree,
mat* mkl_left_matrix, 
Eigen::MatrixXd &V,
long long int vertex_number,
int d,
mat* SS,
int NUMTHREAD,
long long int largest_level_start_index,
long long int common_group_size,
long long int final_group_size,
// vector< vector<std::map<long long int, double>> >  & NUMTHREAD_all_csr_entries,
// std::vector<long long int>& NUMTHREAD_all_csr_offsets,
vector< vector< vector<std::map<long long int, double>> > > & NUMTHREAD_all_csr_entries,
vector< std::vector<long long int> > & NUMTHREAD_all_csr_offsets,
vector<ifstream>& ifs_vec,
vector<vector<long long int>>& nparts_all_count,
vector<long long int>& nnz_nparts_vec,
vector<vector<long long int>>& nparts_read_in_io_all_count,
int top_influenced_index,
int stored_interval
)
{
    int unique_update_times = 0;
    
    auto total_right_matrix_start_time = chrono::system_clock::now();
    
    int nParts = d_row_tree->nParts;

    // for(int iter = 0; iter < d_row_tree->nParts; iter++){
    int iter = top_influenced_index;
    
      int index = largest_level_start_index + iter;

      vector<long long int> &all_count = nparts_all_count[iter];
      vector<long long int> &read_in_io_all_count = nparts_read_in_io_all_count[iter];

      cout << "ppr retrieve " << endl;
      auto ppr_start_time = std::chrono::system_clock::now();
      vector<thread> threads;

      int order = index - largest_level_start_index;

      long long int start_column = order * common_group_size;

      long long int group_size;

      if(order == nParts - 1){
        group_size = final_group_size;
      }
      else{
        group_size = common_group_size;
      }

cout<<0<<endl;

      // cout<<"group_size = "<<group_size<<endl;


      long long int nnz = nnz_nparts_vec[iter];

      long long int submatrix_rows = vertex_number;
      long long int submatrix_cols = group_size;


cout<<1<<endl;

      cout<<"After nnz = "<<nnz<<endl;

      auto hash_coo_time = chrono::system_clock::now();

      // cout<<"sparse matrix: 4"<<endl;

      // mat_coo *ppr_matrix_coo = coo_matrix_new(group_size, vertex_number, nnz);
      mat_coo *ppr_matrix_coo = coo_matrix_new(group_size, vertex_number, nnz);

      ppr_matrix_coo->nnz = nnz;


cout<<2<<endl;



      for (int t = 1; t <= NUMTHREAD; t++){
        // long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
        // long long int end = 0;
        // if (t == NUMTHREAD){
        //   end = start_column + group_size;
        // } else{
        //   end = start_column + t*(group_size/NUMTHREAD);
        // }

        // threads.push_back(thread(PPR_retrieve, 
        // // t-1, 
        // start, end, 
        // // std::ref(all_count[t-1]), 
        // std::ref(all_csr_entries),
        // std::ref(ifs_vec[t-1]),
        // std::ref(offsets),
        // std::ref(read_in_io_all_count[t-1])
        // ));
        

        threads.push_back(thread(PPR_retrieve_nParts_NUMTHREAD, 
        // t-1, 
        // start, end, 
        // std::ref(all_count[t-1]), 
        // std::ref(all_csr_entries),
        std::ref(NUMTHREAD_all_csr_entries[t-1]),

        std::ref(ifs_vec[t-1]),
        // std::ref(offsets),
        std::ref(NUMTHREAD_all_csr_offsets[t-1]),
        std::ref(read_in_io_all_count[t-1])
        ));



      }

      for (int t = 0; t < NUMTHREAD ; t++){
        threads[t].join();
      }
      vector<thread>().swap(threads);


cout<<3<<endl;


    for (int t = 1; t <= NUMTHREAD; t++){

      // long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
      // long long int end = 0;
      // if (t == NUMTHREAD){
      //   end = start_column + group_size;
      // } else{
      //   end = start_column + t*(group_size/NUMTHREAD);
      // }

      // threads.push_back(thread(parallel_store_matrix, 
      // std::ref(all_csr_entries),
      // ppr_matrix_coo, 
      // start,
      // end,
      // start_column,
      // all_count[t-1] ));


      long long int start = (t-1)*(group_size/NUMTHREAD);
      long long int end = 0;
      if (t == NUMTHREAD){
        end = group_size;
      } else{
        end = t*(group_size/NUMTHREAD);
      }

      threads.push_back(thread(parallel_store_matrix_nParts_NUMTHREAD, 
        // std::ref(all_csr_entries),
        std::ref(NUMTHREAD_all_csr_entries[t-1]),
        ppr_matrix_coo, 
        start,
        end,
        // start_column,
        all_count[t-1],
        t,
        stored_interval
      )
      );

    }

    for (int t = 0; t < NUMTHREAD ; t++){
      threads[t].join();
    }
    vector<thread>().swap(threads);


cout<<4<<endl;


      mat_csr* ppr_matrix = csr_matrix_new();

      csr_init_from_coo(ppr_matrix, ppr_matrix_coo);

      coo_matrix_delete(ppr_matrix_coo);

      ppr_matrix_coo = NULL;


cout<<5<<endl;

      // cout<<"mkl_left_matrix->ncols = "<<mkl_left_matrix->ncols<<endl;

      //mkl_left_matrix->ncols == d
      mat *mkl_result_mat = matrix_new(group_size, mkl_left_matrix->ncols);

      auto right_matrix_start_time = chrono::system_clock::now();

      // csr_matrix_matrix_mult(ppr_matrix, mkl_left_matrix, mkl_result_mat);
      csr_matrix_matrix_mult(ppr_matrix, mkl_left_matrix, mkl_result_mat);

cout<<6<<endl;

      csr_matrix_delete(ppr_matrix);
      ppr_matrix = NULL;

cout<<7<<endl;

      mat *temp_mkl_result_mat = matrix_new(group_size, mkl_left_matrix->ncols);


      //This SS is already reversed
      matrix_matrix_mult(mkl_result_mat, SS, temp_mkl_result_mat);
      // second_pointer_matrix_matrix_mult(&mkl_result_mat, &SS, &temp_mkl_result_mat);

cout<<8<<endl;

      matrix_copy(mkl_result_mat, temp_mkl_result_mat);

      // cout<<"mkl_result_mat->nrows = "<<mkl_result_mat->nrows<<endl;
      // cout<<"mkl_result_mat->ncols = "<<mkl_result_mat->ncols<<endl;
      
cout<<9<<endl;

      matrix_delete(temp_mkl_result_mat);
      temp_mkl_result_mat = NULL;

      auto right_matrix_end_time = chrono::system_clock::now();
      auto elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(right_matrix_end_time - right_matrix_start_time);


      // cout<<"U_V.rows() = "<<U_V.rows()<<endl;
      // cout<<"U_V.cols() = "<<U_V.cols()<<endl;


cout<<10<<endl;


      // for(long long int i = 0; i < group_size; i++){
      //     for(long long int j = 0; j < mkl_left_matrix->ncols; j++){
      //         U_V(iter * group_size + i, d + j) = matrix_get_element(mkl_result_mat, i, j);
      //         // if(isnan(U_V(iter * group_size + i, d + j)) || isinf(U_V(iter * group_size + i, d + j))){
      //         //     cout<<"U_V("<<i<<", "<<j<<") = "<<U_V(i, j)<<endl;
      //         // }
      //     }
      // }

      for (int t = 1; t <= NUMTHREAD; t++){
        long long int row_start = (t-1)*(group_size/NUMTHREAD);
        long long int row_end = 0;
        if (t == NUMTHREAD){
          row_end = group_size;
        } else{
          row_end = t * (group_size/NUMTHREAD);
        }

        // threads.push_back(thread(parallel_get_matrix_from_mkl, 
        // mkl_result_mat, 
        // std::ref(U_V), row_start, row_end, 0, d, start_column, d
        // ));

        threads.push_back(thread(parallel_get_matrix_from_mkl, 
        mkl_result_mat, 
        std::ref(V), row_start, row_end, 0, d, start_column, 0
        ));


      }

      for (int t = 0; t < NUMTHREAD ; t++){
        threads[t].join();
      }
      vector<thread>().swap(threads);

cout<<11<<endl;

      // if(iter != d_row_tree->nParts-1){
      //   for(long long int i = 0; i < group_size; i++){
      //       for(long long int j = 0; j < mkl_left_matrix->ncols; j++){
      //           U_V(iter * group_size + i, d + j) = matrix_get_element(mkl_result_mat, i, j);
      //           // if(isnan(U_V(iter * group_size + i, d + j)) || isinf(U_V(iter * group_size + i, d + j))){
      //           //     cout<<"U_V("<<i<<", "<<j<<") = "<<U_V(i, j)<<endl;
      //           // }
      //       }
      //   }
      // }
      // else{
      //   for(long long int i = 0; i < group_size; i++){
      //       for(long long int j = 0; j < mkl_left_matrix->ncols; j++){
      //           U_V(vertex_number - group_size + i, d + j) = matrix_get_element(mkl_result_mat, i, j);
      //           // if(isnan(U_V(vertex_number - group_size + i, d + j)) || isinf(U_V(vertex_number - group_size + i, d + j))){
      //           //     cout<<"U_V("<<i<<", "<<j<<") = "<<U_V(i, j)<<endl;
      //           // }
      //       }
      //   }
      // }


      matrix_delete(mkl_result_mat);
      mkl_result_mat = NULL;

cout<<12<<endl;

    // }
    cout<<"right_unique_update_times = "<<unique_update_times<<endl;

    auto total_right_matrix_end_time = chrono::system_clock::now();
    auto total_elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(total_right_matrix_end_time - total_right_matrix_start_time);
    cout << "Total right matrix cost time: "<< total_elapsed_right_matrix_time.count() << endl;

    // cout << "Total norm cost time = "<< total_norm_time << endl;

}

































void dynamic_single_mkl_right_matrix_multiplication_new(
d_row_tree_mkl* d_row_tree,
mat* mkl_left_matrix, 
Eigen::MatrixXd &V,
long long int vertex_number,
int d,
mat* SS,
int NUMTHREAD,
long long int largest_level_start_index,
long long int common_group_size,
long long int final_group_size,
// // vector< vector<std::map<long long int, double>> >  & NUMTHREAD_all_csr_entries,
// // std::vector<long long int>& NUMTHREAD_all_csr_offsets,
// vector< vector< vector<std::map<long long int, double>> > > & NUMTHREAD_all_csr_entries,
// vector< std::vector<long long int> > & NUMTHREAD_all_csr_offsets,
// vector<ifstream>& ifs_vec,
vector<map<long long int, double>>& all_csr_entries,
vector<long long int>& offsets,
vector<ifstream>& ifs_vec,

vector<vector<long long int>>& nparts_all_count,
vector<long long int>& nnz_nparts_vec,
vector<vector<long long int>>& nparts_read_in_io_all_count,
int top_influenced_index
// , int stored_interval
)
{
    int unique_update_times = 0;
    
    auto total_right_matrix_start_time = chrono::system_clock::now();
    
    int nParts = d_row_tree->nParts;

    // for(int iter = 0; iter < d_row_tree->nParts; iter++){
    int iter = top_influenced_index;
    
      int index = largest_level_start_index + iter;

      vector<long long int> &all_count = nparts_all_count[iter];
      vector<long long int> &read_in_io_all_count = nparts_read_in_io_all_count[iter];

      cout << "ppr retrieve " << endl;
      auto ppr_start_time = std::chrono::system_clock::now();
      vector<thread> threads;

      int order = index - largest_level_start_index;

      long long int start_column = order * common_group_size;

      long long int group_size;

      if(order == nParts - 1){
        group_size = final_group_size;
      }
      else{
        group_size = common_group_size;
      }

cout<<0<<endl;

      // cout<<"group_size = "<<group_size<<endl;


      long long int nnz = nnz_nparts_vec[iter];

      long long int submatrix_rows = vertex_number;
      long long int submatrix_cols = group_size;


cout<<1<<endl;

      cout<<"After nnz = "<<nnz<<endl;

      auto hash_coo_time = chrono::system_clock::now();

      // cout<<"sparse matrix: 4"<<endl;

      // mat_coo *ppr_matrix_coo = coo_matrix_new(group_size, vertex_number, nnz);
      mat_coo *ppr_matrix_coo = coo_matrix_new(group_size, vertex_number, nnz);

      ppr_matrix_coo->nnz = nnz;


cout<<2<<endl;



//       for (int t = 1; t <= NUMTHREAD; t++){
//         // long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
//         // long long int end = 0;
//         // if (t == NUMTHREAD){
//         //   end = start_column + group_size;
//         // } else{
//         //   end = start_column + t*(group_size/NUMTHREAD);
//         // }

//         // threads.push_back(thread(PPR_retrieve, 
//         // // t-1, 
//         // start, end, 
//         // // std::ref(all_count[t-1]), 
//         // std::ref(all_csr_entries),
//         // std::ref(ifs_vec[t-1]),
//         // std::ref(offsets),
//         // std::ref(read_in_io_all_count[t-1])
//         // ));
        

//         threads.push_back(thread(PPR_retrieve_nParts_NUMTHREAD, 
//         // t-1, 
//         // start, end, 
//         // std::ref(all_count[t-1]), 
//         // std::ref(all_csr_entries),
//         std::ref(NUMTHREAD_all_csr_entries[t-1]),

//         std::ref(ifs_vec[t-1]),
//         // std::ref(offsets),
//         std::ref(NUMTHREAD_all_csr_offsets[t-1]),
//         std::ref(read_in_io_all_count[t-1])
//         ));



//       }

//       for (int t = 0; t < NUMTHREAD ; t++){
//         threads[t].join();
//       }
//       vector<thread>().swap(threads);


// cout<<3<<endl;


//     for (int t = 1; t <= NUMTHREAD; t++){

//       // long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
//       // long long int end = 0;
//       // if (t == NUMTHREAD){
//       //   end = start_column + group_size;
//       // } else{
//       //   end = start_column + t*(group_size/NUMTHREAD);
//       // }

//       // threads.push_back(thread(parallel_store_matrix, 
//       // std::ref(all_csr_entries),
//       // ppr_matrix_coo, 
//       // start,
//       // end,
//       // start_column,
//       // all_count[t-1] ));


//       long long int start = (t-1)*(group_size/NUMTHREAD);
//       long long int end = 0;
//       if (t == NUMTHREAD){
//         end = group_size;
//       } else{
//         end = t*(group_size/NUMTHREAD);
//       }

//       threads.push_back(thread(parallel_store_matrix_nParts_NUMTHREAD, 
//         // std::ref(all_csr_entries),
//         std::ref(NUMTHREAD_all_csr_entries[t-1]),
//         ppr_matrix_coo, 
//         start,
//         end,
//         // start_column,
//         all_count[t-1],
//         t,
//         stored_interval
//       )
//       );

//     }

//     for (int t = 0; t < NUMTHREAD ; t++){
//       threads[t].join();
//     }
//     vector<thread>().swap(threads);







      for (int t = 1; t <= NUMTHREAD; t++){
        long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
        long long int end = 0;
        if (t == NUMTHREAD){
          end = start_column + group_size;
        } 
        else{
          end = start_column + t*(group_size/NUMTHREAD);
        }

        threads.push_back(thread(PPR_retrieve, 
        // t-1, 
        start, end, 
        // std::ref(all_count[t-1]), 
        std::ref(all_csr_entries),
        std::ref(ifs_vec[t-1]),
        std::ref(offsets),
        std::ref(read_in_io_all_count[t-1])
        ));

      }

      for (int t = 0; t < NUMTHREAD ; t++){
        threads[t].join();
      }
      vector<thread>().swap(threads);




    for (int t = 1; t <= NUMTHREAD; t++){

      long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
      long long int end = 0;
      if (t == NUMTHREAD){
        end = start_column + group_size;
      } else{
        end = start_column + t*(group_size/NUMTHREAD);
      }

      threads.push_back(thread(parallel_store_matrix, 
      std::ref(all_csr_entries),
      ppr_matrix_coo, 
      start,
      end,
      start_column,
      all_count[t-1] ));

    }

    for (int t = 0; t < NUMTHREAD ; t++){
      threads[t].join();
    }
    vector<thread>().swap(threads);


cout<<4<<endl;


      mat_csr* ppr_matrix = csr_matrix_new();

      csr_init_from_coo(ppr_matrix, ppr_matrix_coo);

      coo_matrix_delete(ppr_matrix_coo);

      ppr_matrix_coo = NULL;


cout<<5<<endl;

      // cout<<"mkl_left_matrix->ncols = "<<mkl_left_matrix->ncols<<endl;

      //mkl_left_matrix->ncols == d
      mat *mkl_result_mat = matrix_new(group_size, mkl_left_matrix->ncols);

      auto right_matrix_start_time = chrono::system_clock::now();

      // csr_matrix_matrix_mult(ppr_matrix, mkl_left_matrix, mkl_result_mat);
      csr_matrix_matrix_mult(ppr_matrix, mkl_left_matrix, mkl_result_mat);

cout<<6<<endl;

      csr_matrix_delete(ppr_matrix);
      ppr_matrix = NULL;

cout<<7<<endl;

      mat *temp_mkl_result_mat = matrix_new(group_size, mkl_left_matrix->ncols);


      //This SS is already reversed
      matrix_matrix_mult(mkl_result_mat, SS, temp_mkl_result_mat);
      // second_pointer_matrix_matrix_mult(&mkl_result_mat, &SS, &temp_mkl_result_mat);

cout<<8<<endl;

      matrix_copy(mkl_result_mat, temp_mkl_result_mat);

      // cout<<"mkl_result_mat->nrows = "<<mkl_result_mat->nrows<<endl;
      // cout<<"mkl_result_mat->ncols = "<<mkl_result_mat->ncols<<endl;
      
cout<<9<<endl;

      matrix_delete(temp_mkl_result_mat);
      temp_mkl_result_mat = NULL;

      auto right_matrix_end_time = chrono::system_clock::now();
      auto elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(right_matrix_end_time - right_matrix_start_time);


      // cout<<"U_V.rows() = "<<U_V.rows()<<endl;
      // cout<<"U_V.cols() = "<<U_V.cols()<<endl;


cout<<10<<endl;


      // for(long long int i = 0; i < group_size; i++){
      //     for(long long int j = 0; j < mkl_left_matrix->ncols; j++){
      //         U_V(iter * group_size + i, d + j) = matrix_get_element(mkl_result_mat, i, j);
      //         // if(isnan(U_V(iter * group_size + i, d + j)) || isinf(U_V(iter * group_size + i, d + j))){
      //         //     cout<<"U_V("<<i<<", "<<j<<") = "<<U_V(i, j)<<endl;
      //         // }
      //     }
      // }

      for (int t = 1; t <= NUMTHREAD; t++){
        long long int row_start = (t-1)*(group_size/NUMTHREAD);
        long long int row_end = 0;
        if (t == NUMTHREAD){
          row_end = group_size;
        } else{
          row_end = t * (group_size/NUMTHREAD);
        }

        // threads.push_back(thread(parallel_get_matrix_from_mkl, 
        // mkl_result_mat, 
        // std::ref(U_V), row_start, row_end, 0, d, start_column, d
        // ));

        threads.push_back(thread(parallel_get_matrix_from_mkl, 
        mkl_result_mat, 
        std::ref(V), row_start, row_end, 0, d, start_column, 0
        ));


      }

      for (int t = 0; t < NUMTHREAD ; t++){
        threads[t].join();
      }
      vector<thread>().swap(threads);

cout<<11<<endl;

      // if(iter != d_row_tree->nParts-1){
      //   for(long long int i = 0; i < group_size; i++){
      //       for(long long int j = 0; j < mkl_left_matrix->ncols; j++){
      //           U_V(iter * group_size + i, d + j) = matrix_get_element(mkl_result_mat, i, j);
      //           // if(isnan(U_V(iter * group_size + i, d + j)) || isinf(U_V(iter * group_size + i, d + j))){
      //           //     cout<<"U_V("<<i<<", "<<j<<") = "<<U_V(i, j)<<endl;
      //           // }
      //       }
      //   }
      // }
      // else{
      //   for(long long int i = 0; i < group_size; i++){
      //       for(long long int j = 0; j < mkl_left_matrix->ncols; j++){
      //           U_V(vertex_number - group_size + i, d + j) = matrix_get_element(mkl_result_mat, i, j);
      //           // if(isnan(U_V(vertex_number - group_size + i, d + j)) || isinf(U_V(vertex_number - group_size + i, d + j))){
      //           //     cout<<"U_V("<<i<<", "<<j<<") = "<<U_V(i, j)<<endl;
      //           // }
      //       }
      //   }
      // }


      matrix_delete(mkl_result_mat);
      mkl_result_mat = NULL;

cout<<12<<endl;

    // }
    cout<<"right_unique_update_times = "<<unique_update_times<<endl;

    auto total_right_matrix_end_time = chrono::system_clock::now();
    auto total_elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(total_right_matrix_end_time - total_right_matrix_start_time);
    cout << "Total right matrix cost time: "<< total_elapsed_right_matrix_time.count() << endl;

    // cout << "Total norm cost time = "<< total_norm_time << endl;

}


















































void dynamic_single_mkl_right_matrix_multiplication_MAE(
d_row_tree_mkl* d_row_tree,
mat* mkl_left_matrix, 
vector<double>& MAE_vec,
long long int vertex_number,
int d,
mat* SS,
int NUMTHREAD,
long long int largest_level_start_index,
long long int common_group_size,
long long int final_group_size,
vector<map<long long int, double>>& all_csr_entries,
vector<long long int>& offsets,
vector<ifstream>& ifs_vec,
vector<vector<long long int>>& nparts_all_count,
vector<long long int>& nnz_nparts_vec,
vector<vector<long long int>>& nparts_read_in_io_all_count,
int top_influenced_index
)
{
    int unique_update_times = 0;
    
    auto total_right_matrix_start_time = chrono::system_clock::now();
    
    int nParts = d_row_tree->nParts;

    int iter = top_influenced_index;
    
      int index = largest_level_start_index + iter;

      vector<long long int> &all_count = nparts_all_count[iter];
      vector<long long int> &read_in_io_all_count = nparts_read_in_io_all_count[iter];

      cout << "ppr retrieve " << endl;
      auto ppr_start_time = std::chrono::system_clock::now();
      vector<thread> threads;

      int order = index - largest_level_start_index;

      long long int start_column = order * common_group_size;

      long long int group_size;

      if(order == nParts - 1){
        group_size = final_group_size;
      }
      else{
        group_size = common_group_size;
      }

cout<<0<<endl;

      long long int nnz = nnz_nparts_vec[iter];
      long long int submatrix_rows = vertex_number;
      long long int submatrix_cols = group_size;


cout<<1<<endl;

      cout<<"After nnz = "<<nnz<<endl;

      auto hash_coo_time = chrono::system_clock::now();

      mat_coo *ppr_matrix_coo = coo_matrix_new(group_size, vertex_number, nnz);

      ppr_matrix_coo->nnz = nnz;


cout<<2<<endl;


      for (int t = 1; t <= NUMTHREAD; t++){
        long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
        long long int end = 0;
        if (t == NUMTHREAD){
          end = start_column + group_size;
        } 
        else{
          end = start_column + t*(group_size/NUMTHREAD);
        }

        threads.push_back(thread(PPR_retrieve, 
        // t-1, 
        start, end, 
        // std::ref(all_count[t-1]), 
        std::ref(all_csr_entries),
        std::ref(ifs_vec[t-1]),
        std::ref(offsets),
        std::ref(read_in_io_all_count[t-1])
        ));

      }

      for (int t = 0; t < NUMTHREAD ; t++){
        threads[t].join();
      }
      vector<thread>().swap(threads);




    for (int t = 1; t <= NUMTHREAD; t++){

      long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
      long long int end = 0;
      if (t == NUMTHREAD){
        end = start_column + group_size;
      } else{
        end = start_column + t*(group_size/NUMTHREAD);
      }

      threads.push_back(thread(parallel_store_matrix, 
      std::ref(all_csr_entries),
      ppr_matrix_coo, 
      start,
      end,
      start_column,
      all_count[t-1] ));

    }

    for (int t = 0; t < NUMTHREAD ; t++){
      threads[t].join();
    }
    vector<thread>().swap(threads);


cout<<4<<endl;


      mat_csr* ppr_matrix = csr_matrix_new();

      csr_init_from_coo(ppr_matrix, ppr_matrix_coo);


cout<<5<<endl;

      // cout<<"mkl_left_matrix->ncols = "<<mkl_left_matrix->ncols<<endl;

      //mkl_left_matrix->ncols == d
      mat *mkl_result_mat = matrix_new(group_size, mkl_left_matrix->ncols);

      auto right_matrix_start_time = chrono::system_clock::now();

      csr_matrix_matrix_mult(ppr_matrix, mkl_left_matrix, mkl_result_mat);

      csr_matrix_delete(ppr_matrix);
      ppr_matrix = NULL;


      matrix_get_mean_absolute_error(mkl_left_matrix, mkl_result_mat, 
                                        ppr_matrix_coo, &MAE_vec[iter]);


      coo_matrix_delete(ppr_matrix_coo);

      ppr_matrix_coo = NULL;


      matrix_delete(mkl_result_mat);
      mkl_result_mat = NULL;


      long long int total_nnz = 0;
      for(int i = 0; i < nnz_nparts_vec.size(); i++){
        total_nnz += nnz_nparts_vec[i];
      }

      double total_MAE = 0.0;

      for(int iter = 0; iter < d_row_tree->nParts; iter++){
        total_MAE += MAE_vec[iter];
      }

      total_MAE /= total_nnz;

      cout<<"Single part update: total_MAE = "<<total_MAE<<endl;


      auto right_matrix_end_time = chrono::system_clock::now();
      auto elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(right_matrix_end_time - right_matrix_start_time);


cout<<10<<endl;


cout<<11<<endl;


cout<<12<<endl;

    cout<<"right_unique_update_times = "<<unique_update_times<<endl;

    auto total_right_matrix_end_time = chrono::system_clock::now();
    auto total_elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(total_right_matrix_end_time - total_right_matrix_start_time);
    cout << "Total right matrix cost time: "<< total_elapsed_right_matrix_time.count() << endl;

}



























void dynamic_single_mkl_right_matrix_multiplication_frobenius(
d_row_tree_mkl* d_row_tree,
mat* mkl_left_matrix, 
vector<double>& frobenius_vec,
long long int vertex_number,
int d,
mat* SS,
int NUMTHREAD,
long long int largest_level_start_index,
long long int common_group_size,
long long int final_group_size,
vector<map<long long int, double>>& all_csr_entries,
vector<long long int>& offsets,
vector<ifstream>& ifs_vec,
vector<vector<long long int>>& nparts_all_count,
vector<long long int>& nnz_nparts_vec,
vector<vector<long long int>>& nparts_read_in_io_all_count,
int top_influenced_index
)
{
    int unique_update_times = 0;
    
    auto total_right_matrix_start_time = chrono::system_clock::now();
    
    int nParts = d_row_tree->nParts;

    int iter = top_influenced_index;
    
      int index = largest_level_start_index + iter;

      vector<long long int> &all_count = nparts_all_count[iter];
      vector<long long int> &read_in_io_all_count = nparts_read_in_io_all_count[iter];

      cout << "ppr retrieve " << endl;
      auto ppr_start_time = std::chrono::system_clock::now();
      vector<thread> threads;

      int order = index - largest_level_start_index;

      long long int start_column = order * common_group_size;

      long long int group_size;

      if(order == nParts - 1){
        group_size = final_group_size;
      }
      else{
        group_size = common_group_size;
      }

cout<<0<<endl;

      long long int nnz = nnz_nparts_vec[iter];
      long long int submatrix_rows = vertex_number;
      long long int submatrix_cols = group_size;


cout<<1<<endl;

      cout<<"After nnz = "<<nnz<<endl;

      auto hash_coo_time = chrono::system_clock::now();

      mat_coo *ppr_matrix_coo = coo_matrix_new(group_size, vertex_number, nnz);

      ppr_matrix_coo->nnz = nnz;


cout<<2<<endl;


      for (int t = 1; t <= NUMTHREAD; t++){
        long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
        long long int end = 0;
        if (t == NUMTHREAD){
          end = start_column + group_size;
        } 
        else{
          end = start_column + t*(group_size/NUMTHREAD);
        }

        threads.push_back(thread(PPR_retrieve, 
        // t-1, 
        start, end, 
        // std::ref(all_count[t-1]), 
        std::ref(all_csr_entries),
        std::ref(ifs_vec[t-1]),
        std::ref(offsets),
        std::ref(read_in_io_all_count[t-1])
        ));

      }

      for (int t = 0; t < NUMTHREAD ; t++){
        threads[t].join();
      }
      vector<thread>().swap(threads);




    for (int t = 1; t <= NUMTHREAD; t++){

      long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
      long long int end = 0;
      if (t == NUMTHREAD){
        end = start_column + group_size;
      } else{
        end = start_column + t*(group_size/NUMTHREAD);
      }

      threads.push_back(thread(parallel_store_matrix, 
      std::ref(all_csr_entries),
      ppr_matrix_coo, 
      start,
      end,
      start_column,
      all_count[t-1] ));

    }

    for (int t = 0; t < NUMTHREAD ; t++){
      threads[t].join();
    }
    vector<thread>().swap(threads);


cout<<4<<endl;


      mat_csr* ppr_matrix = csr_matrix_new();

      csr_init_from_coo(ppr_matrix, ppr_matrix_coo);


cout<<5<<endl;

      auto right_matrix_start_time = chrono::system_clock::now();


      frobenius_vec[iter] = get_matrix_frobenius_A_minus_AVVT(ppr_matrix, mkl_left_matrix);

      cout<<"frobenius_vec["<<iter<<"] = "<<frobenius_vec[iter]<<endl;

      double Original_F_Norm = get_sparse_matrix_frobenius_norm_square(ppr_matrix);

      cout<<"Original ppr matrix frobenius_norm = "<< Original_F_Norm <<endl;

      csr_matrix_delete(ppr_matrix);
      ppr_matrix = NULL;


      coo_matrix_delete(ppr_matrix_coo);

      ppr_matrix_coo = NULL;





      // long long int total_nnz = 0;
      // for(int i = 0; i < nnz_nparts_vec.size(); i++){
      //   total_nnz += nnz_nparts_vec[i];
      // }

      double total_frobenius = 0.0;

      for(int iter = 0; iter < d_row_tree->nParts; iter++){
        total_frobenius += frobenius_vec[iter];
      }

      // total_frobenius /= total_nnz;

      cout<<"Single part update: total_frobenius = "<<total_frobenius<<endl;


      auto right_matrix_end_time = chrono::system_clock::now();
      auto elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(right_matrix_end_time - right_matrix_start_time);
      cout<<"frobenius norm time = "<<elapsed_right_matrix_time.count()<<endl;

cout<<10<<endl;


cout<<11<<endl;


cout<<12<<endl;

    cout<<"right_unique_update_times = "<<unique_update_times<<endl;

    auto total_right_matrix_end_time = chrono::system_clock::now();
    auto total_elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(total_right_matrix_end_time - total_right_matrix_start_time);
    cout << "Total right matrix cost time: "<< total_elapsed_right_matrix_time.count() << endl;

}











































































































void get_left_matrix_from_mkl(mat* mkl, MatrixXd &Eig){
  // Eig.resize(mkl->nrows, mkl->ncols);
  for(long long int i = 0; i < mkl->nrows; i++){
    for(long long int j = 0; j < mkl->ncols; j++){
      Eig(i, j) = matrix_get_element(mkl, i, j);
    }
  }
}










void save_dense_emb(mat* M, string filename)
{
    MKL_INT rows = M->nrows, cols = M->ncols;
    FILE *fid = fopen(filename.c_str(), "wb");
    fwrite(M->d, sizeof(double), rows*cols, fid);
    fclose(fid);
    return;
}















void read_dense_emb(mat* M, string filename)
{
    MKL_INT rows = M->nrows, cols = M->ncols;
    FILE *fid = fopen(filename.c_str(), "rb");
    fread(M->d, sizeof(double), rows*cols, fid);
    fclose(fid);
    return;
}





















