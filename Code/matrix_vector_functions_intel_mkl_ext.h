#pragma once

#include "matrix_vector_functions_intel_mkl.h"
// #define MKL_INT MKL_INT64
#define MKL_INT long long int


/* 
    The rows/cols fields use 1-based indexing. This is due to an undocumented 
    feature of MKL library: if you are trying to cooperate sparse matrix with 
    row-major layout dense matrices, then the interfaces assume the indexing 
    is 0-based. Otherwise if you use column major dense matrices, you should 
    use 1-based indexing.
*/

// typedef struct {
//     int nrows, ncols;
//     long long int nnz; // number of non-zero element in the matrix.
//     long long int capacity; // number of possible nnzs.
//     double *values;
//     int *rows, *cols;
// } mat_coo;

typedef struct {
    MKL_INT nrows, ncols;
    MKL_INT nnz; // number of non-zero element in the matrix.
    MKL_INT capacity; // number of possible nnzs.
    double *values;
    MKL_INT *rows, *cols;
} mat_coo;

typedef struct {
    MKL_INT nrows, ncols;
    MKL_INT nnz; // number of non-zero element in the matrix.
    MKL_INT capacity; // number of possible nnzs.
    double *values;
    MKL_INT *cols;
    int *rows;
} mat_coo_2;


// typedef struct{
//     // public:
//     long long int nnz;
//     int nrows, ncols;
//     double *values;
//     int *cols;
//     int *pointerB, *pointerE;
//     // int test: 10;
//     // ~point(){
//     //     fprintf("mat_csr deleted!\n");
//     // }
// } mat_csr;

typedef struct{
    // public:
    MKL_INT nnz;
    MKL_INT nrows, ncols;
    double *values;
    MKL_INT *cols;
    MKL_INT *pointerB, *pointerE;
    // int test: 10;
    // ~point(){
    //     fprintf("mat_csr deleted!\n");
    // }
} mat_csr;




// initialize with sizes, the only interface that allocates space for coo struct
mat_coo* coo_matrix_new(MKL_INT nrows, MKL_INT ncols, MKL_INT capacity);

mat_coo_2 *coo_2_matrix_new(MKL_INT nrows, MKL_INT ncols, MKL_INT capacity);
// collect allocated space.
void coo_matrix_delete(mat_coo *M);

void coo_2_matrix_delete(mat_coo_2 *M);

void set_coo_matrix_element(mat_coo *M, MKL_INT row, MKL_INT col, double val, MKL_INT force_new);

void coo_matrix_print(mat_coo *M);

void coo_matrix_matrix_mult(mat_coo *A, mat *B, mat *C);

void coo_matrix_copy_to_dense(mat_coo *A, mat *B);

void gen_rand_coo_matrix(mat_coo *M, double density);

// return a pointer, but nothing inside.
mat_csr* csr_matrix_new();

// collect the space
void csr_matrix_delete(mat_csr *M);

void csr_matrix_delete_LP(mat_csr *M);

void csr_matrix_print(mat_csr *M);

// the only interface that allocates space for mat_csr struct and initialize with M
void csr_init_from_coo(mat_csr *D, mat_coo *M);

void csr_init_from_coo_2(mat_csr *D, mat_coo_2 *M);

void csr_matrix_matrix_mult(mat_csr *A, mat *B, mat *C);

void csr_matrix_transpose_matrix_mult(mat_csr *A, mat *B, mat *C);

//Algorithms by Xu Feng
void LUfraction(mat *A, mat *L);

void eigSVD(mat *A, mat **U, mat **S, mat **V);

void frPCA(mat_csr *A, mat **U, mat **S, mat **V, MKL_INT k, MKL_INT p);

void frPCAt(mat_csr *A, mat **U, mat **S, mat **V, MKL_INT k, MKL_INT p);

void frPCAt_onlyUS(mat_csr *A, mat **U, mat **S, MKL_INT k, MKL_INT q);


void randQB_basic_csr(mat_csr *M, MKL_INT k, MKL_INT p, mat **U, mat **S, mat **V);


void randomized_SVD(mat_csr *M, MKL_INT k, MKL_INT p, mat **U, mat **S, mat **V);


// void right_matrix_mkl(mat_csr *transpose_ppr_matrix, mat** mkl_left_matrix, mat** mkl_singular_value_mat, mat** mkl_temp_mul, mat** mkl_result_mat);
void right_matrix_mkl(mat_csr *transpose_ppr_matrix, mat** mkl_left_matrix, mat** mkl_result_mat);


void matrix_get_mean_absolute_error(mat *left, mat *right, mat_coo* original_matrix, double *norm);


double get_matrix_frobenius_A_minus_AVVT(mat_csr *A, mat *V);


double get_sparse_matrix_frobenius_norm_square(mat_csr *A);









