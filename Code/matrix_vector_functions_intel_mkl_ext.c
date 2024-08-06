#include <stdio.h>
// #include <iostream>

#define MKL_INT long long int
// #define MKL_INT MKL_LONG
// #define MKL_INT MKL_INT64

#include "matrix_vector_functions_intel_mkl.h"
#include "matrix_vector_functions_intel_mkl_ext.h"
#include <math.h>
#include <unistd.h>
#include "mkl.h"

// #include"mkl_spblas.h"

#include <omp.h>

double get_matrix_frobenius_A_minus_AVVT(mat_csr *A, mat *V)
{
    int m = A->nrows; // group_size
    int n = A->ncols; // vertex_number
    int k = V->ncols; // d


    double coo_Fnorm = 0;
    double A_Fnorm = 0;


    mat *AV = matrix_new(m, k);

    csr_matrix_matrix_mult(A, V, AV);

    double Ans = get_matrix_frobenius_norm_square(AV);
    printf("AV F norm square (equal to A*V*V^T's) = %.12lf\n", Ans);
    fflush(stdout);
    
    double Ans1 = 0;//, Ans2 = 0;

#pragma omp parallel for reduction(+ : Ans1)
    for (MKL_INT i = 0; i < A->nnz; i++)
    {
        Ans1 += A->values[i] * A->values[i];
    }

    Ans = Ans1 - Ans;

    matrix_delete(AV);
    printf("F norm Error of A - A * V * V ^ T = %.12lf\n", Ans);
    fflush(stdout);

    return Ans;
}

void get_matrix_frobenius_norm_square_non_parallel(mat *M, double *normval)
{
    MKL_INT i;

    // double val, normval = 0.0;  // Initialize normval to 0.0

    MKL_INT multiply_dimension = (int64_t)(M->nrows) * (M->ncols);

    printf("M->nrows = %lld\n", M->nrows);
    printf("M->ncols = %lld\n", M->ncols);

    printf("multiply_dimension = %lld\n", multiply_dimension);

    double val = 0.0;
    // double normval = 0.0;  // Initialize normval to 0.0

    // #pragma omp parallel shared(M) private(i, val)
    // {
    //     #pragma omp for reduction(+:normval)
    for (i = 0; i < multiply_dimension; i++)
    {
        // normval += val * val;
        val = abs(M->d[i] * M->d[i]);
        // printf("%f ", val);
        *normval += val;
        // printf("%f ", normval);
    }
    // printf("\n");
    // }
    // return normval;
}

// double get_matrix_frobenius_A_minus_AVVT(mat_coo *coo_matrix, mat_csr *A, mat *V) {
//     int m = A->nrows; // group_size
//     int n = A->ncols; // vertex_number
//     // V->nrows == vertex_number
//     int k = V->ncols; // d

//     // 先计算 A 的frobenius norm
//     // 计算 A^TV^T * V^TA^T 的frobenius norm, 记作a
//     // 计算A的nnz坐标对应的AV * V^TA^T，记作b
//     // 计算 c = a - b
//     // 计算 A - A的nnz坐标对应的AV * V^TA^T, 记作d
//     // 计算 c + d作为最终结果

//     mat *AV = matrix_new(m, k);

//     csr_matrix_matrix_mult(A, V, AV);

//     mat *KK = matrix_new(k, k);

//     matrix_transpose_matrix_mult(AV, AV, KK);

//     double a = get_matrix_frobenius_norm_square(KK);

//     printf("a = %f\n", a);

//     MKL_INT i;

//     double b = 0.0;

//     // #pragma omp parallel for reduction(+:b) private(i)
//     // for (i = 0; i < coo_matrix->nnz; i++) {
//     //     MKL_INT row = coo_matrix->rows[i] - 1;
//     //     MKL_INT col = coo_matrix->cols[i] - 1;

//     //     double diff = coo_matrix->values[i] * coo_matrix->values[i];
//     //     b += diff * diff;
//     // }

//     double d = 0.0;

//     #pragma omp parallel for reduction(+:d) private(i)
//     for (i = 0; i < coo_matrix->nnz; i++) {
//         MKL_INT row = coo_matrix->rows[i] - 1;
//         MKL_INT col = coo_matrix->cols[i] - 1;

//         vec *left_row = vector_new(AV->ncols);
//         vec *right_row = vector_new(V->ncols);

//         matrix_get_row(AV, row, left_row);
//         matrix_get_row(V, col, right_row);

//         double product_val = vector_dot_product(left_row, right_row);
//         b += product_val;
//         double diff = coo_matrix->values[i] - product_val;
//         d += diff * diff;

//         vector_delete(left_row);
//         vector_delete(right_row);
//     }

//     double c = a - b;

//     printf("b = %f\n", b);

//     printf("c = %f\n", c);

//     printf("d = %f\n", d);

//     return c + d;
// }

// double get_matrix_frobenius_norm_square(mat *M) {
// double get_matrix_frobenius_norm_square(mat *M)
// {
//     MKL_INT i;

//     // double val, normval = 0.0;  // Initialize normval to 0.0

//     MKL_INT multiply_dimension = (int64_t)(M->nrows) * (M->ncols);

//     printf("M->nrows = %lld\n", M->nrows);
//     printf("M->ncols = %lld\n", M->ncols);

//     printf("multiply_dimension = %lld\n", multiply_dimension);

//     double val, normval = 0.0; // Initialize normval to 0.0

// #pragma omp parallel shared(M) private(i, val)
//     {
// #pragma omp for reduction(+ : normval)
//         for (i = 0; i < multiply_dimension; i++)
//         {
//             // val = M->d[i];
//             // normval += val * val;
//             normval += M->d[i] * M->d[i];
//         }
//     }
//     return normval;
// }

// /* matrix frobenius norm */
// double get_matrix_frobenius_norm_square(mat *M){
//     MKL_INT i;
//     double val, normval = 0;
//     #pragma omp parallel shared(M,normval) private(i,val)
//     {
//         #pragma omp for reduction(+:normval)
//         for(i=0; i<((M->nrows)*(M->ncols)); i++){
//             val = M->d[i];
//             normval += val*val;
//         }
//     }
//     return normval;
// }

void matrix_get_mean_absolute_error(mat *left, mat *right, mat_coo *original_matrix, double *norm)
{
    MKL_INT i;
    double local_norm = 0.0;

#pragma omp parallel for reduction(+ : local_norm) private(i)
    for (i = 0; i < original_matrix->nnz; i++)
    {
        MKL_INT row = original_matrix->rows[i] - 1;
        MKL_INT col = original_matrix->cols[i] - 1;

        vec *left_row = vector_new(left->ncols);
        vec *right_row = vector_new(right->ncols);

        matrix_get_row(left, row, left_row);
        matrix_get_row(right, col, right_row);

        double diff = original_matrix->values[i] - vector_dot_product(left_row, right_row);
        local_norm += fabs(diff);

        vector_delete(left_row);
        vector_delete(right_row);
    }

    *norm = local_norm;
    // *norm /= original_matrix->nnz; // Uncomment this line if normalization is intended after the loop
}

void csr_init_from_coo(mat_csr *D, mat_coo *M)
{

    // printf("csr_init_from_coo : 1\n");
    
    D->nrows = M->nrows;
    D->ncols = M->ncols;

    // fflush(stdout); printf("csr_init_from_coo : 2\n");
    // printf("%lld %lld %lld\n", D->nrows, D->ncols, M->nnz);
    // cout << D->nrows << endl;
    // cout << D->ncols << endl;
    // cout << D->nnz << endl;
    // printf("6666\n");
    
    D->pointerB = (MKL_INT *)calloc(D->nrows, sizeof(MKL_INT));

    // sleep(3);
    
    // fflush(stdout); printf("csr_init_from_coo : 3\n");
    // fflush(stdout);
    // // sleep(3);

    D->pointerE = (MKL_INT *)calloc(D->nrows, sizeof(MKL_INT));

    // fflush(stdout); printf("csr_init_from_coo : 3\n");
    // fflush(stdout);
    // sleep(1);

    D->cols = (MKL_INT *)calloc(M->nnz, sizeof(MKL_INT));

    // fflush(stdout); printf("csr_init_from_coo : 4\n");
    // fflush(stdout);
    // sleep(1);

    D->nnz = M->nnz;

    // fflush(stdout); printf("csr_init_from_coo : 5\n");

    D->values = (double *)calloc(M->nnz, sizeof(double));

    // fflush(stdout); printf("csr_init_from_coo : 6\n");

    for (MKL_INT i = 0; i < M->nnz; i++)
    {
        D->values[i] = M->values[i];
    }

    // fflush(stdout); printf("csr_init_from_coo : 7\n");

    MKL_INT current_row, cursor = 0;
    for (current_row = 0; current_row < D->nrows; current_row++)
    {
        D->pointerB[current_row] = cursor + 1;
        while (cursor < M->nnz && M->rows[cursor] - 1 == current_row)
        {
            D->cols[cursor] = M->cols[cursor];

            cursor++;
        }
        D->pointerE[current_row] = cursor + 1;
    }

    // fflush(stdout); printf("csr_init_from_coo : 8\n"); fflush(stdout);

    if (D->pointerB == NULL)
    {
        printf("ERROR! pointerB allocation failure!!!\n");
    }
    if (D->pointerE == NULL)
    {
        printf("ERROR! pointerE allocation failure!!!\n");
    }
    if (D->cols == NULL)
    {
        printf("ERROR! cols allocation failure!!!\n");
    }
}

void csr_init_from_coo_2(mat_csr *D, mat_coo_2 *M)
{

    D->nrows = M->nrows;
    D->ncols = M->ncols;
    
    D->pointerB = (MKL_INT *)calloc(D->nrows, sizeof(MKL_INT));
    D->pointerE = (MKL_INT *)calloc(D->nrows, sizeof(MKL_INT));
    D->cols = M->cols;

    D->nnz = M->nnz;
    D->values = M->values;

    MKL_INT current_row, cursor = 0;
    for (current_row = 0; current_row < D->nrows; current_row++)
    {
        D->pointerB[current_row] = cursor + 1;
        while (cursor < M->nnz && M->rows[cursor] - 1 == current_row)
            cursor++;
        D->pointerE[current_row] = cursor + 1;
    }

    M->values = NULL;
    M->cols = NULL;
    if (D->pointerB == NULL) printf("ERROR! pointerB allocation failure!!!\n");
    if (D->pointerE == NULL) printf("ERROR! pointerE allocation failure!!!\n");
    if (D->cols == NULL) printf("ERROR! cols allocation failure!!!\n");
    
}


// double get_matrix_frobenius_A_minus_AVVT(mat_csr *A, mat *V) // Computes the F norm of matrix A-A*V*V^T  A:(m x n) V:(n x k)
// {
//     mat *LV = matrix_new(A->nrows, V->ncols);
//     // int k = V->ncols;
//     MKL_INT j, k, h, u, k1, k2;
//     k = V->ncols;
//     #pragma omp parallel shared(LV, V, A, k) private(j, h, u)
//     {
//         #pragma omp parallel for
//         for (j = 0; j < A->nrows; ++j)
//             for (h = A->pointerB[j]; h < A->pointerE[j]; ++h)
//                 for (u = 0; u < k; ++u)
//                     matrix_set_element(LV, j, u, matrix_get_element(LV, j, u) + A->values[h] * matrix_get_element(V, A->cols[h], u));
//     }

//     double Ans = 0;

//     #pragma omp parallel shared(LV, V, A) private(k1, k2, j)
//     {
//         #pragma omp parallel for
//         for (k1 = 0; k1 < k; ++k1)
//             for (k2 = 0; k2 < k; ++k2)
//             {
//                 double tot = 0;
//                 for (j = 0; j < A->nrows; ++j)
//                     tot += matrix_get_element(LV, j, k1) * matrix_get_element(LV, j, k2);
//                 for (j = 0; j < A->ncols; ++j)
//                     Ans += tot * matrix_get_element(V, j, k1) * matrix_get_element(V, j, k2);
//             }
//     }

//     MKL_INT kk;
//     #pragma omp parallel shared(LV, V, A) private(kk, j, h)
//     {
//         #pragma omp parallel for

//         for (j = 0; j < A->nrows; ++j)
//             for (h = A->pointerB[j]; h < A->pointerE[j]; ++h)
//             {
//                 double non_update = 0;
//                 for(kk = 0; kk < k; ++kk)
//                     non_update += matrix_get_element(LV, j, kk) * matrix_get_element(V, A->cols[h], kk);
//                 Ans -= non_update * non_update;
//                 non_update = A->values[h] - non_update;
//                 Ans += non_update * non_update;
//             }
//     }

//     return Ans;
// }

// void matrix_get_mean_absolute_error(mat *left, mat *right, mat_coo *original_matrix, double *norm){
//     MKL_INT i;
//     *norm = 0;
//     #pragma omp parallel shared(left, right, original_matrix) private(i)
//     {
//         #pragma omp parallel for
//         for(i=0; i < original_matrix->nnz; i++){
//             MKL_INT row = original_matrix->rows[i] - 1;
//             MKL_INT col = original_matrix->cols[i] - 1;
//             vec * left_row = vector_new(left->ncols);
//             vec * right_row = vector_new(right->ncols);
//             matrix_get_row(left, row, left_row);
//             matrix_get_row(right, col, right_row);
//             *norm += abs( original_matrix->values[i] - vector_dot_product(left_row, right_row) ) ;
//         }
//     }
//     // *norm /= original_matrix->nnz;
// }

/* C = beta*C + alpha*A(1:Anrows, 1:Ancols)[T]*B(1:Bnrows, 1:Bncols)[T] */
void submatrix_submatrix_mult_with_ab(mat *A, mat *B, mat *C,
                                      MKL_INT Anrows, MKL_INT Ancols, MKL_INT Bnrows, MKL_INT Bncols, MKL_INT transa, MKL_INT transb, double alpha, double beta)
{

    MKL_INT opAnrows, opAncols, opBnrows, opBncols;
    if (transa == CblasTrans)
    {
        opAnrows = Ancols;
        opAncols = Anrows;
    }
    else
    {
        opAnrows = Anrows;
        opAncols = Ancols;
    }

    if (transb == CblasTrans)
    {
        opBnrows = Bncols;
        opBncols = Bnrows;
    }
    else
    {
        opBnrows = Bnrows;
        opBncols = Bncols;
    }

    if (opAncols != opBnrows)
    {
        printf("error in submatrix_submatrix_mult()");
        exit(0);
    }

    cblas_dgemm(CblasColMajor, transa, transb,
                opAnrows, opBncols,    // m, n,
                opAncols,              // k
                alpha, A->d, A->nrows, // 1, A, rows of A as declared in memory
                B->d, B->nrows,        // B, rows of B as declared in memory
                beta, C->d, C->nrows   // 0, C, rows of C as declared.
    );
}

void submatrix_submatrix_mult(mat *A, mat *B, mat *C, MKL_INT Anrows, MKL_INT Ancols, MKL_INT Bnrows, MKL_INT Bncols, MKL_INT transa, MKL_INT transb)
{
    double alpha, beta;
    alpha = 1.0;
    beta = 0.0;
    submatrix_submatrix_mult_with_ab(A, B, C, Anrows, Ancols, Bnrows, Bncols, transa, transb, alpha, beta);
}

/* D = M(:,inds)' */
void matrix_get_selected_columns_and_transpose(mat *M, MKL_INT *inds, mat *Mc)
{
    MKL_INT i;
    vec *col_vec;
#pragma omp parallel shared(M, Mc, inds) private(i, col_vec)
    {
#pragma omp parallel for
        for (i = 0; i < (Mc->nrows); i++)
        {
            col_vec = vector_new(M->nrows);
            matrix_get_col(M, inds[i], col_vec);
            matrix_set_row(Mc, i, col_vec);
            vector_delete(col_vec);
            col_vec = NULL;
        }
    }
}

void matrix_set_selected_rows_with_transposed(mat *M, MKL_INT *inds, mat *Mc)
{
    MKL_INT i;
    vec *col_vec;
#pragma omp parallel shared(M, Mc, inds) private(i, col_vec)
    {
#pragma omp parallel for
        for (i = 0; i < (Mc->ncols); i++)
        {
            col_vec = vector_new(Mc->nrows);
            matrix_get_col(Mc, i, col_vec);
            matrix_set_row(M, inds[i], col_vec);
            vector_delete(col_vec);
            col_vec = NULL;
        }
    }
}

void linear_solve_UTxb(mat *A, mat *b)
{
    LAPACKE_dtrtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N', //
                   A->nrows,
                   b->ncols,
                   A->d,
                   A->nrows,
                   b->d,
                   b->nrows);
}

// mat_coo* coo_matrix_new(int nrows, int ncols, long long int capacity) {
//     mat_coo *M = (mat_coo*)malloc(sizeof(mat_coo));
//     M->values = (double*)calloc(capacity, sizeof(double));
//     M->rows = (int*)calloc(capacity, sizeof(int));
//     M->cols = (int*)calloc(capacity, sizeof(int));
//     M->nnz = 0;
//     M->nrows = nrows; M->ncols = ncols;
//     M->capacity = capacity;
//     return M;
// }
mat_coo *coo_matrix_new(MKL_INT nrows, MKL_INT ncols, MKL_INT capacity)
{
    mat_coo *M = (mat_coo *)malloc(sizeof(mat_coo));
    M->values = (double *)calloc(capacity, sizeof(double));
    M->rows = (MKL_INT *)calloc(capacity, sizeof(MKL_INT));
    M->cols = (MKL_INT *)calloc(capacity, sizeof(MKL_INT));
    M->nnz = 0;
    // if (M->values == NULL)
    // {
    //     printf("M->values = NULL!\n");
    // }
    // if (M->rows == NULL)
    // {
    //     printf("M->rows = NULL!\n");
    // }
    // if (M->cols == NULL)
    // {
    //     printf("M->cols = NULL!\n");
    // }
    M->nrows = nrows;
    M->ncols = ncols;
    M->capacity = capacity;
    return M;
}

void coo_matrix_delete(mat_coo *M)
{
    free(M->values);
    M->values = NULL;
    free(M->cols);
    M->cols = NULL;
    free(M->rows);
    M->rows = NULL;
    free(M);
    M = NULL;
}


mat_coo_2 *coo_2_matrix_new(MKL_INT nrows, MKL_INT ncols, MKL_INT capacity)
{
    mat_coo_2 *M = (mat_coo_2 *)malloc(sizeof(mat_coo_2));
    // M->values = (double *)calloc(capacity, sizeof(double));
    M->nnz = 0;
    if (M->values == NULL)
    {
        printf("M->values = NULL!\n");
    }
    if (M->rows == NULL)
    {
        printf("M->rows = NULL!\n");
    }
    if (M->cols == NULL)
    {
        printf("M->cols = NULL!\n");
    }
    M->nrows = nrows;
    M->ncols = ncols;
    M->capacity = capacity;
    return M;
}

void coo_2_matrix_delete(mat_coo_2 *M)
{
    free(M->values);
    M->values = NULL;
    free(M->cols);
    M->cols = NULL;
    free(M->rows);
    M->rows = NULL;
    free(M);
    M = NULL;
}



void coo_matrix_print(mat_coo *M)
{
    MKL_INT i;
    for (i = 0; i < M->nnz; i++)
    {
        printf("(%d, %d: %f), ", *(M->rows + i), *(M->cols + i), *(M->values + i));
    }
    printf("\n");
}

// 0-based interface
void set_coo_matrix_element(mat_coo *M, MKL_INT row, MKL_INT col, double val, MKL_INT force_new)
{
    if (!(row >= 0 && row < M->nrows && col >= 0 && col < M->ncols))
    {
        printf("error: wrong index\n");
        exit(0);
    }
    if (!force_new)
    {
        MKL_INT i;
        for (i = 0; i < M->nnz; i++)
        {
            if (*(M->rows + i) == row + 1 && *(M->cols + i) == col + 1)
            {
                *(M->values + i) = val;
                return;
            }
        }
    }
    if (M->nnz < M->capacity)
    {
        *(M->rows + M->nnz) = row + 1;
        *(M->cols + M->nnz) = col + 1;
        *(M->values + M->nnz) = val;
        M->nnz = M->nnz + 1;
        return;
    }
    printf("error: capacity exceeded. capacity=%d, nnz=%d\n", M->capacity, M->nnz);
    exit(0);
}

void coo_matrix_matrix_mult(mat_coo *A, mat *B, mat *C)
{
    /*
    void mkl_dcoomm (
        const char *transa , const MKL_INT *m , const MKL_INT *n ,
        const MKL_INT *k , const double *alpha , const char *matdescra ,
        const double *val , const MKL_INT *rowind , const MKL_INT *colind ,
        const MKL_INT *nnz , const double *b , const MKL_INT *ldb ,
        const double *beta , double *c , const MKL_INT *ldc );
    */
    double alpha = 1.0, beta = 0.0;
    const char *trans = "N";
    const char *metadescra = "GXXF";
    mkl_dcoomm(
        trans, &(A->nrows), &(C->ncols),
        &(A->ncols), &(alpha), metadescra,
        A->values, A->rows, A->cols,
        &(A->nnz), B->d, &(B->nrows),
        &(beta), C->d, &(C->nrows));
}

void coo_matrix_transpose_matrix_mult(mat_coo *A, mat *B, mat *C)
{
    /*
    void mkl_dcoomm (
        const char *transa , const MKL_INT *m , const MKL_INT *n ,
        const MKL_INT *k , const double *alpha , const char *matdescra ,
        const double *val , const MKL_INT *rowind , const MKL_INT *colind ,
        const MKL_INT *nnz , const double *b , const MKL_INT *ldb ,
        const double *beta , double *c , const MKL_INT *ldc );
    */
    double alpha = 1.0, beta = 0.0;
    const char *trans = "T";
    const char *metadescra = "GXXF";
    mkl_dcoomm(
        trans, &(A->nrows), &(C->ncols),
        &(A->ncols), &(alpha), metadescra,
        A->values, A->rows, A->cols,
        &(A->nnz), B->d, &(B->nrows),
        &(beta), C->d, &(C->nrows));
}

void coo_matrix_copy_to_dense(mat_coo *A, mat *B)
{
    MKL_INT i, j;
    // printf("z1\n");
    for (MKL_INT i = 0; i < B->nrows; i++)
    {
        for (MKL_INT j = 0; j < B->ncols; j++)
        {
            matrix_set_element(B, i, j, 0.0);
        }
    }
    // printf("z2\n");
    for (MKL_INT i = 0; i < A->nnz; i++)
    {
        matrix_set_element(B, *(A->rows + i) - 1, *(A->cols + i) - 1, *(A->values + i));
    }
    // printf("z3\n");
}

double get_rand_uniform(VSLStreamStatePtr stream)
{
    double ans;
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &ans, 0.0, 1.0);
    return ans;
}

double get_rand_normal(VSLStreamStatePtr stream)
{
    double ans;
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &ans, 0.0, 1.0);
    return ans;
}

void gen_rand_coo_matrix(mat_coo *M, double density)
{
    VSLStreamStatePtr stream_u;
    VSLStreamStatePtr stream_n;
    // vslNewStream( &stream_u, BRNG, time(NULL));
    // vslNewStream( &stream_n, BRNG, time(NULL));
    vslNewStream(&stream_u, BRNG, 123);
    vslNewStream(&stream_n, BRNG, 456);
    MKL_INT i, j;
    for (i = 0; i < M->nrows; i++)
    {
        for (j = 0; j < M->ncols; j++)
        {
            if (get_rand_uniform(stream_u) < density)
            {
                set_coo_matrix_element(M, i, j, get_rand_normal(stream_n), 1);
            }
        }
    }
}

void coo_matrix_sort_element(mat_coo *A)
{
    MKL_INT i, j;
    // seletion sort
    for (i = 0; i < A->nnz; i++)
    {
        for (j = i + 1; j < A->nnz; j++)
        {
            if ((A->rows[i] > A->rows[j]) ||
                (A->rows[i] == A->rows[j] && A->cols[i] > A->cols[j]))
            {
                double dtemp;
                MKL_INT itemp;
                itemp = A->rows[i];
                A->rows[i] = A->rows[j];
                A->rows[j] = itemp;
                itemp = A->cols[i];
                A->cols[i] = A->cols[j];
                A->cols[j] = itemp;
                dtemp = A->values[i];
                A->values[i] = A->values[j];
                A->values[j] = dtemp;
            }
        }
    }
}

void csr_matrix_delete(mat_csr *M)
{

    // printf("csr_matrix_delete : 1 \n");
    free(M->values);
    M->values = NULL;

    // printf("csr_matrix_delete : 2 \n");
    free(M->cols);
    M->cols = NULL;

    // printf("csr_matrix_delete : 3 \n");
    free(M->pointerB);

    M->pointerB = NULL;

    // printf("csr_matrix_delete : 4 \n");
    free(M->pointerE);

    M->pointerE = NULL;

    // printf("csr_matrix_delete : 5 \n");
    free(M);

    M = NULL;
    // printf("csr_matrix_delete : 6 \n");
}

void csr_matrix_print(mat_csr *M)
{
    MKL_INT i;
    for (i = 0; i < M->nnz; i++)
    {
        printf("%d:%f ", i, M->values[i]);
    }
    for (i = 0; i < M->nnz; i++)
    {
        printf("%d:%d ", i, M->cols[i]);
    }
    for (i = 0; i < M->nrows; i++)
    {
        printf("%d:%d\t", i, M->pointerB[i]);
    }
    for (i = 0; i < M->nrows; i++)
    {
        printf("%d:%d\t", i, M->pointerE[i]);
    }
}

mat_csr *csr_matrix_new()
{
    mat_csr *M = (mat_csr *)malloc(sizeof(mat_csr));

    if (M == NULL)
    {
        printf("csr_matrix M allocation failure!!!\n");
    }

    return M;
}

// void csr_init_from_coo(mat_csr *D, mat_coo *M) {
//     D->nrows = M->nrows;
//     D->ncols = M->ncols;

//     D->pointerB = (int*)calloc(D->nrows, sizeof(int));
//     D->pointerE = (int*)calloc(D->nrows, sizeof(int));

//     D->cols = (int*)calloc(M->nnz, sizeof(int));

//     D->nnz = M->nnz;

//     D->values = (double*)calloc(M->nnz, sizeof(double));

//     for(int i = 0; i < M->nnz; i++){
//         D->values[i] = M->values[i];
//     }

//     int current_row, cursor=0;
//     for (current_row = 0; current_row < D->nrows; current_row++) {
//         D->pointerB[current_row] = cursor+1;
//         while (M->rows[cursor]-1 == current_row) {
//             D->cols[cursor] = M->cols[cursor];

//             cursor++;
//         }
//         D->pointerE[current_row] = cursor+1;
//     }
//     if(D->pointerB == NULL){
//         printf("pointerB allocation failure!!!\n");
//     }
//     if(D->pointerE == NULL){
//         printf("pointerE allocation failure!!!\n");
//     }
//     if(D->cols == NULL){
//         printf("cols allocation failure!!!\n");
//     }

// }

// void csr_matrix_matrix_mult(mat_csr *A, mat *B, mat *C) {
//     /* void mkl_dcsrmm (
//         const char *transa , const MKL_INT *m , const MKL_INT *n ,
//         const MKL_INT *k , const double *alpha , const char *matdescra ,
//         const double *val , const MKL_INT *indx , const MKL_INT *pntrb ,
//         const MKL_INT *pntre , const double *b , const MKL_INT *ldb ,
//         const double *beta , double *c , const MKL_INT *ldc );
//     */
//     char * transa = "N";
//     double alpha = 1.0, beta = 0.0;
//     const char *matdescra = "GXXF";

//     printf("A->nrows = %d\n", A->nrows);
//     printf("A->ncols = %d\n", A->ncols);
//     printf("B->nrows = %d\n", B->nrows);
//     printf("B->ncols = %d\n", B->ncols);
//     printf("C->nrows = %d\n", C->nrows);
//     printf("C->ncols = %d\n", C->ncols);

//     mkl_dcsrmm(transa, &(A->nrows), &(C->ncols),
//         &(A->ncols), &alpha, matdescra,
//         A->values, A->cols, A->pointerB,
//         A->pointerE, B->d, &(B->nrows),
//         &beta, C->d, &(C->nrows));
// }

// void csr_matrix_transpose_matrix_mult(mat_csr *A, mat *B, mat *C) {
//     /* void mkl_dcsrmm (
//         const char *transa , const MKL_INT *m , const MKL_INT *n ,
//         const MKL_INT *k , const double *alpha , const char *matdescra ,
//         const double *val , const MKL_INT *indx , const MKL_INT *pntrb ,
//         const MKL_INT *pntre , const double *b , const MKL_INT *ldb ,
//         const double *beta , double *c , const MKL_INT *ldc );
//     */
//     char * transa = "T";
//     double alpha = 1.0, beta = 0.0;
//     const char *matdescra = "GXXF";

//     printf("A->nrows = %d\n", A->nrows);
//     printf("A->ncols = %d\n", A->ncols);
//     printf("B->nrows = %d\n", B->nrows);
//     printf("B->ncols = %d\n", B->ncols);
//     printf("C->nrows = %d\n", C->nrows);
//     printf("C->ncols = %d\n", C->ncols);

//     mkl_dcsrmm(transa, &(A->nrows), &(C->ncols),
//         &(A->ncols), &alpha, matdescra,
//         A->values, A->cols, A->pointerB,
//         A->pointerE, B->d, &(B->nrows),
//         &beta, C->d, &(C->nrows));
// }

void csr_matrix_matrix_mult(mat_csr *A, mat *B, mat *C)
{
    /* void mkl_dcsrmm (
        const char *transa , const MKL_INT *m , const MKL_INT *n ,
        const MKL_INT *k , const double *alpha , const char *matdescra ,
        const double *val , const MKL_INT *indx , const MKL_INT *pntrb ,
        const MKL_INT *pntre , const double *b , const MKL_INT *ldb ,
        const double *beta , double *c , const MKL_INT *ldc );
    */
    char *transa = "N";
    double alpha = 1.0, beta = 0.0;
    const char *matdescra = "GXXF";

    // printf("A->nrows = %d\n", A->nrows);
    // printf("A->ncols = %d\n", A->ncols);
    // printf("B->nrows = %d\n", B->nrows);
    // printf("B->ncols = %d\n", B->ncols);
    // printf("C->nrows = %d\n", C->nrows);
    // printf("C->ncols = %d\n", C->ncols);

    mkl_dcsrmm(transa, &(A->nrows), &(C->ncols),
               &(A->ncols), &alpha, matdescra,
               A->values, A->cols, A->pointerB,
               A->pointerE, B->d, &(B->nrows),
               &beta, C->d, &(C->nrows));
}

void csr_matrix_transpose_matrix_mult(mat_csr *A, mat *B, mat *C)
{
    /* void mkl_dcsrmm (
        const char *transa , const MKL_INT *m , const MKL_INT *n ,
        const MKL_INT *k , const double *alpha , const char *matdescra ,
        const double *val , const MKL_INT *indx , const MKL_INT *pntrb ,
        const MKL_INT *pntre , const double *b , const MKL_INT *ldb ,
        const double *beta , double *c , const MKL_INT *ldc );
    */
    char *transa = "T";
    double alpha = 1.0, beta = 0.0;
    const char *matdescra = "GXXF";

    // printf("A->nrows = %d\n", A->nrows);
    // printf("A->ncols = %d\n", A->ncols);
    // printf("B->nrows = %d\n", B->nrows);
    // printf("B->ncols = %d\n", B->ncols);
    // printf("C->nrows = %d\n", C->nrows);
    // printf("C->ncols = %d\n", C->ncols);

    mkl_dcsrmm(transa, &(A->nrows), &(C->ncols),
               &(A->ncols), &alpha, matdescra,
               A->values, A->cols, A->pointerB,
               A->pointerE, B->d, &(B->nrows),
               &beta, C->d, &(C->nrows));
}

double get_sparse_matrix_frobenius_norm_square(mat_csr *A)
{
    MKL_INT i;
    double val, normval = 0;
#pragma omp parallel shared(A, normval) private(i, val)
    {
#pragma omp for reduction(+ : normval)
        for (i = 0; i < (A->nnz); i++)
        {
            val = A->values[i];
            normval += val * val;
        }
    }
    return normval;
}