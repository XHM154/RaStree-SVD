#include "matrix_vector_functions_intel_mkl_ext.h"


/*[L, ~] = lu(A) as in MATLAB*/
void LUfraction(mat *A, mat *L)
{

    matrix_copy(L, A);

    // int *ipiv = (int *)malloc(sizeof(int)*L->nrows);    
    //Waiting for verifying, possible error, the way to malloc pointer arrays
    MKL_INT *ipiv = (MKL_INT *)malloc(sizeof(MKL_INT)*L->nrows);    

    LAPACKE_dgetrf (LAPACK_COL_MAJOR, L->nrows, L->ncols, L->d, L->nrows, ipiv);    

    MKL_INT i,j;

    #pragma omp parallel private(i,j) 
    {
    #pragma omp for     
        for(i=0;i<L->ncols;i++)
        {
            for(j=0;j<i;j++)
            {
                L->d[i*L->nrows+j] = 0;
            }
            L->d[i*L->nrows+i] = 1;
        }
    }

    {    
        for(i=L->ncols-1;i>=0;i--)
        {

            //possible error
            MKL_INT ipi = ipiv[i]-1;

            for(j=0;j<L->ncols;j++)
            {
                double temp = L->d[j*L->nrows+ipi];
                L->d[j*L->nrows+ipi] = L->d[j*L->nrows+i];
                L->d[j*L->nrows+i] = temp;
            }
        }
    }
}




void tLUfraction(mat *A, mat *L)
{

    matrix_copy(L, A);

    // int *ipiv = (int *)malloc(sizeof(int)*L->nrows);    
    MKL_INT *ipiv = (MKL_INT *)malloc(sizeof(MKL_INT)*L->nrows);    

    LAPACKE_dgetrf (LAPACK_COL_MAJOR, L->nrows, L->ncols, L->d, L->nrows, ipiv);    

    MKL_INT i,j;

    #pragma omp parallel private(i,j) 
    {
    #pragma omp for     
        for(i=0;i<L->ncols;i++)
        {
            
            for(j=0;j<i;j++)
            {
                L->d[i*L->nrows+j] = 0;
            }

            L->d[i*L->nrows+i] = 1;
        }
    }

    {    
        for(i=L->ncols-1;i>=0;i--)
        {

            MKL_INT ipi = ipiv[i]-1;

            for(j=0;j<L->ncols;j++)
            {
                double temp = L->d[j*L->nrows+ipi];
                L->d[j*L->nrows+ipi] = L->d[j*L->nrows+i];
                L->d[j*L->nrows+i] = temp;
            }
        }

    }

}









/*[U, S, V] = eigSVD(A)*/
void eigSVD(mat* A, mat **U, mat **S, mat **V)
{
    matrix_transpose_matrix_mult(A, A, *V);
    LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', (*V)->ncols, (*V)->d, (*V)->ncols, (*S)->d);
    
    //VV is of dimension (k+s, k+s), so both ncols as dimensions are appropriate
    mat *V1 = matrix_new((*V)->ncols, (*V)->ncols);

    matrix_copy(V1, (*V));
    
    MKL_INT i, j;
    double eps = 1e-4;
    #pragma omp parallel shared(V1,S) private(i,j) 
    {
    #pragma omp for 
        for(i=0; i<V1->ncols; i++)
        {   
            
            if((*S)->d[i] < 0){
                (*S)->d[i] = 0;
            }
            (*S)->d[i] = sqrt((*S)->d[i]);

            if((*S)->d[i] <= eps){
                for(j=0; j<V1->nrows;j++)            
                    V1->d[i*V1->nrows+j] = 0;
                    (*S)->d[i] = 0;
            }
            else{
                for(j=0; j<V1->nrows;j++)            
                    V1->d[i*V1->nrows+j] /= (*S)->d[i];
            }
        }
    }
    mat *AA=matrix_new(A->nrows,A->ncols);
    matrix_copy(AA,A);
    matrix_matrix_mult(AA, V1, (*U));
    matrix_delete(AA);
    matrix_delete(V1);
    AA = NULL;
    V1 = NULL;
}















void frPCAt(mat_csr *A, mat **U, mat **S, mat **V, MKL_INT k, MKL_INT q)
{

    MKL_INT s = 5;    
    printf("frPCAt:1\n");
    mat *Q = matrix_new(A->nrows, k+s);

    printf("frPCAt:2\n");
    mat *Qt = matrix_new(A->ncols, k+s);

    printf("frPCAt:3\n");
    mat *SS = matrix_new(k+s, 1);

    printf("frPCAt:4\n");
    mat *VV = matrix_new(k+s, k+s);

    printf("frPCAt:5\n");

    if(q%2 == 0)
    {
    printf("frPCAt:6\n");

        initialize_random_matrix(Q);

    printf("frPCAt:7\n");
        csr_matrix_transpose_matrix_mult(A, Q, Qt);

        if(q==2)
        {
    printf("frPCAt:8\n");
            eigSVD(Qt, &Qt, &SS, &VV);
        }
        else
        {
    printf("frPCAt:9\n");
            LUfraction(Qt, Qt);
        }
    }
    else
    {
    printf("frPCAt:10\n");
        initialize_random_matrix(Qt);
    }
    MKL_INT niter = (q-1)/2, i;
    for(i=1;i<=niter;i++)
    {

    printf("frPCAt:11\n");
        csr_matrix_matrix_mult(A, Qt, Q);

    printf("frPCAt:12\n");
        csr_matrix_transpose_matrix_mult(A, Q, Qt);

        if(i==niter)
        {
    printf("frPCAt:13\n");
            eigSVD(Qt, &Qt, &SS, &VV);
        }
        else
        {
    printf("frPCAt:14\n");
            LUfraction(Qt, Qt);
        }
    }

    printf("frPCAt:15\n");
    csr_matrix_matrix_mult(A, Qt, Q);

    printf("frPCAt:16\n");
    eigSVD(Q, &Q, &SS, &VV);

    MKL_INT inds[k]; 
    for(i=s;i<k+s;i++)
    {

        inds[i-s] = i;

    }
    printf("frPCAt:17\n");

    mat *VV2 = matrix_new(k+s, k);

    printf("frPCAt:18\n");
    matrix_get_selected_columns(Q, inds, *U);

    printf("frPCAt:19\n");
    matrix_get_selected_rows(SS, inds, *S);

    printf("frPCAt:20\n");
    matrix_get_selected_columns(VV, inds, VV2);

    printf("frPCAt:21\n");
    matrix_matrix_mult(Qt, VV2, (*V));

    printf("frPCAt:22\n");
    matrix_delete(Q);
    printf("frPCAt:23\n");
    matrix_delete(Qt);

    printf("frPCAt:24\n");
    matrix_delete(SS);
    printf("frPCAt:25\n");
    matrix_delete(VV);
    printf("frPCAt:26\n");
    matrix_delete(VV2);
    Q = NULL;
    Qt = NULL;
    SS = NULL;
    VV = NULL;
    VV2 = NULL;

}






void frPCAt_onlyUS(mat_csr *A, mat **U, mat **S, MKL_INT k, MKL_INT q)
{
    printf("Before Setting Threads\n");
    // omp_set_num_threads(48);
    printf("After Setting Threads\n");

    MKL_INT s = 5;    
    printf("frPCAt:1\n");
    mat *Q = matrix_new(A->nrows, k+s);

    printf("frPCAt:2\n");
    mat *Qt = matrix_new(A->ncols, k+s);

    printf("frPCAt:3\n");
    mat *SS = matrix_new(k+s, 1);

    printf("frPCAt:4\n");
    mat *VV = matrix_new(k+s, k+s);

    printf("frPCAt:5\n");

    if(q%2 == 0)
    {
    printf("frPCAt:6\n");

        initialize_random_matrix(Q);

    printf("frPCAt:7\n");
        
        csr_matrix_transpose_matrix_mult(A, Q, Qt);

        if(q==2)
        {
    printf("frPCAt:8\n");
            eigSVD(Qt, &Qt, &SS, &VV);
        }
        else
        {
    printf("frPCAt:9\n");
            LUfraction(Qt, Qt);
        }
    }
    else
    {
    printf("frPCAt:10\n");
        initialize_random_matrix(Qt);
    }
    MKL_INT niter = (q-1)/2, i;
    for(i=1;i<=niter;i++)
    {

    printf("frPCAt:11\n");
        csr_matrix_matrix_mult(A, Qt, Q);

    printf("frPCAt:12\n");
        csr_matrix_transpose_matrix_mult(A, Q, Qt);

        if(i==niter)
        {
    printf("frPCAt:13\n");
            eigSVD(Qt, &Qt, &SS, &VV);
        }
        else
        {
    printf("frPCAt:14\n");
            LUfraction(Qt, Qt);
        }
    }

    printf("frPCAt:15\n");
    csr_matrix_matrix_mult(A, Qt, Q);

    printf("frPCAt:16\n");
    eigSVD(Q, &Q, &SS, &VV);

    // int inds[k]; 
    MKL_INT inds[k];     

    for(i=s;i<k+s;i++)
    {

        inds[i-s] = i;

    }
    printf("frPCAt:17\n");

    mat *VV2 = matrix_new(k+s, k);

    printf("frPCAt:18\n");

    printf("in:U->nrows = %d", (*U)->nrows);
    matrix_get_selected_columns(Q, inds, *U);
    printf("in:U->nrows = %d", (*U)->nrows);

    printf("frPCAt:19\n");
    matrix_get_selected_rows(SS, inds, *S);


    printf("frPCAt:22\n");
    matrix_delete(Q);
    printf("frPCAt:23\n");
    matrix_delete(Qt);

    printf("frPCAt:24\n");
    matrix_delete(SS);
    printf("frPCAt:25\n");
    matrix_delete(VV);
    printf("frPCAt:26\n");

    Q = NULL;
    Qt = NULL;
    SS = NULL;
    VV = NULL;


}














void frPCA(mat_csr *A, mat **U, mat **S, mat **V, MKL_INT k, MKL_INT q)
{
    
    printf("frPCA!\n");
    
    MKL_INT s = 5;    

    mat *Q = matrix_new(A->nrows, k+s);

    mat *Qt = matrix_new(A->ncols, k+s);

    mat *UU = matrix_new(A->ncols, k+s);

    mat *SS = matrix_new(k+s, 1);

    mat *VV = matrix_new(k+s, k+s);

    if(q%2 == 0)
    {
        
        initialize_random_matrix(Qt);

        csr_matrix_matrix_mult(A, Qt, Q);

        if(q==2)
        {

            eigSVD(Q, &Q, &SS, &VV);

        }
        else
        {
            LUfraction(Q, Q);
        }
    }
    else
    {
        initialize_random_matrix(Q);
    }
    int niter = (q-1)/2, i;
    for(i=1;i<=niter;i++)
    {

        csr_matrix_transpose_matrix_mult(A, Q, Qt);

        csr_matrix_matrix_mult(A, Qt, Q);

        if(i==niter)
        {
            eigSVD(Q, &Q, &SS, &VV);
        }
        else
        {
            LUfraction(Q, Q);
        }
    }

    csr_matrix_transpose_matrix_mult(A, Q, Qt);

    eigSVD(Qt, &UU, &SS, &VV);

    MKL_INT inds[k]; 
    for(i=s;i<k+s;i++)
    {
        inds[i-s] = i;
    }

    mat *VV2 = matrix_new(k+s, k);

    matrix_get_selected_columns(UU, inds, *V);

    matrix_get_selected_rows(SS, inds, *S);

    matrix_get_selected_columns(VV, inds, VV2);

    matrix_matrix_mult(Q, VV2, (*U));

    matrix_delete(Q);
    matrix_delete(Qt);
    matrix_delete(UU);
    matrix_delete(SS);
    matrix_delete(VV);
    matrix_delete(VV2);
    Q = NULL;
    Qt = NULL;
    UU = NULL;
    SS = NULL;
    VV = NULL;
    VV2 = NULL;
}








void randQB_basic_csr(mat_csr *M, MKL_INT k, MKL_INT p, mat **U, mat **S, mat **V) {
    MKL_INT m, n, i, l=k+5;
    m = M->nrows; n = M->ncols;
    
    mat *Q = matrix_new(m, l);
    mat *B = matrix_new(l, n);
    mat *Vt = matrix_new(l, n);
    mat *VV = matrix_new(n, l);
    mat *UU = matrix_new(l, l);
    mat *UUk = matrix_new(l, k);
    mat *SS = matrix_new(l, l);
    *U = matrix_new(m ,k); 
    *V = matrix_new(n, k);
    *S = matrix_new(k, k);
    
    // samples
    mat *R, *G, *Bt;
    R = matrix_new(n, l);
    G = Q;
    Bt = R;
    initialize_random_matrix(R);
    csr_matrix_matrix_mult(M, R, G);
    QR_factorization_getQ_inplace(G);
    
    // power iteration
    if (p > 0) {
        for (i = 0; i < p; i++) {
            csr_matrix_transpose_matrix_mult(M, G, R);
            QR_factorization_getQ_inplace(R);
            csr_matrix_matrix_mult(M, R, G);
            QR_factorization_getQ_inplace(G);
        }
    }
    
    //QR_factorization_getQ_inplace(G);
    csr_matrix_transpose_matrix_mult(M, Q, Bt);
    matrix_build_transpose(B, Bt);
    
    singular_value_decomposition(B, UU, SS, Vt);
    matrix_build_transpose(VV, Vt);
    // int inds[k]; 
    
    MKL_INT inds[k]; 
    
    for(i=0;i<k;i++)
    {
        inds[i] = i;
    }
    matrix_get_selected_columns(UU, inds, UUk);
    matrix_matrix_mult(Q, UUk, (*U));
    matrix_get_selected_columns(VV, inds, (*V));
    matrix_copy_first_k_rows_and_columns(*S, SS);
    
    matrix_delete(R);
    matrix_delete(Q);
    matrix_delete(B);
    matrix_delete(VV);
    matrix_delete(Vt);
    matrix_delete(UU);
    matrix_delete(UUk);
    matrix_delete(SS);

    R = NULL;
    Q = NULL;
    B = NULL;
    VV = NULL;
    Vt = NULL;
    UU = NULL;
    UUk = NULL;
    SS = NULL;
    
}


void randomized_SVD(mat_csr *A, MKL_INT r, MKL_INT iter, mat **U, mat **S, mat **V) {

    MKL_INT A_rows = A->nrows;
    MKL_INT A_cols = A->ncols;

    mat * Q = matrix_new(A_cols, r);

    initialize_random_matrix(Q);

    mat* AQ = matrix_new(A_rows, Q->ncols);

    csr_matrix_matrix_mult(A, Q, AQ);

    QR_Decomposition(AQ);

    mat * AQ_temp = matrix_new(A_cols, Q->ncols);


    for(int i = 0; i < iter; i++){

        csr_matrix_transpose_matrix_mult(A, AQ, AQ_temp);

        QR_Decomposition(AQ_temp);

        csr_matrix_matrix_mult(A, AQ_temp, AQ);

        QR_Decomposition(AQ);

    }

    mat* C = matrix_new(AQ->ncols, A->ncols);

    mat* C_Trans = matrix_new(C->ncols, C->nrows);

    csr_matrix_transpose_matrix_mult(A, AQ, C_Trans);

    matrix_build_transpose(C, C_Trans);

    mat* UU = matrix_new(C->nrows, C->nrows);

    singular_value_decomposition(C, UU, (*S), (*V));

    matrix_matrix_mult(AQ, UU, (*U));

    matrix_delete(Q);
    matrix_delete(AQ);
    matrix_delete(AQ_temp);

    matrix_delete(UU);
    matrix_delete(C);
    matrix_delete(C_Trans);

    Q = NULL;
    AQ = NULL;
    AQ_temp = NULL;
    UU = NULL;
    C = NULL;
    C_Trans = NULL;
    
}












void right_matrix_mkl(mat_csr *ppr_matrix, mat** mkl_left_matrix, mat** mkl_result_mat){

    csr_matrix_matrix_mult(ppr_matrix, (*mkl_left_matrix), (*mkl_result_mat));

}