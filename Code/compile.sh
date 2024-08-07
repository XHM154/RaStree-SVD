
source /opt/intel/oneapi/setvars.sh 



export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/mkl/2023.0.0/lib/intel64/

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/compiler/2023.0.0/linux/compiler/lib/intel64_lin/




icc -O3 -DMKL_ILP64 -I./ -qmkl -no-multibyte-chars -qopenmp -c -o ./frpca.o ./frpca.c

icc -O3 -DMKL_ILP64 -I./ -qmkl -no-multibyte-chars -qopenmp -c -o ./matrix_vector_functions_intel_mkl.o ./matrix_vector_functions_intel_mkl.c

icc -O3 -DMKL_ILP64 -I./ -qmkl -no-multibyte-chars -qopenmp -c -o ./matrix_vector_functions_intel_mkl_ext.o ./matrix_vector_functions_intel_mkl_ext.c

icc -O3 -DMKL_ILP64 -I./ -I/usr/include/eigen3 -qopenmp -std=c++11 -qmkl -c Matrix_RaSTreeSVD_ppr_undirected.cpp -o Matrix_RaSTreeSVD_ppr_undirected.o 

icc -O3 -DMKL_ILP64 -I./ -I/usr/include/eigen3 -qopenmp -std=c++11 -qmkl -c Matrix_RaSTreeSVD_ppr_directed.cpp -o Matrix_RaSTreeSVD_ppr_directed.o 

icc -O3 -DMKL_ILP64 -I./ -I/usr/include/eigen3 -qopenmp -std=c++11 -qmkl -c Matrix_RaSTreeSVD_image.cpp -o Matrix_RaSTreeSVD_image.o 

icc -o RASTREESVD_PPR_U ./frpca.o ./matrix_vector_functions_intel_mkl.o ./matrix_vector_functions_intel_mkl_ext.o Matrix_RaSTreeSVD_ppr_undirected.o   -L/opt/intel/oneapi/mkl/2023.0.0/lib/intel64/ -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -qopenmp -lpthread -lm -ldl -lboost_serialization

icc -o RASTREESVD_PPR_D ./frpca.o ./matrix_vector_functions_intel_mkl.o ./matrix_vector_functions_intel_mkl_ext.o Matrix_RaSTreeSVD_ppr_directed.o   -L/opt/intel/oneapi/mkl/2023.0.0/lib/intel64/ -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -qopenmp -lpthread -lm -ldl -lboost_serialization

icc -o RASTREESVD_IMAGE ./frpca.o ./matrix_vector_functions_intel_mkl.o ./matrix_vector_functions_intel_mkl_ext.o Matrix_RaSTreeSVD_image.o   -L/opt/intel/oneapi/mkl/2023.0.0/lib/intel64/ -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -qopenmp -lpthread -lm -ldl -lboost_serialization