#!/bin/bash

source /opt/intel/oneapi/setvars.sh 

export MKL_NUM_THREADS=64


./RASTREESVD_PPR_U YouTube-u Graph_Dataset/ 0.000001 64 1138499 128 0 >   ./Output/test1.txt
./RASTREESVD_PPR_U YouTube-u Graph_Dataset/ 0.000001 64 1138499 128 1 3 >   ./Output/test2.txt
./RASTREESVD_IMAGE image_align_celeba 116412 64 202599 256 0 >   ./Output/test3.txt
./RASTREESVD_IMAGE image_align_celeba 116412 64 202599 256 1 3 >   ./Output/test4.txt
