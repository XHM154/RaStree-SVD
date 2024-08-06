#!/bin/bash

source /opt/intel/oneapi/setvars.sh 

export MKL_NUM_THREADS=64

#-------------------Compilation of RSTreeSVD with Dynamic Version-----------------------

if [ "$#" -ge 1 ]; then 
    if [ "$1" = 1 ]; then
        echo "Dynamic-Compile.sh executed!"
        ./Dynamic-Compile.sh > output_compile.log 2>&1
        echo "Dynamic-Compile.sh done!"
        if grep -q "error" "./output_compile.log"; then
            echo "Compile Error!"
            echo "Check Error in output_compile.log!"
            exit
        else
            echo "Compilation finish with no error!"
        fi
    fi
fi

#-------------------End-----------

./RASTREESVD_PPR_U YouTube-u Graph_Dataset/ 0.5 12 0.000001 64 2_4_8 1138499 128 0 3 0 >   ./Output/test1.txt
./RASTREESVD_PPR_U YouTube-u Graph_Dataset/ 0.5 12 0.000001 64 2_4_8 1138499 128 1 3 0 >   ./Output/test2.txt
