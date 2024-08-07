# RaStree-SVD

## Environment
- Ubuntu
- C++ 11
- GCC
- Intel C++ Compiler

## Preparation


First, we create necessary folders.

```sh
mkdir ./Graph_Dataset
mkdir ./IO_File
mkdir ./PPR_info
mkdir ./Output
```

Place the prepared graph data [file].txt in the folder [Graph_Dataset]. 

Note that the file must only contains information of the edges, and the node numbers start from 0.

Each row in the file contains two numbers, which are the starting and ending nodes of an edge [outNode] [inNode].

Note that directed graphs and undirected graphs should be processed separately.



## Compilations
```sh
bash compile.sh
```


At the same time, we use "queryname" to refer to a dataset, for example, it should be "YouTube-u" for the YouTube dataset.

Because directed and undirected graphs are usually processed differently, we often write two type of programs as "U" and "D" to process them, respectively.





### Parameter of proximity matrices

[queryname] -> dataset

[NUM_threads] -> number of threads

[vertex_number] -> the number of vertices in the graph

[d] -> output dimension of the algorithm

[Type] -> [0 for static SVD] / [1 for dynamic SVD]

[dynamic_type] (only works when [Type] == 1) -> [3 for Exp.3] / [4 for Exp.4]



### Execute the program for proximity matrices of graphs
```sh
RASTREESVD_PPR_U [queryname] Graph_Dataset/ [NUM_threads] [vertex_number] [d] [Type] [dynamic_type]  # For Undirected Graph
RASTREESVD_PPR_D [queryname] Graph_Dataset/ [NUM_threads] [vertex_number] [d] [Type] [dynamic_type]  # For Directed Graph

# Example
./RASTREESVD_PPR_U YouTube-u Graph_Dataset/ 0.000001 64 1138499 128 0 >   ./Output/test1.txt
./RASTREESVD_PPR_U YouTube-u Graph_Dataset/ 0.000001 64 1138499 128 1 3 >   ./Output/test2.txt
```


### Data Pre-processing for image matrix
The image data [image_align_celeba] can be downloaded from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html.

Extract the compressed file [img_align_celeba.zip] to the folder [/image_matrix]. 

After extraction, all jpg images are contained in the folder [/image_matrix/image_align_celeba].

Next, execute the following command:
```sh
cd ./image_matrix
python3 process.py 202599 64
```


### Parameter of image matrices

[queryname] -> dataset

[column_size] -> column dimension of a single picture

[NUM_threads] -> number of threads

[image_number] -> the number of images

[d] -> output dimension of the algorithm

[Type] -> [0 for static SVD] / [1 for dynamic SVD]

[dynamic_type] (only works when [Type] == 1) -> [3 for Exp.3] / [4 for Exp.4]


### Execute the program for image matrix
```sh
RASTREESVD_IMAGE [queryname] [column_size] [NUM_threads] [image_number] [d] [Type] [dynamic_type]  

# Example
./RASTREESVD_IMAGE image_align_celeba 116412 64 202599 256 0 >   ./Output/test3.txt
./RASTREESVD_IMAGE image_align_celeba 116412 64 202599 256 1 3 >   ./Output/test4.txt
```