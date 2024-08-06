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
#include <stack>
#include <unordered_map>

// #include "Graph.h"
#include "Graph_dynamic.h"

#include <fstream>
#include <cstring>
#include <thread>
#include <mutex>
#include "Eigen/Sparse"
#include "Eigen/Dense"
#include <chrono>
#include <climits>

#include<assert.h>

#include<queue>

#include "ppr_computation_store.h"

#include "General.h"

#include "General_dynamic.h"

#include<assert.h>
#include <unordered_map>
#include<cmath>

#include<list>

#include<atomic>


#include <unistd.h>

#include <boost/functional/hash.hpp>
#include <boost/algorithm/string.hpp>

#include "../emhash/hash_table7.hpp"

#include <random>

using namespace Eigen;

using namespace std;



using third_float_map = emhash7::HashMap<int, float>;
using third_bool_map = emhash7::HashMap<int, bool>;
using third_int_map = emhash7::HashMap<int, int>;


typedef std::pair<double, long long int> P_d_l;
typedef std::pair<long long, double> P_l_d;
typedef std::pair<long long, pair<double, double>> P_l_d_d;






bool Check_Exist_PPR(string queryname, double rmax)
{
  string file_path = "PPR_info/" + queryname + ".txt";
  std::ifstream file(file_path.c_str());
  if(!file.is_open())
  {
    cout << "PPR info log fail to be opened!\n";
    file.close();
    return 0;
  }

  double rrr = 0;
  file >> rrr;

  if(!file)
  {
    cout << "No Value in the PPR log!\n";
    file.close();
    return 0;
  }

  if(fabs(rrr - rmax) < 1e-12)
  {
    cout << "Same current PPR as the previous one!\n";
    file.close();
    return 1;
  }
  else {
    cout << "Different current PPR as the previous one!\n";
    file.close();
    return 0;
  }

}

void Overwrite_PPR_info(string queryname, double rmax)
{
  string file_path = "PPR_info/" + queryname + ".txt";
  std::ofstream file(file_path.c_str());
  file << fixed << setprecision(12) << rmax;
  file.close();
}


bool maxScoreCmp(const pair<double, pair<int, int>>& a, const pair<double, pair<int, int>>& b){
    return a.first > b.first;
}


const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");



void Memory_use_Monitoring_Gen()
{
  // std::thread::id threadId = std::this_thread::get_id();
  pid_t threadId = getpid();
  std::stringstream ss;
  ss << threadId;
  string command = "./Mem_used_record.sh " + ss.str();
  int result = system(command.c_str());

  if(result != 0) 
    cerr << "failed to monitor memory use" << endl;
  else 
    cout << ss.str() << " memory get monitored now!" << endl;
}

void Output_Memory_Use()
{
  std::ifstream file("Mem_Record.log");
  cout << "\n-------------------Memory Use Details-------------------------\n";
  if(file.is_open())
  {
    std::string line1, line2, line3;
    std::getline(file, line1);
    std::getline(file, line2);
    std::getline(file, line3);

    cout << line1 << endl;
    cout << line2 << endl;
    cout << line3 << endl;
  }
  else cout << "memory file does not exist!" << endl;
  
  cout << "-------------------End of the Details-------------------------\n\n";
}



int main(int argc,  char **argv)
{

  cout << "Start!\n";
  auto start_time = std::chrono::system_clock::now();
  srand((unsigned)time(0));
  char *endptr;

  srand(789331);
  int nnn = rand() % 3 + 1;
  for(int i = 0; i < 10; ++i)
  nnn = rand() % 3 + 1;

  string queryname = argv[1];

  string querypath = argv[2];

  clock_t start = clock();
  double alpha = strtod(argv[3], &endptr);
  int pass = strtod(argv[4], &endptr);
  double backward_theta = strtod(argv[5], &endptr);
  int NUMTHREAD = strtod(argv[6], &endptr);

  // int nParts = strtod(argv[8], &endptr);
  string Hierarchy_Structure_string = argv[7];
  vector<int> Hierarchy_Structure;
  vector<string> Hierarchy_Structure_token;
  boost::split(Hierarchy_Structure_token, Hierarchy_Structure_string, boost::is_any_of("_"));

  for(auto x : Hierarchy_Structure_token)
    Hierarchy_Structure.push_back(std::stoi(x));

  int nParts = 1;
  for(auto x : Hierarchy_Structure)
    nParts *= x;

  long long int vertex_number = strtod(argv[8], &endptr);

  int Graph_Type = 0;
  // long long int total_nnz_number = -1;

  cout << "argc : " << argc << endl;

  int d_1_level = 0;

  double residuemax = backward_theta; // PPR error up to residuemax

  double reservemin = backward_theta; // Retain approximate PPR larger than reservemin

  cout << "alpha: " << alpha << ", residuemax: " << residuemax << ", reservemin: " << reservemin <<endl;
  cout << "nParts: "<< nParts << ", vertex_number: "<<vertex_number<<endl; 

  fflush(stdout);

  int total_d = atoi(argv[9]) * 2;

  int d = total_d / 2;

  int count_nParts = 0;


  string graph_path =  querypath + queryname + ".txt";
  
  int snapshots_number = 0;

  
  double Initial_val_percent = 0;
  int erase_way = 0;
  erase_way = atoi(argv[10]);
  int Exp_type = strtod(argv[11], &endptr);
  
  if(Exp_type == 3) snapshots_number = 5, Initial_val_percent = 0.5;
  else if(Exp_type == 4) snapshots_number = 10, Initial_val_percent = 0.9;
  else {
    cout << "Error\n";
    return 0;
  }
  
  double threshold_percent = 0;
  if(erase_way != 0 && argc > 12) threshold_percent = strtod(argv[12], &endptr);

  cout << "snapshots_number = " << snapshots_number << endl;

  // snapshots_number = 5;
  // snapshots_number = 10;

  //Switch Graph Type

  // Graph* g = new Graph();
  UGraph* g = new UGraph();

  g->initializeDynamicUGraph(vertex_number);
  // g->initializeDirectedDynamicGraph(vertex_number);

  vector<pair<int, int>> edge_vec;


  // Use the first snapshot to construct, other snapshots to update
  g->inputDynamicGraph(graph_path.c_str(), edge_vec);
  // g->inputDirectedDynamicGraph(graph_path.c_str(), edge_vec);









  // Construction
  unordered_map<int, vector<long long int>> vec_mapping;

  vector<int> vertex_mapping;

  vector<int> inner_group_mapping;

  vertex_mapping.resize(vertex_number);

  inner_group_mapping.resize(vertex_number);

  long long int common_group_size;

  common_group_size = vertex_number / nParts;

  long long int final_group_size = vertex_number - (nParts - 1) * common_group_size;

  cout<<"common_group_size = "<<common_group_size<<endl;

  int* Group_Size = new int[nParts]();
  int* Start_Column_List = new int[nParts]();

  for(int i = 0; i < nParts; ++i)
    Group_Size[i] = (i < nParts - 1) ? common_group_size : final_group_size;
    
  Start_Column_List[0] = 0;
  for(int i = 1; i < nParts; ++i)
    Start_Column_List[i] = Start_Column_List[i - 1] + Group_Size[i - 1];

  for(int i = 0; i < nParts; ++i)
    for(int j = 0; j < Group_Size[i]; ++j)
      vertex_mapping[Start_Column_List[i] + j] = i;

  // for(int t = 0; t < vertex_number; t++){
  //   long long int index = t / common_group_size;
  //   if(index != nParts){
  //     vertex_mapping[t] = index;

  //   }
  //   else{
  //     vertex_mapping[t] = index - 1;
  //   }
  // }

  for(int i = 0; i < nParts; ++i)
    for(int j = 0; j < Group_Size[i]; ++j)
      vec_mapping[i].push_back(Start_Column_List[i] + j);

  // for(int i = 0; i < nParts; i++){
  //   if(i != nParts - 1){
  //     for(long long int j = i * common_group_size; j < (i+1) * common_group_size; j++){
  //       vec_mapping[i].push_back(j);
  //     }
  //   }
  //   else{
  //     for(long long int j = 0; j < final_group_size; j++){
  //       vec_mapping[i].push_back(common_group_size * (nParts - 1) + j);
  //     }
  //   }
  // }

  for(int i = 0; i < vec_mapping.size(); i++){
    for(int j = 0; j < vec_mapping[i].size(); j++){
      inner_group_mapping[vec_mapping[i][j]] = j;
    }
  }



  int hierarchy_n = 2;
  d_row_tree_mkl* d_row_tree = new d_row_tree_mkl(vertex_number, d, nParts, hierarchy_n, vec_mapping);





  mat* U_cur_iter;

  U_cur_iter = matrix_new(vertex_number, d);





int level_p = log(nParts) / log(hierarchy_n) + 1;

cout << Hierarchy_Structure.size() << endl;
int TOTAL_NODES = 1, level_node = 1;
for(auto x : Hierarchy_Structure)
{
  level_node *= x;
  TOTAL_NODES += level_node;
}


int* father_node = new int[TOTAL_NODES]();
int* number_of_son = new int[TOTAL_NODES]();
int* finish_son_number = new int[TOTAL_NODES]();
vector<vector<int>> son_node_list(TOTAL_NODES);

level_node = 1;
int total_nodes = 1;

for(int i = 0; i < Hierarchy_Structure.size(); ++i)
{
  int fans_out = Hierarchy_Structure[i];
  int start_node = total_nodes - level_node;
  level_node *= fans_out;
  total_nodes += level_node;
  for(int j = 0; j < level_node; ++j)
    father_node[total_nodes - level_node + j] = start_node + (j / fans_out);
  cout << i + 1 <<"-level with " << level_node << " numbered from ";
  cout << total_nodes - level_node << " to " << total_nodes - 1 << endl;
}

for(int i = 1; i < total_nodes; ++i)
{
  number_of_son[father_node[i]]++;
  son_node_list[father_node[i]].push_back(i);
}

d_row_tree->level_p = Hierarchy_Structure.size() + 1;
d_row_tree->total_nodes = total_nodes;
d_row_tree->largest_level_start_index = total_nodes - nParts;
d_row_tree->largest_level_end_index = total_nodes - 1;
d_row_tree->hierarchy_matrix_vec.resize(total_nodes);
d_row_tree->near_n_matrix_vec.resize(total_nodes);
d_row_tree->less_near_n.resize(total_nodes - nParts);



cout << "Finish Setting for the tree structure." << endl;
cout << "Total Nodes Number : " << total_nodes << endl;

int largest_level_start_index = total_nodes - nParts;

int d_1 = d / 2 + 1;


vector<int> factorization_d_vec(total_nodes, 0);

vector<int> selection_d_vec(total_nodes, 0);

vector<double> F_norm_near_n_matrix(total_nodes, 0);

  for(int i = total_nodes - 1; i >= 1; --i)
  if(i >= largest_level_start_index){
    factorization_d_vec[i] = d / number_of_son[father_node[i]] + 1;
    selection_d_vec[i] = d / number_of_son[father_node[i]] + 1;
  } else {
    factorization_d_vec[i] = d / number_of_son[father_node[i]] + 1;
    
    int tot = 0;
    for(auto x : son_node_list[i])
      tot += factorization_d_vec[x];

    selection_d_vec[i] = d;
    
    factorization_d_vec[i] = min(factorization_d_vec[i], selection_d_vec[i]);
  }

  selection_d_vec[0] = factorization_d_vec[0] = d;




vector< vector<double> > low_approx_value_vec(total_nodes);

for(int i = 0; i < total_nodes; i++){
  low_approx_value_vec[i].resize(factorization_d_vec[i]);
}













// vector<int> SVD_flag_vec(vertex_number, -1);
vector<int> SVD_flag_vec(total_nodes, -1);

// vector<int> enqueue_flag_vec(vertex_number, -1);
vector<int> enqueue_flag_vec(total_nodes, -1);

//   Maintain a vector of $flag\_vec$ 
//   with $flag\_vec[i] = false$ for each block, to denote
//   the SVD is still incomplete.

  

// //   Set an empty queue $Q$.

Queue Q =
{
  //.arr = 
  (MKL_INT*)malloc( sizeof(MKL_INT) * total_nodes ),
  //.capacity = 
  total_nodes,
  //.front =
  0,
  //.rear = 
  0
};



mat *SS = matrix_new(d, d);

std::stack<int> Stack;

int count_top = nParts;

for(int i = 0; i < nParts; i++)
  Stack.push(nParts - i - 1 + largest_level_start_index);


for(int i = 0; i < hierarchy_n; i++){
  if(enqueue_flag_vec[largest_level_start_index + i] != 0){
    enqueue(&Q, largest_level_start_index + i);
    enqueue_flag_vec[largest_level_start_index + i] = 0;
  }
}

count_top -= hierarchy_n;


vector<int> nparts_binary_snapshot_vec(nParts, 0);


vector<std::map<long long int, double>> all_csr_entries(vertex_number);

std::vector<long long int> offsets(vertex_number, 0);


vector<std::ifstream> ifs_vec(NUMTHREAD);

vector<std::ofstream> ofs_vec(NUMTHREAD);

vector<vector<int>> selection_result(total_nodes);













vector<vector<long long int>> nparts_all_count(nParts, vector<long long int>(NUMTHREAD, 0));

vector<long long int> nnz_nparts_vec(nParts, 0);




vector<vector<long long int>> nparts_write_out_io_all_count(nParts, vector<long long int>(NUMTHREAD, 0));

vector<vector<long long int>> nparts_read_in_io_all_count(nParts, vector<long long int>(NUMTHREAD, 0));



vector<string> embedding_matrix_name_vec(d_row_tree->total_nodes);


for(int i = 0; i < d_row_tree->total_nodes; i++){
  embedding_matrix_name_vec[i] = "IO_File/" + queryname + "_embedding_matrix_" + to_string(i) + ".bin";
}


vector<int> random_seed(nParts);
for(int i = 0; i < nParts; ++i)
  random_seed[i] = 6789 + i * 2;

double UpperBound = snapshots_number / (1 - Initial_val_percent);
double LowerBound = 0;

double Initial_val = UpperBound + 1e-7 - snapshots_number * 1.0;

cout << LowerBound << " " << UpperBound << " " << Initial_val << " " << endl;
cout << "Snapshot number = " << snapshots_number << endl;

pair<double, pair<double, double>> Preserve_Para;

Preserve_Para.first = Initial_val;
Preserve_Para.second.first = LowerBound;
Preserve_Para.second.second = UpperBound;
cout << "LowerBound = " << LowerBound << endl;
cout << "UpperBound = " << UpperBound << endl;
cout << "Initial_val = " << Initial_val << endl;

vector<vector<double>> Frobeniusnorm_per_snapshots(nParts);
for(int i = 0; i < nParts; ++i)
  Frobeniusnorm_per_snapshots[i].assign(snapshots_number + 1, 0.0);

vector<double> MAE_vec(nParts, 0.0);


double* Frobenius_norm_each_col = new double[vertex_number]{0};
double** Frobenius_norm_each_row = new double* [nParts];
// pair<int, double>** ppr_nnz = new pair<int, double>*[vertex_number]{NULL};
double** ppr_nnz = new double*[vertex_number]{NULL};
int** ppr_col = new int*[vertex_number]{NULL};

for(int i = 0; i < nParts; ++i)
  Frobenius_norm_each_row[i] = new double[Group_Size[i]]{0};

double* snapshot_num_columns = new double[vertex_number]{0};
double** snapshot_num_rows = new double*[nParts]{NULL};
  
std::mt19937 gen_snapshot(random_seed[0]);
std::uniform_real_distribution<> dist_snapshot(LowerBound, UpperBound);

for(int i = 0; i < vertex_number; ++i)
{
  double temp = dist_snapshot(gen_snapshot);
  snapshot_num_columns[i] = temp;
  
}

for(int i = 0; i < nParts; ++i)
{
  snapshot_num_rows[i] = new double[Group_Size[i]]{0};
  for(int j = 0; j < Group_Size[i]; ++j)
    snapshot_num_rows[i][j] = dist_snapshot(gen_snapshot);
}

double Total_ppr_time = 0;
double Total_ppr_retain_time = 0;
double save_emb_time = 0;
auto hierarchy_Start_time = std::chrono::system_clock::now();

mat_csr** leaf_ppr_matrix = new mat_csr*[total_nodes]();

//Input top non-zeros elements from file with sorted elements


bool Exist_tag = Check_Exist_PPR(queryname, backward_theta);
// Exist_tag = 0;

if(Exist_tag)
{
  std::string offset_path = "IO_File/"+ queryname + "_offset.bin";
  cout << "Offset file path as : " << offset_path << endl;

  std::ifstream offset_ifs;
  offset_ifs.open(offset_path, std::ios::binary);
  boost::archive::binary_iarchive offset_ia(offset_ifs, boost::archive::no_header | boost::archive::no_tracking);
  offset_ia >> offsets;
  offset_ifs.close();

  for(int i = 0; i < NUMTHREAD; i++){
    string stored_filename = "IO_File/" + queryname + "_thread_" + to_string(i) + ".bin";
    ifs_vec[i].open(stored_filename, std::ios::binary);
  }

} else 
  for(int i = 0; i < NUMTHREAD; i++){
    string stored_filename = "IO_File/" + queryname + "_thread_" + to_string(i) + ".bin";
    // std::fstream fs;
    ofs_vec[i].open(stored_filename, std::ios::binary);
    
  }

  std::thread mem_rec(Memory_use_Monitoring_Gen);
  

while(!Stack.empty()){
  
  int index = Stack.top();
  Stack.pop();

  cout<<"index = "<<index<<endl;

  string stored_filename = "IO_File/" + queryname + "_embedding_matrix_" + to_string(index) + ".bin";

  if(index == 0){
    stored_filename = "IO_File/" + queryname + "_embedding_matrix_" + to_string(index + 1) + ".bin";
  }


  if(index < largest_level_start_index){

        double Fnorm_tot = 0;
        int atleast = 0;


        cout << "Node " << index << " Merge from " << number_of_son[index] << " son nodes." << endl;
        
        int top_d = -selection_d_vec[index];

        for(auto x : son_node_list[index])
        {


          if(x < largest_level_start_index)
          {
            low_approx_value_vec[x].resize(factorization_d_vec[x]);
            d_row_tree->hierarchy_matrix_vec[x] = matrix_new(vertex_number, factorization_d_vec[x]);
          }
          
          if (x >= largest_level_start_index)
          {
            cout << "FrPCA has already done on bottom node number " << x << endl; 
          }
          else 
          {
            cout << "Do Truncated SVD on a non-leaf node number " << x << " with " << factorization_d_vec[x] << " dimensions." << endl;
            dense_sub_svd_store_function(
              factorization_d_vec[x],
              pass, 
              d_row_tree->near_n_matrix_vec[x], 
              d_row_tree->hierarchy_matrix_vec[x],
              low_approx_value_vec[x]
            );

            // cout << "NNode " << x << " with fnorm sqr = " << get_matrix_frobenius_norm_square(d_row_tree->hierarchy_matrix_vec[x]) << "\n";
            cout << "Singular Values on a non-leaf node number " << x << " with " << factorization_d_vec[x] << " dimensions as below:" << endl;
            
            double temp_sum = 0;
            for(auto x : low_approx_value_vec[x])
            {
              cout << x << endl;
              temp_sum += x * x;
            }
            
  
            cout << "Truncated SVD done on node " << x << endl;

          
            matrix_delete(d_row_tree->near_n_matrix_vec[x]);
            d_row_tree->near_n_matrix_vec[x] = NULL;

            cout << "Matrix Space Free!" << endl;
          }
        }

        cout<<"selection_d_vec[index] = "<<selection_d_vec[index]<<endl;

        for(int x : son_node_list[index])
          top_d += factorization_d_vec[x];
        
        cout<<"top_d = "<<top_d<<endl;


        vector<int>& result = selection_result[index];
        result.resize(number_of_son[index], 0);

        take_out_index_OfSons(std::ref(low_approx_value_vec), son_node_list[index], factorization_d_vec, result, number_of_son[index], top_d);


        cout<<endl;
        for(int i = 0; i < number_of_son[index]; i++){
            cout<<"result["<<i<<"] = "<<result[i]<<endl;
        }
        cout<<endl;

        



        cout<<"dense 2"<<endl;
        
        d_row_tree->near_n_matrix_vec[index] = matrix_new(d_row_tree->row_dim, selection_d_vec[index]);

        cout<<"dense 3"<<endl;

        F_norm_near_n_matrix[index] = 0;
        int current_row = 0;

        for(int i = 0; i < number_of_son[index]; i++)
        {
          int x = son_node_list[index][i];

          matrix_copy_A_start_end_cols_B_start_end_cols(
            d_row_tree->near_n_matrix_vec[index], d_row_tree->hierarchy_matrix_vec[x], 
          current_row, current_row + factorization_d_vec[x] - result[i], 
          result[i], factorization_d_vec[x]);

          current_row += factorization_d_vec[x] - result[i];

          for(int tempi = result[i]; tempi < factorization_d_vec[x]; ++tempi)
            F_norm_near_n_matrix[index] += sqr(low_approx_value_vec[x][tempi]);

          cout << x << " dense 6"<<endl;

            // matrix_delete(d_row_tree->hierarchy_matrix_vec[x]);

          cout << current_row <<" after delete"<<endl;
          
            // d_row_tree->hierarchy_matrix_vec[x] = NULL;

        }

cout<<"before dense_sub_svd_function"<<endl;


        if(index != 0)
        {

          auto dense_svd_start_time = chrono::system_clock::now();

          cout<<"d_row_tree->near_n_matrix_vec[index]->nrows = "<<d_row_tree->near_n_matrix_vec[index]->nrows<<endl;
          cout<<"d_row_tree->near_n_matrix_vec[index]->ncols = "<<d_row_tree->near_n_matrix_vec[index]->ncols<<endl;

          auto dense_end_svd_time = chrono::system_clock::now();
          auto dense_elapsed_svd_time = chrono::duration_cast<std::chrono::seconds>(dense_end_svd_time - dense_svd_start_time);
          cout<< "dense_svd cost time = "<<dense_elapsed_svd_time.count() << endl;

          

          auto save_dense_start_time = chrono::system_clock::now();
          

          auto save_dense_end_time = chrono::system_clock::now();
          auto save_dense_elapsed_time = chrono::duration<double>(save_dense_end_time - save_dense_start_time);
          // auto save_dense_elapsed_time = chrono::duration_cast<std::chrono::seconds>(save_dense_end_time - save_dense_start_time);
          cout<< "save dense_dense cost time = "<<fixed <<setprecision(12) <<save_dense_elapsed_time.count() << endl;
          save_emb_time += save_dense_elapsed_time.count();

          ++finish_son_number[father_node[index]];
          if(finish_son_number[father_node[index]] == number_of_son[father_node[index]])
            Stack.push(father_node[index]);
          
        }
        else
        {
            
          cout<<"Final SVD"<<endl;

          auto svd_start_time = chrono::system_clock::now();

          cout<<"d_row_tree->near_n_matrix_vec[0]->nrows = "<<d_row_tree->near_n_matrix_vec[0]->nrows<<endl;

          cout<<"d_row_tree->near_n_matrix_vec[0]->ncols = "<<d_row_tree->near_n_matrix_vec[0]->ncols<<endl;


          mat *Vt = matrix_new(d, d_row_tree->near_n_matrix_vec[0]->ncols);

          truncated_singular_value_decomposition(d_row_tree->near_n_matrix_vec[0], U_cur_iter, SS, Vt, d);
          
          matrix_delete(Vt);

          matrix_delete(d_row_tree->near_n_matrix_vec[0]);

          Vt = NULL;
          d_row_tree->near_n_matrix_vec[0] = NULL;

          low_approx_value_vec[0].resize(factorization_d_vec[0]);
          for(int iter = 0; iter < factorization_d_vec[0]; ++iter)
          low_approx_value_vec[0][factorization_d_vec[0] - 1 - iter] = matrix_get_element(SS, iter, iter);

          cout << "Singular Values of Node 0 as below: \n";
          for(auto x : low_approx_value_vec[0])
            cout << x << endl;


          auto end_svd_time = chrono::system_clock::now();
          auto elapsed_svd_time = chrono::duration_cast<std::chrono::seconds>(end_svd_time - svd_start_time);
          cout<< "Final SVD cost time = "<<elapsed_svd_time.count() << endl;

          cout<<"Final SVD finished!"<<endl;
          break;
        }



    }
  
  else{
            // NUMTHREAD = 1;
            cout<<"sparse matrix ?"<<endl;
            // return;
              long long int nnz = 0;
              int order = index - largest_level_start_index;
              vector<long long int> &all_count = nparts_all_count[order];
              

              cout << "ppr computation " << endl;
              auto ppr_start_time = std::chrono::system_clock::now();
              vector<thread> threads;


              vector<long long int> &write_out_io_all_count = nparts_write_out_io_all_count[order];


              long long int start_column = order * common_group_size;

              long long int group_size;

              if(order == nParts - 1){
                group_size = final_group_size;
              }
              else{
                group_size = common_group_size;
              }

              cout << "Ready to PushForward\n" << endl;

              ++count_nParts;
              int random_num = rand() % 7 + 3;
              double residuemax_ = 2.0 / random_num / count_nParts / pow(1.1, count_nParts);
              residuemax_ = residuemax;
              double reservemin_ = residuemax_;
              
              cout << "residuemax = reservemin = " << fixed <<setprecision(12) << residuemax_ << "  for index " << index << endl;
              
              vector<long long int> &read_in_io_all_count = nparts_read_in_io_all_count[order];

              for (int t = 1; t <= NUMTHREAD; t++){
                long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
                long long int end = 0;
                if (t == NUMTHREAD){
                  end = start_column + group_size;
                } else{
                  end = start_column + t * (group_size / NUMTHREAD);
                }

                if(Exist_tag == 0)
                  threads.push_back(thread(AnytimeUndirectedRowbasedMixedPush, t-1, start, end, g, residuemax_, reservemin_,
                    std::ref(all_count[t - 1]), alpha, std::ref(all_csr_entries), std::ref(ofs_vec[t-1]), 
                    std::ref(offsets),
                    std::ref(write_out_io_all_count[t - 1]),
                    1, ppr_col, ppr_nnz
                  ));
                else 
                  threads.push_back(thread(PPR_retrieve_less_space, start, end, 
                    std::ref(all_csr_entries),
                    std::ref(ifs_vec[t - 1]),
                    std::ref(offsets),
                    std::ref(read_in_io_all_count[t - 1]),
                    &all_count[t - 1], ppr_col, ppr_nnz
                  ));

              }

              for (int t = 0; t < NUMTHREAD ; t++){
                threads[t].join();
              }
              vector<thread>().swap(threads);


              auto ppr_end_time = chrono::system_clock::now();
              auto elapsed_ppr_time = chrono::duration<double>(ppr_end_time - ppr_start_time);
              cout<< "ppr cost time = "<<fixed<<setprecision(12)<<elapsed_ppr_time.count()<<endl;
              Total_ppr_time += elapsed_ppr_time.count();

              for(int i = 0; i < NUMTHREAD; i++){
                // cout<<"before: all_count["<<i<<"] = "<<all_count[i]<<endl;
                nnz += all_count[i];
                all_count[i] = nnz - all_count[i];
              }

              nnz_nparts_vec[order] = nnz;
              cout<<"nnz = "<<nnz<<endl;
            


          auto sparse_svd_start_time = chrono::system_clock::now();

        
        low_approx_value_vec[index].resize(factorization_d_vec[index]);
        d_row_tree->hierarchy_matrix_vec[index] = matrix_new(vertex_number, factorization_d_vec[index]);

        cout<<"factorization_d_vec[index] = "<<factorization_d_vec[index]<<endl;
        cout<<"low_approx_value_vec[index].size() = "<<low_approx_value_vec[index].size()<<endl;

        pair<double, pair<double, double>> Ppara = Preserve_Para;
        Ppara.first = sqrt((Ppara.first - LowerBound) * (UpperBound - LowerBound));
        Ppara.first = Ppara.first + LowerBound;

          sparse_sub_svd_function_take_out_sparse_matrix_preserve_someofelements(
            &leaf_ppr_matrix[index],
            vertex_number,
            all_csr_entries,
            Start_Column_List[order],
            Group_Size[order],
            nnz,
            all_count,
            NUMTHREAD,
            Preserve_Para,
            random_seed[order],
            erase_way, erase_way, 
            Frobeniusnorm_per_snapshots[order],
            // Frobenius_norm_each_col,
            ppr_col, ppr_nnz,
            snapshot_num_columns,
            snapshot_num_rows[order],
            // Frobenius_norm_each_row[order],
            &Total_ppr_retain_time
          );

        double kk_t = 0;
        long long cc_t = 0;
        
        cout << "Do FrPCA on a leaf node number " << index << " with " << factorization_d_vec[index] << " dimensions." << endl;
        
        
        sparse_matrix_take_out_singular_values(
            low_approx_value_vec[index], 
            factorization_d_vec[index], pass, d_row_tree->hierarchy_matrix_vec[index],
            leaf_ppr_matrix[index], vertex_number, leaf_ppr_matrix[index]->nnz
            );

            // cout << "NNode " << index << " with fnorm sqr = " << get_matrix_frobenius_norm_square(d_row_tree->hierarchy_matrix_vec[index]) << "\n";
                    
            cout << "Singular values on a leaf node number " << index << " with " << factorization_d_vec[index] << " dimensions." << endl;
        
            
            // cout << "NNode " << x << " with ori fnorm sqr = " << get_matrix_frobenius_norm_square(d_row_tree->near_n_matrix_vec[x]);
            

        leaf_ppr_matrix[index] = NULL;
        cout << "FrPCA finished on node " << index << endl;

    auto sparse_end_svd_time = chrono::system_clock::now();
    auto sparse_elapsed_svd_time = chrono::duration_cast<std::chrono::seconds>(sparse_end_svd_time - sparse_svd_start_time);
    cout<< "sparse_svd cost time = "<<sparse_elapsed_svd_time.count()<<endl;
        
    auto save_sparse_start_time = chrono::system_clock::now();


    auto save_sparse_end_time = chrono::system_clock::now();
    // auto save_sparse_elapsed_time = chrono::duration_cast<std::chrono::seconds>(save_sparse_end_time - save_sparse_start_time);
    auto save_sparse_elapsed_time = chrono::duration<double>(save_sparse_end_time - save_sparse_start_time);
    cout<< "save sparse_dense matrix cost time = " << fixed << setprecision(12) << save_sparse_elapsed_time.count() << endl;
    save_emb_time += save_sparse_elapsed_time.count();


        cout<<"Sparse index = "<<index<<endl;
        cout<<"---------------------------"<<endl;


        
        ++finish_son_number[father_node[index]];
        if(finish_son_number[father_node[index]] == number_of_son[father_node[index]])
          Stack.push(father_node[index]);

        cout<<"sparse matrix finished"<<endl;

  }

}

Overwrite_PPR_info(queryname, backward_theta);

if(Exist_tag == 1)
{
  for(int i = 0; i < NUMTHREAD; i++)
    ifs_vec[i].close();
} else 
{
  for(int i = 0; i < NUMTHREAD; i++){
    ofs_vec[i].close();
  }
}


auto hierarchy_End_time = std::chrono::system_clock::now();

auto hierarchy_Time_Consume = std::chrono::duration<double>(hierarchy_End_time - hierarchy_Start_time);

  cout << "----------------------------\n";
  cout << "PPR Computation time = " << fixed << setprecision(12) << Total_ppr_time << endl;
  cout << "PPR Retain time = " << fixed << setprecision(12) << Total_ppr_retain_time << endl;
  cout << "Save Embedding time = " << fixed << setprecision(12) << save_emb_time << endl;
  cout << "Hierarchy consumed time = " << (hierarchy_Time_Consume.count() - Total_ppr_time - Total_ppr_retain_time - save_emb_time) << endl;
  
  cout << "\n-------------------------------------------" << endl;



if(!Exist_tag)
{
  string offset_outpath = "IO_File/" + queryname + "_offset" + ".bin";
  std::ofstream offset_ofs;
  offset_ofs.open(offset_outpath, std::ios::binary);
  boost::archive::binary_oarchive offset_oa(offset_ofs,
                                            boost::archive::no_header | boost::archive::no_tracking);
  offset_oa << offsets;
  offset_ofs.close();
}

  std::string V_result_path = "IO_File/"+ queryname + "_" + to_string(residuemax) + "_" + to_string(d) + "_V.bin";
  std::ofstream V_ofs(V_result_path, std::ios::binary);
  V_ofs.write(reinterpret_cast<char*>(U_cur_iter->d), sizeof(double) * U_cur_iter->nrows * (long long)U_cur_iter->ncols);
  V_ofs.close();

  cout<<"finish hierarchy"<<endl;

  // MatrixXd U_V(vertex_number, total_d);

  MatrixXd U_out(vertex_number, d);
  MatrixXd V_out(vertex_number, d);




  auto static_right_matrix_start_time = chrono::system_clock::now();



  for(int i = 0; i < NUMTHREAD; i++){
    string stored_filename = "IO_File/" + queryname + "_thread_" + to_string(i) + ".bin";

    ifs_vec[i].open(stored_filename, std::ios::binary);
  }
  

    Output_Memory_Use();

    mem_rec.detach();

    thread Mem11(Memory_use_Monitoring_Gen);

    std::chrono::seconds duration11(3);
    std::this_thread::sleep_for(duration11);

    Output_Memory_Use();
    Mem11.detach();


    double* leaf_fnorm_square = new double[nParts]();
    long long int* leaf_nnz = new long long[nParts]();
    double* L_1_1_norm = new double[nParts]();
    
    pair<double, pair<double, double>> TPpara = Preserve_Para;

    double temp112 = max((UpperBound - LowerBound) * (UpperBound - TPpara.first), 0.0);
    double temp1 = sqrt(Preserve_Para.first * (UpperBound - LowerBound));

    if(erase_way > 0)
      TPpara.first = temp1;
      
    mkl_right_matrix_multiplication_V_frobenius_error_with_recover_update(
      d_row_tree, U_cur_iter,
      MAE_vec,
      vertex_number, d, SS, NUMTHREAD,
      largest_level_start_index, 
      Group_Size, 
      all_csr_entries,
      offsets,
      ifs_vec,
      nparts_all_count,
      nnz_nparts_vec,
      nparts_read_in_io_all_count,
      random_seed, TPpara, ppr_col, ppr_nnz, erase_way,
      snapshot_num_columns,
      snapshot_num_rows,
      leaf_fnorm_square,
      leaf_nnz,
      L_1_1_norm
    );


if(erase_way == 0)
  return 0;
  


  auto static_right_matrix_end_time = chrono::system_clock::now();
  auto static_elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(static_right_matrix_end_time - static_right_matrix_start_time);
  // cout << "Right matrix consumed time = "<< static_elapsed_right_matrix_time.count() << endl;


  cout << "Preparation for Dynamic Part." << endl;

  double total_leaf_fnorm = 0;
  long long total_leaf_nnz = 0;
  double total_L11_norm = 0;

  for(int i = 0 ; i < nParts; ++i)
    total_leaf_fnorm += leaf_fnorm_square[i],
    total_leaf_nnz += leaf_nnz[i];




  for(int i = 0; i < NUMTHREAD; i++){
    ifs_vec[i].close();
  }

  mat* U_original = U_cur_iter;
  U_cur_iter = matrix_new(vertex_number, d);
  matrix_copy(U_cur_iter, U_original);


  vector<double> accumulated_update_fnorm(nParts, 0);


  vector<double> fnorm_sum_snap(snapshots_number + 1);

  double Total_ppr_fnorm = 0;
  for(int i = 0; i <= snapshots_number; ++i)
  {
    for(int j = 0; j < nParts; ++j)
      fnorm_sum_snap[i] += Frobeniusnorm_per_snapshots[j][i];
    Total_ppr_fnorm += fnorm_sum_snap[i];
  }

  
  double Current_snapshot_fnorm = fnorm_sum_snap[0];

  double Amortized_update_time = 0;

  for(int iter = 1; iter <= snapshots_number; iter++)
  {
    pair<double, pair<double, double>> Ppara = Preserve_Para;
    
    double temp11 = max((UpperBound - LowerBound) * (UpperBound - iter - Ppara.first), 0.0);

    if(erase_way > 0)
      // Ppara.first = UpperBound - sqrt(temp11);
      Ppara.first = sqrt((Ppara.first + iter) * (UpperBound - LowerBound));


    Current_snapshot_fnorm += fnorm_sum_snap[iter];
    double Address_snapshot_fnorm = Current_snapshot_fnorm;

    total_leaf_fnorm = 0;
    total_leaf_nnz = 0;
    total_L11_norm = 0;
    for(int i = 0 ; i < nParts; ++i)
    {
      total_leaf_fnorm += leaf_fnorm_square[i],
      total_leaf_nnz += leaf_nnz[i];
      total_L11_norm += L_1_1_norm[i];
    }

    double average_L11_norm = total_L11_norm / total_leaf_nnz;


    for(int i = 0; i < NUMTHREAD; i++){
      string stored_filename = "IO_File/" + queryname + "_thread_" + to_string(i) + ".bin";

      ifs_vec[i].open(stored_filename, std::ios::binary);
    }

    cout<<"Current Update Snapshot Number = "<<iter<<endl;

    double ppr_tot_time = 0;

    edge_vec.clear();
    malloc_trim(0);
    
    cout<<"edge_vec.size() = "<<edge_vec.size()<<endl;


    vector<thread> refresh_threads;

    int top_influenced_index = 0;

    queue<int> Update_Queue;
    bool* InQueue = new bool[total_nodes + 1]();
    
    if(iter <= snapshots_number)
    {
      for(int i = 0; i < nParts; ++i){
        accumulated_update_fnorm[i] += Frobeniusnorm_per_snapshots[i][iter];
        // Address_snapshot_fnorm -= accumulated_update_fnorm[i];
      }
      

    
      cout << "update martix for snapshot " << iter << " is generated!" << endl;

      vector<P_d_l> update_order(nParts);

      double total_update_fnorm = 0;
      // long long total_update_nnz = 0;
      for(int  i = 0; i < nParts; ++i)
      {
        update_order[i].first = accumulated_update_fnorm[i];
        update_order[i].second = i;

        total_update_fnorm += update_order[i].first;
        cout << i << " update F norm square = " << update_order[i].first << endl;
      }

      // cout << "Total update nnz = " << total_update_nnz << endl;

      sort(update_order.begin(), update_order.end(), std::greater<P_d_l>());

      int update_number = 0;
      cout << "total_update_fnorm = " << total_update_fnorm << endl;
      cout << "sqr(threshold_percent) * total_leaf_fnorm = " << sqr(threshold_percent) * Address_snapshot_fnorm << endl;
      cout << "total_leaf_fnorm = " << Address_snapshot_fnorm << endl;

      if(threshold_percent < 1e-10)
        for(int i = 0; i < nParts; ++i)
        Update_Queue.push(i + largest_level_start_index);
      else
      while(total_update_fnorm > sqr(threshold_percent) * Address_snapshot_fnorm + 1e-8)
      {
        Update_Queue.push(largest_level_start_index + update_order[update_number].second);
        total_update_fnorm -= update_order[update_number].first;
        // Address_snapshot_fnorm += update_order[update_number].first;
        ++update_number;
        cout << "update number = " << update_number << " 's fnorm = " << update_order[update_number].first << endl;
      }
    } 



    cout << " Number of Leaf vertices in Updata Queue = " << Update_Queue.size() << endl;
    
    double Error_Before_Update = 0;
    // if(1 == 0)
    mkl_right_matrix_multiplication_V_frobenius_error_with_recover_update(
          d_row_tree, U_original, MAE_vec, vertex_number, d, SS, NUMTHREAD,
          largest_level_start_index, Group_Size, 
          all_csr_entries, offsets, ifs_vec, nparts_all_count, nnz_nparts_vec,
          nparts_read_in_io_all_count, random_seed, Ppara, ppr_col, ppr_nnz, erase_way,
          snapshot_num_columns,
          snapshot_num_rows,
          leaf_fnorm_square, leaf_nnz, L_1_1_norm,
          &Error_Before_Update
        );



    std::thread mem_rec_iter(Memory_use_Monitoring_Gen);

    auto iter_start_time = chrono::system_clock::now();    

    int Count_Leaf_Num = 0;
    int Count_NonLeaf_Num = 0;
    long long Count_Leaf_nnz = 0;

    while(!Update_Queue.empty())
    {
      int index = Update_Queue.front();
      Update_Queue.pop();
      cout << "Current Node Number : " << index << endl;

      if(index >= largest_level_start_index)
      {
        auto ppr_refresh_time = std::chrono::system_clock::now();
        top_influenced_index = index - largest_level_start_index;
        Count_Leaf_nnz += leaf_nnz[top_influenced_index];
        Count_Leaf_Num++;

        cout << "Current Leaf Node Number : " << top_influenced_index << endl;
        long long int start_column = Start_Column_List[top_influenced_index];

        auto ppr_start_time = chrono::system_clock::now();

        long long int group_size = Group_Size[top_influenced_index];

        vector<long long int> &all_count = nparts_all_count[top_influenced_index];

        int read_in_file_iter_index = nparts_binary_snapshot_vec[top_influenced_index];

        // int write_out_file_iter_index = iter;

        vector<long long int> &write_out_io_all_count = nparts_write_out_io_all_count[top_influenced_index];
        vector<long long int> &read_in_io_all_count = nparts_read_in_io_all_count[top_influenced_index];

        for (int t = 1; t <= NUMTHREAD; t++){
          long long int start = start_column + (t-1)*(group_size/NUMTHREAD);
          long long int end = 0;
          if (t == NUMTHREAD){
            end = start_column + group_size;
          } 
          else{
            end = start_column + t * (group_size / NUMTHREAD);
          }

          long long ttt = 0;
          refresh_threads.push_back(thread(PPR_retrieve_less_space, start, end, 
          std::ref(all_csr_entries),
          std::ref(ifs_vec[t - 1]),
          std::ref(offsets),
          std::ref(read_in_io_all_count[t - 1]),
          &ttt, ppr_col, ppr_nnz
          ));

        }
        
        for (int t = 0; t < NUMTHREAD ; t++){
          refresh_threads[t].join();
        }
        vector<thread>().swap(refresh_threads);
        
        long long nnz = nnz_nparts_vec[top_influenced_index];

        sparse_sub_svd_function_take_out_sparse_matrix_preserve_someofelements(
          &leaf_ppr_matrix[index],
          vertex_number,
          all_csr_entries,
          Start_Column_List[top_influenced_index],
          Group_Size[top_influenced_index],
          nnz,
          all_count,
          NUMTHREAD,
          Ppara,
          random_seed[top_influenced_index],
          0, erase_way, Frobeniusnorm_per_snapshots[top_influenced_index],
          ppr_col, ppr_nnz,
          snapshot_num_columns,
          snapshot_num_rows[top_influenced_index]
        );

        accumulated_update_fnorm[top_influenced_index] = 0;


        nparts_binary_snapshot_vec[top_influenced_index] = top_influenced_index;

        nnz_nparts_vec[top_influenced_index] = nnz;

        cout << "nnz = " << nnz << endl;

        auto finish_ppr_refresh_time = chrono::system_clock::now();
        auto elapsed_ppr_refresh_time = chrono::duration<double>(finish_ppr_refresh_time - ppr_refresh_time);
        cout<< "Iter = "<<iter << ", total ppr time: "<< elapsed_ppr_refresh_time.count() << endl;


        int largest_level_start_index = d_row_tree->largest_level_start_index;

        cout << "singular values of original matrix " << index - largest_level_start_index << " are as below:\n";

        for(auto x : low_approx_value_vec[index])
          cout << x << endl;

        cout << "End of singular values of original matrix.\n";

        int sparse_unique_update_times = 0;
        
        if(d_row_tree != NULL && d_row_tree->hierarchy_matrix_vec[index] != NULL)
        {
          matrix_delete(d_row_tree->hierarchy_matrix_vec[index]);
          cout << "Out of Date singular matrix deleted!" << endl;
        } 
          else cout << "singular right matrix already deleted.\n";


        d_row_tree->hierarchy_matrix_vec[index] = matrix_new(vertex_number, factorization_d_vec[index]);

        auto ppr_end_time = chrono::system_clock::now();
        ppr_tot_time += chrono::duration<double>(ppr_end_time - ppr_start_time).count();
        
        auto _sstart_point = chrono::system_clock::now();

          cout << "Do FrPCA on a leaf node number " << index << " with " << factorization_d_vec[index] << " dimensions." << endl;
          sparse_matrix_take_out_singular_values(
              low_approx_value_vec[index], 
              factorization_d_vec[index], pass, d_row_tree->hierarchy_matrix_vec[index],
              leaf_ppr_matrix[index], vertex_number, leaf_ppr_matrix[index]->nnz
          );

        auto _eend_point = chrono::system_clock::now();
        auto _ddduration = chrono::duration<double>(_eend_point - _sstart_point);

        cout << "Frpca Computation Time = " << _ddduration.count() << endl;


          leaf_ppr_matrix[index] = NULL;
          cout << "FrPCA finished on node " << index << endl;


          // cout<<"Sparse index = " << index << endl;
          cout<<"d_row_tree->hierarchy_matrix_vec[index]->d[0] = "<<d_row_tree->hierarchy_matrix_vec[index]->d[0]<<endl;
          cout<<"d_row_tree->hierarchy_matrix_vec[index]->nrows = "<<d_row_tree->hierarchy_matrix_vec[index]->nrows<<endl;
          cout<<"d_row_tree->hierarchy_matrix_vec[index]->ncols = "<<d_row_tree->hierarchy_matrix_vec[index]->ncols<<endl;
          cout<<"---------------------------"<<endl;


          cout<<"finish dynamic sparse"<<endl;

      }
      else {
        
        Count_NonLeaf_Num++;

        cout << "Node " << index << " Merge from " << number_of_son[index] << " son nodes." << endl;
        
        int top_d = -selection_d_vec[index];

        cout << "selection_d_vec[index] = " << selection_d_vec[index] << endl;

        for(auto x : low_approx_value_vec[index])
          cout << x << endl;
          
        // cout << "Maximun Singular Value of original dense Submaxtrix is " << low_approx_value_vec[index][factorization_d_vec[index] - 1] << endl;
        // cout << "Minimun Singular Value of original dense Submaxtrix is " << low_approx_value_vec[index][0] << endl;

        for(int x : son_node_list[index])
          top_d += factorization_d_vec[x];
        
        cout << "top_d = " << top_d << endl;

        vector<int>& result = selection_result[index];

        // vector<int> result(number_of_son[index], 0);

        // take_out_index_OfSons(std::ref(low_approx_value_vec), son_node_list[index], factorization_d_vec, result, number_of_son[index], top_d);

        for(int i = 0; i < number_of_son[index]; i++){
            cout<<"result["<<i<<"] = "<<result[i]<<endl;
        }
        cout<<endl;

        
        d_row_tree->near_n_matrix_vec[index] = matrix_new(d_row_tree->row_dim, selection_d_vec[index]);

        cout<<"dense 3"<<endl;

        F_norm_near_n_matrix[index] = 0;
        int current_row = 0;

        for(int i = 0; i < number_of_son[index]; i++)
        {
          int x = son_node_list[index][i];

          matrix_copy_A_start_end_cols_B_start_end_cols(
            d_row_tree->near_n_matrix_vec[index], d_row_tree->hierarchy_matrix_vec[x], 
          current_row, current_row + factorization_d_vec[x] - result[i], 
          result[i], factorization_d_vec[x]);

          current_row += factorization_d_vec[x] - result[i];

          for(int tempi = result[i]; tempi < factorization_d_vec[x]; ++tempi)
            F_norm_near_n_matrix[index] += sqr(low_approx_value_vec[x][tempi]);

        }

        if(index != 0)
        {
            cout << "Do Truncated SVD on a non-leaf node number " << index << " with " << factorization_d_vec[index] << " dimensions." << endl;
            dense_sub_svd_store_function(
              factorization_d_vec[index],
              pass, 
              d_row_tree->near_n_matrix_vec[index], 
              d_row_tree->hierarchy_matrix_vec[index],
              low_approx_value_vec[index]
            );

            cout << "Truncated SVD done on node " << index << endl;
          
            matrix_delete(d_row_tree->near_n_matrix_vec[index]);
            d_row_tree->near_n_matrix_vec[index] = NULL;

            cout << "Matrix Space Free!" << endl;
        }
        else
        {
            
          cout<<"Final SVD"<<endl;

          auto svd_start_time = chrono::system_clock::now();

          cout<<"d_row_tree->near_n_matrix_vec[0]->nrows = "<<d_row_tree->near_n_matrix_vec[0]->nrows<<endl;

          cout<<"d_row_tree->near_n_matrix_vec[0]->ncols = "<<d_row_tree->near_n_matrix_vec[0]->ncols<<endl;


          mat *Vt = matrix_new(d, d_row_tree->near_n_matrix_vec[0]->ncols);

          truncated_singular_value_decomposition(d_row_tree->near_n_matrix_vec[0], U_cur_iter, SS, Vt, d);

          matrix_delete(Vt);

          matrix_delete(d_row_tree->near_n_matrix_vec[0]);

          Vt = NULL;
          d_row_tree->near_n_matrix_vec[0] = NULL;

          auto end_svd_time = chrono::system_clock::now();
          auto elapsed_svd_time = chrono::duration_cast<std::chrono::seconds>(end_svd_time - svd_start_time);
          cout<< "Final SVD cost time = "<<elapsed_svd_time.count() << endl;

          cout<<"Final SVD finished!"<<endl;
          break;
        }

      }
      
      if(index != 0 && InQueue[father_node[index]] == 0)
      {
        InQueue[father_node[index]] = 1;
        Update_Queue.push(father_node[index]);
      }

            
    }

    delete[] InQueue;
    InQueue = NULL;

 
    cout<<"Writing back embedding for the "<<iter<<" round"<<endl;
    

    auto iter_end_time = chrono::system_clock::now();
    auto elapsed_iter_time = chrono::duration<double>(iter_end_time - iter_start_time);


    Output_Memory_Use();
    mem_rec_iter.detach();

    double Error_After_Update = 0;

      mkl_right_matrix_multiplication_V_frobenius_error_with_recover_update(
        d_row_tree, U_cur_iter, MAE_vec, vertex_number, d, SS, NUMTHREAD,
        largest_level_start_index, Group_Size, 
        all_csr_entries, offsets, ifs_vec, nparts_all_count, nnz_nparts_vec,
        nparts_read_in_io_all_count, random_seed, Ppara, ppr_col, ppr_nnz, erase_way,
        snapshot_num_columns,
        snapshot_num_rows,
        leaf_fnorm_square, leaf_nnz, L_1_1_norm,
        &Error_After_Update
      );
    

    cout << "Error (A - A * V * V^T)/A Before Update = " << Error_Before_Update << endl;
    cout << "Error (A - A * V * V^T)/A After  Update = " << Error_After_Update << endl;

    cout << "Count for Updated Leaf Vertex = " << Count_Leaf_Num << endl;
    cout << "Count for Updated Non-Leaf Vertex = " << Count_NonLeaf_Num << endl;
    cout << "Count for Updated Leaf NNZ = " << Count_Leaf_nnz << endl;
    cout << "Count for all NNZ = " << total_leaf_nnz << endl;

    
    cout << fixed << setprecision(12) << "PPR time at snapshot " << iter << " = " << ppr_tot_time << endl;
    cout << "Iter = "<<iter<<", time = "<< elapsed_iter_time.count() - ppr_tot_time << endl; 

    Amortized_update_time += (elapsed_iter_time.count() - ppr_tot_time);


    for(int i = 0; i < NUMTHREAD; i++){
      ifs_vec[i].close();
    }

    
    auto right_matrix_end_time = chrono::system_clock::now();
    auto elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(right_matrix_end_time - iter_end_time);

}

  Amortized_update_time /= snapshots_number;

  cout << "Amortized update time = " << Amortized_update_time << '\n';

    for(int i = 0; i < NUMTHREAD; i++){
      string stored_filename = "IO_File/" + queryname + "_thread_" + to_string(i) + ".bin";

      ifs_vec[i].open(stored_filename, std::ios::binary);
    }



    for(int i = 0; i < NUMTHREAD; i++){
      ifs_vec[i].close();
    }


  auto end_write_time = chrono::system_clock::now();

  auto elapsed_time = chrono::duration_cast<std::chrono::seconds>(end_write_time - start_time);
  cout << "total embedding time: "<< elapsed_time.count() << endl;


}