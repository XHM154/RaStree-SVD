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




void Read_in_dynamic_PPR(
  int t_minus_1,
  string& all_csr_filename,
  string& dynamic_ppr_filename,
  std::ifstream& ifs_all_csr,
  std::ifstream& ifs_dynamic_ppr,
  // vector<std::map<long long int, double>> & thread_all_csr_entries,
  // vector<unordered_map<long long int, double>> & thread_forward_pi_forward_residue_backward_pi_backward_residue_entries,
  vector< vector<std::map<long long int, double>> > & Partition_thread_all_csr_entries,
  // vector< vector<unordered_map<long long int, double>> > & Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries,
  vector< vector< third_float_map > > & Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries,

  // long long int & all_csr_offsets,
  // long long int & dynamic_ppr_offsets
vector< long long int > & Partition_all_csr_offsets,

vector< long long int > & Partition_dynamic_ppr_offsets
)
{


    cout<<all_csr_filename<<endl;
    cout<<dynamic_ppr_filename<<endl;
    

    ifs_all_csr.open(all_csr_filename, std::ios::binary);
    ifs_dynamic_ppr.open(dynamic_ppr_filename, std::ios::binary);
    


    auto start_io_read_time = chrono::system_clock::now();

    boost::archive::binary_iarchive ia_all_csr(ifs_all_csr,
        boost::archive::no_header | boost::archive::no_tracking);


    if (!ifs_all_csr.good())
    {
      cout<<t_minus_1<<": !ifs_all_csr.good()"<<endl;
      cout << "\nError state = " << ifs_all_csr.rdstate();
      cout << "\ngood() = " << ifs_all_csr.good();
      cout << "\neof() = " << ifs_all_csr.eof();
      cout << "\nfail() = " << ifs_all_csr.fail();
      cout << "\nbad() = " << ifs_all_csr.bad() << endl;

      if (ifs_all_csr.fail()) {
          cout<<"before perror"<<endl;
          perror(all_csr_filename.c_str());
          cout<<"after perror"<<endl;
      }
      
      cerr << "ifs_all_csr Error: " << strerror(errno);

    }


    boost::archive::binary_iarchive ia_dynamic_ppr(ifs_dynamic_ppr,
        boost::archive::no_header | boost::archive::no_tracking);

    if (!ifs_dynamic_ppr.good())
    {
      cout<<t_minus_1<<": !ifs_dynamic_ppr.good()"<<endl;
      cout << "\nError state = " << ifs_dynamic_ppr.rdstate();
      cout << "\ngood() = " << ifs_dynamic_ppr.good();
      cout << "\neof() = " << ifs_dynamic_ppr.eof();
      cout << "\nfail() = " << ifs_dynamic_ppr.fail();
      cout << "\nbad() = " << ifs_dynamic_ppr.bad() << endl;

      if (ifs_dynamic_ppr.fail()) {
          cout<<"before perror"<<endl;
          perror(dynamic_ppr_filename.c_str());
          cout<<"after perror"<<endl;
      }

      cerr << "ifs_dynamic_ppr Error: " << strerror(errno);

    }
    

    auto end_io_read_time = chrono::system_clock::now();
    auto elapsed_read_time = chrono::duration_cast<std::chrono::seconds>(end_io_read_time - start_io_read_time);
    cout << "Read in PPR time = "<< elapsed_read_time.count() << endl;




    // cout<<k <<" : start_loop"<<endl;

    // ifs_all_csr.seekg(all_csr_offsets);
    // // cout<<"finish ifs_all_csr.seekg(all_csr_offsets);"<<endl;      
    // ia_all_csr >> thread_all_csr_entries;
    // // cout<<"finish ia_all_csr >> thread_all_csr_entries;"<<endl;
    // ifs_all_csr.clear();
    // // cout<<"finish ifs_all_csr.clear();"<<endl;


    
    // ifs_dynamic_ppr.seekg(dynamic_ppr_offsets);            
    // // cout<<"finish ifs_dynamic_ppr.seekg(dynamic_ppr_offsets);"<<endl;      
    // ia_dynamic_ppr >> thread_forward_pi_forward_residue_backward_pi_backward_residue_entries;
    // // cout<<"finish ia_dynamic_ppr >> thread_forward_pi_forward_residue_backward_pi_backward_residue_entries;"<<endl;
    // ifs_dynamic_ppr.clear();
    // // cout<<"finish ifs_dynamic_ppr.clear();"<<endl;



    for(int i = 0; i < Partition_all_csr_offsets.size(); i++){
      ifs_all_csr.seekg(Partition_all_csr_offsets[i]);
      // cout<<"finish ifs_all_csr.seekg(all_csr_offsets);"<<endl;      
      ia_all_csr >> Partition_thread_all_csr_entries[i];
      // cout<<"finish ia_all_csr >> thread_all_csr_entries;"<<endl;
      ifs_all_csr.clear();
      // cout<<"finish ifs_all_csr.clear();"<<endl;


      
      ifs_dynamic_ppr.seekg(Partition_dynamic_ppr_offsets[i]);            
      // cout<<"finish ifs_dynamic_ppr.seekg(dynamic_ppr_offsets);"<<endl;      
      ia_dynamic_ppr >> Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[i];
      // cout<<"finish ia_dynamic_ppr >> thread_forward_pi_forward_residue_backward_pi_backward_residue_entries;"<<endl;
      ifs_dynamic_ppr.clear();
      // cout<<"finish ifs_dynamic_ppr.clear();"<<endl;
    }




    ifs_all_csr.close();
    
    ifs_dynamic_ppr.close();

}







void Undirected_Dynamic_PPR(
  
int start, int end, 
UGraph* g, double residuemax, 
double alpha, 
int vertex_number,
double reservemin,


// vector<std::map<long long int, double>> & thread_all_csr_entries,

// vector<unordered_map<long long int, double>> & thread_forward_pi_forward_residue_backward_pi_backward_residue_entries,


// vector< vector<std::map<long long int, double>> > & thread_all_csr_entries,

// vector< vector<unordered_map<long long int, double>> > & thread_forward_pi_forward_residue_backward_pi_backward_residue_entries,



vector< vector<std::map<long long int, double>> > & Partition_thread_all_csr_entries,

// vector< vector<unordered_map<long long int, double>> > & Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries,
vector< vector< third_float_map > > & Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries,


std::ifstream& ifs_all_csr,
std::ifstream& ifs_dynamic_ppr,

// long long int & all_csr_offsets,
// long long int & dynamic_ppr_offsets,

vector< long long int > & Partition_all_csr_offsets,
vector< long long int > & Partition_dynamic_ppr_offsets,


std::ofstream& ofs_all_csr,
std::ofstream& ofs_dynamic_ppr,



long long int& all_count,
int t_minus_1,

string& queryname,

vector<pair<int, int>>& edge_vec,

int read_in_file_iter_index,
int write_out_file_iter_index,

int stored_interval

)
{

    bool *flags = new bool[vertex_number];


    all_count = 0;


    string all_csr_filename = "IO_File/" + queryname + "_thread_" + to_string(t_minus_1) + "_all_csr" + "_iter" + to_string(read_in_file_iter_index) + ".bin";
    string dynamic_ppr_filename = "IO_File/" + queryname + "_thread_" + to_string(t_minus_1)  + "_dynamic_ppr" + "_iter" + to_string(read_in_file_iter_index) + ".bin";



    Read_in_dynamic_PPR(
      t_minus_1,
      all_csr_filename,
      dynamic_ppr_filename,
      ifs_all_csr,
      ifs_dynamic_ppr,
      // thread_all_csr_entries,
      // thread_forward_pi_forward_residue_backward_pi_backward_residue_entries,
      Partition_thread_all_csr_entries,
      Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries,

      // all_csr_offsets,
      // dynamic_ppr_offsets
      Partition_all_csr_offsets,
      Partition_dynamic_ppr_offsets
    );





    string all_csr_filename_write = "IO_File/" + queryname + "_thread_" + to_string(t_minus_1) + "_all_csr" + "_iter" + to_string(write_out_file_iter_index) + ".bin";

    string dynamic_ppr_filename_write = "IO_File/" + queryname + "_thread_" + to_string(t_minus_1)  + "_dynamic_ppr" + "_iter" + to_string(write_out_file_iter_index) + ".bin";

    
    ofs_all_csr.open(all_csr_filename_write, std::ios::binary);

    ofs_dynamic_ppr.open(dynamic_ppr_filename_write, std::ios::binary);

    cout<<"open finished: "<<t_minus_1<<endl;

    //write out
    boost::archive::binary_oarchive oa_all_csr(ofs_all_csr, 
      boost::archive::no_header | boost::archive::no_tracking);

    if (!ofs_all_csr.good())
    {
      cout<<t_minus_1<<": !ofs_all_csr.good()"<<endl;
    }


    boost::archive::binary_oarchive oa_dynamic_ppr(ofs_dynamic_ppr, 
      boost::archive::no_header | boost::archive::no_tracking);

    if (!ofs_dynamic_ppr.good())
    {
      cout<<t_minus_1<<": !ofs_dynamic_ppr.good()"<<endl;
    }

    cout<<"before ppr"<<endl;

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







    for(int k = start; k < end; k++){

      // unordered_map<long long int, double>& forward_pi_entries = thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[4 * (k - start) + 0];
      // unordered_map<long long int, double>& forward_residue_entries = thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[4 * (k - start) + 1];
      // unordered_map<long long int, double>& backward_pi_entries = thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[4 * (k - start) + 2];
      // unordered_map<long long int, double>& backward_residue_entries = thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[4 * (k - start) + 3];
      
      // map<long long int, double> & all_csr_entries = thread_all_csr_entries[k - start];


      int interval_index = (k - start) / stored_interval;

      if(interval_index == Partition_thread_all_csr_entries.size()){
        interval_index--;
      }

      int inner_index = k - start - interval_index * stored_interval;

      // unordered_map<long long int, double>& forward_pi_entries = Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[interval_index][4 * inner_index + 0];
      // unordered_map<long long int, double>& forward_residue_entries = Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[interval_index][4 * inner_index + 1];    
      // unordered_map<long long int, double>& backward_pi_entries = Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[interval_index][4 * inner_index + 2];    
      // unordered_map<long long int, double>& backward_residue_entries = Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[interval_index][4 * inner_index + 3];

      cout<<"before"<<endl;
      third_float_map& forward_pi_entries = Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[interval_index][4 * inner_index + 0];
      third_float_map& forward_residue_entries = Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[interval_index][4 * inner_index + 1];    
      third_float_map& backward_pi_entries = Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[interval_index][4 * inner_index + 2];    
      third_float_map& backward_residue_entries = Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[interval_index][4 * inner_index + 3];

      map<long long int, double> all_csr_entries = Partition_thread_all_csr_entries[interval_index][inner_index];
      cout<<"after"<<endl;


      // cout<<"thread_all_csr_entries[k - start].size() = "<<thread_all_csr_entries[k - start].size()<<endl;



      forback_Q.front = 0;
      forback_Q.rear = 0;

      record_Q.front = 0;
      record_Q.rear = 0;



      // third_int_map in_degree_map;
      // third_int_map out_degree_map;
      unordered_map<int, int> degree_map;
      //记得对于undirected图，(u, v)要作为(u, v)和(v, u)插入两次
      // in_degree_map.reserve(2 * edge_vec.size());
      // out_degree_map.reserve(2 * edge_vec.size());
      degree_map.reserve(2 * edge_vec.size());

      for(int i = 0; i < edge_vec.size(); i++){
        int from_node = edge_vec[i].first;
        int to_node = edge_vec[i].second;

        double j = max(g->former_degree[from_node], degree_map[from_node]);

        if(j == 0){
          // break;
          // pi_transpose_map[from_node] += alpha * residue_transpose[from_node];
          backward_residue_entries[from_node] = 0;
          degree_map[from_node] = 1;
          continue;
        }

        degree_map[from_node] = j + 1;

        backward_residue_entries[from_node] += 
        ( 
          (1 - alpha) * backward_pi_entries[to_node] - backward_pi_entries[from_node] 
          - alpha * backward_residue_entries[from_node] + alpha * (from_node == k ? 1 : 0)
        )
        / double(j + 1) / alpha;

        if(backward_residue_entries[from_node] > residuemax && !flags[from_node]){
          // queue_vec.push_back(from_node);
          enqueue(&forback_Q, from_node);
          enqueue(&record_Q, from_node);
          flags[from_node] = true;
        }

      }

      for(int i = 0; i < edge_vec.size(); i++){
        int from_node = edge_vec[i].second;
        int to_node = edge_vec[i].first;

        double j = max(g->former_degree[from_node], degree_map[from_node]);

        if(j == 0){
          // break;
          // pi_transpose_map[from_node] += alpha * residue_transpose[from_node];
          backward_residue_entries[from_node] = 0;
          degree_map[from_node] = 1;
          continue;
        }

        degree_map[from_node] = j + 1;

        backward_residue_entries[from_node] += 
        ( 
          (1 - alpha) * backward_pi_entries[to_node] - backward_pi_entries[from_node] 
          - alpha * backward_residue_entries[from_node] + alpha * (from_node == k ? 1 : 0)
        )
        / double(j + 1) / alpha;

        if(backward_residue_entries[from_node] > residuemax && !flags[from_node]){
          // queue_vec.push_back(from_node);
          enqueue(&forback_Q, from_node);
          enqueue(&record_Q, from_node);
          flags[from_node] = true;
        }

      }

      unordered_map<int, int>().swap(degree_map);




      while(!isEmpty(&forback_Q)){
        
        int backward_v = get_front(&forback_Q);
      


        if(backward_residue_entries[backward_v] > residuemax){
          for(int j = 0; j < g->degree[backward_v]; j++){

            int backward_u = g->AdjList[backward_v][j];
            

            backward_residue_entries[backward_u] += (1-alpha) * backward_residue_entries[backward_v] / g->degree[backward_u];

            enqueue(&record_Q, backward_u);

            if(backward_residue_entries[backward_u] > residuemax && !flags[backward_u]){

              enqueue(&forback_Q, backward_u);
              enqueue(&record_Q, backward_u);

              flags[backward_u] = true;
            }
          }

          backward_pi_entries[backward_v] += alpha * backward_residue_entries[backward_v];
          // all_csr_entries[k][backward_v] += alpha * backward_residue_entries[backward_v] / reservemin;
          all_csr_entries[backward_v] += alpha * backward_residue_entries[backward_v] / reservemin;

        if(all_csr_entries[backward_v] == 0){
          cout<<"0 == all_csr_entries[backward_v] += alpha * backward_residue_entries[backward_v] / reservemin;"<<endl;
        }

          backward_residue_entries[backward_v] = 0;
        }
        
        flags[backward_v] = false;

        forback_Q.front++;
      }





      forback_Q.front = 0;
      forback_Q.rear = 0;

      record_Q.front = 0;
      record_Q.rear = 0;



      // unordered_map<int, int> degree_map;
      //记得对于undirected图，(u, v)要作为(u, v)和(v, u)插入两次
      // in_degree_map.reserve(2 * edge_vec.size());
      // out_degree_map.reserve(2 * edge_vec.size());
      degree_map.reserve(2 * edge_vec.size());

      for(int i = 0; i < edge_vec.size(); i++){
        int from_node = edge_vec[i].first;
        int to_node = edge_vec[i].second;
        
        double j = max(g->former_degree[from_node], degree_map[from_node]);

        if(j == 0){
          // break;
          // pi_map[from_node] += alpha * residue[from_node];
          forward_residue_entries[from_node] = 0;
          degree_map[from_node] = 1;
          continue;
        }

        degree_map[from_node] = j + 1;

        if(forward_pi_entries.find(from_node) == forward_pi_entries.end() || forward_pi_entries[from_node] == 0)
        {
          continue;
        }

        // all_csr_entries[k][from_node] += forward_pi_entries[from_node] * (double)(1.0) / double(j) / reservemin;
        all_csr_entries[from_node] += forward_pi_entries[from_node] * (double)(1.0) / double(j) / reservemin;

        if(all_csr_entries[from_node] == 0){
          cout<<"0 == all_csr_entries[from_node] += forward_pi_entries[from_node] * (double)(1.0) / double(j) / reservemin;"<<endl;
        }

        forward_pi_entries[from_node] *= (double)(j + 1.0) / double(j);

        forward_residue_entries[from_node] -= forward_pi_entries[from_node] / (double)(j+1.0) / alpha;
        forward_residue_entries[to_node] += (1 - alpha) * forward_pi_entries[from_node] / (double)(j+1.0) / alpha;


        if(forward_residue_entries[from_node] > residuemax && !flags[from_node]){
          enqueue(&forback_Q, from_node);
          enqueue(&record_Q, from_node);
          flags[from_node] = true;
        }
        if(forward_residue_entries[to_node] > residuemax && !flags[to_node]){
          enqueue(&forback_Q, to_node);
          enqueue(&record_Q, to_node);
          flags[to_node] = true;
        }
      }




      for(int i = 0; i < edge_vec.size(); i++){
        int from_node = edge_vec[i].second;
        int to_node = edge_vec[i].first;
        
        double j = max(g->former_degree[from_node], degree_map[from_node]);

        if(j == 0){
          // break;
          // pi_map[from_node] += alpha * residue[from_node];
          forward_residue_entries[from_node] = 0;
          degree_map[from_node] = 1;
          continue;
        }

        degree_map[from_node] = j + 1;

        if(forward_pi_entries.find(from_node) == forward_pi_entries.end() || forward_pi_entries[from_node] == 0)
        {
          continue;
        }

        // all_csr_entries[k][from_node] += forward_pi_entries[from_node] * (double)(1.0) / double(j) / reservemin;
        all_csr_entries[from_node] += forward_pi_entries[from_node] * (double)(1.0) / double(j) / reservemin;

        if(all_csr_entries[from_node] == 0){
          cout<<"0 == all_csr_entries[from_node] += forward_pi_entries[from_node] * (double)(1.0) / double(j) / reservemin;"<<endl;
        }

        forward_pi_entries[from_node] *= (double)(j + 1.0) / double(j);

        forward_residue_entries[from_node] -= forward_pi_entries[from_node] / (double)(j+1.0) / alpha;
        forward_residue_entries[to_node] += (1 - alpha) * forward_pi_entries[from_node] / (double)(j+1.0) / alpha;


        if(forward_residue_entries[from_node] > residuemax && !flags[from_node]){
          enqueue(&forback_Q, from_node);
          enqueue(&record_Q, from_node);
          flags[from_node] = true;
        }
        if(forward_residue_entries[to_node] > residuemax && !flags[to_node]){
          enqueue(&forback_Q, to_node);
          enqueue(&record_Q, to_node);
          flags[to_node] = true;
        }
      }

      unordered_map<int, int>().swap(degree_map);



      while(!isEmpty(&forback_Q)){
        int transposed_forward_v = get_front(&forback_Q);
        

        if(forward_residue_entries[transposed_forward_v] / g->degree[transposed_forward_v] > residuemax){

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

            forward_residue_entries[transposed_forward_u] += (1-alpha) * forward_residue_entries[transposed_forward_v] / g->degree[transposed_forward_v];
            enqueue(&record_Q, transposed_forward_u);
            
            
            if(forward_residue_entries[transposed_forward_u] / g->degree[transposed_forward_u] > residuemax && !flags[transposed_forward_u]){
              enqueue(&forback_Q, transposed_forward_u);
              enqueue(&record_Q, transposed_forward_u);
              flags[transposed_forward_u] = true;
            }

          }
          forward_pi_entries[transposed_forward_v] += alpha * forward_residue_entries[transposed_forward_v];

          // all_csr_entries[k][transposed_forward_v] += alpha * backward_residue_entries[transposed_forward_v] / reservemin;
          all_csr_entries[transposed_forward_v] += alpha * forward_residue_entries[transposed_forward_v] / reservemin;

        if(all_csr_entries[transposed_forward_v] == 0){
          cout<<"0 == all_csr_entries[transposed_forward_v] += alpha * forward_residue_entries[transposed_forward_v] / reservemin;"<<endl;
        }


          forward_residue_entries[transposed_forward_v] = 0;
        }
        
        flags[transposed_forward_v] = false;

        forback_Q.front++;
      }




      all_count += all_csr_entries.size();
      

      if(Partition_thread_all_csr_entries[interval_index].size() - 1 == inner_index){
        
        Partition_all_csr_offsets[interval_index] = ofs_all_csr.tellp();
        oa_all_csr << Partition_thread_all_csr_entries[interval_index];

        Partition_dynamic_ppr_offsets[interval_index] = ofs_dynamic_ppr.tellp();
        oa_dynamic_ppr << Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[interval_index];

        for(int i = 0; i < Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[interval_index].size(); i++){
          Partition_thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[interval_index][i].clear();
        }
      }


      // if(k == start){
      //   int i = 0;
      //   for(auto key_val: all_csr_entries){
      //     cout<<"all_csr_entries["<<i<<"].first = "<<key_val.first<<endl;
      //     cout<<"all_csr_entries["<<i<<"].second = "<<key_val.second<<endl;
      //     i++;
      //   }
      // }

      if(k == start){
        cout<<"thread"<<t_minus_1<<": "<<all_csr_entries.size()<<endl;
      }    
      // cout<<"all_count = "<<all_count<<endl;


    }



    // all_csr_offsets = ofs_all_csr.tellp();
    // cout<<"finish all_csr_offsets = ofs_all_csr.tellp();"<<endl;      
    // oa_all_csr << thread_all_csr_entries;

    // dynamic_ppr_offsets = ofs_dynamic_ppr.tellp();
    // oa_dynamic_ppr << thread_forward_pi_forward_residue_backward_pi_backward_residue_entries;





    // for(int k = start; k < end; k++){

    //   unordered_map<long long int, double>& forward_pi_entries = thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[4 * (k - start) + 0];
    //   unordered_map<long long int, double>& forward_residue_entries = thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[4 * (k - start) + 1];
    //   unordered_map<long long int, double>& backward_pi_entries = thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[4 * (k - start) + 2];
    //   unordered_map<long long int, double>& backward_residue_entries = thread_forward_pi_forward_residue_backward_pi_backward_residue_entries[4 * (k - start) + 3];

    //   forward_pi_entries.clear();
    //   forward_residue_entries.clear();

    //   backward_pi_entries.clear();
    //   backward_residue_entries.clear();

    // }

    // cur_part_nnz = cur_nnz;



    cout<<"before close: "<<t_minus_1<<endl;

    ofs_all_csr.close();
    
    ofs_dynamic_ppr.close();


    cout<<"close finished: "<<t_minus_1<<endl;



    cout<<"Undirected_Dynamic_PPR: Finished"<<endl;



}
    // }























// void Directed_Dynamic_PPR(int start, int end, Graph* g, double residuemax, 
// double alpha, 
// int iter,
// int vertex_number,

// double reservemin,


// vector<map<long long int, double>>& all_csr_entries,
// vector<unordered_map<long long int, double>>& forward_pi_entries,
// vector<unordered_map<long long int, double>>& forward_residue_entries,
// vector<unordered_map<long long int, double>>& backward_pi_entries,
// vector<unordered_map<long long int, double>>& backward_residue_entries,


// std::ifstream& ifs_all_csr,
// std::ifstream& ifs_forward_pi,
// std::ifstream& ifs_forward_residue,
// std::ifstream& ifs_backward_pi,
// std::ifstream& ifs_backward_residue,


// std::vector<long long int>& all_csr_offsets,
// std::vector<long long int>& forward_pi_offsets,
// std::vector<long long int>& forward_residue_offsets,
// std::vector<long long int>& backward_pi_offsets,
// std::vector<long long int>& backward_residue_offsets,




// std::ofstream& ofs_all_csr,
// std::ofstream& ofs_forward_pi,
// std::ofstream& ofs_forward_residue,
// // std::ofstream& ofs_forward_hitting_times,
// std::ofstream& ofs_backward_pi,
// std::ofstream& ofs_backward_residue,
// // std::ofstream& ofs_backward_hitting_times,


// long long int& cur_part_nnz,

// long long int& read_in_io_all_count
// )

// {
//     bool *flags = new bool[vertex_number];

//     auto start_io_read_time = chrono::system_clock::now();

//     boost::archive::binary_iarchive ia_all_csr(ifs_all_csr,
//         boost::archive::no_header | boost::archive::no_tracking);

//     boost::archive::binary_iarchive ia_forward_pi(ifs_forward_pi,
//         boost::archive::no_header | boost::archive::no_tracking);

//     boost::archive::binary_iarchive ia_forward_residue(ifs_forward_residue,
//         boost::archive::no_header | boost::archive::no_tracking);

//     boost::archive::binary_iarchive ia_backward_pi(ifs_backward_pi,
//         boost::archive::no_header | boost::archive::no_tracking);

//     boost::archive::binary_iarchive ia_backward_residue(ifs_backward_residue,
//         boost::archive::no_header | boost::archive::no_tracking);





//     //write out
//     boost::archive::binary_oarchive oa_all_csr(ofs_all_csr, 
//       boost::archive::no_header | boost::archive::no_tracking);

//     boost::archive::binary_oarchive oa_forward_pi(ofs_forward_pi, 
//       boost::archive::no_header | boost::archive::no_tracking);
//     boost::archive::binary_oarchive oa_forward_residue(ofs_forward_residue, 
//       boost::archive::no_header | boost::archive::no_tracking);
      
//     boost::archive::binary_oarchive oa_backward_pi(ofs_backward_pi, 
//       boost::archive::no_header | boost::archive::no_tracking);
//     boost::archive::binary_oarchive oa_backward_residue(ofs_backward_residue, 
//       boost::archive::no_header | boost::archive::no_tracking);


//     for(int it = start; it < end; it++){

//         ifs_all_csr.seekg(all_csr_offsets[it]);
//         ifs_forward_pi.seekg(forward_pi_offsets[it]);            
//         ifs_forward_residue.seekg(forward_residue_offsets[it]);
//         ifs_backward_pi.seekg(backward_pi_offsets[it]);            
//         ifs_backward_residue.seekg(backward_residue_offsets[it]);

//         ia_all_csr >> all_csr_entries[it];
//         ia_forward_pi >> forward_pi_entries[it];
//         ia_forward_residue >> forward_residue_entries[it];
//         ia_backward_pi >> backward_pi_entries[it];
//         ia_backward_residue >> backward_residue_entries[it];
//         // std::cout<<"in_map_vec[i].size() = "<<in_map_vec[i].size()<<std::endl;
    
//         // all_count += all_csr_entries[it].size();
    
//         ifs_all_csr.clear();
//         ifs_forward_pi.clear();
//         ifs_forward_residue.clear();
//         ifs_backward_pi.clear();
//         ifs_backward_residue.clear();
    
//     }

//     auto end_io_read_time = chrono::system_clock::now();
//     auto elapsed_read_time = chrono::duration_cast<std::chrono::seconds>(end_io_read_time - start_io_read_time);
//     read_in_io_all_count = elapsed_read_time.count();



//     //Create Queue
//     Queue forback_Q =
//     {
//       //.arr = 
//       (MKL_INT*)malloc( sizeof(MKL_INT) * (INT_MAX-1) ),
//       //.capacity = 
//       // INT_MAX,
//       INT_MAX - 1,
//       //.front =
//       0,
//       //.rear = 
//       0
//     };


//     //The queue to record used elements in this node
//     Queue record_Q =
//     {
//       //.arr = 
//       (MKL_INT*)malloc(sizeof(MKL_INT) * (INT_MAX-1) ),
//       //.capacity = 
//       // INT_MAX,
//       INT_MAX - 1,
//       //.front =
//       0,
//       //.rear = 
//       0
//     };


//     long long int cur_nnz = 0;
//     for(int i = 0; i < vertex_number; i++){

//         // 这里的start到end应该改成关心的节点部分，
//         // 比如改过的边的k-hop内的范围，然后这些节点再分成NUNMTHREAD个
//         //for backward, (from_node, to_node)就是(from_node, to_node)

//         //这里的k就是取nparts之一的去改
//         for(int k = start; k < end; k++){


//           // while(!isEmpty(&record_Q)){
//           //   int current = get_front(&record_Q);
//           //   flags[current] = false;
//           //   // pi[current] = 0;
//           //   // residue[current] = 0;
//           //   dequeue(&record_Q);
//           // }


//           forback_Q.front = 0;
//           forback_Q.rear = 0;

//           record_Q.front = 0;
//           record_Q.rear = 0;


//           //Backward Push
//           int from_node = i;
//           int to_node;
//           for(int j = g->former_degree[from_node]; j < g->degree[from_node]; j++){
//             to_node = g->AdjList[from_node][j];

//             backward_residue_entries[k][from_node] += ( (1 - alpha) 
//               * backward_pi_entries[k][to_node] - backward_pi_entries[k][from_node]
//               - alpha * backward_residue_entries[k][from_node] + alpha * (k == from_node) )
//               / (g->degree[from_node] + 1) / alpha;

//             enqueue(&record_Q, from_node);

//             if(backward_residue_entries[k][from_node] > residuemax && !flags[from_node]){
//               enqueue(&forback_Q, from_node);
//               flags[from_node] = true;
//             }


//             while(!isEmpty(&forback_Q)){
              
//               int backward_v = get_front(&forback_Q);
            


//               if(backward_residue_entries[k][backward_v] > residuemax){
//                 for(int j = 0; j < g->indegree[backward_v]; j++){

//                   int backward_u = g->inAdjList[backward_v][j];
                  

//                   backward_residue_entries[k][backward_u] += (1-alpha) * backward_residue_entries[k][backward_v] / g->outdegree[backward_u];

//                   enqueue(&record_Q, backward_u);

//                   if(backward_residue_entries[k][backward_u] > residuemax && !flags[backward_u]){

//                     enqueue(&forback_Q, backward_u);
//                     enqueue(&record_Q, backward_u);

//                     flags[backward_u] = true;
//                   }
//                 }
//                 backward_pi_entries[k][backward_v] += alpha * backward_residue_entries[k][backward_v];
//                 backward_residue_entries[k][backward_v] = 0;
//               }
              
//               flags[backward_v] = false;

//               forback_Q.front++;
//             }



//             for(int i = 0; i < record_Q.rear; i++){
//               int backward_index = record_Q.arr[i];
//               //times number entered into the queue
//               if(backward_pi_entries[k][backward_index] != 0){ 
//                 all_csr_entries[k][backward_index] += backward_pi_entries[k][backward_index] / reservemin;
//                 // flags[backward_index] = true;
//               }
//               flags[backward_index] = false;
//             }





//                 //Forward transpose Push
//               // }
//               //for forward_transpose, so inserting (from_node, to_node) 
//               // is equal to inserting (to_node, from_node) now
//               //But for the undirected case, these two are the same.
//               // from_node = i;
//               // for(int j = g->former_degree[from_node]; j < g->degree[from_node]; j++){



//               // while(!isEmpty(&record_Q)){
//               //   int current = get_front(&record_Q);
//               //   flags[current] = false;
//               //   dequeue(&record_Q);
//               //   // residue[current] = 0;
//               //   // pi[current] = 0;
//               // }

//               forback_Q.front = 0;
//               forback_Q.rear = 0;

//               record_Q.front = 0;
//               record_Q.rear = 0;

//               if(j == 0){
//                 continue;

//                 // if(forward_residue_entries[k][to_node] > residuemax && !flags[k][to_node]){
//                 //     enqueue(&queue_list[k], to_node);
//                 //     flags[k][to_node] = true;
//                 // }

//               }

//               forward_pi_entries[k][from_node] *= (j + 1) / j;

//               forward_residue_entries[k][from_node] -= forward_pi_entries[k][from_node] / (j+1) / alpha;

//               forward_residue_entries[k][to_node] += (1 - alpha) * forward_pi_entries[k][from_node] / (j+1) / alpha;

//               enqueue(&record_Q, from_node);
//               enqueue(&record_Q, to_node);

//               if(forward_residue_entries[k][from_node] > residuemax && !flags[k][from_node]){
//                 enqueue(&forback_Q, from_node);
//                 flags[k][from_node] = true;
//               }

//               if(forward_residue_entries[k][to_node] > residuemax && !flags[k][to_node]){
//                 enqueue(&forback_Q, to_node);
//                 flags[k][to_node] = true;
//               }




//               while(!isEmpty(&forback_Q)){
//                 int transposed_forward_v = get_front(&forback_Q);
                

//                 if(forward_residue_entries[k][transposed_forward_v] / g->indegree[transposed_forward_v] > residuemax){

//                   for(int j = 0; j < g->indegree[transposed_forward_v]; j++){
//                     int transposed_forward_u = g->inAdjList[transposed_forward_v][j];

//                     // // already add self loop, this can be omitted
//                     // if(g->indegree[transposed_forward_u] == 0){
//                     //   //处理dead end
//                     //   cout<<"Forward indegree[forward_u] == 0"<<endl;
//                     //   transposed_forward_pi[transposed_forward_u] += alpha * transposed_forward_residue[transposed_forward_v];
//                     //   transposed_forward_residue[src] += (1-alpha) * transposed_forward_residue[transposed_forward_v];
//                     //   continue;
//                     // }

//                     forward_residue_entries[k][transposed_forward_u] += (1-alpha) * forward_residue_entries[k][transposed_forward_v] / g->indegree[transposed_forward_v];
//                     enqueue(&record_Q, transposed_forward_u);
                    
                    
//                     if(forward_residue_entries[k][transposed_forward_u] / g->indegree[transposed_forward_u] > residuemax && !flags[transposed_forward_u]){
//                       enqueue(&forback_Q, transposed_forward_u);
//                       enqueue(&record_Q, transposed_forward_u);
//                       flags[transposed_forward_u] = true;
//                     }

//                   }
//                   forward_pi_entries[k][transposed_forward_v] += alpha * forward_residue_entries[k][transposed_forward_v];
//                   forward_residue_entries[k][transposed_forward_v] = 0;
//                 }
                
//                 flags[transposed_forward_v] = false;

//                 forback_Q.front++;
//               }




//               for(int i = 0; i < record_Q.rear; i++){
//                 int transposed_forward_index = record_Q.arr[i];
//                 //times number entered into the queue
//                 if(forward_pi_entries[k][transposed_forward_index] != 0){ 
//                   all_csr_entries[k][transposed_forward_index] += forward_pi_entries[transposed_forward_index] / reservemin;
//                   // flags[transposed_forward_index] = true;
//                 }
//                 flags[transposed_forward_index] = false;
//               }




              
//           }

//           cur_nnz += all_csr_entries[k].size();


//           all_csr_offsets[it] = ofs_all_csr.tellp();
//           oa_all_csr << all_csr_entries[it];


//           forward_pi_offsets[k] = ofs_forward_pi.tellp();
//           forward_residue_offsets[k] = ofs_forward_residue.tellp();
//           oa_forward_pi << forward_pi_entries[k];
//           oa_forward_residue << forward_residue_entries[k];


//           forward_pi_entries[k].clear();
//           forward_residue_entries[k].clear();


//           backward_pi_offsets[k] = ofs_backward_pi.tellp();
//           backward_residue_offsets[k] = ofs_backward_residue.tellp();
//           oa_backward_pi << backward_pi_entries[k];
//           oa_backward_residue << backward_residue_entries[k];

//           backward_pi_entries[k].clear();
//           backward_residue_entries[k].clear();


//         }
//     }


//     cur_part_nnz = cur_nnz;


// }
//     // }






































// void nodegree_DenseDynamicForwardPush(int start, int end, UGraph* g, double residuemax, 
// double alpha, 
// vector<int>& labeled_node_vec, 
// float** residue, 
// float** pi,


// bool** flags, 
// Queue* queue_list,

// unordered_map<int, int> &row_index_mapping,


// int col_dim,

// int nParts,


// d_row_tree_mkl* subset_tree,
// vector<int> &inner_group_mapping,
// vector<int> &indicator

// )
// {

//   int vertices_number = g->n;  

  
//   for(int it = start; it < end; it++){

//     int src = labeled_node_vec[it];

//     while(!isEmpty(&queue_list[it])){

//       int v = get_front(&queue_list[it]);


//       if(g->degree[v] == 0){
//         flags[it][v] = false;


//         pi[it][v] = alpha * residue[it][v];

//         queue_list[it].front++;


//         continue;
//       }



//       if(residue[it][v] > residuemax){
//         for(int j = 0; j < g->degree[v]; j++){
//           int u = g->AdjList[v][j];
//           residue[it][u] += (1-alpha) * residue[it][v] / g->degree[v];


//           if(g->degree[u] == 0){
//             pi[it][u] += alpha * residue[it][u];
//             residue[it][src] += (1-alpha) * residue[it][u];
//             continue;
//           }
          

//           if(residue[it][u] > residuemax && !flags[it][u]){
//             enqueue(&queue_list[it], u);
//             flags[it][u] = true;
//           }

//         }
//         pi[it][v] += alpha * residue[it][v];




//         int col_v = v / col_dim;
        
//         if(col_v == nParts){
//           col_v--;
//         }

//         int inner_col_index = inner_group_mapping[v];


//         subset_tree->dense_mat_mapping[col_v](
//           it, inner_col_index) += alpha * residue[it][v];

//         residue[it][v] = 0;
//       }
      
//       flags[it][v] = false;

//       queue_list[it].front++;
//     }


//     queue_list[it].front = 0;
//     queue_list[it].rear = 0;


//   }

//   return;

// }



























































































// void Log_sparse_matrix_entries_with_norm_computation_LP(
// int i,    
// double reservemin, 

// d_row_tree_mkl* subset_tree,
// unordered_map<int, vector<int>> &vec_mapping,

// vector<int>& update_mat_tree_record,
// int iter,
// double delta,
// int count_labeled_node,
// int d,
// vector<long long int>& record_submatrices_nnz
// ){


    
//     SparseMatrix<double, 0, int> &old_mat_mapping = subset_tree->mat_mapping[i];

//     int temp_row_dim = subset_tree->row_dim;
//     int current_group_size = vec_mapping[i].size();



//     SparseMatrix<double, 0, int> current_mat_mapping;


//     current_mat_mapping.resize(temp_row_dim, current_group_size);
//     current_mat_mapping = subset_tree->dense_mat_mapping[i].sparseView();


//     long long int temp_record_submatrices_nnz = 0;

//     for (int k_iter=0; k_iter<current_mat_mapping.outerSize(); ++k_iter){
//         for (SparseMatrix<double, ColMajor, int>::InnerIterator it(current_mat_mapping, k_iter); it; ++it){
//             if(it.value() > reservemin){
//                 it.valueRef() = log10(1 + it.value()/reservemin);

//                 temp_record_submatrices_nnz++;
//             }
//             else if(it.value() == 0){

//             }
//             else{
//                 it.valueRef() = log10(1 + it.value()/reservemin);
//                 temp_record_submatrices_nnz++;
//             }

//         }
//     }



//     double A_norm = current_mat_mapping.norm();



//     double Ei_norm = (current_mat_mapping - old_mat_mapping).norm();



//     delta = delta * sqrt(2);



//     if( subset_tree->norm_B_Bid_difference_vec[i] + Ei_norm < delta * A_norm){
//       update_mat_tree_record[i] = -1;
//       current_mat_mapping.resize(0, 0);
//       current_mat_mapping.data().squeeze();

//     }
//     else{
//       update_mat_tree_record[i] = iter;
//       old_mat_mapping.resize(0, 0);
//       old_mat_mapping.data().squeeze();
//       subset_tree->mat_mapping[i] = current_mat_mapping;

//       record_submatrices_nnz[i] = temp_record_submatrices_nnz;

//     }




// }



































































































// void Log_sparse_matrix_entries_LP(
// int i,    
// double reservemin, 
// d_row_tree_mkl* subset_tree,
// unordered_map<int, vector<int>> &vec_mapping,
// vector<long long int>& record_submatrices_nnz
// ){

//     SparseMatrix<double, 0, int> &current_mat_mapping = subset_tree->mat_mapping[i];

//     current_mat_mapping.resize(0, 0);




//     record_submatrices_nnz[i] = 0;




//     int temp_row_dim = subset_tree->row_dim;
//     int current_group_size = vec_mapping[i].size();
//     current_mat_mapping.resize(temp_row_dim, current_group_size);
//     current_mat_mapping = subset_tree->dense_mat_mapping[i].sparseView();
//     for (int k_iter=0; k_iter<subset_tree->mat_mapping[i].outerSize(); ++k_iter){
//         for (SparseMatrix<double, ColMajor, int>::InnerIterator it(subset_tree->mat_mapping[i], k_iter); it; ++it){
//             if(it.value() > reservemin){
//                 it.valueRef() = log10(1 + it.value()/reservemin);
//                 record_submatrices_nnz[i]++;
//             }
//             else{

//                 it.valueRef() = log10(1 + it.value()/reservemin);
//                 record_submatrices_nnz[i]++;
//             }
            

//         }
//     }

// }











































































// void sparse_sub_svd_function_with_norm_computation(int d, int pass, 
// int update_j, 
// vector<long long int>& record_submatrices_nnz,
// d_row_tree_mkl* subset_tree,
// int largest_level_start_index,
// int current_out_iter,
// int lazy_update_start_iter)
// {


//   Eigen::SparseMatrix<double, 0, int> &submatrix = subset_tree->mat_mapping[update_j];


//   mat* matrix_vec_t = subset_tree->hierarchy_matrix_vec[largest_level_start_index + update_j];


//   SparseMatrix<double, RowMajor, long> ppr_matrix_temp(submatrix.rows(), submatrix.cols());


//   ppr_matrix_temp = submatrix;


//   long long int nnz = record_submatrices_nnz[update_j];


//   assert(nnz < INT_MAX);
//   auto hash_coo_time = chrono::system_clock::now();


//   mat_coo *ppr_matrix_coo = coo_matrix_new(submatrix.rows(), submatrix.cols(), nnz);
//   ppr_matrix_coo->nnz = nnz;

//   long nnz_iter=0;
//   double ppr_norm =0;

//   for (int k=0; k<ppr_matrix_temp.outerSize(); ++k){
//     for (SparseMatrix<double, RowMajor, long int>::InnerIterator it(ppr_matrix_temp, k); it; ++it){
//       double value1 = it.value();
//       if(value1 == 0){

//       }
//       else{
//         ppr_matrix_coo->rows[nnz_iter] = it.row() + 1;
//         ppr_matrix_coo->cols[nnz_iter] = it.col() + 1;
//         ppr_matrix_coo->values[nnz_iter] = value1;
//         ppr_norm += ppr_matrix_coo->values[nnz_iter]*ppr_matrix_coo->values[nnz_iter];
//         nnz_iter ++;
//       }
//     }
//   }

//   auto coo_csr_time = chrono::system_clock::now();
//   auto elapsed_sparse_coo_time = chrono::duration_cast<std::chrono::seconds>(coo_csr_time- hash_coo_time);

//   mat_csr* ppr_matrix = csr_matrix_new();
//   csr_init_from_coo(ppr_matrix, ppr_matrix_coo);


//   coo_matrix_delete(ppr_matrix_coo);
//   ppr_matrix_coo = NULL;

//   mat *U = matrix_new(submatrix.rows(), d);
//   mat *S = matrix_new(d, 1);

//   mat *V = matrix_new(submatrix.cols(), d);

//   frPCA(ppr_matrix, &U, &S, &V, d, pass);

//   mat * S_full = matrix_new(d, d);
//   for(int i = 0; i < d; i++){
//     matrix_set_element(S_full, i, i, matrix_get_element(S, i, 0));
//   }


//   matrix_matrix_mult(U, S_full, matrix_vec_t);

//   if(current_out_iter >= lazy_update_start_iter - 1){
//     auto norm_start_time = chrono::system_clock::now();

//     mat * V_transpose_matrix = matrix_new(d, submatrix.cols());

//     matrix_build_transpose(V_transpose_matrix, V);

//     mat * final_matrix_shape_for_frobenius = matrix_new(submatrix.rows(), submatrix.cols());


//     matrix_matrix_mult(matrix_vec_t, V_transpose_matrix, final_matrix_shape_for_frobenius);


//     matrix_delete(V_transpose_matrix);

//     V_transpose_matrix = NULL;


//     for (int k=0; k<ppr_matrix_temp.outerSize(); ++k){
//       for (SparseMatrix<double, RowMajor, long int>::InnerIterator it(ppr_matrix_temp, k); it; ++it){
//         double value1 = it.value();
//         if(value1 == 0){

//         }
//         else{
//           double XY_value = matrix_get_element(final_matrix_shape_for_frobenius, it.row(), it.col());
//           matrix_set_element(final_matrix_shape_for_frobenius, it.row(), it.col(), XY_value - value1);
//         }
//       }
//     }



//     subset_tree->norm_B_Bid_difference_vec[update_j] = get_matrix_frobenius_norm(final_matrix_shape_for_frobenius);

//     matrix_delete(final_matrix_shape_for_frobenius);

//     final_matrix_shape_for_frobenius = NULL;


//     auto norm_end_time = chrono::system_clock::now();
//     auto elapsed_norm_time = chrono::duration_cast<std::chrono::seconds>(norm_end_time - norm_start_time);
//   }


//   ppr_matrix_temp.resize(0,0);
//   ppr_matrix_temp.data().squeeze();




//   matrix_delete(U);
//   matrix_delete(S);
//   matrix_delete(V);
//   matrix_delete(S_full);
//   U = NULL;
//   S = NULL;
//   V = NULL;
//   S_full = NULL;
  
//   csr_matrix_delete(ppr_matrix);

//   ppr_matrix = NULL;

// }






























































