#include <iostream>
#include <sycl/sycl.hpp>
#include <queue>
#include <set>
#include "common.h"

struct CSR {
  std::vector<size_t> row_offsets;
  std::vector<size_t> col_indices;
};

CSR read_CSR_from_file(std::string fname) {
  std::ifstream f(fname);
  if (!f.is_open()) {
    std::cerr << "Error opening file " << fname << std::endl;
    exit(1);
  }
  std::string line;
  std::vector<size_t> row_offsets;
  std::vector<size_t> col_indices;
  std::getline(f, line);
  std::istringstream iss(line);
  size_t val;
  while (iss >> val)
    row_offsets.push_back(val);
  std::getline(f, line);
  iss = std::istringstream(line);
  while (iss >> val)
    col_indices.push_back(val);


  return {row_offsets, col_indices};
}

template<size_t sg_size>
class BFSKernel; // kernel forward declaration

template <size_t sg_size>
class BFS {
protected:
  size_t local_size;
  size_t global_size;

  std::vector<size_t> row_offsets;
  std::vector<size_t> col_indices;
  std::vector<int> parents;

  PrefetchedBuffer<size_t, 1> row_offsets_buf;
  PrefetchedBuffer<size_t, 1> col_indices_buf;
  PrefetchedBuffer<int, 1> parents_buf;

  BenchmarkArgs args;

public:
  BFS(BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    std::string fname = args.cli.get<std::string>("--path");
    local_size = args.local_size;
    global_size = args.problem_size;

    auto csr = read_CSR_from_file(fname);
    row_offsets = csr.row_offsets;
    col_indices = csr.col_indices;

    parents.resize(row_offsets.size() - 1);
    std::fill(parents.begin(), parents.end(), -1);

    row_offsets_buf.initialize(args.device_queue, row_offsets.data(), row_offsets.size());
    col_indices_buf.initialize(args.device_queue, col_indices.data(), col_indices.size());
    parents_buf.initialize(args.device_queue, parents.data(), parents.size());
  }

  void run(std::vector<sycl::event>& events) {
    events.push_back(args.device_queue.submit([&](sycl::handler& cgh) {
      const size_t num_nodes = row_offsets.size() - 1;
      auto row_offsets_acc = row_offsets_buf.template get_access<sycl::access::mode::read>(cgh);
      auto col_indices_acc = col_indices_buf.template get_access<sycl::access::mode::read>(cgh);
      auto parents_acc = parents_buf.template get_access<sycl::access::mode::read_write>(cgh);

      sycl::range<1> global_range {local_size}; // we want only one workgroup in order to synchronize with barriers
      sycl::range<1> local_range {local_size};
      sycl::nd_range<1> ndrange {global_range, local_range};

      using mask_t = uint32_t;
      const size_t MASK_SIZE = sizeof(mask_t) * 8;
      const size_t NUM_MASKS = num_nodes / MASK_SIZE + 1; // the number of masks needed to represent all nodes
      sycl::local_accessor<mask_t, 1> frontier{sycl::range<1>{NUM_MASKS}, cgh};
      sycl::local_accessor<mask_t, 1> next{sycl::range<1>{NUM_MASKS}, cgh};
      sycl::local_accessor<mask_t, 1> running{sycl::range<1>{1}, cgh};

      cgh.parallel_for<BFSKernel<sg_size>>(ndrange, [=, num_nodes=num_nodes](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(sg_size)]] {
        sycl::atomic_ref<mask_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space> running_ar{running[0]};
        auto local_size = item.get_local_range(0);
        auto wg = item.get_group();
        auto id = item.get_global_id(0);

        if (id == 0) {
          running_ar.store(1);
          frontier[0] = next[0] = 1;
          parents_acc[0] = 0;
        }

        sycl::group_barrier(wg);
        while (running_ar.load()) {
          for (size_t i = id; i < NUM_MASKS; i += local_size) {
            frontier[i] = next[i];
            next[i] = 0;
          }
          sycl::group_barrier(wg);

          for (size_t node_id = id; node_id < num_nodes; node_id += local_size) {
            int node_mask_offet = node_id / MASK_SIZE; // to access the right mask
            mask_t node_bit = 1 << (node_id % MASK_SIZE); // to access the right bit in the mask 
            sycl::atomic_ref<mask_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space> next_ar{next[node_mask_offet]};

            if (parents_acc[node_id] == -1) {
              for (int i = row_offsets_acc[node_id]; i < row_offsets_acc[node_id + 1]; i++) {
                size_t neighbor = col_indices_acc[i];
                int neighbor_mask_offset = neighbor / MASK_SIZE;
                mask_t neighbor_bit = 1 << (neighbor % MASK_SIZE);

                if (frontier[neighbor_mask_offset] & neighbor_bit) {
                  parents_acc[node_id] = neighbor;
                  next_ar |= node_bit;
                  break;
                }
              }
            }
          }
          if (id == 0) running_ar.store(0);
          sycl::group_barrier(wg);
          for (size_t i = id; i < NUM_MASKS; i += local_size) {
            running_ar += next[i];
          }
          sycl::group_barrier(wg);
        }       
        
      }); // end parallel_for
    })); // end events.push back
  } // end run


  bool verify(VerificationSetting& ver) {
    parents_buf.reset();
    if (parents[0] != 0) {
      return false;
    }
    for (int i = 1; i < parents.size(); i++) {
      if (parents[i] == -1) {
        return false;
      }
      auto parent = parents[i];
      bool found = false;
      for (int j = row_offsets[parent]; j < row_offsets[parent + 1]; j++) {
        if (col_indices[j] == i) {
          found = true;
          break;
        }
      }
      if (!found) {
        return false;
      }
    }
    return true;
  }

  static std::string getBenchmarkName() { 
    std::stringstream name;
    name << "BFS";
    name << "_sg";
    name << sg_size;
    return name.str();
  }
};


int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  if (app.deviceSupportsSG(8)) {
    app.run<BFS<8>>();
  }

  if (app.deviceSupportsSG(16)) {
    app.run<BFS<16>>();   
  }

  if (app.deviceSupportsSG(32)) {
    app.run<BFS<32>>();
  }
  
  return 0;
}
