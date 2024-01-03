#include <iostream>
#include <random>
#include <vector>
#include <sycl/sycl.hpp>

#include "common.h"

namespace s = sycl;

template<typename T,  unsigned int sparsity, size_t sg_size>
class SpGEMMKernel; // kernel forward declaration

template<typename T>
struct CSRMatrix {
  std::vector<T> values;
  std::vector<int> column_indices;
  std::vector<int> row_pointers;
};

template<typename T>
struct SYCL_CSRMatrix {
  PrefetchedBuffer<T, 1> values;
  PrefetchedBuffer<int, 1> column_indices;
  PrefetchedBuffer<int, 1> row_pointers;
};

template<typename T>
void createRandomSparseCSR(size_t numRows, size_t numCols, float sparsity, CSRMatrix<T>& csr) {
  csr.values.clear();
  csr.column_indices.clear();
  csr.row_pointers.assign(numRows + 1, 0);

  int nnz = 0;

  for (int i = 0; i < numRows; i++) {
    for (int j = 0; j < numCols; ++j) {
      if (rand() / (RAND_MAX + 1.0) > sparsity) {
        csr.values.push_back(1);
        csr.column_indices.push_back(j);
        nnz++;
      }
    }
    csr.row_pointers[i + 1] = nnz;
  }
}

template <class T, unsigned int sparsity, size_t sg_size>
class SpGEMM {
protected:
  size_t num_iters;

  CSRMatrix<T> csrA;
  CSRMatrix<T> csrB;
  std::vector<T> c;

  SYCL_CSRMatrix<T> sycl_csrA;
  SYCL_CSRMatrix<T> sycl_csrB;
  PrefetchedBuffer<T, 1> c_buf;

  size_t size;
  BenchmarkArgs args;

public:
  SpGEMM(BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size = args.problem_size; // input size defined by the user
    auto seed = args.cli.get<unsigned int>("--seed");
    assert (sparsity >= 0 && sparsity <= 100);

    srand(seed);

    createRandomSparseCSR(size, size, sparsity / 100.0f, csrA);
    createRandomSparseCSR(size, size, sparsity / 100.0f, csrB);
    c.resize(size * size);

    sycl_csrA.values.initialize(args.device_queue, csrA.values.data(), s::range{csrA.values.size()});
    sycl_csrA.column_indices.initialize(args.device_queue, csrA.column_indices.data(), s::range{csrA.column_indices.size()});
    sycl_csrA.row_pointers.initialize(args.device_queue, csrA.row_pointers.data(), s::range{csrA.row_pointers.size()});

    sycl_csrB.values.initialize(args.device_queue, csrB.values.data(), s::range{csrB.values.size()});
    sycl_csrB.column_indices.initialize(args.device_queue, csrB.column_indices.data(), s::range{csrB.column_indices.size()});
    sycl_csrB.row_pointers.initialize(args.device_queue, csrB.row_pointers.data(), s::range{csrB.row_pointers.size()});

    c_buf.initialize(args.device_queue, c.data(), s::range{c.size()});
  }

  void run(std::vector<s::event>& events) {
    events.push_back(args.device_queue.submit([&](s::handler& cgh) {
      auto valuesA = sycl_csrA.values.template get_access<s::access_mode::read>(cgh);
      auto col_indicesA = sycl_csrA.column_indices.template get_access<s::access_mode::read>(cgh);
      auto row_pointersA = sycl_csrA.row_pointers.template get_access<s::access_mode::read>(cgh);

      auto valuesB = sycl_csrB.values.template get_access<s::access_mode::read>(cgh);
      auto col_indicesB = sycl_csrB.column_indices.template get_access<s::access_mode::read>(cgh);
      auto row_pointersB = sycl_csrB.row_pointers.template get_access<s::access_mode::read>(cgh);

      auto valuesC = c_buf.template get_access<s::access_mode::discard_write>(cgh);

      cgh.parallel_for<SpGEMMKernel<T, sparsity, sg_size>>(sycl::range<2>({size, size}), [=](sycl::item<2> item) [[intel::reqd_sub_group_size(sg_size)]] {
        int row = item.get_id(0);
        int col = item.get_id(1);

        T sum = 0;

        int row_startA = row_pointersA[row];
        int row_endA = row_pointersA[row + 1];

        for (int k = row_startA; k < row_endA; ++k) {
          int colA = col_indicesA[k];
          T valA = valuesA[k];

          // Find the corresponding row in B
          int row_startB = row_pointersB[colA];
          int row_endB = row_pointersB[colA + 1];

          for (int l = row_startB; l < row_endB; ++l) {
            int colB = col_indicesB[l];
            T valB = valuesB[l];

            if (colB == col) {
              sum += valA * valB;
              break; // No need to continue searching in the same column
            }
          }
        }

        valuesC[item.get_linear_id()] = sum;
      });
    }));
  }


  bool verify(VerificationSetting& ver) {
    c_buf.reset();
    if (!ver.enabled) {
      return true;
    }
    for (int row = 0; row < size; row++) {
      for (int col = 0; col < size; col++) {
        T sum = 0;

        int row_startA = csrA.row_pointers[row];
        int row_endA = csrA.row_pointers[row + 1];

        for (int k = row_startA; k < row_endA; ++k) {
          int colA = csrA.column_indices[k];
          T valA = csrA.values[k];

          // Find the corresponding row in B
          int row_startB = csrB.row_pointers[colA];
          int row_endB = csrB.row_pointers[colA + 1];

          for (int l = row_startB; l < row_endB; ++l) {
            int colB = csrB.column_indices[l];
            T valB = csrB.values[l];

            if (colB == col) {
              sum += valA * valB;
              break; // No need to continue searching in the same column
            }
          }
        }

        if (c[row * size + col] != sum) {
          return false;
        }
      }
    }

    return true;
  }

  static std::string getBenchmarkName() { 
    std::stringstream name;
    name << "SpGEMM";
    name << "_sp" << sparsity;
    name << "_" << ReadableTypename<T>::name;
    name << "_sg" << sg_size;
    return name.str();
  }
};

template<unsigned int sparsity, size_t sg_size>
void run_helper(BenchmarkApp& app) {
  if (app.deviceSupportsSG(sg_size)) {
    app.run<SpGEMM<int, sparsity, sg_size>>();
    app.run<SpGEMM<long long, sparsity, sg_size>>();
    app.run<SpGEMM<float, sparsity, sg_size>>();
    if (app.deviceSupportsFP64()) {
      app.run<SpGEMM<double, sparsity, sg_size>>();
    }
  }
}

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  run_helper<25, 8>(app);
  run_helper<25, 16>(app);
  run_helper<25, 32>(app);
  run_helper<50, 8>(app);
  run_helper<50, 16>(app);
  run_helper<50, 32>(app);
  run_helper<75, 8>(app);
  run_helper<75, 16>(app);
  run_helper<75, 32>(app);
  
  return 0;
}
