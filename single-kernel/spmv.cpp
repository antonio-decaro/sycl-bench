#include <iostream>
#include <random>
#include <vector>
#include <sycl/sycl.hpp>

#include "common.h"

namespace s = sycl;

template<typename T,  unsigned int sparsity, size_t sg_size>
class SpMVKernel; // kernel forward declaration

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
class SpMV {
protected:
  size_t num_iters;

  CSRMatrix<T> csrA;
  std::vector<T> b;
  std::vector<T> c;

  SYCL_CSRMatrix<T> sycl_csrA;
  PrefetchedBuffer<T, 1> b_buf;
  PrefetchedBuffer<T, 1> c_buf;

  size_t size;
  BenchmarkArgs args;

public:
  SpMV(BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size = args.problem_size; // input size defined by the user
    auto seed = args.cli.get<unsigned int>("--seed");
    assert (sparsity >= 0 && sparsity <= 100);

    srand(seed);

    createRandomSparseCSR(size, size, sparsity / 100.0f, csrA);

    b.resize(size);
    std::fill(b.begin(), b.end(), 1);
    c.resize(size);

    sycl_csrA.values.initialize(args.device_queue, csrA.values.data(), s::range{csrA.values.size()});
    sycl_csrA.column_indices.initialize(args.device_queue, csrA.column_indices.data(), s::range{csrA.column_indices.size()});
    sycl_csrA.row_pointers.initialize(args.device_queue, csrA.row_pointers.data(), s::range{csrA.row_pointers.size()});

    b_buf.initialize(args.device_queue, b.data(), s::range{b.size()});
    c_buf.initialize(args.device_queue, c.data(), s::range{c.size()});
  }

  void run(std::vector<s::event>& events) {
    events.push_back(args.device_queue.submit([&](s::handler& cgh) {
      auto valuesA = sycl_csrA.values.template get_access<s::access_mode::read>(cgh);
      auto col_indices = sycl_csrA.column_indices.template get_access<s::access_mode::read>(cgh);
      auto row_pointers = sycl_csrA.row_pointers.template get_access<s::access_mode::read>(cgh);

      auto valuesB = b_buf.template get_access<s::access_mode::read>(cgh);
      auto valuesC = c_buf.template get_access<s::access_mode::discard_write>(cgh);

      cgh.parallel_for<SpMVKernel<T, sparsity, sg_size>>(sycl::range<1>(size), [=, size=size](sycl::item<1> item) [[intel::reqd_sub_group_size(sg_size)]] {
        int row = item[0];
        T dot_product = 0;
        for (int j = row_pointers[row]; j < row_pointers[row + 1]; j++) {
          int col = col_indices[j];
          dot_product += valuesA[col] * valuesB[col];
        }
        valuesC[row] = dot_product;
      });
    }));
  }


  bool verify(VerificationSetting& ver) {
    c_buf.reset();
    if (!ver.enabled) {
      return true;
    }
    for (int row = 0; row < size; row++) {
      T dot_product = 0;
      for (int j = csrA.row_pointers[row]; j < csrA.row_pointers[row + 1]; j++) {
        int col = csrA.column_indices[j];
        dot_product += csrA.values[col] * b[col];
      }
      if (c[row] != dot_product) {
        return false;
      }
    }

    return true;
  }

  static std::string getBenchmarkName() { 
    std::stringstream name;
    name << "SpMV";
    name << "_" << ReadableTypename<T>::name;
    name << "_sg" << sg_size;
    return name.str();
  }
};

template<unsigned int sparsity, size_t sg_size>
void run_helper(BenchmarkApp& app) {
  if (app.deviceSupportsSG(sg_size)) {
    app.run<SpMV<int, sparsity, sg_size>>();
    app.run<SpMV<long long, sparsity, sg_size>>();
    app.run<SpMV<float, sparsity, sg_size>>();
    if (app.deviceSupportsFP64()) {
      app.run<SpMV<double, sparsity, sg_size>>();
    }
  }
}

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  run_helper<70, 8>(app);
  run_helper<70, 16>(app);
  run_helper<70, 32>(app);
  
  return 0;
}
