#include <iostream>
#include <random>
#include <vector>
#include <sycl/sycl.hpp>
#include <oneapi/mkl/spblas.hpp>

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
struct CSCMatrix {
  std::vector<T> values;
  std::vector<int> row_indices;
  std::vector<int> column_pointers;
};

template<typename T>
struct SYCL_CSRMatrix {
  PrefetchedBuffer<T, 1> values;
  PrefetchedBuffer<int, 1> column_indices;
  PrefetchedBuffer<int, 1> row_pointers;
};

template<typename T>
struct SYCL_CSCMatrix {
  PrefetchedBuffer<T, 1> values;
  PrefetchedBuffer<int, 1> row_indices;
  PrefetchedBuffer<int, 1> column_pointers;
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

template<typename T>
void createRandomSparseCSC(size_t numRows, size_t numCols, float sparsity, CSCMatrix<T>& csc) {
  csc.values.clear();
  csc.row_indices.clear();
  csc.column_pointers.assign(numCols + 1, 0);

  int nnz = 0;

  for (int j = 0; j < numCols; j++) {
    for (int i = 0; i < numRows; ++i) {
      if (rand() / (RAND_MAX + 1.0) > sparsity) {
        csc.values.push_back(1);
        csc.row_indices.push_back(i);
        nnz++;
      }
    }
    csc.column_pointers[j + 1] = nnz;
  }
}

template <class T, unsigned int sparsity, size_t sg_size>
class SpGEMM {
protected:
  size_t num_iters;

  CSRMatrix<T> csrA;
  CSCMatrix<T> cscB;
  std::vector<T> c;

  SYCL_CSRMatrix<T> sycl_csrA;
  SYCL_CSCMatrix<T> sycl_cscB;
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
    createRandomSparseCSC(size, size, sparsity / 100.0f, cscB);
    c.resize(size * size);

    sycl_csrA.values.initialize(args.device_queue, csrA.values.data(), s::range{csrA.values.size()});
    sycl_csrA.column_indices.initialize(args.device_queue, csrA.column_indices.data(), s::range{csrA.column_indices.size()});
    sycl_csrA.row_pointers.initialize(args.device_queue, csrA.row_pointers.data(), s::range{csrA.row_pointers.size()});

    sycl_cscB.values.initialize(args.device_queue, cscB.values.data(), s::range{cscB.values.size()});
    sycl_cscB.column_pointers.initialize(args.device_queue, cscB.column_pointers.data(), s::range{cscB.column_pointers.size()});
    sycl_cscB.row_indices.initialize(args.device_queue, cscB.row_indices.data(), s::range{cscB.row_indices.size()});

    c_buf.initialize(args.device_queue, c.data(), s::range{c.size()});
  }

  void run(std::vector<s::event>& events) {
    events.push_back(args.device_queue.submit([&](s::handler& cgh) {
      auto csr_values = sycl_csrA.values.template get_access<s::access_mode::read>(cgh);
      auto csr_column_indices = sycl_csrA.column_indices.template get_access<s::access_mode::read>(cgh);
      auto csr_row_pointers = sycl_csrA.row_pointers.template get_access<s::access_mode::read>(cgh);

      auto csc_values = sycl_cscB.values.template get_access<s::access_mode::read>(cgh);
      auto csc_column_pointers = sycl_cscB.column_pointers.template get_access<s::access_mode::read>(cgh);
      auto csc_row_indices = sycl_cscB.row_indices.template get_access<s::access_mode::read>(cgh);

      auto valuesC = c_buf.template get_access<s::access_mode::discard_write>(cgh);

      cgh.parallel_for<SpGEMMKernel<T, sparsity, sg_size>>(sycl::range<2>({size, size}), [=, size=size](sycl::item<2> item) [[intel::reqd_sub_group_size(sg_size)]] {
        int rowC = item.get_id(0);
        int colC = item.get_id(1);

        T sum = 0;

        for (int k = 0; k < size; ++k) {
          int rowA = csc_row_indices[k];

          if (rowC == rowA) {
            int colB = csc_column_pointers[k];
            int colBEnd = csc_column_pointers[k + 1];

            for (int j = colB; j < colBEnd; ++j) {
              int colA = csr_column_indices[j];
              T valA = csr_values[j];
              T valB = csc_values[k];
              sum += valA * valB;
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

        for (int k = 0; k < size; ++k) {
          int rowA = cscB.row_indices[k];

          if (row == rowA) {
            int colB = cscB.column_pointers[k];
            int colBEnd = cscB.column_pointers[k + 1];

            for (int j = colB; j < colBEnd; ++j) {
              int colA = csrA.column_indices[j];
              T valA = csrA.values[j];
              T valB = cscB.values[k];
              sum += valA * valB;
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
    app.run<SpGEMM<float, sparsity, sg_size>>();
    if (app.deviceSupportsFP64()) {
      app.run<SpGEMM<double, sparsity, sg_size>>();
    }
  }
}

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  run_helper<20, 8>(app);
  run_helper<20, 16>(app);
  run_helper<20, 32>(app);
  run_helper<50, 8>(app);
  run_helper<50, 16>(app);
  run_helper<50, 32>(app);
  run_helper<80, 8>(app);
  run_helper<80, 16>(app);
  run_helper<80, 32>(app);
  
  return 0;
}
