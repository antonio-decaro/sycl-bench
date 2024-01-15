#include <iostream>
#include <sycl/sycl.hpp>

#include "common.h"


namespace s = sycl;
class MatrixMulAccKernel; // kernel forward declaration

template <class T, size_t sg_size>
class MatrixMulAcc {
protected:
  size_t num_iters;
  std::vector<T> a;
  std::vector<T> b;
  std::vector<T> c;

  PrefetchedBuffer<T, 1> a_buf;
  PrefetchedBuffer<T, 1> b_buf;
  PrefetchedBuffer<T, 1> c_buf;

  size_t size;
  BenchmarkArgs args;

public:
  MatrixMulAcc(BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size = args.problem_size; // input size defined by the user
    a.resize(size * size);
    b.resize(size * size);
    c.resize(size * size);

    std::fill(a.begin(), a.end(), 1);
    std::fill(b.begin(), b.end(), 1);
    std::fill(c.begin(), c.end(), 0);

    a_buf.initialize(args.device_queue, a.data(), s::range<1>{size * size});
    b_buf.initialize(args.device_queue, b.data(), s::range<1>{size * size});
    c_buf.initialize(args.device_queue, c.data(), s::range<1>{size * size});
  }

  void run(std::vector<sycl::event>& events) {
    const size_t local_size = args.local_size;
    events.push_back(args.device_queue.submit([&](s::handler& cgh) {
      auto in_A = a_buf.template get_access<s::access_mode::read>(cgh);
      auto in_B = b_buf.template get_access<s::access_mode::read>(cgh);
      auto out = c_buf.template get_access<s::access_mode::read_write>(cgh);
      cgh.parallel_for(s::nd_range<2>{{size, size}, {1, local_size}}, [=, num_iters=args.num_iters, size=this->size](s::nd_item<2> gid) [[intel::reqd_sub_group_size(sg_size)]] {
        int gidx = gid.get_global_id(0);
        int gidy = gid.get_global_id(1);
        for(int iter = 0; iter < num_iters; iter++) {
          out[gidx * size + gidy] = 0;
          for(int k = 0; k < size; k++) {
            out[gidx * size + gidy] += in_A[gidx * size + k] * in_B[k * size + gidy];
          }
        }
      }); // end parallel for
    })); // end events.push back
  }


  bool verify(VerificationSetting& ver) {
    c_buf.reset();
    for(int i = 0; i < size * size; i++)
      if(size != c[i])
        return false;


    return true;
  }

  static std::string getBenchmarkName() { 
    std::stringstream name;
    name << "MatrixMul_";
    name << ReadableTypename<T>::name;
    name << "_sg";
    name << sg_size;
    return name.str();
  }
};


int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  if (app.deviceSupportsSG(8)) {
    app.run<MatrixMulAcc<int, 8>>();
    app.run<MatrixMulAcc<long long, 8>>();
    app.run<MatrixMulAcc<float, 8>>();
    if (app.deviceSupportsFP64()) {
      app.run<MatrixMulAcc<double, 8>>();
    }
  }

  if (app.deviceSupportsSG(16)) {
    app.run<MatrixMulAcc<int, 16>>();
    app.run<MatrixMulAcc<long long, 16>>();
    app.run<MatrixMulAcc<float, 16>>();
    if (app.deviceSupportsFP64()) {
      app.run<MatrixMulAcc<double, 16>>();
    }
  }

  if (app.deviceSupportsSG(32)) {
    app.run<MatrixMulAcc<int, 32>>();
    app.run<MatrixMulAcc<long long, 32>>();
    app.run<MatrixMulAcc<float, 32>>();
    if (app.deviceSupportsFP64()) {
      app.run<MatrixMulAcc<double, 32>>();
    }
  }
  
  return 0;
}
