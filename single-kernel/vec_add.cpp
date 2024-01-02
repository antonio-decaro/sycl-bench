#include "common.h"

#include <iostream>

// Opening sycl namespace is unsupported on hipSYCL
// (mainly due to CUDA/HIP design issues), better
// avoid it
// using namespace sycl;
namespace s = sycl;
template <typename T, size_t sg_size>
class VecAddKernel;

template <typename T, size_t sg_size>
class VecAddBench {
protected:
  std::vector<T> input1;
  std::vector<T> input2;
  std::vector<T> output;
  BenchmarkArgs args;

  PrefetchedBuffer<T, 1> input1_buf;
  PrefetchedBuffer<T, 1> input2_buf;
  PrefetchedBuffer<T, 1> output_buf;

public:
  VecAddBench(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    // host memory intilization
    input1.resize(args.problem_size);
    input2.resize(args.problem_size);
    output.resize(args.problem_size);

    for(size_t i = 0; i < args.problem_size; i++) {
      input1[i] = static_cast<T>(i);
      input2[i] = static_cast<T>(i);
      output[i] = static_cast<T>(0);
    }

    input1_buf.initialize(args.device_queue, input1.data(), s::range<1>(args.problem_size));
    input2_buf.initialize(args.device_queue, input2.data(), s::range<1>(args.problem_size));
    output_buf.initialize(args.device_queue, output.data(), s::range<1>(args.problem_size));
  }

  void run(std::vector<sycl::event>& events) {
    events.push_back(args.device_queue.submit([&](sycl::handler& cgh) {
      auto in1 = input1_buf.template get_access<s::access::mode::read>(cgh);
      auto in2 = input2_buf.template get_access<s::access::mode::read>(cgh);
      // Use discard_write here, otherwise the content of the host buffer must first be copied to device
      auto out = output_buf.template get_access<s::access::mode::discard_write>(cgh);
      sycl::range<1> ndrange{args.problem_size};

      cgh.parallel_for<class VecAddKernel<T, sg_size>>(ndrange, [=, iters=args.num_iters](sycl::id<1> gid) [[intel::reqd_sub_group_size(sg_size)]] {
        for (int _ = 0; _ < iters; _++)
          out[gid] = in1[gid] + in2[gid]; 
      });
    }));
  }

  bool verify(VerificationSetting& ver) {
    // Triggers writeback
    output_buf.reset();

    bool pass = true;
    for(size_t i = ver.begin[0]; i < ver.begin[0] + ver.range[0]; i++) {
      auto expected = input1[i] + input2[i];
      if(expected != output[i]) {
        pass = false;
        break;
      }
    }
    return pass;
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "VectorAddition_";
    name << ReadableTypename<T>::name;
    name << "_sg";
    name << sg_size;
    
    return name.str();
  }
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  if (app.deviceSupportsSG(8)) {
    app.run<VecAddBench<int, 8>>();
    app.run<VecAddBench<long long, 8>>();
    app.run<VecAddBench<float, 8>>();
    if(app.deviceSupportsFP64())
      app.run<VecAddBench<double, 8>>();
  }

  if (app.deviceSupportsSG(16)) {
    app.run<VecAddBench<int, 16>>();
    app.run<VecAddBench<long long, 16>>();
    app.run<VecAddBench<float, 16>>();
    if(app.deviceSupportsFP64())
      app.run<VecAddBench<double, 16>>();
  }

  if (app.deviceSupportsSG(32)) {
    app.run<VecAddBench<int, 32>>();
    app.run<VecAddBench<long long, 32>>();
    app.run<VecAddBench<float, 32>>();
    if(app.deviceSupportsFP64())
      app.run<VecAddBench<double, 32>>();
  }

  return 0;
}
