#include <iostream>
#include <sycl/sycl.hpp>

#include "bitmap.h"
#include "common.h"


namespace s = sycl;
template<size_t sg_size>
class SobelBenchKernel; // kernel forward declaration

/*
  A Sobel filter with a convolution matrix 3x3.
  Input and output are two-dimensional buffers of floats.
 */
template<size_t sg_size>
class SobelBench {
protected:
  std::vector<sycl::float4> input;
  std::vector<sycl::float4> output;

  size_t w, h; // size of the input picture
  size_t size; // user-defined size (input and output will be size x size)
  BenchmarkArgs args;


  PrefetchedBuffer<sycl::float4, 2> input_buf;
  PrefetchedBuffer<sycl::float4, 2> output_buf;

public:
  SobelBench(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size = args.problem_size; // input size defined by the user
    input.resize(size * size);
    load_bitmap_mirrored("./Brommy.bmp", size, input);
    output.resize(size * size);

    input_buf.initialize(args.device_queue, input.data(), s::range<2>(size, size));
    output_buf.initialize(args.device_queue, output.data(), s::range<2>(size, size));
  }

  void run(std::vector<sycl::event>& events) {
    const size_t local_size = args.local_size;
    events.push_back(args.device_queue.submit([&](sycl::handler& cgh) {
      auto in = input_buf.get_access<s::access::mode::read>(cgh);
      auto out = output_buf.get_access<s::access::mode::discard_write>(cgh);
      sycl::nd_range<2> ndrange{{size, size}, {local_size, 1}};

      // Sobel kernel 3x3
      const float kernel[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};

      cgh.parallel_for<class SobelBenchKernel<sg_size>>(ndrange, [in, out, kernel, size_ = size](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(sg_size)]] {
        int x = item.get_global_id(0);
        int y = item.get_global_id(1);
        sycl::float4 Gx = sycl::float4(0, 0, 0, 0);
        sycl::float4 Gy = sycl::float4(0, 0, 0, 0);
        const int radius = 3;

        // constant-size loops in [0,1,2]
        for(int x_shift = 0; x_shift < 3; x_shift++) {
          for(int y_shift = 0; y_shift < 3; y_shift++) {
            // sample position
            uint xs = x + x_shift - 1; // [x-1,x,x+1]
            uint ys = y + y_shift - 1; // [y-1,y,y+1]
            // for the same pixel, convolution is always 0
            if(x == xs && y == ys)
              continue;
            // boundary check
            if(xs < 0 || xs >= size_ || ys < 0 || ys >= size_)
              continue;

            // sample color
            sycl::float4 sample = in[{xs, ys}];

            // convolution calculation
            int offset_x = x_shift + y_shift * radius;
            int offset_y = y_shift + x_shift * radius;

            float conv_x = kernel[offset_x];
            sycl::float4 conv4_x = sycl::float4(conv_x);
            Gx += conv4_x * sample;

            float conv_y = kernel[offset_y];
            sycl::float4 conv4_y = sycl::float4(conv_y);
            Gy += conv4_y * sample;
          }
        }
        // taking root of sums of squares of Gx and Gy
        sycl::float4 color = hypot(Gx, Gy);
        sycl::float4 minval = sycl::float4(0.0, 0.0, 0.0, 0.0);
        sycl::float4 maxval = sycl::float4(1.0, 1.0, 1.0, 1.0);
        out[x][y] = clamp(color, minval, maxval);
      });
    }));
  }


  bool verify(VerificationSetting& ver) {
    // Triggers writeback
    output_buf.reset();
    save_bitmap("sobel3.bmp", size, output);

    const float kernel[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    bool pass = true;
    int radius = 3;
    for(size_t i = ver.begin[0]; i < ver.begin[0] + ver.range[0]; i++) {
      int x = i % size;
      int y = i / size;
      sycl::float4 Gx, Gy;
      for(uint x_shift = 0; x_shift < 3; x_shift++)
        for(uint y_shift = 0; y_shift < 3; y_shift++) {
          uint xs = x + x_shift - 1;
          uint ys = y + y_shift - 1;
          if(x == xs && y == ys)
            continue;
          if(xs < 0 || xs >= size || ys < 0 || ys >= size)
            continue;
          sycl::float4 sample = input[xs + ys * size];
          int offset_x = x_shift + y_shift * radius;
          int offset_y = y_shift + x_shift * radius;
          float conv_x = kernel[offset_x];
          sycl::float4 conv4_x = sycl::float4(conv_x);
          Gx += conv4_x * sample;
          float conv_y = kernel[offset_y];
          sycl::float4 conv4_y = sycl::float4(conv_y);
          Gy += conv4_y * sample;
        }

      sycl::float4 color = hypot(Gx, Gy);
      sycl::float4 minval = sycl::float4(0.0, 0.0, 0.0, 0.0);
      sycl::float4 maxval = sycl::float4(1.0, 1.0, 1.0, 1.0);
      sycl::float4 expected = clamp(color, minval, maxval);
      sycl::float4 dif = fdim(output[i], expected);
      float length = sycl::length(dif);
      if(length > 0.01f) {
        pass = false;
        break;
      }
    }
    return pass;
  }


  static std::string getBenchmarkName() { 
    std::stringstream name;
    name << "Sobel3_sg";
    name << sg_size;
  
    return name.str();
  }

}; // SobelBench class


int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  if (app.deviceSupportsSG(8)) {
    app.run<SobelBench<8>>();
  }
  if (app.deviceSupportsSG(16)) {
    app.run<SobelBench<16>>();
  }
  if (app.deviceSupportsSG(32)) {
    app.run<SobelBench<32>>();
  }
  return 0;
}
