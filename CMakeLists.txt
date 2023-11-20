cmake_minimum_required(VERSION 3.5)
project(sycl-bench)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "CMake Build Type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Due to CMake limitations, hipSYCL requires C++ standard to be set manually
set(CMAKE_SYCL_FLAGS "${CMAKE_SYCL_FLAGS} -std=c++17")

if(CMAKE_GENERATOR STREQUAL "Ninja")
  set(CMAKE_SYCL_FLAGS "${CMAKE_SYCL_FLAGS} -fdiagnostics-color=always")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
  set(COMPUTECPP_USER_FLAGS "${COMPUTECPP_USER_FLAGS} -fdiagnostics-color=always")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include-bloom-filter)

include_directories(${CMAKE_SOURCE_DIR}/polybench/common)

set(SYCL_IMPL "" CACHE STRING "Select SYCL implementation")
set_property(CACHE SYCL_IMPL PROPERTY STRINGS "dpcpp" "AdaptiveCpp" "triSYCL")
option(ENABLED_TIME_EVENT_PROFILING "Enable time profiling with sycl event" OFF)

set(supported_implementations
  AdaptiveCpp
  dpcpp
  triSYCL
)# SYCL Bench variables
option(SYCL_BENCH_ENABLE_FP64_BENCHMARKS "Enable FP64 benchmarks" ON)



list(FIND supported_implementations ${SYCL_IMPL} impl_idx)

if(NOT SYCL_IMPL OR impl_idx EQUAL -1)
  message(FATAL_ERROR "Please specify SYCL_IMPL (one of: ${supported_implementations})")
endif()

if(SYCL_IMPL STREQUAL "AdaptiveCpp")
  find_package(AdaptiveCpp REQUIRED)
elseif(SYCL_IMPL STREQUAL "dpcpp")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

  if(DPCPP_WITH_CUDA_BACKEND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda")

    set(CUDA_ARCH "" CACHE STRING "CUDA device architecture e.g. sm_70")

    if(NOT CUDA_ARCH STREQUAL "")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xsycl-target-backend --cuda-gpu-arch=${CUDA_ARCH}")
    endif()
  endif()

  if(DPCPP_WITH_ROCM_BACKEND)
    set(ROCM_ARCH "" CACHE STRING "ROCm device architecture e.g. gfx908")

    if(NOT ROCM_ARCH STREQUAL "")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=amd_gpu_${ROCM_ARCH}")
    endif()
  endif()

  if(DPCPP_WITH_LZ_BACKEND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=spir64_gen")

    set(LZ_ARCH "" CACHE STRING "Level Zero device architecture e.g. acm-g10")

    if(NOT LZ_ARCH STREQUAL "")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xsycl-target-backend \"-device ${LZ_ARCH}\"")
    endif()
  endif()
elseif(SYCL_IMPL STREQUAL "triSYCL")
  find_package(TriSYCL MODULE REQUIRED)
endif()

set(benchmarks

  # micro/arith.cpp
  # micro/DRAM.cpp
  # micro/host_device_bandwidth.cpp
  # micro/pattern_L2.cpp
  # micro/sf.cpp
  # micro/local_mem.cpp

  intel/vec_add.cpp
  intel/matrix_mul.cpp
)

# Setting variables
add_compile_definitions(SYCL_BENCH_ENABLE_FP64_BENCHMARKS=$<BOOL:${SYCL_BENCH_ENABLE_FP64_BENCHMARKS}>)

foreach(benchmark IN LISTS benchmarks)
  get_filename_component(target ${benchmark} NAME_WE)

  add_executable(${target} ${benchmark})

  if(ENABLED_TIME_EVENT_PROFILING)
    target_compile_definitions(${target} PUBLIC SYCL_BENCH_ENABLE_QUEUE_PROFILING=1)
  endif()
  if(SYCL_IMPL STREQUAL "AdaptiveCpp")
  add_sycl_to_target(TARGET ${target} SOURCES ${benchmark})
  endif()

  if(SYCL_IMPL STREQUAL "dpcpp")
    target_compile_definitions(${target} PRIVATE __DPCPP__)
  endif()
  

  install(TARGETS ${target} RUNTIME DESTINATION bin/benchmarks/)
  get_filename_component(dir ${benchmark} DIRECTORY)
  set_property(TARGET ${target} PROPERTY FOLDER ${dir})
endforeach(benchmark)

# The "compiletime" target should only be used in the context of the compile time evaluation script
# set_target_properties(compiletime PROPERTIES EXCLUDE_FROM_ALL 1)

install(PROGRAMS bin/run-suite DESTINATION bin/)
install(FILES ${PROJECT_SOURCE_DIR}/Brommy.bmp DESTINATION share/)
