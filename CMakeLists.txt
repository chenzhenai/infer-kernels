cmake_minimum_required(VERSION 3.18)

# 在 project() 之前选定架构，否则 CUDA 语言初始化会用编译器默认值。
# 覆盖方式：cmake -DCMAKE_CUDA_ARCHITECTURES=90a ...
#   8.x -> 80（A100）  9.x -> 90a（H100）  10.0–10.2 -> 100a（B200）  10.3+ -> 103a（B300）
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(_ik_arch "80")
  find_program(_ik_smi nvidia-smi)
  if(_ik_smi)
    execute_process(
      COMMAND ${_ik_smi} --query-gpu=compute_cap --format=csv,noheader
      OUTPUT_VARIABLE _ik_cap
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      RESULT_VARIABLE _ik_smi_rc)
    if(_ik_smi_rc EQUAL 0 AND _ik_cap MATCHES "([0-9]+)[.]([0-9]+)")
      set(_major "${CMAKE_MATCH_1}")
      set(_minor "${CMAKE_MATCH_2}")
      if(_major STREQUAL "8")
        set(_ik_arch "80")
      elseif(_major STREQUAL "9")
        set(_ik_arch "90a")
      elseif(_major STREQUAL "10" AND _minor GREATER_EQUAL 3)
        set(_ik_arch "103a")
      elseif(_major STREQUAL "10")
        set(_ik_arch "100a")
      else()
        message(WARNING "Compute capability ${_major}.${_minor} is outside 8/9/10; using sm_80")
      endif()
    endif()
  else()
    message(STATUS "nvidia-smi not found; default CMAKE_CUDA_ARCHITECTURES=80")
  endif()
  set(CMAKE_CUDA_ARCHITECTURES "${_ik_arch}" CACHE STRING "80, 90a, 100a, or 103a")
endif()

# project() 会用该架构做编译器测试；版本不够时先给出明确错误，而不是 “compiler is broken”。
find_program(_ik_nvcc nvcc)
if(_ik_nvcc)
  execute_process(
    COMMAND ${_ik_nvcc} --version
    OUTPUT_VARIABLE _ik_nvcc_ver
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(_ik_nvcc_ver MATCHES "release ([0-9]+[.][0-9]+)")
    set(_ik_cuda_ver "${CMAKE_MATCH_1}")
    if(CMAKE_CUDA_ARCHITECTURES MATCHES "103a" AND _ik_cuda_ver VERSION_LESS 12.9)
      message(FATAL_ERROR "sm_103a needs CUDA 12.9 or newer (nvcc ${_ik_cuda_ver})")
    endif()
    if(CMAKE_CUDA_ARCHITECTURES MATCHES "100a" AND _ik_cuda_ver VERSION_LESS 12.8)
      message(FATAL_ERROR "sm_100a needs CUDA 12.8 or newer (nvcc ${_ik_cuda_ver})")
    endif()
  endif()
endif()

project(infer_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(device_info tools/device_info.cu)
add_executable(oob_probe tools/oob_probe.cu)
foreach(tgt device_info oob_probe)
  set_target_properties(${tgt} PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O2;-lineinfo>)
endforeach()
