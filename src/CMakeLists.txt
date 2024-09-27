#src cmake
cmake_minimum_required(VERSION 3.10)

set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(
  APPEND SRC_LIST
  "${dir}/Convolution.cpp"
  "${dir}/Convolution.cu"
)

set(SRC_LIST "${SRC_LIST}" PARENT_SCOPE)
