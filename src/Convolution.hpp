#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <vector>
#include <array>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>

template<typename T> 
concept Arithmetic = std::integral<T> || std::floating_point<T>;

template<std::size_t N> 
concept Kernel = N>2 && static_cast<bool>(N%2) && N<100;

template<std::size_t N, std::size_t BlockSize> 
concept CUDABlock = BlockSize>=N;

template <typename T,std::size_t N>
void convolution1D
(
    const std::vector<T>& data,
    const std::array<T,N>& kernel,
    std::vector<T>& result
) 
requires Arithmetic<T> && Kernel<N>;

template <typename T,std::size_t N>
void convolution2D
(
    const std::vector<T>& data,
    size_t width,
    const std::array<std::array<T,N>,N>& kernel,
    std::vector<T>& result
) 
requires Arithmetic<T> && Kernel<N>;

template <typename T,std::size_t N>
void convolution3D
(
    const std::vector<T>& data,
    size_t width,
    size_t depth,
    const std::array<std::array<std::array<T,N>,N>,N>& kernel
) 
requires Arithmetic<T> && Kernel<N>;

#endif
