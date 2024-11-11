#ifndef MERGE_H
#define MERGE_H

#include <vector>
#include <array>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>

template<typename T> 
concept Arithmetic = std::integral<T> || std::floating_point<T>;

template<std::size_t BlockSize, std::size_t ThreadSize> 
concept CUDABlock = BlockSize>0 && ThreadSize>0;

template <typename T, std::size_t BlockSize, std::size_t ThreadSize>
void merge
(
    const std::vector<T>& A,
    const std::vector<T>& B,
    std::vector<T>& C
) 
requires Arithmetic<T> && CUDABlock<BlockSize,ThreadSize>;

#endif
