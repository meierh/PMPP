#ifndef PARALLELSCAN_H
#define PARALLELSCAN_H

#include <vector>
#include <array>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>

template<typename T> 
concept Arithmetic = std::integral<T> || std::floating_point<T>;

template<std::size_t BlockSize, std::size_t BlockChunkSize> 
concept CUDABlock = BlockChunkSize>=BlockSize;

template <typename T, std::size_t BlockSize, std::size_t BlockChunkSize>
void parallelInclusiveScan
(
    const std::vector<T>& data,
    std::vector<T>& result
) 
requires Arithmetic<T> && CUDABlock<BlockSize,BlockChunkSize>;

#endif
