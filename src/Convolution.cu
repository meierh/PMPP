#include "Convolution.hpp"

template <typename T,std::size_t N, std::size_t CacheSize>
__global__ void convolution_1D
(
    T* data,
    std::size_t xDimLength
)
{
    std::size_t dataOffset = blockIdx.x*CacheSize;
    __shared__ T kernelMem[CacheSize];
    
    std::size_t myInd=0;
    while(threadIdx.x+myInd < CacheSize)
    {
        kernelMem[threadIdx.x+myInd] = data[dataOffset+threadIdx.x+myInd];
    }
    __syncthreads();
    
}
