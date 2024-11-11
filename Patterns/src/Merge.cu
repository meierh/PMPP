#include "Merge.hpp"

template <typename T>
__device__ void merge_sequential
(
    T* A,
    std::size_t Al,
    T* B,
    std::size_t Bl,
    T* C
)
{
    std::size_t Ai = 0;
    std::size_t Bi = 0;
    std::size_t Ci = 0;
    
    while((Ai<Al) && (Bi<Bl))
    {
        if(A[Ai] <= B[Bi])
        {
            C[Ci] = A[Ai];
            Ai++;
        }
        else
        {
            C[Ci] = B[Bi];
            Bi++;
        }
        Ci++;        
    }
    
    if(Ai==Al)
    {
        while(Bi<Bl)
        {
            C[Ci] = B[Bi];
            Bi++;
            Ci++;
        }
    }
    else
    {
        while(Ai<Al)
        {
            C[Ci] = A[Ai];
            Ai++;
            Ci++;
        }
    }
}

template <typename T, std::size_t BlockSize, std::size_t ThreadSize>
__global__ void merge_kernel
(
    T* A,
    std::size_t Al,
    T* B,
    std::size_t Bl,
    T* C
)
{
    std::size_t Cl = Al+Bl;
    std::size_t totalThreadIdx = blockIdx.x+blockDim.x + threadIdx.x;
    std::size_t offsetC = blockIdx.x*BlockSize*ThreadSize + threadIdx.x*ThreadSize;
    std::size_t offsetCnext = offsetC+ThreadSize;
    offsetCnext = (offsetCnext>Cl) ? Cl : offsetCnext;
    
    __shared__ T locAB[BlockSize][ThreadSize];
    __shared__ T locAsize[BlockSize];
    __shared__ T locC[BlockSize][ThreadSize];
    __shared__ std::size_t offsetA[BlockSize+1];
    
    offsetA[threadIdx.x] = co_rank();
    if(threadIdx.x==0)
        offsetA[BlockSize] = co_rank();
    __syncthreads();
    
    //Read data into AB
    Asize[threadIdx.x] = offsetA[threadIdx.x+1]-offsetA[threadIdx.x];
    std::size_t locABOffset=0;
    for(std::size_t Ai=offsetA[threadIdx.x]; Ai<offsetA[threadIdx.x+1]; Ai++)
    {
        locAB[threadIdx.x][locABOffset] = A[Ai];
        locABOffset++;
    }
    std::size_t offsetB = offsetC - offsetA[threadIdx.x];
    std::size_t offsetBnext = offsetCnext - offsetA[threadIdx.x+1];
    for(std::size_t Bi=offsetB; Bi<offsetBnext; Bi++)
    {
        locAB[threadIdx.x][locABOffset] = B[Bi];
        locABOffset++;
    }
    __syncthreads();
    
    merge_sequential(&(locAB[0]),Asize[threadIdx.x],&(locAB[Asize[threadIdx.x]]),offsetBnext-offsetB,locC);
    __syncthreads();
    
    //Write data into C
    for(std::size_t locCoffset=0; locCoffset<ThreadSize; locCoffset++)
    {
        C[offsetC + locCoffset*BlockSize + threadIdx.x] = locC[locCoffset*BlockSize + threadIdx.x];
    }
}

template <typename T, std::size_t BlockSize, std::size_t ThreadSize>
void merge
(
    const std::vector<T>& A,
    const std::vector<T>& B,
    std::vector<T>& C
) 
requires Arithmetic<T> && CUDABlock<BlockSize,ThreadSize>
{
    
}
