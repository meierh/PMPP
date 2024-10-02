#include "ParallelScan.hpp"

template <typename T>
__device__ void readChunkIntoCache
(
    T* d_data,
    std::size_t dataOffset,
    std::size_t dataRange,
    T* cachePtr
)
requires Arithmetic<T>
{
    std::size_t myIndx=0;
    while(threadIdx.x+myIndx < dataRange)
    {
        cachePtr[threadIdx.y+myIndx] = d_data[threadIdx.x+myIndx+dataOffset];
        myIndx+=blockDim.x;
    }
}

template <typename T>
__device__ void brentKungAlgorithm
(
    T* cachePtr,
    std::size_t dataRange
)
requires Arithmetic<T>
{
    for(std::size_t stride = 1; stride <= blockDim.x; stride *= 2)
    {
        __syncthreads();
        std::size_t index = (threadIdx.x+1)*2*stride-1;
        if(index<dataRange)
            cachePtr[index] += cachePtr[index-stride];
    }
}

template <typename T>
__device__ void updateChunkOnCarry
(
    T* d_data,
    std::size_t dataOffset,
    std::size_t dataRange
)
requires Arithmetic<T>
{
    T carryValue = d_data[dataOffset-1];
    
    std::size_t myIndx=0;
    while(threadIdx.x+myIndx < dataRange)
    {
        d_data[threadIdx.x+myIndx+dataOffset] = d_data[threadIdx.x+myIndx+dataOffset] + carryValue;
        myIndx+=blockDim.x;
    }
}

template <typename T>
__device__ void writeChunkIntoMemory
(
    T* cachePtr,
    std::size_t dataOffset,
    std::size_t dataRange,
    T* d_data
)
requires Arithmetic<T>
{
    std::size_t myIndx=0;
    while(threadIdx.x+myIndx < dataRange)
    {
        d_data[threadIdx.x+myIndx+dataOffset] = cachePtr[threadIdx.y+myIndx];
        myIndx+=blockDim.x;
    }
}

template <std::size_t BlockChunkSize>
__device__ void computeDataRange
(
    const std::size_t chunkIdx,
    const std::size_t dataLength,
    std::size_t& dataOffset,
    std::size_t& dataRange
)
{
    dataOffset = chunkIdx*BlockChunkSize;
    std::size_t dataNextOffset = (dataOffset+BlockChunkSize) < dataLength ? dataLength : dataOffset+BlockChunkSize;
    dataRange = dataNextOffset-dataOffset;
}

template <typename T, std::size_t BlockSize, std::size_t BlockChunkSize>
__global__ void parallelInclusiveScanKernel
(
    T* d_data,
    std::size_t dataLength,
    std::uint32_t* d_blockUpdateFront,
    std::uint32_t* d_blockUpdateFrontDone,
    T* d_result
)
requires Arithmetic<T> && CUDABlock<BlockSize,BlockChunkSize>
{
    std::size_t dataOffset, dataRange;
    computeDataRange<BlockChunkSize>(blockIdx.x,dataLength,dataOffset,dataRange);
    
    __shared__ T kernelMem[BlockChunkSize];
    
    readChunkIntoCache<T>(d_data,dataOffset,dataRange,kernelMem);
    __syncthreads();
    
    brentKungAlgorithm(kernelMem,dataRange);
    __syncthreads();
    
    writeChunkIntoMemory(kernelMem,dataOffset,dataRange,d_result);
    __syncthreads();
        
    for(std::uint32_t updateStep=0; updateStep<gridDim.x; updateStep++)
    {
        std::uint32_t* updateFrontVar = &(d_blockUpdateFront[updateStep]);
        std::uint32_t myUpdateInd = atomicAdd(updateFrontVar,1);
        std::uint32_t myUpdateChunk = myUpdateInd+updateStep;
        if(updateStep==0)
        {
            if(myUpdateChunk>0)
            {
                computeDataRange<BlockChunkSize>(myUpdateChunk,dataLength,dataOffset,dataRange);
                updateChunkOnCarry(d_result,dataOffset,dataRange);
            }
        }
        else
        {
            std::uint32_t* prevUpdateFrontDoneVar = &(d_blockUpdateFrontDone[updateStep-1]);
            if(threadIdx.x==0)
            {
                while((*prevUpdateFrontDoneVar) <  myUpdateInd)
                {}
            }
            __syncthreads();
            computeDataRange<BlockChunkSize>(myUpdateChunk,dataLength,dataOffset,dataRange);
            updateChunkOnCarry(d_result,dataOffset,dataRange);
        }
        std::uint32_t* updateFrontDoneVar = &(d_blockUpdateFrontDone[updateStep]);
        atomicAdd(updateFrontDoneVar,1);
    }
}

template <typename T, std::size_t BlockSize, std::size_t BlockChunkSize>
void parallelInclusiveScan
(
    const std::vector<T>& data,
    std::vector<T>& result
) 
requires Arithmetic<T> && CUDABlock<BlockSize,BlockChunkSize>
{
    result.resize(data.size());
    std::fill(result.begin(),result.end(),0);
    
    const std::size_t nbrOfBlocks = ceil((float)result.size()/BlockChunkSize);
    std::vector<std::uint32_t> blockUpdateFront(nbrOfBlocks);
    std::fill(blockUpdateFront.begin(),blockUpdateFront.end(),0);
    
    T* d_data;
    std::uint32_t* d_blockUpdateFront;
    std::uint32_t* d_blockUpdateFrontDone;
    T* d_result;
    cudaMalloc((void**)&d_data,data.size()*sizeof(T));
    cudaMalloc((void**)&d_blockUpdateFront,nbrOfBlocks*sizeof(std::uint32_t));
    cudaMalloc((void**)&d_blockUpdateFrontDone,nbrOfBlocks*sizeof(std::uint32_t));
    cudaMalloc((void**)&d_result,data.size()*sizeof(T));
    
    cudaMemcpy(d_data,data.data(),data.size()*sizeof(T),cudaMemcpyHostToDevice);
    cudaMemcpy(d_blockUpdateFront,blockUpdateFront.data(),nbrOfBlocks*sizeof(std::uint32_t),cudaMemcpyHostToDevice);
    cudaMemcpy(d_blockUpdateFrontDone,blockUpdateFront.data(),nbrOfBlocks*sizeof(std::uint32_t),cudaMemcpyHostToDevice);
    cudaMemcpy(d_result,result.data(),result.size()*sizeof(T),cudaMemcpyHostToDevice);
    
    std::size_t GridSize = ceil((float)data.size()/BlockSize);
    std::cout<<"BlockSize:"<<BlockSize<<" GridSize:"<<GridSize<<std::endl;
    parallelInclusiveScanKernel<T,BlockSize,BlockChunkSize><<<GridSize,BlockSize>>>(d_data,data.size(),d_blockUpdateFront,d_blockUpdateFrontDone,d_result);
    
    result.resize(data.size());
    cudaMemcpy(result.data(),d_result,data.size()*sizeof(T),cudaMemcpyDeviceToHost);
    
    cudaFree(d_data);
    cudaFree(d_blockUpdateFront);
    cudaFree(d_blockUpdateFrontDone);
    cudaFree(d_result);
}

template void parallelInclusiveScan<int,32,512>(const std::vector<int>& data, std::vector<int>& result);


