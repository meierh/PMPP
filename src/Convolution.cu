#include "Convolution.hpp"

template <typename T,std::size_t N, std::size_t BlockSize>
__global__ void convolution_1D
(
    T* data,
    std::size_t xDimLength,
    T* kernel,
    T* result
)
requires Arithmetic<T> && Kernel<N> && CUDABlock<N,BlockSize>
{
    std::size_t dataOffset = blockIdx.x*BlockSize;
    std::size_t blockLimit = BlockSize < xDimLength-dataOffset ? BlockSize : xDimLength-dataOffset;
        
    __shared__ T kernelMem[N];
    std::size_t myInd=0;
    while(threadIdx.x+myInd < N)
    {
        kernelMem[threadIdx.x+myInd] = kernel[threadIdx.x+myInd];
        myInd+=blockDim.x;
    }
    __syncthreads();
    
    
    constexpr std::size_t overlap = (N-1)/2;
    __shared__ T dataCache[BlockSize+2*overlap];
    //Fill left overlap
    myInd=0;
    while(threadIdx.x+myInd < overlap)
    {
        int localDataInd = dataOffset-overlap+threadIdx.x+myInd;
        std::size_t localMemInd = threadIdx.x+myInd;
        if(localDataInd<0)
            dataCache[localMemInd] = 0;
        else
            dataCache[localMemInd] = data[localDataInd];
        myInd+=blockDim.x;
    }
    //Fill right overlap
    myInd=0;
    while(threadIdx.x+myInd < overlap)
    {
        int localDataInd = dataOffset+blockLimit+threadIdx.x+myInd;
        std::size_t localMemInd = threadIdx.x+myInd+blockLimit+overlap;
        if(localDataInd>=xDimLength)
            dataCache[localMemInd] = 0;
        else
            dataCache[localMemInd] = data[localDataInd];
        myInd+=blockDim.x;
    }
    //Fill remaining cache
    myInd=0;
    while(threadIdx.x+myInd < blockLimit)
    {
        int localDataInd = threadIdx.x+myInd+dataOffset;
        std::size_t localMemInd = threadIdx.x+myInd+overlap;
        dataCache[localMemInd] = data[localDataInd];
        myInd+=blockDim.x;
    }
    __syncthreads();
    
    myInd=0;
    while(threadIdx.x+myInd < blockLimit)
    {
        std::size_t imageCacheOffset = threadIdx.x+myInd+overlap;
        T sum = 0;
        for(std::size_t kernelInd=0; kernelInd<N; kernelInd++)
        {
            std::size_t imageCacheInd = imageCacheOffset-overlap+kernelInd;
            sum += dataCache[imageCacheInd]*kernelMem[kernelInd];
        }
        result[dataOffset+threadIdx.x+myInd] = sum;
        myInd+=blockDim.x;
    }
}

template <typename T,std::size_t N>
void convolution1D
(
    const std::vector<T>& data,
    const std::array<T,N>& kernel,
    std::vector<T>& result
) 
requires Arithmetic<T> && Kernel<N>
{
    T* d_data;
    T* d_kernel;
    T* d_result;
    cudaMalloc((void**)&d_data,data.size()*sizeof(T));
    cudaMalloc((void**)&d_kernel,N*sizeof(T));    
    cudaMalloc((void**)&d_result,data.size()*sizeof(T));
    
    cudaMemcpy(d_data,data.data(),data.size()*sizeof(T),cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel,kernel.data(),N*sizeof(T),cudaMemcpyHostToDevice);
    
    constexpr std::size_t BlockSize = 12;
    std::size_t GridSize = ceil((float)data.size()/BlockSize);
    std::cout<<"BlockSize:"<<BlockSize<<" GridSize:"<<GridSize<<std::endl;
    convolution_1D<T,N,BlockSize><<<GridSize,BlockSize>>>(d_data,data.size(),d_kernel,d_result);
    
    result.resize(data.size());
    cudaMemcpy(result.data(),d_result,data.size()*sizeof(T),cudaMemcpyDeviceToHost);
    
    cudaFree(d_data);
    cudaFree(d_kernel);
    cudaFree(d_result);
}

template void convolution1D<int,3>(const std::vector<int>& data,const std::array<int,3>& kernel,std::vector<int>& result);

template <typename T,std::size_t N, std::size_t BlockSize>
__global__ void convolution_2D
(
    T* data,
    std::size_t dataLen,
    std::size_t xWidth,
    T* kernel,
    T* result
)
requires Arithmetic<T> && Kernel<N> && CUDABlock<N,BlockSize>
{
    std::size_t xOffset = blockIdx.x*BlockSize;
    std::size_t yOffset = blockIdx.x*BlockSize;
    std::size_t yWidth = dataLen/xWidth;
    std::size_t blockXLimit = BlockSize < xWidth-xOffset ? BlockSize : xWidth-xOffset;
    std::size_t blockYLimit = BlockSize < yWidth-yOffset ? BlockSize : yWidth-yOffset;
    
    __shared__ T kernelMem[N][N];
    std::size_t myIndx=0;
    std::size_t myIndy=0;
    while(threadIdx.y+myIndy < N)
    {
        myIndx=0;
        while(threadIdx.x+myIndx < N)
        {
            kernelMem[threadIdx.y+myIndy][threadIdx.x+myIndx] = kernel[(threadIdx.y+myIndy)*N+threadIdx.x+myIndx];
            myIndx+=blockDim.x;
        }
        myIndy+=blockDim.y;
    }
    __syncthreads();
    
    
    constexpr std::size_t overlap = (N-1)/2;
    __shared__ T dataCache[BlockSize+2*overlap][BlockSize+2*overlap];
    myIndx=0;
    myIndy=0;
    while(threadIdx.y+myIndy < blockYLimit+2*overlap)
    {
        std::size_t imageYInd = yOffset+threadIdx.y+myIndy-overlap;
        myIndx=0;
        while(threadIdx.x+myIndx < blockXLimit+2*overlap)
        {
            std::size_t imageXInd = xOffset+threadIdx.x+myIndx-overlap;
            if( threadIdx.x+myIndx < overlap || threadIdx.y+myIndy < overlap ||
                threadIdx.x+myIndx >= BlockSize+overlap || threadIdx.y+myIndy >= BlockSize+overlap)
                dataCache[threadIdx.y+myIndy][threadIdx.x+myIndx] = 0;
            else
                dataCache[threadIdx.y+myIndy][threadIdx.x+myIndx] = data[imageYInd*xWidth+imageXInd];
            myIndx+=blockDim.x;
        }
        myIndy+=blockDim.y;
    }
    __syncthreads();
    
    myIndx=0;
    myIndy=0;
    while(threadIdx.y+myIndy < blockYLimit)
    {
        std::size_t locImageYInd = threadIdx.y+myIndy+overlap;
        myIndx=0;
        while(threadIdx.x+myIndx < blockXLimit)
        {
            std::size_t locImageXInd = threadIdx.x+myIndx+overlap;
            T sum = 0;
            for(std::size_t kernelIndy=0; kernelIndy<N; kernelIndy++)
            {
                for(std::size_t kernelIndx=0; kernelIndx<N; kernelIndx++)
                {
                    std::size_t imageCacheIndy = locImageYInd-overlap+kernelIndy;
                    std::size_t imageCacheIndx = locImageXInd-overlap+kernelIndx;
                    sum += dataCache[imageCacheIndy][imageCacheIndx]*kernelMem[kernelIndy][kernelIndy];
                }
            }
            result[(yOffset+locImageYInd)*xWidth+xOffset+locImageXInd] = sum;
            myIndx+=blockDim.x;
        }
        myIndy+=blockDim.y;
    }
}

template <typename T,std::size_t N>
void convolution2D
(
    const std::vector<T>& data,
    size_t width,
    const std::array<std::array<T,N>,N>& kernel,
    std::vector<T>& result
) 
requires Arithmetic<T> && Kernel<N>
{
    T* d_data;
    T* d_kernel;
    T* d_result;
    cudaMalloc((void**)&d_data,data.size()*sizeof(T));
    cudaMalloc((void**)&d_kernel,N*N*sizeof(T));    
    cudaMalloc((void**)&d_result,data.size()*sizeof(T));
    
    cudaMemcpy(d_data,data.data(),data.size()*sizeof(T),cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel,kernel.data(),N*N*sizeof(T),cudaMemcpyHostToDevice);
    
    constexpr std::size_t BlockSize = 12;
    std::size_t GridSize = ceil((float)data.size()/BlockSize);
    std::cout<<"BlockSize:"<<BlockSize<<" GridSize:"<<GridSize<<std::endl;
    convolution_2D<T,N,BlockSize><<<GridSize,BlockSize>>>(d_data,data.size(),d_kernel,d_result);
    
    result.resize(data.size());
    cudaMemcpy(result.data(),d_result,data.size()*sizeof(T),cudaMemcpyDeviceToHost);
    
    cudaFree(d_data);
    cudaFree(d_kernel);
    cudaFree(d_result);
}
