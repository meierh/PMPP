#include "Sort.hpp"

template <typename TUINT, std::uint8_t radixWidth>
__device__ std::uint8_t toBucket
(
    TUINT value,
    std::uint8_t radixSection,
    bool leadBitSign
)
requires Arithmetic<TUINT> && IsUint<TUINT> && RadixWidth<TUINT,radixWidth>
{
    std::uint8_t nbrOfBits = sizeof(TUINT)*8;
    if(leadBitSign)
    {
        value = value ^ (1 << (nbrOfBits-1));
    }
    std::uint8_t bitOffset = radixSection*radixWidth;
    std::uint8_t bitRange = bitOffset+radixWidth;
    TUINT bitMask = 0;
    for(std::uint8_t bitI=bitOffset; bitI<bitRange; bitI++)
    {
        bitMask |= (1<<bitI);
    }
    TUINT maskedValue = value & bitMask;
    maskedValue = maskedValue >> bitOffset;
    return maskedValue;    
}

template <typename TUINT, std::size_t radixWidth>
__global__ void computeBucketValue
(
    const TUINT* data,
    std::size_t dataLen,
    std::uint8_t radixSection,
    bool leadBitSign,
    std::uint8_t* d_toBucket
)
requires Arithmetic<TUINT> && IsUint<TUINT> && RadixWidth<TUINT,radixWidth>
{
    std::uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<dataLen)
    {
        d_toBucket[tid] = toBucket<TUINT,leadBitSign,radixWidth>(data[tid],radixSection);
    }
}

template <typename TUINT, std::size_t BlockSize, std::size_t BlockChunkSize>
__global__ void computeMoveInfo
(
    const std::uint8_t* d_toBucket,
    std::size_t dataLen,
    std::uint64_t* d_perBucketInsertionValue,
    std::uint8_t nbrBuckets
)
requires Arithmetic<TUINT> && IsUint<TUINT>
{
    
}

template <typename TUINT>
__global__ void moveValues
(
    const TUINT* d_preMovedData,
    std::size_t dataLen,
    const std::uint8_t* d_toBucket,
    const std::uint64_t* d_perBucketInsertionValue,
    TUINT* d_postMovedData
)
requires Arithmetic<TUINT> && IsUint<TUINT>
{
    std::uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<dataLen)
    {
        std::uint8_t targetBucket = d_toBucket[tid];
        std::uint64_t targetIndex = d_perBucketInsertionValue[targetBucket*dataLen+tid];
        d_postMovedData[targetIndex] = d_preMovedData[tid];
    }
}

template <typename T, std::size_t radixWidth, std::size_t BlockSize, std::size_t BlockChunkSize>
void radixSort
(
    const std::vector<T>& data,
    std::vector<T>& result
) 
requires Arithmetic<T>
{
    //std::vector<ArithToUint<T>::>
    constexpr ArithToLeadBitSign<T> signedLeadBit;
    constexpr bool signedLead = signedLeadBit.leadBitSign;
    typedef typename ArithToUint<T>::type TUINT;

    constexpr std::uint8_t nbrBuckets = 1<<radixWidth;
    std::vector<std::uint64_t> perBucketInsertionVal(nbrBuckets*data.size());
    
    std::uint8_t nbrRadixSections = 8*sizeof(TUINT) / radixWidth;

    TUINT* d_data;
    TUINT* d_data_cp;
    cudaMalloc((void**)&d_data,data.size()*sizeof(TUINT));
    cudaMalloc((void**)&d_data_cp,data.size()*sizeof(TUINT));
    cudaMemcpy(d_data,data.data(),data.size()*sizeof(TUINT),cudaMemcpyHostToDevice);
    
    std::uint8_t* d_toBucket;
    cudaMalloc((void**)&d_toBucket,data.size()*sizeof(std::uint8_t));
    
    std::uint64_t* d_perBucketInsertionValue;
    cudaMalloc((void**)&d_perBucketInsertionValue,perBucketInsertionVal.size()*sizeof(std::uint64_t));
    
    TUINT* d_result;
    cudaMalloc((void**)&d_result,data.size()*sizeof(TUINT));
    
    std::size_t GridSize = ceil((float)data.size()/BlockSize);
    std::cout<<"BlockSize:"<<BlockSize<<" GridSize:"<<GridSize<<std::endl;
    
    bool leadRadix = false;
    for(std::uint8_t radixSection=0; radixSection<nbrRadixSections; radixSection++)
    {
        if(signedLead)
        {
            if(radixSection==nbrRadixSections-1)
                leadRadix = true;
        }
        computeBucketValue<TUINT,radixWidth>
            <<<GridSize,BlockSize>>>(d_data,data.size(),radixSection,leadRadix,d_toBucket);
        cudaDeviceSynchronize();
        
        computeInsertionValue<TUINT,BlockSize,BlockChunkSize>
            <<<GridSize,BlockSize>>>(d_toBucket,data.size(),d_perBucketInsertionValue,nbrBuckets);
        cudaDeviceSynchronize();
        
        moveValues<TUINT>
            <<<GridSize,BlockSize>>>(d_data,data.size(),d_toBucket,d_perBucketInsertionValue,d_data_cp);
        cudaDeviceSynchronize();
        cudaMemcpy(d_data,d_data_cp,data.size()*sizeof(TUINT),cudaMemcpyDeviceToDevice);
    }
    
    radixSortStep<TUINT,signedLead,radixWidth,BlockSize><<<GridSize,BlockSize>>>(d_data,data.size(),d_result);
    
    result.resize(data.size());
    cudaMemcpy(result.data(),d_result,data.size()*sizeof(T),cudaMemcpyDeviceToHost);
    
    cudaFree(d_data);
    cudaFree(d_perBucketInsertionValue);
    cudaFree(d_result);
}

template<> void radixSort<float,32>(const std::vector<float>& data, std::vector<float>& result);
