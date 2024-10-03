#include "Sort.hpp"

template <typename TUINT, bool leadBitSign, std::uint8_t radixWidth, std::uint8_t radixSection>
__device__ std::uint8_t toBucket
(
    TUINT value
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

template <typename TUINT, bool leadBitSign, std::size_t radixWidth, std::size_t BlockSize>
__global__ void computeInsertionValue
(
    TUINT* data,
    std::size_t dataLen,
    std::uint64_t* d_perBucketInsertionValue,
    std::uint8_t nbrBuckets,
    std::uint8_t* d_toBucket,
    TUINT* result
)
requires Arithmetic<TUINT> && IsUint<TUINT> && RadixWidth<TUINT,radixWidth>
{
    
}

template <typename T, std::size_t radixWidth, std::size_t BlockSize>
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

    std::uint8_t nbrBuckets = 2;
    for(std::uint8_t i=0; i<radixWidth-1; i++)
        nbrBuckets *= 2;
    std::vector<std::uint64_t> perBucketInsertionVal(nbrBuckets*data.size());

    TUINT* d_data;
    std::uint64_t* d_perBucketInsertionValue;
    std::uint8_t* d_toBucket;
    TUINT* d_result;
    cudaMalloc((void**)&d_data,data.size()*sizeof(TUINT));
    cudaMalloc((void**)&d_perBucketInsertionValue,perBucketInsertionVal.size()*sizeof(std::uint64_t));
    cudaMalloc((void**)&d_toBucket,d_data.size()*sizeof(std::uint8_t));
    cudaMalloc((void**)&d_result,data.size()*sizeof(TUINT));

    cudaMemcpy(d_data,data.data(),data.size()*sizeof(T),cudaMemcpyHostToDevice);
    
    std::size_t GridSize = ceil((float)data.size()/BlockSize);
    std::cout<<"BlockSize:"<<BlockSize<<" GridSize:"<<GridSize<<std::endl;
    radixSortStep<TUINT,signedLead,radixWidth,BlockSize><<<GridSize,BlockSize>>>(d_data,data.size(),d_result);
    
    result.resize(data.size());
    cudaMemcpy(result.data(),d_result,data.size()*sizeof(T),cudaMemcpyDeviceToHost);
    
    cudaFree(d_data);
    cudaFree(d_perBucketInsertionValue);
    cudaFree(d_result);
}

template<> void radixSort<float,32>(const std::vector<float>& data, std::vector<float>& result);
