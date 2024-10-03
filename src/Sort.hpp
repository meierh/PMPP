#ifndef SORT_H
#define SORT_H

#include <vector>
#include <array>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>

template<typename T> 
concept Arithmetic = std::integral<T> || std::floating_point<T>;

template<typename Arith> struct ArithToUint;
template<> struct ArithToUint<std::uint8_t>  { typedef std::uint8_t  type; };
template<> struct ArithToUint<std::uint16_t> { typedef std::uint16_t type; };
template<> struct ArithToUint<std::uint32_t> { typedef std::uint32_t type; };
template<> struct ArithToUint<std::uint64_t> { typedef std::uint64_t type; };
template<> struct ArithToUint<std::int8_t>  { typedef std::uint8_t  type; };
template<> struct ArithToUint<std::int16_t> { typedef std::uint16_t type; };
template<> struct ArithToUint<std::int32_t> { typedef std::uint32_t type; };
template<> struct ArithToUint<std::int64_t> { typedef std::uint64_t type; };
template<> struct ArithToUint<float> { typedef std::uint32_t type; };
template<> struct ArithToUint<double> { typedef std::uint64_t type; };

/*
template<typename Arith> 
class ArithToUint2
{
    
    if(sizeof(Arith)==1)
        typedef std::uint8_t type;
    else if(sizeof(Arith)==2)
        typedef std::uint8_t type;
    
}
requires Arithmetic<Arith>;
*/


template<typename Arith> struct ArithToLeadBitSign;
template<> struct ArithToLeadBitSign<std::uint8_t>  { bool leadBitSign = false; };
template<> struct ArithToLeadBitSign<std::uint16_t> { bool leadBitSign = false; };
template<> struct ArithToLeadBitSign<std::uint32_t> { bool leadBitSign = false; };
template<> struct ArithToLeadBitSign<std::uint64_t> { bool leadBitSign = false; };
template<> struct ArithToLeadBitSign<std::int8_t>  { bool leadBitSign = true; };
template<> struct ArithToLeadBitSign<std::int16_t> { bool leadBitSign = true; };
template<> struct ArithToLeadBitSign<std::int32_t> { bool leadBitSign = true; };
template<> struct ArithToLeadBitSign<std::int64_t> { bool leadBitSign = true; };
template<> struct ArithToLeadBitSign<float> { bool leadBitSign = true; };
template<> struct ArithToLeadBitSign<double> { bool leadBitSign = true; };

/*
template<typename T> 
concept SizeMatch = sizeof(T)==sizeof(ArithToUint<T>::type);
*/

template<std::size_t BlockSize, std::size_t ThreadSize> 
concept CUDABlock = BlockSize>0 && ThreadSize>0;

template<typename T> 
concept IsUint =    std::is_same<T,std::uint8_t >::value || std::is_same<T,std::uint16_t>::value ||
                    std::is_same<T,std::uint32_t>::value || std::is_same<T,std::uint64_t>::value;

template<typename TUINT, std::size_t radixWidth> 
concept RadixWidth = (sizeof(TUINT)*8)>=radixWidth && ((sizeof(TUINT)*8)%radixWidth)==0 && radixWidth<=8 && radixWidth>0;
                    
template <typename T, std::size_t BlockSize>
void radixSort
(
    const std::vector<T>& data,
    std::vector<T>& result
)
requires Arithmetic<T>;

#endif
