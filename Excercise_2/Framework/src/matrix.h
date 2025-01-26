#pragma once

#include <cstdint>
#include <string>

#include "pointer.h"

template<typename T>
struct cpu_matrix
{
	std::uint64_t rows;
	std::uint64_t cols;

	host_ptr<T[]> data;
};

template<typename T>
struct gpu_matrix
{
	std::uint64_t rows;
	std::uint64_t cols;

	cuda_ptr<T[]> data;
};

template<typename T>
cpu_matrix<T> make_cpu_matrix(std::uint64_t rows, std::uint64_t cols);

template<typename T>
gpu_matrix<T> make_gpu_matrix(std::uint64_t rows, std::uint64_t cols);

template<typename T>
gpu_matrix<T> to_gpu(cpu_matrix<T> const& mat);

template<typename T>
cpu_matrix<T> to_cpu(gpu_matrix<T> const& mat);
