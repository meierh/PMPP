#include "matrix.h"

#include <fstream>
#include <limits>
#include <cuda_runtime.h>

template<typename T>
cpu_matrix<T> make_cpu_matrix(std::uint64_t rows, std::uint64_t cols)
{
	cpu_matrix<T> mat;
	mat.rows = rows;
	mat.cols = cols;
	mat.data = make_host_array<T>(rows * cols);
	return mat;
}

template cpu_matrix<float> make_cpu_matrix(std::uint64_t rows, std::uint64_t cols);


template<typename T>
gpu_matrix<T> make_gpu_matrix(std::uint64_t rows, std::uint64_t cols)
{
	gpu_matrix<T> mat;
	mat.rows = rows;
	mat.cols = cols;
	mat.data = make_managed_cuda_array<T>(rows * cols);
	return mat;
}

template gpu_matrix<float> make_gpu_matrix(std::uint64_t rows, std::uint64_t cols);


template<typename T>
gpu_matrix<T> to_gpu(cpu_matrix<T> const& mat)
{
	gpu_matrix<T> cpy;
	cpy.rows = mat.rows;
	cpy.cols = mat.cols;
	cpy.data = make_managed_cuda_array<T>(mat.rows * mat.cols);

	cudaMemcpy(cpy.data.get(), mat.data.get(), mat.rows * mat.cols * sizeof(T), cudaMemcpyHostToDevice);
	return cpy;
}

template gpu_matrix<float> to_gpu(cpu_matrix<float> const& mat);


template<typename T>
cpu_matrix<T> to_cpu(gpu_matrix<T> const& mat)
{
	cpu_matrix<T> cpy;
	cpy.rows = mat.rows;
	cpy.cols = mat.cols;
	cpy.data = make_host_array<T>(mat.rows * mat.cols);

	cudaMemcpy(cpy.data.get(), mat.data.get(), mat.rows * mat.cols * sizeof(T), cudaMemcpyDeviceToHost);
	return cpy;
}

template cpu_matrix<float> to_cpu(gpu_matrix<float> const& mat);
