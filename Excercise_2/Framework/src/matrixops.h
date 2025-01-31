#pragma once

#include "matrix.h"

cpu_matrix<float> generate_vectors(std::uint64_t num_vecs);

cpu_matrix<float> generate_vector(std::uint64_t num_vecs);
cpu_matrix<float> generate_matrix(std::uint64_t rows, std::uint64_t cols);

double compare_matrices(cpu_matrix<float> const& m1, cpu_matrix<float> const& m2);

void scale_vectors(cpu_matrix<float>& dst, cpu_matrix<float> const& src, float a);
void scale_vectors(gpu_matrix<float>& dst, gpu_matrix<float> const& src, float a);
void scale_vectors_optimized(gpu_matrix<float>& dst, gpu_matrix<float> const& src, float a);

void transpose_matrix(cpu_matrix<float>& dst, cpu_matrix<float> const& src);
void transpose_matrix(gpu_matrix<float>& dst, gpu_matrix<float> const& src);
void transpose_matrix_optimized(gpu_matrix<float>& dst, gpu_matrix<float> const& src);

void compute_matrix_vector_product(cpu_matrix<float>& dst, cpu_matrix<float> const& m, cpu_matrix<float> const& v);
void compute_matrix_vector_product(gpu_matrix<float>& dst, gpu_matrix<float> const& m, gpu_matrix<float> const& v);
void compute_matrix_vector_product_optimized(gpu_matrix<float>& dst, gpu_matrix<float> const& m, gpu_matrix<float> const& v);

void cwise_op_vectors(cpu_matrix<float>& dst, cpu_matrix<float> const& src);
void cwise_op_vectors(gpu_matrix<float>& dst, gpu_matrix<float> const& src);
void cwise_op_vectors_optimized(gpu_matrix<float>& dst, gpu_matrix<float> const& src);

__global__ void scale_vectors_kernel(
	float* dst_data,
	float const* src_data,
	std::uint64_t num_vecs, std::uint64_t vec_dims,
	float a
);
__global__ void scale_vectors_kernel_optimized(
	float* dst_data,
	float const* src_data,
	std::uint64_t num_vecs, std::uint64_t vec_dims,
	float a
);

constexpr int tilesize = 32;
constexpr int blockrows = 8;

__global__ void transpose_matrix_kernel(
	float* dst_data,
	float const* src_data,
	std::uint64_t rows, std::uint64_t cols
);
__global__ void transpose_matrix_kernel_optimized(
	float* dst_data,
	float const* src_data,
	std::uint64_t rows, std::uint64_t cols
);

constexpr int tilesize_matrix_vec = 32;
__global__ void compute_matrix_vector_product_kernel(
	float* dst_data,
	float const* src_data,
	std::uint64_t rows, std::uint64_t cols,
	float const* v_data
);

constexpr int tilesizeX_matrix_vec = 128;
constexpr int tilesizeY_matrix_vec = 1;
__global__ void compute_matrix_vector_product_kernel_optimized(
	float* dst_data,
	float const* src_data,
	std::uint64_t rows, std::uint64_t cols,
	float const* v_data
);

constexpr int vecs_per_block = 32;
constexpr int dims_per_vec = 4;

__global__ void cwise_op_vectors_kernel(
	float* dst_data,
	float const* src_data,
	std::uint64_t num_vecs
);
__global__ void cwise_op_vectors_kernel_optimized(
	float* dst_data,
	float const* src_data,
	std::uint64_t num_vecs
);
