#include "matrixops.h"

unsigned int compute_dim(std::uint64_t global_size, int block_size)
{
	return static_cast<unsigned int>((global_size / block_size) + (global_size % block_size > 0 ? 1 : 0));
}

void scale_vectors(gpu_matrix<float>& dst, gpu_matrix<float> const& src, float a)
{
	dim3 block_size = { 64 };
	dim3 grid_size = { compute_dim(src.rows, block_size.x) };
	scale_vectors_kernel<<<grid_size, block_size>>>(dst.data.get(), src.data.get(), src.rows, src.cols, a);
}

void scale_vectors_optimized(gpu_matrix<float>& dst, gpu_matrix<float> const& src, float a)
{
	dim3 block_size = { 256 };
	dim3 grid_size = { compute_dim(src.rows*src.cols, block_size.x) };
	scale_vectors_kernel_optimized<<<grid_size, block_size>>>(dst.data.get(), src.data.get(), src.rows, src.cols, a);
}

void transpose_matrix(gpu_matrix<float>& dst, gpu_matrix<float> const& src)
{
	const dim3 block_size = { tilesize, blockrows };
	dim3 grid_size = { compute_dim(src.cols, block_size.x), compute_dim(src.rows, block_size.x) };
	transpose_matrix_kernel<<<grid_size, block_size>>>(dst.data.get(), src.data.get(), src.rows, src.cols);
}

void transpose_matrix_optimized(gpu_matrix<float>& dst, gpu_matrix<float> const& src)
{
	const dim3 block_size = { tilesize, blockrows };
	dim3 grid_size = { compute_dim(src.cols, block_size.x), compute_dim(src.rows, block_size.x) };
	transpose_matrix_kernel_optimized<<<grid_size, block_size>>>(dst.data.get(), src.data.get(), src.rows, src.cols);
}

void compute_matrix_vector_product(gpu_matrix<float>& dst, gpu_matrix<float> const& m, gpu_matrix<float> const& v)
{
	const dim3 block_size = { tilesize_matrix_vec };
	dim3 grid_size = { compute_dim(m.rows, block_size.x) };
	compute_matrix_vector_product_kernel<<<grid_size, block_size>>>(dst.data.get(), m.data.get(), m.rows, m.cols, v.data.get());
}

#include <iostream>
void compute_matrix_vector_product_optimized(gpu_matrix<float>& dst, gpu_matrix<float> const& m, gpu_matrix<float> const& v)
{
	const dim3 block_size = {tilesizeX_matrix_vec, tilesizeY_matrix_vec};
	dim3 grid_size = {compute_dim(m.cols,block_size.x), compute_dim(m.rows,block_size.y)};
	std::cout<<"m dims cols:"<<m.cols<<" rows:"<<m.rows<<std::endl;
	std::cout<<"block_size ("<<block_size.x<<","<<block_size.y<<")"<<std::endl;
	std::cout<<"grid_size ("<<grid_size.x<<","<<grid_size.y<<")"<<std::endl;
	cudaMemset(dst.data.get(),0,m.rows*sizeof(float));
	compute_matrix_vector_product_kernel_optimized<<<grid_size, block_size>>>(dst.data.get(), m.data.get(), m.rows, m.cols, v.data.get());
}

void cwise_op_vectors(gpu_matrix<float>& dst, gpu_matrix<float> const& src)
{
	dim3 block_size = { vecs_per_block * dims_per_vec };
	dim3 grid_size = { compute_dim(src.rows * src.cols, block_size.x) };
	cwise_op_vectors_kernel<<<grid_size, block_size>>>(dst.data.get(), src.data.get(), src.rows);
}

void cwise_op_vectors_optimized(gpu_matrix<float>& dst, gpu_matrix<float> const& src)
{
	dim3 block_size = { 1024 };
	dim3 grid_size = { compute_dim(src.rows * src.cols, block_size.x) };
	cwise_op_vectors_kernel_optimized<<<grid_size, block_size>>>(dst.data.get(), src.data.get(), src.rows);
}
