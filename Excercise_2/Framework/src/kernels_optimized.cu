#include "matrixops.h"

__global__ void scale_vectors_kernel_optimized(
	float* dst_data,
	float const* src_data,
	std::uint64_t num_vecs, std::uint64_t vec_dims,
	float a
)
{
	auto ind = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(ind >= num_vecs*vec_dims)
		return;
	
	dst_data[ind] = src_data[ind]*a;
}

__global__ void transpose_matrix_kernel_optimized(
	float* dst_data,
	float const* src_data,
	std::uint64_t rows, std::uint64_t cols
)
{
	__shared__ float tile[tilesize][tilesize+1];

	auto c = tilesize * blockIdx.x + threadIdx.x;
	auto r = tilesize * blockIdx.y + threadIdx.y;

	for(int j=0; j<tilesize; j+=blockrows)
	{
		if((r + j) < rows && c < cols)
			tile[threadIdx.y+j][threadIdx.x] = src_data[(r+j)*cols+c];
	}
	__syncthreads();

	auto nc = tilesize * blockIdx.y + threadIdx.x;
	auto nr = tilesize * blockIdx.x + threadIdx.y;

	for(int j = 0; j < tilesize; j += blockrows)
	{
		if(nc < rows && (nr + j) < cols)
			dst_data[(nr+j) * rows + nc] = tile[threadIdx.x][threadIdx.y + j];
	}
}

__global__ void compute_matrix_vector_product_kernel_optimized(
	float* dst_data,
	float const* m_data,
	std::uint64_t rows, std::uint64_t cols,
	float const* v_data
)
{
	__shared__ float matrixBuffer[tilesizeY_matrix_vec][tilesizeX_matrix_vec];
	matrixBuffer[threadIdx.y][threadIdx.x] = 0;
	__shared__ float vectorBuffer[tilesizeX_matrix_vec];
	if(threadIdx.x==0)
		vectorBuffer[threadIdx.y] = 0;
	__syncthreads();
	
	std::uint64_t c_offset = tilesizeX_matrix_vec * blockIdx.x;	
	std::uint64_t c = threadIdx.x+c_offset;
	std::uint64_t r_offset = tilesizeY_matrix_vec * blockIdx.y;
	std::uint64_t r = threadIdx.y+r_offset;
	
	if(c<cols && r<rows)
	{
		matrixBuffer[threadIdx.y][threadIdx.x] = m_data[r*cols+c];
		if(threadIdx.y==0)
			vectorBuffer[threadIdx.x] = v_data[c];
	}
	__syncthreads();
	
	matrixBuffer[threadIdx.y][threadIdx.x] *= vectorBuffer[threadIdx.x];
	__syncthreads();
	
	for(int stride=tilesizeX_matrix_vec/2; stride>0; stride/=2)
	{
		if(threadIdx.x<stride)
		{
			matrixBuffer[threadIdx.y][threadIdx.x] += matrixBuffer[threadIdx.y][threadIdx.x+stride];
		}
		__syncthreads();
	}
	
	if(threadIdx.x==0 && r<rows)
		atomicAdd(dst_data+r,matrixBuffer[threadIdx.y][0]);
}

__global__ void cwise_op_vectors_kernel_optimized(
	float* dst_data,
	float const* src_data,
	std::uint64_t num_vecs
)
{
	float4* dst_data_vec = (float4*) dst_data;
	float4 const* src_data_vec = (float4 const*) src_data;

	auto vec_idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(vec_idx >= num_vecs)
		return;

	float4 src_vector = src_data_vec[vec_idx];
	float4 dst_vector = src_vector;
	
	dst_vector.x = asinf(cosf(src_vector.x-src_vector.y));
	dst_vector.y = acosf(sinf(src_vector.y+src_vector.x));
	dst_vector.z = asinf(cosf(src_vector.z*src_vector.w));
	dst_vector.w = acosf(sinf(src_vector.w*src_vector.w));
	
	dst_data_vec[vec_idx] = dst_vector;
}
