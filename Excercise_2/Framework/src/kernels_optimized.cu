#include "matrixops.h"

__global__ void scale_vectors_kernel_optimized(
	float* dst_data,
	float const* src_data,
	std::uint64_t num_vecs, std::uint64_t vec_dims,
	float a
)
{
	std::uint64_t offset = blockDim.x*blockIdx.x*vec_dims;
	std::uint64_t maxLen = num_vecs*vec_dims;
	for(std::uint64_t d=0; d<vec_dims; d++)
	{
		std::uint64_t ind = threadIdx.x + d*blockDim.x + offset;
		if(ind<maxLen)
			dst_data[ind] = src_data[ind]*a;
	}
}

__global__ void transpose_matrix_kernel_optimized(
	float* dst_data,
	float const* src_data,
	std::uint64_t rows, std::uint64_t cols
)
{
	__shared__ float tile[tilesize+1][tilesize];

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
	__shared__ float matrixBuffer[tilesize][tilesize];
	float acc = 0;
	
	std::uint64_t r_offset = tilesize * blockIdx.x;
		
	for(std::uint64_t c_offset=0; c_offset<cols; c_offset+=tilesize)
	{
		for(std::uint64_t t_r=0; t_r<tilesize; t_r++)
		{
			std::uint64_t r = r_offset+t_r;
			if(r<rows)
			{
				std::uint64_t c = c_offset+threadIdx.x;
				if(c<cols)
				{
					const float m_data_one = m_data[r*cols+c];
					matrixBuffer[t_r][threadIdx.x] = m_data_one;
				}
			}
		}
		__syncthreads();

		for(std::uint64_t t=0; t<tilesize; t++)
		{
			std::uint64_t c = c_offset+t;
			if(c<cols)
			{
				acc += matrixBuffer[threadIdx.x][t] * v_data[c];
			}
		}
	}
	if(r_offset+threadIdx.x<rows)
	{
		dst_data[r_offset+threadIdx.x] = acc;
	}
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
	if(vec_idx < num_vecs)
	{
		float4 src_vector = src_data_vec[vec_idx];
		float4 dst_vector;
		
		dst_vector.x = asinf(cosf(src_vector.x-src_vector.y));
		dst_vector.y = acosf(sinf(src_vector.y+src_vector.x));
		dst_vector.z = asinf(cosf(src_vector.z*src_vector.w));
		dst_vector.w = acosf(sinf(src_vector.w*src_vector.w));
		
		dst_data_vec[vec_idx] = dst_vector;
	}
}
