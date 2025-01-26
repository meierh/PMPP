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
	auto vec_idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(vec_idx < num_vecs)
	{
		float src_vector[dims_per_vec], dst_vector[dims_per_vec];
		src_vector[0] = src_data[vec_idx*dims_per_vec+0];
		src_vector[1] = src_data[vec_idx*dims_per_vec+1];
		src_vector[2] = src_data[vec_idx*dims_per_vec+2];
		src_vector[3] = src_data[vec_idx*dims_per_vec+3];
		
		dst_vector[0] = asinf(cosf(src_vector[0]-src_vector[1]));
		dst_vector[1] = acosf(sinf(src_vector[1]+src_vector[0]));
		dst_vector[2] = asinf(cosf(src_vector[2]*src_vector[3]));
		dst_vector[3] = acosf(sinf(src_vector[3]*src_vector[3]));
		
		dst_data[vec_idx*dims_per_vec+0] = dst_vector[0];
		dst_data[vec_idx*dims_per_vec+1] = dst_vector[1];
		dst_data[vec_idx*dims_per_vec+2] = dst_vector[2];
		dst_data[vec_idx*dims_per_vec+3] = dst_vector[3];
	}
}
