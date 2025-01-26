#include "matrixops.h"

__global__ void scale_vectors_kernel(
	float* dst_data,
	float const* src_data,
	std::uint64_t num_vecs, std::uint64_t vec_dims,
	float a
)
{
	auto x_index = blockDim.x * blockIdx.x + threadIdx.x;

	if(x_index >= num_vecs)
		return;

	for(std::uint64_t i = 0; i < vec_dims; ++i)
		dst_data[x_index * vec_dims + i] = src_data[x_index * vec_dims + i] * a;
}

__global__ void transpose_matrix_kernel(
	float* dst_data,
	float const* src_data,
	std::uint64_t rows, std::uint64_t cols
)
{
	__shared__ float tile[tilesize][tilesize];

	auto c = tilesize * blockIdx.x + threadIdx.x;
	auto r = tilesize * blockIdx.y + threadIdx.y;

	auto lc = threadIdx.x;
	auto lr = threadIdx.y;

	for(int j = 0; j < tilesize; j += blockrows)
	{
		if((r + j) < rows && c < cols)
			tile[lr + j][lc] = src_data[(r + j) * cols + c];
	}

	__syncthreads();

	auto nc = tilesize * blockIdx.y + threadIdx.x;
	auto nr = tilesize * blockIdx.x + threadIdx.y;

	for(int j = 0; j < tilesize; j += blockrows)
	{
		if(nc < rows && (nr + j) < cols)
			dst_data[(nr + j) * rows + nc] = tile[lc][lr + j];
	}
}

__global__ void compute_matrix_vector_product_kernel(
	float* dst_data,
	float const* m_data,
	std::uint64_t rows, std::uint64_t cols,
	float const* v_data
)
{
	auto r = blockDim.x * blockIdx.x + threadIdx.x;

	if(r < rows)
	{
		float acc = 0.f;
		for(int c = 0; c < cols; ++c)
			acc += m_data[r * cols + c] * v_data[c];

		dst_data[r] = acc;
	}
}

__global__ void cwise_op_vectors_kernel(
	float* dst_data,
	float const* src_data,
	std::uint64_t num_vecs
)
{
	auto x_index = blockDim.x * blockIdx.x + threadIdx.x;
	auto vec_idx = x_index / dims_per_vec;

	if(vec_idx >= num_vecs)
		return;

	if(x_index % dims_per_vec == 0)
		dst_data[vec_idx * dims_per_vec + 0] = asinf(cosf(src_data[vec_idx * dims_per_vec + 0] - src_data[vec_idx * dims_per_vec + 1]));
	else if(x_index % dims_per_vec == 1)
		dst_data[vec_idx * dims_per_vec + 1] = acosf(sinf(src_data[vec_idx * dims_per_vec + 1] + src_data[vec_idx * dims_per_vec + 0]));
	else if(x_index % dims_per_vec == 2)
		dst_data[vec_idx * dims_per_vec + 2] = asinf(cosf(src_data[vec_idx * dims_per_vec + 2] * src_data[vec_idx * dims_per_vec + 3]));
	else if(x_index % dims_per_vec == 3)
		dst_data[vec_idx * dims_per_vec + 3] = acosf(sinf(src_data[vec_idx * dims_per_vec + 3] * src_data[vec_idx * dims_per_vec + 3]));
}