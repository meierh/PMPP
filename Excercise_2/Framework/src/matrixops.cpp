#include "matrixops.h"

#include <cmath>
#include <limits>

float make_coeff(int x, int a, int b)
{
	int nums[] = {
		(x + 1) * a,
		(x + 2) * b,
		(x + 3) * 3,
	};
	return static_cast<float>(((nums[0] << 1) ^ (nums[1] << 3) ^ (nums[2] << 4) ^ x ^ (nums[0] >> 2) ^ (nums[1] >> 4) ^ (nums[2] >> 1)) % 255) / 127.f - 1.f;
}

cpu_matrix<float> generate_vectors(std::uint64_t num_vecs)
{
	cpu_matrix<float> m = make_cpu_matrix<float>(num_vecs, 4);

	auto normalize_vec = [](float& x, float& y, float& z) {
		auto n = std::sqrt(x * x + y * y + z * z);

		if(n <= std::numeric_limits<float>::epsilon())
		{
			x = 0.f;
			y = 0.f;
			z = 0.f;
			return;
		}

		x /= n;
		y /= n;
		z /= n;
	};

	for(std::uint64_t r = 0; r < m.rows; ++r)
	{
		float x, y, z, w;

		x = make_coeff(r, 17, 5);
		y = make_coeff(r, 13, 7);
		z = make_coeff(r, 19, 5);
		w = 1.f;

		normalize_vec(x, y, z);

		m.data[r * m.cols + 0] = x;
		m.data[r * m.cols + 1] = y;
		m.data[r * m.cols + 2] = z;
		m.data[r * m.cols + 3] = w;
	}

	return m;
}

cpu_matrix<float> generate_vector(std::uint64_t rows)
{
	cpu_matrix<float> m = make_cpu_matrix<float>(rows, 1);

	for(std::uint64_t r = 0; r < m.rows; ++r)
	{
		m.data[r] = make_coeff(r, 17, 19);
	}

	return m;
}

cpu_matrix<float> generate_matrix(std::uint64_t rows, std::uint64_t cols)
{
	cpu_matrix<float> m = make_cpu_matrix<float>(rows, cols);

	for(std::uint64_t r = 0; r < m.rows; ++r)
	{
		for(std::uint64_t c = 0; c < m.cols; ++c)
			m.data[r * m.cols + c] = make_coeff(c * 13 + r * 17, 23, 7);
	}

	return m;
}

double compare_matrices(cpu_matrix<float> const& m1, cpu_matrix<float> const& m2)
{
	double err = 0.;

	for(std::uint64_t r = 0; r < m1.rows; ++r)
	{
		for(std::uint64_t c = 0; c < m1.cols; ++c)
			err += std::abs(m1.data[r * m1.cols + c] - m2.data[r * m2.cols + c]);
	}

	return (err / (m1.rows * m1.cols));
}

void scale_vectors(cpu_matrix<float>& dst, cpu_matrix<float> const& src, float a)
{
	for(std::uint64_t v = 0; v < src.rows; ++v)
	{
		for(std::uint64_t i = 0; i < src.cols; ++i)
			dst.data[v * src.cols + i] = src.data[v * src.cols + i] * a;
	}
}

void transpose_matrix(cpu_matrix<float>& dst, cpu_matrix<float> const& src)
{
	for(std::uint64_t r = 0; r < src.rows; ++r)
	{
		for(std::uint64_t c = 0; c < src.cols; ++c)
			dst.data[c * src.rows + r] = src.data[r * src.cols + c];
	}
}

void compute_matrix_vector_product(cpu_matrix<float>& dst, cpu_matrix<float> const& m, cpu_matrix<float> const& v)
{
	for(std::uint64_t r = 0; r < m.rows; ++r)
	{
		float acc = 0.f;
		for(std::uint64_t c = 0; c < m.cols; ++c)
			acc += m.data[r * m.cols + c] * v.data[c];

		dst.data[r] = acc;
	}
}

void cwise_op_vectors(cpu_matrix<float>& dst, cpu_matrix<float> const& src)
{
	for(std::uint64_t v = 0; v < src.rows; ++v)
	{
		dst.data[v * src.cols + 0] = std::asin(cos(src.data[v * src.cols + 0] - src.data[v * src.cols + 1]));
		dst.data[v * src.cols + 1] = std::acos(sin(src.data[v * src.cols + 1] + src.data[v * src.cols + 0]));
		dst.data[v * src.cols + 2] = std::asin(cos(src.data[v * src.cols + 2] * src.data[v * src.cols + 3]));
		dst.data[v * src.cols + 3] = std::acos(sin(src.data[v * src.cols + 3] * src.data[v * src.cols + 3]));
	}
}