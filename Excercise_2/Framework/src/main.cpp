
#include "pointer.h"
#include "matrix.h"
#include "matrixops.h"

#include <array>

void print_matrix(cpu_matrix<float> const& m)
{
	for(std::uint64_t r = 0; r < m.rows; ++r)
	{
		std::printf(" (");
		for(std::uint64_t c = 0; c < m.cols; ++c)
			std::printf("%8.5f ", m.data[r * m.cols + c]);

		std::printf(")\n");
	}
	std::printf("\n");
}

template<bool useOptim>
cpu_matrix<float> test_scale(cpu_matrix<float> const& src, cpu_matrix<float> const& ref, float a)
{
	auto gpu_vectors = to_gpu(src);
	auto err = cudaGetLastError();
	if(err)
		std::fprintf(stderr, "Error copying matrix: '%s'\n", cudaGetErrorString(err));

	auto gpu_vectors_transformed = make_gpu_matrix<float>(src.rows, src.cols);
	err = cudaGetLastError();
	if(err)
		std::fprintf(stderr, "Error allocating matrix: '%s'\n", cudaGetErrorString(err));

	if(useOptim)
		scale_vectors_optimized(gpu_vectors_transformed, gpu_vectors, a);
	else
		scale_vectors(gpu_vectors_transformed, gpu_vectors, a);

	err = cudaGetLastError();
	if(err)
		std::fprintf(stderr, "Error executing scale_vectors: '%s'\n", cudaGetErrorString(err));

	auto cpu_vectors_transformed = to_cpu(gpu_vectors_transformed);
	std::printf("test_scale<%s> Avg error: %e\n", (useOptim ? "true" : "false"), compare_matrices(ref, cpu_vectors_transformed));
	return cpu_vectors_transformed;
}

template<bool useOptim>
cpu_matrix<float> test_matrix_transpose(cpu_matrix<float> const& src, cpu_matrix<float> const& ref)
{
	auto gpu_matrix = to_gpu(src);
	auto err = cudaGetLastError();
	if(err)
		std::fprintf(stderr, "Error copying matrix: '%s'\n", cudaGetErrorString(err));

	auto gpu_matrix_transposed = make_gpu_matrix<float>(src.cols, src.rows);
	err = cudaGetLastError();
	if(err)
		std::fprintf(stderr, "Error allocating matrix: '%s'\n", cudaGetErrorString(err));

	if(useOptim)
		transpose_matrix_optimized(gpu_matrix_transposed, gpu_matrix);
	else
		transpose_matrix(gpu_matrix_transposed, gpu_matrix);
		
	err = cudaGetLastError();
	if(err)
		std::fprintf(stderr, "Error executing transpose_matrix: '%s'\n", cudaGetErrorString(err));

	auto cpu_matrix_transposed = to_cpu(gpu_matrix_transposed);
	std::printf("test_matrix_transpose<%s> Avg error: %e\n", (useOptim ? "true" : "false"), compare_matrices(ref, cpu_matrix_transposed));
	return cpu_matrix_transposed;
}

template<bool useOptim>
cpu_matrix<float> test_matrix_vector_product(cpu_matrix<float> const& m, cpu_matrix<float> const& v, cpu_matrix<float> const& ref)
{
	auto gpu_matrix = to_gpu(m);
	auto err = cudaGetLastError();
	if(err)
		std::fprintf(stderr, "Error copying matrix: '%s'\n", cudaGetErrorString(err));

	auto gpu_vector = to_gpu(v);
	err = cudaGetLastError();
	if(err)
		std::fprintf(stderr, "Error copying vector: '%s'\n", cudaGetErrorString(err));

	auto gpu_result = make_gpu_matrix<float>(m.rows, 1);
	err = cudaGetLastError();
	if(err)
		std::fprintf(stderr, "Error allocating matrix: '%s'\n", cudaGetErrorString(err));

	if(useOptim)
		compute_matrix_vector_product_optimized(gpu_result, gpu_matrix, gpu_vector);
	else
		compute_matrix_vector_product(gpu_result, gpu_matrix, gpu_vector);

	err = cudaGetLastError();
	if(err)
		std::fprintf(stderr, "Error executing compute_matrix_vector_product: '%s'\n", cudaGetErrorString(err));

	auto cpu_result = to_cpu(gpu_result);
	std::printf("test_matrix_vector_product<%s> Avg error: %e\n", (useOptim ? "true" : "false"), compare_matrices(ref, cpu_result));
	return cpu_result;
}

template<bool useOptim>
cpu_matrix<float> test_cwise(cpu_matrix<float> const& src, cpu_matrix<float> const& ref)
{
	auto gpu_vectors = to_gpu(src);
	auto err = cudaGetLastError();
	if(err)
		std::fprintf(stderr, "Error copying matrix: '%s'\n", cudaGetErrorString(err));

	auto gpu_vectors_transformed = make_gpu_matrix<float>(src.rows, src.cols);
	err = cudaGetLastError();
	if(err)
		std::fprintf(stderr, "Error allocating matrix: '%s'\n", cudaGetErrorString(err));

	if(useOptim)
		cwise_op_vectors_optimized(gpu_vectors_transformed, gpu_vectors);
	else
		cwise_op_vectors(gpu_vectors_transformed, gpu_vectors);

	err = cudaGetLastError();
	if(err)
		std::fprintf(stderr, "Error executing scale_vectors: '%s'\n", cudaGetErrorString(err));

	auto cpu_vectors_transformed = to_cpu(gpu_vectors_transformed);
	std::printf("test_cwise<%s> Avg error: %e\n", (useOptim ? "true" : "false"), compare_matrices(ref, cpu_vectors_transformed));
	return cpu_vectors_transformed;
}

int main(int argc, char ** argv)
{
	std::uint64_t vecsize = 32*1024*1024;

	std::uint64_t numrows = 16*1024;
	std::uint64_t numcols = 16*1024;

	std::array<bool, 4> task{ false, false, false, false };

	bool runUnopt = false;
	bool runOpt = false;

	bool verbose = false;

	for(int i = 1; i < argc; ++i)
	{
		std::string arg = argv[i];

		if(arg == "-v")
			verbose = true;
		else if(arg == "-u")
			runUnopt = true;
		else if(arg == "-o")
			runOpt = true;
		else if(arg == "-t1")
			task[0] = true;
		else if(arg == "-t2")
			task[1] = true;
		else if(arg == "-t3")
			task[2] = true;
		else if(arg == "-t4")
			task[3] = true;
		else if(arg == "-s")
		{
			vecsize = 8;
			numrows = 8;
			numcols = 8;
		}
		else if(arg == "-m")
		{
			vecsize = 64;
			numrows = 64;
			numcols = 64;
		}
	}

	if(!runOpt && !runUnopt)
	{
		runOpt = true;
		runUnopt = true;
	}

	if(!task[0] && !task[1] && !task[2] && !task[3])
		task = { true, true, true, true };

	if(task[0])
	{
		auto vecs = generate_vectors(vecsize);
		auto ref = make_cpu_matrix<float>(vecs.rows, vecs.cols);
		if(verbose)
			print_matrix(vecs);
		scale_vectors(ref, vecs, 1.5f);
		if(verbose)
			print_matrix(ref);
		if(runUnopt)
		{
			auto v1 = test_scale<false>(vecs, ref, 1.5f);
			if(verbose)
				print_matrix(v1);
		}
		if(runOpt)
		{
			auto v2 = test_scale<true>(vecs, ref, 1.5f);
			if(verbose)
				print_matrix(v2);
		}
	}

	if(task[1])
	{
		auto mat = generate_matrix(numrows, numcols);
		auto ref = make_cpu_matrix<float>(mat.cols, mat.rows);
		if(verbose)
			print_matrix(mat);
		transpose_matrix(ref, mat);
		if(verbose)
			print_matrix(ref);
		if(runUnopt)
		{
			auto v1 = test_matrix_transpose<false>(mat, ref);
			if(verbose)
				print_matrix(v1);
		}
		if(runOpt)
		{
			auto v2 = test_matrix_transpose<true>(mat, ref);
			if(verbose)
				print_matrix(v2);
		}
	}

	if(task[2])
	{
		/*
		numrows = 10; numcols = 10;
		verbose = true;
		*/
		
		auto mat = generate_matrix(numrows, numcols);
		auto vec = generate_vector(mat.cols);
		auto ref = make_cpu_matrix<float>(mat.rows, 1);

		if(verbose)
			print_matrix(mat);
		if(verbose)
			print_matrix(vec);

		compute_matrix_vector_product(ref, mat, vec);
		if(verbose)
			print_matrix(ref);
		if(runUnopt)
		{
			auto v1 = test_matrix_vector_product<false>(mat, vec, ref);
			if(verbose)
				print_matrix(v1);
		}
		if(runOpt)
		{
			auto v2 = test_matrix_vector_product<true>(mat, vec, ref);
			if(verbose)
				print_matrix(v2);
		}
	}

	if(task[3])
	{
		auto vecs = generate_matrix(vecsize, 4);
		auto ref = make_cpu_matrix<float>(vecs.rows, dims_per_vec);
		if(verbose)
			print_matrix(vecs);
		cwise_op_vectors(ref, vecs);
		if(verbose)
			print_matrix(ref);
		if(runUnopt)
		{
			auto v1 = test_cwise<false>(vecs, ref);
			if(verbose)
				print_matrix(v1);
		}
		if(runOpt)
		{
			auto v2 = test_cwise<true>(vecs, ref);
			if(verbose)
				print_matrix(v2);
		}
	}

	return 0;
}
