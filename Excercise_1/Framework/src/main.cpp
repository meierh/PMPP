
#include "image.h"
#include "pointer.h"
#include "filtering.h"

std::string basePath = "/media/helge/Seagate4TB/Uni/Semester_23/PMPP/PMPP/Excercise_1/Framework/";

gpu_image create_grayscale(gpu_image const& src)
{
	auto gray_gpu = make_gpu_image(src.width, src.height);
	auto err = cudaGetLastError();
	if(err)
		std::fprintf(stderr, "Error allocating image: '%s'\n", cudaGetErrorString(err));

	to_grayscale(gray_gpu, src);
	err = cudaGetLastError();
	if(err)
		std::fprintf(stderr, "Error executing grayscale kernel: '%s'\n", cudaGetErrorString(err));

	{
		auto gray_cpu = to_cpu(gray_gpu);
		save(basePath+"out/cornell_grayscale.ppm", gray_cpu);
	}
	return gray_gpu;
}

gpu_image create_edgedetect(gpu_image const& src, bool horizontal)
{
	auto filter_cpu = make_cpu_filter(3, 3);
	if(horizontal)
	{
		filter_cpu.data[0] =  1.f;
		filter_cpu.data[1] =  0.f;
		filter_cpu.data[2] = -1.f;
		filter_cpu.data[3] =  2.f;
		filter_cpu.data[4] =  0.f;
		filter_cpu.data[5] = -2.f;
		filter_cpu.data[6] =  1.f;
		filter_cpu.data[7] =  0.f;
		filter_cpu.data[8] = -1.f;
	}
	else
	{
		filter_cpu.data[0] =  1.f;
		filter_cpu.data[1] =  2.f;
		filter_cpu.data[2] =  1.f;
		filter_cpu.data[3] =  0.f;
		filter_cpu.data[4] =  0.f;
		filter_cpu.data[5] =  0.f;
		filter_cpu.data[6] = -1.f;
		filter_cpu.data[7] = -2.f;
		filter_cpu.data[8] = -1.f;
	}

	auto filter_gpu = to_gpu(filter_cpu);
	auto err = cudaGetLastError();
	if(err)
		std::fprintf(stderr, "Error copying image: '%s'\n", cudaGetErrorString(err));

	auto filtered_gpu = make_gpu_image(src.width, src.height);
	err = cudaGetLastError();
	if(err)
		std::fprintf(stderr, "Error allocating image: '%s'\n", cudaGetErrorString(err));

	apply_convolution(filtered_gpu, src, filter_gpu, true);
	err = cudaGetLastError();
	if(err)
		std::fprintf(stderr, "Error executing convolution kernel: '%s'\n", cudaGetErrorString(err));

	{
		auto filtered_cpu = to_cpu(filtered_gpu);
		if(horizontal)
			save(basePath+"out/cornell_filtered_h.ppm", filtered_cpu);
		else
			save(basePath+"out/cornell_filtered_v.ppm", filtered_cpu);
	}

	return filtered_gpu;
}

template<int num_threads, int num_bins>
gpu_image create_histogram(gpu_image const& img)
{
	auto hist_gpu = make_gpu_matrix<std::uint32_t>(num_bins, 1);
	auto err = cudaGetLastError();
	if(err)
		std::fprintf(stderr, "Error allocating histogram: '%s'\n", cudaGetErrorString(err));

	auto plot_gpu = make_gpu_image(1024, 1024);
	err = cudaGetLastError();
	if(err)
		std::fprintf(stderr, "Error allocating image: '%s'\n", cudaGetErrorString(err));

	compute_histogram<num_threads, num_bins>(hist_gpu, img);
	err = cudaGetLastError();
	if(err)
		std::fprintf(stderr, "Error computing histogram: '%s'\n", cudaGetErrorString(err));

	draw_histogram<num_bins>(plot_gpu, hist_gpu, 1200);
	err = cudaGetLastError();
	if(err)
		std::fprintf(stderr, "Error drawing histogram: '%s'\n", cudaGetErrorString(err));

	{
		auto plot_cpu = to_cpu(plot_gpu);
		save(basePath+"out/cornell_hist.ppm", plot_cpu);
	}

#if 0
	auto hist_cpu = to_cpu(hist_gpu);
	std::uint64_t sum = 0;
	for(int b = 0; b < hist_cpu.width; ++b)
	{
		std::printf("%d %d\n", b, hist_cpu.data[b]);
		sum += hist_cpu.data[b];
	}
	std::printf("Total histogram pixels: %zu\n", sum);
#endif
	return plot_gpu;
}

int main()
{
	cpu_image base_img = load(basePath+"img/Cornell_Box_with_3_balls_of_different_materials.ppm");
	save(basePath+"out/cornell_unchanged.ppm", base_img);

	auto base_gpu = to_gpu(base_img);
	auto gray_gpu = create_grayscale(base_gpu);

	auto filter_h_cpu = make_cpu_filter(3, 3);
	
	auto filtered_h_gpu = create_edgedetect(gray_gpu, true);
	auto filtered_v_gpu = create_edgedetect(gray_gpu, false);

	auto plot_gpu = create_histogram<64, 32>(base_gpu);

	return 0;
}
