#include "filtering.h"

unsigned int compute_dim(std::uint64_t global_size, int block_size)
{
	return static_cast<unsigned int>((global_size / block_size) + (global_size % block_size > 0 ? 1 : 0));
}


__global__ void gray_scale_kernel(
	std::uint32_t* dst_data,
	std::uint32_t* src_data, 
	std::uint64_t w, std::uint64_t h
)
{
	//TODO: 1.3) Implement conversion
	std::uint64_t xGlobal = blockIdx.x*blockDim.x+threadIdx.x;
	std::uint64_t yGlobal = blockIdx.y*blockDim.y+threadIdx.y;
	if(xGlobal<w && yGlobal<h)
	{
		std::uint64_t arrIdx = yGlobal*w+xGlobal;
		std::uint32_t pixel = src_data[arrIdx];
		unsigned char r = pixel & 0xff;
		unsigned char g = (pixel >> 8) & 0xff;
		unsigned char b = (pixel >> 16) & 0xff;
		unsigned char gray = 0.2126*r+0.7152*g+0.0722*b;
		r = g = b = gray;
		std::uint32_t r_ = r;
		pixel |= r_;
		std::uint32_t g_ = g;
		pixel |= g_ << 8;
		std::uint32_t b_ = b;
		pixel |= b_ << 16;
		dst_data[arrIdx] = pixel;
	}
}

void to_grayscale(gpu_image& dst, gpu_image const& src)
{
	dim3 block_size = { 32, 32 };
	dim3 grid_size = { compute_dim(src.width, block_size.x), compute_dim(src.height, block_size.y) };
	gray_scale_kernel<<<grid_size, block_size>>>(dst.data.get(), src.data.get(), src.width, src.height);
}


__global__ void convolution_kernel(
	std::uint32_t* dst_data,
	std::uint32_t* src_data, 
	std::uint64_t w, std::uint64_t h, 
	float* filter_data,
	std::uint64_t fw, std::uint64_t fh,
	bool use_abs_value
)
{
	//TODO: 1.4) Implement convolution
	std::uint64_t xGlobal = blockIdx.x*blockDim.x+threadIdx.x;
	std::uint64_t yGlobal = blockIdx.y*blockDim.y+threadIdx.y;
	if(xGlobal<w && yGlobal<h)
	{
		std::uint64_t lh = fh/2;
		std::uint64_t lw = fw/2;
		float sum_r = 0;
		float sum_g = 0;
		float sum_b = 0;
		std::uint64_t arrIdx = yGlobal*w+xGlobal;
		for(std::uint64_t fhi=0; fhi<fh; fhi++)
		{
			std::int64_t fyGlobal = (std::int64_t)yGlobal-lh+fhi;
			for(std::uint64_t fwi=0; fwi<fw; fwi++)
			{
				std::int64_t fxGlobal = (std::int64_t)xGlobal-lw+fwi;
				float filterVal = filter_data[fhi*fw+fwi];
				
				bool inImage = fxGlobal>=0 && fxGlobal<w && fyGlobal>=0 && fyGlobal<h;
				if(inImage)
				{
					std::uint64_t arrIdx = fyGlobal*w+fxGlobal;
					std::uint32_t pixel = src_data[arrIdx];
					unsigned char r = pixel & 0xff;
					unsigned char g = (pixel >> 8) & 0xff;
					unsigned char b = (pixel >> 16) & 0xff;
					std::uint32_t r_ = r;
					std::uint32_t g_ = g;
					std::uint32_t b_ = b;
					sum_r += r_*filterVal;
					sum_g += g_*filterVal;
					sum_b += b_*filterVal;
				}
			}
		}
		if(use_abs_value)
		{
			sum_r = std::abs(sum_r)/2;
			sum_g = std::abs(sum_g)/2;
			sum_b = std::abs(sum_b)/2;
		}
		std::uint32_t pixel = 0;
		std::uint32_t res_r = sum_r;
		std::uint32_t res_g = sum_g;
		std::uint32_t res_b = sum_b;
		pixel |= res_r;
		pixel |= res_g << 8;
		pixel |= res_b << 16;
		
		dst_data[arrIdx] = pixel;
	}
}

void apply_convolution(gpu_image& dst, gpu_image const& src, gpu_filter const& filter, bool use_abs_value)
{
	dim3 block_size = { 32, 32 };
	dim3 grid_size = { compute_dim(src.width, block_size.x), compute_dim(src.height, block_size.y) };
	convolution_kernel<<<grid_size, block_size>>>(dst.data.get(), src.data.get(), src.width, src.height, filter.data.get(), filter.width, filter.height, use_abs_value);
}


template<int num_threads, int num_bins>
__global__ void histogram_kernel(
	std::uint32_t* hist_data,
	std::uint32_t* img_data,
	std::uint64_t w, std::uint64_t h,
	std::uint8_t channel_flags = 1
)
{
	//TODO: 1.5) Implement histogram computation
	__shared__ std::uint32_t histo[num_bins][num_threads];
	constexpr unsigned char binSpan = 256/num_bins;
	for(int binI=0; binI<num_bins; binI++)
	{
		for(int threadI=0; threadI<num_threads; threadI++)
		{
			histo[binI][threadI] = 0;
		}
	}
	
	std::uint64_t totalLen = w*h;
	std::uint64_t blockSlice = totalLen/gridDim.x + 1;
	std::uint64_t totalOffset = blockIdx.x*blockSlice;
	for(std::uint64_t localI=threadIdx.x; localI<blockSlice; localI+=num_threads)
	{
		std::uint32_t pixel = img_data[localI+totalOffset];
		unsigned char c = pixel << (8*(channel_flags-1));
		uint channel = c;
		uint ind = channel/binSpan;
		histo[ind][threadIdx.x]++;
	}
	
	for(int binI=threadIdx.x; binI<num_bins; binI+=num_threads)
	{
		for(int threadI=1; threadI<num_threads; threadI++)
		{
			histo[binI][0] += histo[binI][threadI];
		}
	}
	
	for(int binI=threadIdx.x; binI<num_bins; binI+=num_threads)
	{
		atomicAdd(hist_data+binI,histo[0][binI]);
	}
}

template<int num_threads, int num_bins>
void compute_histogram(gpu_matrix<std::uint32_t>& hist, gpu_image const& img)
{
	dim3 block_size = { num_threads };
	dim3 grid_size = { compute_dim(img.width, block_size.x) };
	histogram_kernel<num_threads, num_bins><<<grid_size, block_size>>>(hist.data.get(), img.data.get(), img.width, img.height);
}

template void compute_histogram<64, 32>(gpu_matrix<std::uint32_t>& hist, gpu_image const& img);


template<int num_bins>
__global__ void draw_hist_kernel(
	std::uint32_t* img_data,
	std::uint32_t* hist_data,
	std::uint64_t w, std::uint64_t h,
	std::uint32_t scale
)
{
	auto x_index = blockDim.x * blockIdx.x + threadIdx.x;
	auto y_index = blockDim.y * blockIdx.y + threadIdx.y;

	if(x_index >= w)
		return;
	if(y_index >= h)
		return;

	auto invidx = h - y_index;
	if((hist_data[x_index / (w / num_bins)] / scale) > invidx)
		img_data[x_index + y_index * w] = (250) | (180 << 8) | (33 << 16);
	else
		img_data[x_index + y_index * w] = (50) | (50 << 8) | (50 << 16);
}

template<int num_bins>
void draw_histogram(gpu_image const& img, gpu_matrix<std::uint32_t>& hist, std::uint32_t scale)
{
	dim3 block_size = { 32, 32 };
	dim3 grid_size = { compute_dim(img.width, block_size.x), compute_dim(img.height, block_size.y) };
	draw_hist_kernel<32><<<grid_size, block_size>>>(img.data.get(), hist.data.get(), img.width, img.height, scale);
}

template void draw_histogram<32>(gpu_image const& img, gpu_matrix<std::uint32_t>& hist, std::uint32_t scale);
