#pragma once

#include "image.h"

void to_grayscale(gpu_image& dst, gpu_image const& src);

void apply_convolution(gpu_image& dst, gpu_image const& src, gpu_filter const& filter, bool use_abs_value = false);

template<int num_threads, int num_bins>
void compute_histogram(gpu_matrix<std::uint32_t>& hist, gpu_image const& img);

template<int num_bins>
void draw_histogram(gpu_image const& img, gpu_matrix<std::uint32_t>& hist, std::uint32_t scale);