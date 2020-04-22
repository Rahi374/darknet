#include "dark_cuda.h"
#include "yolo_v2_class.hpp"

#include <stdio.h>

__global__ void mat_to_image_resize_gpu(float *dst, int dst_w, int dst_h, unsigned char *src, int src_w, int src_h, int src_step)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;

	int sx = (double)(idx + 0.5) * ((double)src_w / (double)dst_w);
	int sy = (double)(idy + 0.5) * ((double)src_h / (double)dst_h);

	/* subsample, convert RGB to BGR, and convert cv::Mat to image_t */
	if (idx < dst_w || idy < dst_h) {
		dst[0 * dst_w * dst_h + idy * dst_w + idx] = src[sy * src_step + 3 * sx + 2] / 255.0f;
		dst[1 * dst_w * dst_h + idy * dst_w + idx] = src[sy * src_step + 3 * sx + 1] / 255.0f;
		dst[2 * dst_w * dst_h + idy * dst_w + idx] = src[sy * src_step + 3 * sx + 0] / 255.0f;
	}
}

int cuda_mat_to_image_resize(image_t *dst, int dst_w, int dst_h,
			     unsigned char *src, int src_w, int src_h, int src_step)
{
	unsigned char *d_src;
	float *d_dst;

	unsigned int src_size = src_h * src_step * sizeof(unsigned char);
	unsigned int dst_size = 3 * dst_w * dst_h * sizeof(float);

	CHECK_CUDA(cudaMalloc((unsigned char**)&d_src, src_size));
	CHECK_CUDA(cudaMalloc((float**)&d_dst, dst_size));

	dim3 dimGrid(ceil(dst_w/(float)32), ceil(dst_h/(float)32));
	dim3 dimBlock(32, 32);

	CHECK_CUDA(cudaMemcpy(d_src, src, src_size, cudaMemcpyHostToDevice));
	mat_to_image_resize_gpu<<<dimGrid, dimBlock>>>(d_dst, dst_w, dst_h, d_src, src_w, src_h, src_step);
	CHECK_CUDA(cudaGetLastError());

	cudaDeviceSynchronize();

	CHECK_CUDA(cudaMemcpy(dst->data, d_dst, dst_size, cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaFree(d_src));
	CHECK_CUDA(cudaFree(d_dst));

	return 0;
}
